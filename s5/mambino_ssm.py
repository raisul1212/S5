"""MambinoSSM: drop-in replacement for S5SSM that adds Mambino's
proprioceptive predictor branch + additive-PC W_eps feedback path,
while keeping S5's parameter-efficient design (tiny complex state at
dim P, HiPPO init, bidirectional MAIN scan with shared B and split C).

Architecture (the additive_pc branch of NCB v031 SSMCore, plugged into
S5's framework with complex P-dim state):

  Predictor scan (FORWARD ONLY, complex diag at P):
    s(t) = Lambda_s_bar * s(t-1) + B_s_bar @ x(t)
    x_hat(t) = C_s @ s(t-1)             [shifted; uses past predictor state]

  ε(t) = x(t) - x_hat(t)                [raw difference at H, no RMSNorm]

  Main scan input (additive PC, NO M-gate):
    u(t) = B_bar @ x(t) + W_eps_bar @ ε(t)

  Main scan (BIDIRECTIONAL, complex diag at P):
    h_fwd, h_bwd both via S5's apply_ssm machinery with B_bar input
      replaced by u(t)-style sequences
    y = Re(C_fwd @ h_fwd + C_bwd @ h_bwd) + D * x

  Intrinsic loss:
    L_int = mean(||ε||^2)               [sown for logging; lambda_pc * L_int
                                          added to task loss in train step]

References to NCB v031 (frozen):
  - ssm_core.py:520-534  ("add-DSPC" design comment block, distinguishes
    additive PC from original DSPC; no M-gate, raw ε)
  - ssm_core.py:1654-1708 (the additive_pc=True compute path in
    _forward_parallel; u = x_tilde @ B^T + eps @ W_eps; M = 1)
  - ssm_core.py:1350-1367 (_maybe_bidirectional_scan; main scan only)
  - ssm_core.py:1614      (predictor scan parallel_scan_diag, no
    bidirectional wrapper -> predictor is forward only)

This faithfully ports the architecture that produced 0.4965 on LRA-ListOps
(SLURM 11143665, 2026-06-28).  Layer-scale and v0.6 selectivity gates
(eps_gate, input_gate, output_gate) are intentionally OMITTED to keep
parameter overhead vs vanilla S5 minimal -- they are orthogonal to the
core predictive-coding contribution and add d_C-resolution Dense layers
that do not translate to S5's P-dim state philosophy.

Param overhead vs S5 alone (per layer, complex P, with bidirectional main
+ forward-only predictor):
  Main SSM (Lambda, B, C_fwd, C_bwd, D, log_step): 6,296  [== S5SSM]
  Predictor branch (Lambda_s, B_s, C_s, log_step_s): 4,120  [NEW]
  W_eps additive PC matrix (P,H,2 complex):  2,048           [NEW]
  + S5's half_glu2 Dense + BN handled by SequenceLayer wrapper.

Total per layer (incl. S5 wrapper): 29,280  (vs S5 alone 23,064)
"""
from functools import partial
import jax
import jax.numpy as np
from jax.nn.initializers import lecun_normal, normal
from flax import linen as nn

from .ssm import (
    discretize_zoh,
    discretize_bilinear,
    binary_operator,
    apply_ssm,
)
from .ssm_init import init_VinvB, init_CV, init_log_steps, trunc_standard_normal


class MambinoSSM(nn.Module):
    """ S5SSM with an added Mambino predictor branch + W_eps additive PC path.

        Args (mirror S5SSM signature so it can drop into SequenceLayer):
            H, P: model + state dims (S5's d_model + ssm_size)
            Lambda_re_init, Lambda_im_init: per-block HiPPO eigenvalues (real, imag)
            V, Vinv: HiPPO conjugation matrices (used by B and C init)
            C_init: "lecun_normal" | "trunc_standard_normal" | "complex_normal"
            discretization: "zoh" | "bilinear"
            dt_min, dt_max: timescale init range
            conj_sym: whether to enforce conj-sym halving (S5 default True)
            clip_eigs: clip Lambda_re to (-inf, -1e-4) (S5 hyperparam)
            bidirectional: bidirectional MAIN scan (predictor stays forward-only)
            step_rescale: discretization step rescale (S5 hyperparam)
    """
    H: int
    P: int
    Lambda_re_init: np.ndarray
    Lambda_im_init: np.ndarray
    V: np.ndarray
    Vinv: np.ndarray
    C_init: str
    discretization: str
    dt_min: float
    dt_max: float
    conj_sym: bool
    clip_eigs: bool
    bidirectional: bool
    step_rescale: float = 1.0

    def setup(self):
        """Initialize main SSM parameters (identical to S5SSM) plus
        Mambino's predictor branch + W_eps additive feedback matrix.
        """
        # ── conj-sym handling ──
        if self.conj_sym:
            local_P = 2 * self.P
        else:
            local_P = self.P

        # ──────────────────────────────────────────────────────────────
        # MAIN SSM PARAMETERS (identical to S5SSM so BfastandCdecay
        # tagging picks them up automatically; same names so all
        # downstream S5 machinery applies unchanged)
        # ──────────────────────────────────────────────────────────────
        self.Lambda_re = self.param(
            "Lambda_re", lambda rng, shape: self.Lambda_re_init, (None,))
        self.Lambda_im = self.param(
            "Lambda_im", lambda rng, shape: self.Lambda_im_init, (None,))
        if self.clip_eigs:
            self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            self.Lambda = self.Lambda_re + 1j * self.Lambda_im

        # Main B matrix (input projection at H to state at P)
        B_init = lecun_normal()
        B_shape = (local_P, self.H)
        self.B = self.param(
            "B",
            lambda rng, shape: init_VinvB(B_init, rng, shape, self.Vinv),
            B_shape)
        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]

        # Main C readouts (S5 convention: C1 = forward, C2 = backward)
        if self.C_init == "trunc_standard_normal":
            C_init_fn = trunc_standard_normal
            C_shape = (self.H, local_P, 2)
        elif self.C_init == "lecun_normal":
            C_init_fn = lecun_normal()
            C_shape = (self.H, local_P, 2)
        elif self.C_init == "complex_normal":
            C_init_fn = normal(stddev=0.5 ** 0.5)
        else:
            raise NotImplementedError(
                f"C_init method {self.C_init!r} not implemented")

        if self.C_init == "complex_normal":
            if self.bidirectional:
                C_complex = self.param("C", C_init_fn, (self.H, 2 * self.P, 2))
                self.C_tilde = C_complex[..., 0] + 1j * C_complex[..., 1]
            else:
                C_complex = self.param("C", C_init_fn, (self.H, self.P, 2))
                self.C_tilde = C_complex[..., 0] + 1j * C_complex[..., 1]
        else:
            if self.bidirectional:
                self.C1 = self.param(
                    "C1",
                    lambda rng, shape: init_CV(C_init_fn, rng, shape, self.V),
                    C_shape)
                self.C2 = self.param(
                    "C2",
                    lambda rng, shape: init_CV(C_init_fn, rng, shape, self.V),
                    C_shape)
                C1 = self.C1[..., 0] + 1j * self.C1[..., 1]
                C2 = self.C2[..., 0] + 1j * self.C2[..., 1]
                self.C_tilde = np.concatenate((C1, C2), axis=-1)
            else:
                self.C = self.param(
                    "C",
                    lambda rng, shape: init_CV(C_init_fn, rng, shape, self.V),
                    C_shape)
                self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        # D feedthrough matrix (real, per H channel)
        self.D = self.param("D", normal(stddev=1.0), (self.H,))

        # Main discretization step
        self.log_step = self.param(
            "log_step", init_log_steps,
            (self.P, self.dt_min, self.dt_max))
        step = self.step_rescale * np.exp(self.log_step[:, 0])

        # Discretize main SSM
        if self.discretization == "zoh":
            self.Lambda_bar, self.B_bar = discretize_zoh(
                self.Lambda, B_tilde, step)
        elif self.discretization == "bilinear":
            self.Lambda_bar, self.B_bar = discretize_bilinear(
                self.Lambda, B_tilde, step)
        else:
            raise NotImplementedError(
                f"Discretization {self.discretization!r} not implemented")

        # ──────────────────────────────────────────────────────────────
        # PREDICTOR BRANCH (forward-only causal predictor)
        # Same HiPPO init as main scan; same shapes.
        # Name params with _s suffix so they get tagged correctly by
        # BfastandCdecay (extended in train_helpers.py to recognize
        # Lambda_s_re, Lambda_s_im, log_step_s as "ssm" group; B_s as
        # "none" group; C_s as "regular" group).
        # ──────────────────────────────────────────────────────────────
        self.Lambda_s_re = self.param(
            "Lambda_s_re", lambda rng, shape: self.Lambda_re_init, (None,))
        self.Lambda_s_im = self.param(
            "Lambda_s_im", lambda rng, shape: self.Lambda_im_init, (None,))
        if self.clip_eigs:
            self.Lambda_s = np.clip(self.Lambda_s_re, None, -1e-4) + 1j * self.Lambda_s_im
        else:
            self.Lambda_s = self.Lambda_s_re + 1j * self.Lambda_s_im

        self.B_s = self.param(
            "B_s",
            lambda rng, shape: init_VinvB(B_init, rng, shape, self.Vinv),
            B_shape)
        B_s_tilde = self.B_s[..., 0] + 1j * self.B_s[..., 1]

        # Predictor C: forward only (predictor is causal, predicts NEXT input)
        C_s_shape = (self.H, local_P, 2)
        self.C_s = self.param(
            "C_s",
            lambda rng, shape: init_CV(C_init_fn, rng, shape, self.V),
            C_s_shape)
        C_s_complex = self.C_s[..., 0] + 1j * self.C_s[..., 1]
        # No bidirectional for predictor -> use single C_s (no concat)
        self.C_s_tilde = C_s_complex

        # Predictor discretization step (separate from main)
        self.log_step_s = self.param(
            "log_step_s", init_log_steps,
            (self.P, self.dt_min, self.dt_max))
        step_s = self.step_rescale * np.exp(self.log_step_s[:, 0])

        if self.discretization == "zoh":
            self.Lambda_s_bar, self.B_s_bar = discretize_zoh(
                self.Lambda_s, B_s_tilde, step_s)
        elif self.discretization == "bilinear":
            self.Lambda_s_bar, self.B_s_bar = discretize_bilinear(
                self.Lambda_s, B_s_tilde, step_s)

        # ──────────────────────────────────────────────────────────────
        # W_eps: additive PC feedback matrix.  Maps ε at H back into
        # the main scan state at P.  Complex, same shape as B.
        # Init: small (Xavier-uniform scaled), so the additive feedback
        # starts as a tiny correction (recovers vanilla SSM behaviour
        # at init; learns to use the predictive-coding signal).
        # ──────────────────────────────────────────────────────────────
        W_eps_init = lecun_normal()
        self.W_eps_param = self.param(
            "W_eps",
            lambda rng, shape: init_VinvB(W_eps_init, rng, shape, self.Vinv),
            (local_P, self.H))
        W_eps_tilde = self.W_eps_param[..., 0] + 1j * self.W_eps_param[..., 1]

        # Discretize W_eps the same way B is discretized (same scan dynamics
        # apply -- it's another input projection into the main state)
        if self.discretization == "zoh":
            _, self.W_eps_bar = discretize_zoh(
                self.Lambda, W_eps_tilde, step)
        elif self.discretization == "bilinear":
            _, self.W_eps_bar = discretize_bilinear(
                self.Lambda, W_eps_tilde, step)

    def _apply_predictor_scan(self, input_sequence):
        """Run the predictor scan (forward only, no bidirectional)
        and produce x_hat(t) = C_s @ s(t-1) (causally shifted).

        Args:
            input_sequence: (L, H) input
        Returns:
            x_hat: (L, H) one-step-ahead predictions at H.  x_hat[0] is
                zeros (no past state at t=0).
        """
        L = input_sequence.shape[0]
        # Predictor scan: s(t) = Lambda_s_bar * s(t-1) + B_s_bar @ x(t)
        Lambda_elements = self.Lambda_s_bar * np.ones((L, self.Lambda_s_bar.shape[0]))
        Bu_elements = jax.vmap(lambda u: self.B_s_bar @ u)(input_sequence)
        _, s_seq = jax.lax.associative_scan(binary_operator,
                                            (Lambda_elements, Bu_elements))
        # x_hat(t) = C_s @ s(t-1)  ==> causal shift by 1.  s(t-1) for t=0
        # is zero state (S5's apply_ssm starts at h(0)=0 effectively; the
        # scan's first output is s(1) = Lambda * 0 + Bu = Bu_elements[0]).
        # Our shifted sequence: s_shifted[t] = s[t-1] for t>=1, zero for t=0.
        zero_state = np.zeros_like(s_seq[0:1])
        s_shifted = np.concatenate([zero_state, s_seq[:-1]], axis=0)
        # x_hat(t) = Re(C_s @ s_shifted(t))  (matches S5's read convention)
        if self.conj_sym:
            x_hat = jax.vmap(lambda x: 2 * (self.C_s_tilde @ x).real)(s_shifted)
        else:
            x_hat = jax.vmap(lambda x: (self.C_s_tilde @ x).real)(s_shifted)
        return x_hat

    def _apply_main_scan_with_additive_pc(self, x_seq, eps_seq):
        """Run the main scan with the additive PC input:
            u(t) = B_bar @ x(t) + W_eps_bar @ eps(t)
        Then apply C_tilde to produce the output (bidirectional concat
        handled the same way as S5's apply_ssm).

        Args:
            x_seq: (L, H) original input sequence
            eps_seq: (L, H) surprise sequence (raw, at H)
        Returns:
            ys: (L, H) main scan output.
        """
        L = x_seq.shape[0]
        # Compute per-timestep input to main scan
        Bu_x = jax.vmap(lambda u: self.B_bar @ u)(x_seq)        # (L, local_P)
        Bu_eps = jax.vmap(lambda u: self.W_eps_bar @ u)(eps_seq)  # (L, local_P)
        Bu_elements = Bu_x + Bu_eps                              # additive PC

        Lambda_elements = self.Lambda_bar * np.ones((L, self.Lambda_bar.shape[0]))

        # Forward scan
        _, xs_fwd = jax.lax.associative_scan(
            binary_operator, (Lambda_elements, Bu_elements))

        if self.bidirectional:
            _, xs_bwd = jax.lax.associative_scan(
                binary_operator, (Lambda_elements, Bu_elements), reverse=True)
            xs = np.concatenate((xs_fwd, xs_bwd), axis=-1)
        else:
            xs = xs_fwd

        if self.conj_sym:
            return jax.vmap(lambda x: 2 * (self.C_tilde @ x).real)(xs)
        else:
            return jax.vmap(lambda x: (self.C_tilde @ x).real)(xs)

    def __call__(self, input_sequence):
        """Forward pass.  Matches S5SSM signature exactly.

        Args:
            input_sequence: (L, H) input sequence
        Returns:
            output: (L, H) — drop-in replacement for S5SSM's return.

        Intrinsic loss = mean(||eps||^2) is sown into Flax 'intermediates'
        collection so the train loop can aggregate across all blocks for
        logging and add lambda_pc * L_int to the task loss.
        """
        # ── Predictor branch: x_hat(t) = C_s @ s(t-1) ──
        x_hat = self._apply_predictor_scan(input_sequence)            # (L, H)

        # ── Surprise (raw, no RMSNorm, no M-gate) ──
        eps = input_sequence - x_hat                                   # (L, H)

        # ── Main scan with additive PC: u(t) = B @ x(t) + W_eps @ eps(t) ──
        ys = self._apply_main_scan_with_additive_pc(input_sequence, eps)  # (L, H)

        # ── Feedthrough: y = ys + D * x ──
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)            # (L, H)
        output = ys + Du                                                # (L, H)

        # ── Intrinsic loss: mean ||eps||^2 ──
        # Sow into 'intermediates' so the train step can collect across
        # all blocks and add lambda_pc * sum(L_int_per_block) to task loss.
        intrinsic_loss = np.mean(eps * eps)
        self.sow("intermediates", "intrinsic_loss", intrinsic_loss)

        return output


def init_MambinoSSM(H, P, Lambda_re_init, Lambda_im_init, V, Vinv,
                    C_init, discretization, dt_min, dt_max,
                    conj_sym, clip_eigs, bidirectional):
    """Factory matching init_S5SSM signature exactly so MambinoSSM can
    be swapped in via a flag with no other changes."""
    return partial(MambinoSSM,
                   H=H, P=P,
                   Lambda_re_init=Lambda_re_init,
                   Lambda_im_init=Lambda_im_init,
                   V=V, Vinv=Vinv,
                   C_init=C_init,
                   discretization=discretization,
                   dt_min=dt_min, dt_max=dt_max,
                   conj_sym=conj_sym,
                   clip_eigs=clip_eigs,
                   bidirectional=bidirectional)
