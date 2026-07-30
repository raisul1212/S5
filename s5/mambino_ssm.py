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

References to NCB v031 (frozen ancestor):
  - ssm_core.py:520-534  ("add-DSPC" design comment block; distinguishes
    additive PC from original DSPC — no M-gate, raw ε)
  - ssm_core.py:1614      (predictor scan parallel_scan_diag, no
    bidirectional wrapper -> predictor is forward only)
  - ssm_core.py:1618-1620 (causal shift s(t-1) and x_hat readout)
  - ssm_core.py:1654-1708 (additive_pc=True compute path in
    _forward_parallel; u = x_tilde @ B^T + eps @ W_eps; M = 1)
  - ssm_core.py:1350-1367 (_maybe_bidirectional_scan for MAIN scan)

Port fidelity: the six core add-DSPC equations (forward-only predictor,
causal-shifted readout x_hat(t) = C_s @ s(t-1), raw ε with no RMSNorm,
M-gate = 1, additive input u = B x + W_ε ε, intrinsic loss mean(||ε||²))
are all preserved.  This ports the architecture that produced 0.4965 on
LRA-ListOps (SLURM 11143665, 2026-06-28).

PORT DIVERGENCES vs v031 ancestor (all intentional, all documented for
reviewer transparency):

  1. Main-scan bidirectional combination:
       v031 sums:      h = h_fwd + h_bwd, then y = C @ h  (shared C)
       port concats:   xs = [xs_fwd | xs_bwd], y = C̃ @ xs where
                       C̃ = [C1 | C2] is S5's separate-C convention
     Sum-with-shared-C is the C1 = C2 special case of concat-with-
     separate-C, so the port is a strict superset in output-projection
     capacity.  This is the standard S5 convention (Smith et al. 2023).

  2. Selectivity stack OMITTED (v031 ssm_core.py:1681-1702):
       - task_coupling (ScaleGradient on ε)
       - eps_gate (per-token sigmoid gate on ε before W_ε)
       - input_gate (per-token sigmoid gate on x̃ before B)
     These add d_C-resolution Dense layers that do not translate to
     S5's P-dim state philosophy; orthogonal to core PC contribution.

  3. Mixture-of-A K>1 experts OMITTED (v031 ssm_core.py:1605-1612):
     Port stays at K=1.  Per v031 comment, "K=1 reduces to the original
     single-A DSPC behaviour exactly" — so K=1 is a faithful subset,
     not a divergence in the math.

  4. Log-scale second readout OMITTED (v031 ssm_core.py:1644-1652):
     v0.7.x nll_gaussian / free_energy losses are LOSS-side variants;
     port keeps only the raw ε² intrinsic loss.  Main signal path is
     identical in v031 too (raw ε reaches main scan in all variants).

Complex-state ports (mechanical, not semantic changes vs real-state v031):
  - v031 uses real state; port uses complex diag state at P (S5 convention).
    Predictor readout is 2·Re(C_s @ s) in port vs s @ C_s.T in v031;
    both produce real-valued predictions at H.

Param overhead vs S5 alone (per layer, complex P, with bidirectional main
+ forward-only predictor):
  Main SSM (Lambda, B, C_fwd, C_bwd, D, log_step): 6,296  [== S5SSM]
  Predictor branch (Lambda_s, B_s, C_s, log_step_s): 4,120  [NEW]
  W_eps additive PC matrix (P,H,2 complex):  2,048           [NEW]
  + S5's half_glu2 Dense + BN handled by SequenceLayer wrapper.

Total per layer (incl. S5 wrapper): 29,280  (vs S5 alone 23,064)
"""
from functools import partial
import math
import jax
import jax.numpy as np
from jax.nn.initializers import lecun_normal, normal
from flax import linen as nn

from .ssm import (
    discretize_zoh,
    discretize_bilinear,
    binary_operator,
    apply_ssm,
    inject_analog_noise,
    quantize_adc,
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
    bidir_predictor: bool = False  # v0.7: symmetric bidirectional predictor
    # Chip analysis knobs -- default 0/0 = no-op (identical to vanilla eval).
    # Under noise_sigma > 0: inject Gaussian noise at ALL analog compute
    # sites: DAC in, B / C / B_s / C_s / W_eps crossbars, eps subtraction,
    # ADC out.  Mambino has ~9 injection sites per layer vs pure S5's ~5 --
    # the "analog scalability" thesis is that Mambino tolerates this extra
    # noise exposure via redundant pathways.
    noise_sigma: float = 0.0
    adc_bits: int = 0
    dac_bits: int = 0
    # See S5SSM for semantics.  Per-layer bools set by SequenceLayer via
    # crossings_every: False means this SSM does not sit at a boundary
    # and skips DAC-in or ADC-out quantization.  Crossbar noise still
    # applied (analog compute is noisy regardless of boundary layout).
    dac_in_enabled: bool = True
    adc_out_enabled: bool = True
    # State cell noise -- see S5SSM for physics.  Applied to BOTH main
    # scan output AND predictor scan output (both use analog state cells).
    #   retention_sigma: correlated (exact scan model)
    #   read_sigma: independent (post-scan Gaussian)
    retention_sigma: float = 0.0
    read_sigma: float = 0.0
    # -- v2 surprise gate (Cluster A: roles 1+3+4 = state-write x adaptive
    # gain x segmentation-via-sign).  Per-token signed adaptive gate on the
    # eps -> W_eps write:  g(t)=tanh(kappa*z(t)+bias), z=running-EMA-normalized
    # ||eps||,  u(t)=B_bar@x(t) + g(t)*(W_eps_bar@eps(t)).  Off (default) =>
    # byte-identical param tree + forward (kill-switch; existing ckpts reload).
    # Adds 2 scalars/layer.
    surprise_gate: bool = False
    gate_alpha: float = 0.9        # EMA decay of the running normalizer
    gate_range: str = "signed"     # "signed"=tanh [-1,1] (push/pop) | "unsigned"=sigmoid [0,1]
    gate_kappa_init: float = 0.0   # sensitivity init (0 => g starts flat at tanh(bias))
    gate_bias_init: float = 2.0    # bias init (+2 => g~0.96 = ~v1 write, livelier kappa grad)
    gate_detach: bool = False      # also stop-gradient eps in the WRITE (calibration fix)
    # -- v2 Cluster B (role 2 = inference-time LEARNING).  A surprise-gated
    # fast-weight associative memory, added as a PARALLEL branch:
    #     k,q = norm(W_k x),   norm(W_q x)     shared address space (from x)
    #     v   = norm(W_v eps)                  SIGNED content (the first moment)
    #     s   = sigmoid(fw_kappa*z + fw_bias)  UNSIGNED write strength
    #     M_t = gamma*M_{t-1} + (1-gamma)*s*(v k^T)      M in R^{d x d}, M_0 = 0
    #     o_t = M_{t-1} q_t                    EXCLUSIVE read (read before write)
    #     y  += W_o o_t
    # M is STATE, not parameters -- it is written and read within one forward
    # pass, which is what makes this inference-time learning.  The 4-roles
    # theorem gives role 2 the signed first moment for its DIRECTION and the
    # magnitude for its WRITE STRENGTH; hence v from eps, gate from ||eps||.
    #
    # Three deliberate choices (Fable-audited, see paper_v2_ppac/memos/
    # cluster_b_implementation_plan.md):
    #  * EXCLUSIVE read.  An inclusive read (M_t) lets the layer learn
    #    W_q ~ W_k so that k_t.q_t ~ 1, degenerating into a gated
    #    INSTANTANEOUS rank-d path that uses no memory at all.  Still causal,
    #    so no test would catch it -- but it would void the whole claim.
    #  * (1-gamma) coupling, no free eta.  Makes M a bounded EMA (||M|| <= 1
    #    with normalized v,k) instead of scaling as 1/(1-gamma), and leaves
    #    ONE scale on the path instead of eta x W_o (whose zero-init
    #    combination is a permanently dead branch).
    #  * v is normalized TOO.  Otherwise ||v|| ~ ||eps|| puts surprise into
    #    the write twice -- once via the gate, once via the content -- which
    #    would blunt the gated-vs-constant ablation.
    #
    # Attached in __call__ AFTER the main scan and OUTSIDE it, never folded
    # into Bu_elements: that keeps the fast weight a SIBLING of the main scan
    # (both consume x,eps; neither consumes the other), so it adds width, not
    # depth, and the diagonal associative scan keeps its shape + kill-switch.
    # Off (default) => byte-identical param tree AND forward.
    fast_weight: bool = False
    fw_dim: int = 8                 # d: q/k/v dim; M is d x d
    fw_proj: str = "shared"         # "shared" (1 down-proj + 3 dxd mixers) | "separate"
    fw_rule: str = "hebb"           # "hebb" (B0) | "delta" (B1, not implemented yet)
    fw_kq_source: str = "x"         # address space:  "x" | "eps"
    fw_v_source: str = "eps"        # written content: "eps" | "x"
    fw_impl: str = "chunk"          # "seq" (reference) | "scan" | "chunk"
    fw_chunk: int = 64              # chunk length C for fw_impl="chunk"
    fw_gate_mode: str = "surprise"  # "surprise" (kappa learned) | "const" (kappa
                                    # pinned to 0, bias learned -> the iso-param
                                    # ablation) | "off" (s == 1, ungated)
    fw_kappa_init: float = 0.0
    fw_bias_init: float = 0.0       # sigmoid(0)=0.5 => half-strength write at init
    fw_gamma_init: float = 0.95     # decay; stored as a logit when trainable
    fw_gamma_trainable: bool = True  # False => gamma frozen at fw_gamma_init exactly
                                     # (fw_gamma_init=0 gives the no-memory control)
    fw_norm_qkv: bool = True
    fw_out_init: str = "zeros"      # zero-init W_o => branch is an exact no-op at
                                    # step 0, so training STARTS at the baseline
    fw_read: str = "exclusive"      # "exclusive" (M_{t-1}) | "inclusive" (M_t)

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

        # Predictor readouts.
        # Forward-only (default): single C_s, predicts x(t) from s_fwd(t-1).
        # Bidirectional (self.bidir_predictor=True): add C_s2 for the
        # backward direction; predicts x(t) from s_bwd(t+1) (anti-causal
        # future context).  Total predictor readout is concat(C_s, C_s2)
        # applied to concat(s_fwd_shifted, s_bwd_shifted), mirroring the
        # main scan's C_tilde = concat(C1, C2).
        C_s_shape = (self.H, local_P, 2)
        self.C_s = self.param(
            "C_s",
            lambda rng, shape: init_CV(C_init_fn, rng, shape, self.V),
            C_s_shape)
        C_s_complex = self.C_s[..., 0] + 1j * self.C_s[..., 1]
        if self.bidir_predictor:
            self.C_s2 = self.param(
                "C_s2",
                lambda rng, shape: init_CV(C_init_fn, rng, shape, self.V),
                C_s_shape)
            C_s2_complex = self.C_s2[..., 0] + 1j * self.C_s2[..., 1]
            # Concat readout matches main scan's C_tilde = concat(C1, C2).
            self.C_s_tilde = np.concatenate((C_s_complex, C_s2_complex), axis=-1)
        else:
            # Forward-only: single C_s (no concat).
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

        # ── v2 surprise-gate params (2 scalars).  Allocated ONLY when the
        # gate is on, so surprise_gate=False leaves the param tree
        # byte-identical to today's MambinoSSM and existing checkpoints
        # reload unchanged. ──
        if self.surprise_gate:
            self.gate_kappa = self.param(
                "gate_kappa",
                lambda rng, shape: np.full(shape, self.gate_kappa_init),
                (1,))
            self.gate_bias = self.param(
                "gate_bias",
                lambda rng, shape: np.full(shape, self.gate_bias_init),
                (1,))

        # ── v2 Cluster-B fast-weight params.  Guarded exactly like the gate
        # above, so fast_weight=False leaves the param tree byte-identical. ──
        if self.fast_weight:
            if self.fw_rule != "hebb":
                raise NotImplementedError(
                    f"fw_rule={self.fw_rule!r}: only 'hebb' (B0) is implemented. "
                    "The delta rule (B1) has a data-dependent MATRIX transition "
                    "(I - beta k k^T), which is not an associative scan and needs "
                    "the chunked WY algorithm; see the Cluster B plan Sec 5.")
            d = self.fw_dim
            proj_init = lecun_normal()
            if self.fw_proj == "shared":
                # One H->d down-projection reused for q/k/v via three dxd
                # mixers: 2Hd + 3d^2 params instead of separate's 4Hd.
                self.fw_P = self.param("fw_P", proj_init, (self.H, d))
                self.fw_Mq = self.param("fw_Mq", proj_init, (d, d))
                self.fw_Mk = self.param("fw_Mk", proj_init, (d, d))
                self.fw_Mv = self.param("fw_Mv", proj_init, (d, d))
            elif self.fw_proj == "separate":
                self.fw_W_q = self.param("fw_W_q", proj_init, (self.H, d))
                self.fw_W_k = self.param("fw_W_k", proj_init, (self.H, d))
                self.fw_W_v = self.param("fw_W_v", proj_init, (self.H, d))
            else:
                raise ValueError(f"fw_proj must be 'shared' or 'separate', got {self.fw_proj!r}")
            # Zero-init W_o (default) makes the branch an exact no-op at step 0.
            # NOT a dead gradient: dL/dW_o = (dL/dy) o^T is nonzero as long as
            # o is not identically zero, so W_o escapes on the first update and
            # every upstream gradient turns on behind it.  This is why eta was
            # deleted rather than kept and zero-initialised -- W_o=0 AND eta=0
            # together WOULD be permanently dead.
            self.fw_W_o = self.param(
                "fw_W_o",
                (lambda rng, shape: np.zeros(shape))
                if self.fw_out_init == "zeros" else proj_init,
                (d, self.H))
            if self.fw_gamma_trainable:
                g0 = min(max(float(self.fw_gamma_init), 1e-4), 1.0 - 1e-4)
                self.fw_gamma_logit = self.param(
                    "fw_gamma_logit",
                    lambda rng, shape: np.full(shape, math.log(g0 / (1.0 - g0))),
                    (1,))
            # Always allocated (even when fw_gate_mode='off') so the param tree
            # is identical across gate modes -- that is what makes the
            # surprise-vs-constant ablation exactly iso-parameter.
            self.fw_kappa = self.param(
                "fw_kappa",
                lambda rng, shape: np.full(shape, self.fw_kappa_init),
                (1,))
            self.fw_bias = self.param(
                "fw_bias",
                lambda rng, shape: np.full(shape, self.fw_bias_init),
                (1,))

    def _surprise_z(self, eps_seq):
        """EMA-normalized surprise magnitude z(t).  Pure (no params), so both
        gates can share it:

            s(t)   = ||eps(t)||_2                     (surprise magnitude at H)
            mu, m2 = causal EMA of s, s^2             (running normalizer)
            z(t)   = (s(t) - mu(t)) / sqrt(var + eps) (adaptive threshold)

        Cluster A applies tanh(kappa_A z + b_A) on top (signed push/pop on the
        state write); Cluster B applies sigmoid(kappa_B z + b_B) (unsigned
        write strength for the fast weight).  Each owns its OWN kappa/bias --
        z is the only thing they share.

        Scan-safe: pointwise ops + one causal-EMA associative scan (a linear
        recurrence, the same primitive as the SSM scan); the main scan is
        untouched, so parallelism is preserved.  Reads stop_gradient(eps) so
        the gate DECISION never trains the predictor (the gain is a top-down,
        task-driven signal; the surprise stays the predictor's).
        Returns (L,) real.
        """
        s = np.linalg.norm(jax.lax.stop_gradient(eps_seq), axis=-1)   # (L,)
        L = s.shape[0]
        a = self.gate_alpha
        A = np.full((L, 1), a)
        _, mu = jax.lax.associative_scan(
            binary_operator, (A, ((1.0 - a) * s)[:, None]))
        _, m2 = jax.lax.associative_scan(
            binary_operator, (A, ((1.0 - a) * (s * s))[:, None]))
        mu = mu[:, 0]
        m2 = m2[:, 0]
        # Adam-style bias-correction of the EMAs: without it the warmup
        # underestimate makes early tokens spuriously surprising (z(0)~3
        # regardless of input).  Dividing by (1-alpha^(t+1)) gives z(0)=0
        # and self-corrects to the true EMA as t grows (no masking needed).
        corr = 1.0 - a ** (np.arange(L) + 1.0)          # (L,)
        mu = mu / corr
        m2 = m2 / corr
        sig = np.sqrt(np.maximum(m2 - mu * mu, 0.0) + 1e-6)
        return (s - mu) / sig                                         # (L,)

    def _surprise_gate(self, eps_seq):
        """Per-token signed adaptive gate g(t) on the eps -> W_eps write.

            s(t)   = ||eps(t)||_2                     (surprise magnitude at H)
            mu, s2 = causal EMA of s, s^2             (running normalizer)
            z(t)   = (s(t) - mu(t)) / sqrt(var+eps)   (adaptive threshold)
            g(t)   = tanh(kappa*z + bias)             ([-1,1]: push+/hold0/pop-)

        Scan-safe: pointwise ops + one causal-EMA associative scan (a linear
        recurrence, the same primitive as the SSM scan); the main scan is
        untouched, so parallelism is preserved.  The gate DECISION reads
        stop_gradient(eps) so it never trains the predictor (the gain is a
        top-down, task-driven signal; the surprise stays the predictor's).
        Returns (L, 1) real.
        """
        pre = self.gate_kappa[0] * self._surprise_z(eps_seq) + self.gate_bias[0]
        if self.gate_range == "unsigned":
            g = jax.nn.sigmoid(pre)
        else:
            g = np.tanh(pre)
        return g[:, None]                                             # (L, 1)

    # ──────────────────────────────────────────────────────────────────
    # Cluster B — surprise-gated fast weight (role 2)
    # ──────────────────────────────────────────────────────────────────
    def _fw_project(self, x_seq, eps_seq):
        """Project q, k (address space) and v (written content) to dim d.

        Sources are two INDEPENDENT flags so all four combinations are
        explicit.  Default kq<-x, v<-eps: the theorem gives role 2 the signed
        first moment for its direction (hence v from eps) while the address
        space stays anchored on x, so a later query can actually hit what an
        earlier surprise wrote.  Sourcing k from eps as well would only let
        lookups fire when surprise PATTERNS recur -- a far weaker retrieval
        condition.  Returns (q, k, v), each (L, d).
        """
        src = {"x": x_seq, "eps": eps_seq}
        kq_in, v_in = src[self.fw_kq_source], src[self.fw_v_source]
        if self.fw_proj == "shared":
            q = (kq_in @ self.fw_P) @ self.fw_Mq
            k = (kq_in @ self.fw_P) @ self.fw_Mk
            v = (v_in @ self.fw_P) @ self.fw_Mv
        else:
            q = kq_in @ self.fw_W_q
            k = kq_in @ self.fw_W_k
            v = v_in @ self.fw_W_v
        if self.fw_norm_qkv:
            def l2n(u):
                return u / np.sqrt(np.sum(u * u, axis=-1, keepdims=True) + 1e-6)
            q, k, v = l2n(q), l2n(k), l2n(v)
        return q, k, v

    def _fw_seq(self, q, k, v, s, gamma):
        """GROUND TRUTH implementation: the recurrence written out literally,
        one step at a time.  O(L) sequential and slow, but correct by
        construction -- `scan` and `chunk` are checked against this."""
        d = q.shape[-1]

        def step(M, inp):
            q_t, k_t, v_t, s_t = inp
            o_pre = M @ q_t                                   # read BEFORE write
            M_new = gamma * M + (1.0 - gamma) * s_t * np.outer(v_t, k_t)
            o_post = M_new @ q_t                              # read AFTER write
            return M_new, (o_pre if self.fw_read == "exclusive" else o_post)

        _, o = jax.lax.scan(step, np.zeros((d, d), dtype=q.dtype), (q, k, v, s))
        return o

    def _fw_scan(self, q, k, v, s, gamma):
        """Parallel via associative_scan over the FLATTENED d^2 state.

        gamma is a SCALAR shared by every entry of M, so M_t = gamma*M_{t-1} +
        W_t decomposes into d^2 independent scalar linear recurrences -- the
        same monoid `binary_operator` already implements for the SSM scan.
        The scan is inclusive, so the exclusive read is produced by SHIFTING
        the output, never by altering the scan.  Costs O(L*d^2) memory, which
        is why `chunk` exists."""
        L, d = q.shape
        W = ((1.0 - gamma) * s[:, None, None]
             * v[:, :, None] * k[:, None, :]).reshape(L, d * d)
        A = np.full((L, d * d), gamma, dtype=W.dtype)
        _, M = jax.lax.associative_scan(binary_operator, (A, W))   # inclusive M_t
        M = M.reshape(L, d, d)
        if self.fw_read == "exclusive":
            M = np.concatenate([np.zeros_like(M[0:1]), M[:-1]], axis=0)
        return np.einsum("lij,lj->li", M, q)

    def _fw_chunk(self, q, k, v, s, gamma):
        """Chunked: O(L*d + n*d^2) memory.  The one that scales, and the path
        B1's delta rule will need.

        With 0-based local index i in a chunk starting at global nC, and
        S = M_{nC-1} the carried state:
            M_{nC+i} = gamma^(i+1) S + sum_{i'<=i} gamma^(i-i') w_{nC+i'}
        so  inclusive: carry exponent i+1, decay gamma^(i-i')   masked i >= i'
            exclusive: carry exponent i,   decay gamma^(i-1-i') masked i >  i'
        Both keep every exponent >= 0.

        The decay matrix is built EXPLICITLY rather than by the usual
        "rescale k by gamma^(-i')" trick, which is numerically unsafe: fp32
        precision collapses once gamma^(-C) > 1/eps, i.e. below gamma ~ 0.878
        at C=128, and bf16 breaks essentially always."""
        L, d = q.shape
        C = min(int(self.fw_chunk), L)
        pad = (-L) % C
        if pad:
            q = np.pad(q, ((0, pad), (0, 0)))
            k = np.pad(k, ((0, pad), (0, 0)))
            v = np.pad(v, ((0, pad), (0, 0)))
            s = np.pad(s, ((0, pad),))
        n = (L + pad) // C
        qc, kc = q.reshape(n, C, d), k.reshape(n, C, d)
        vc, sc = v.reshape(n, C, d), s.reshape(n, C)
        i = np.arange(C)
        excl = (self.fw_read == "exclusive")
        e = i[:, None] - i[None, :] - (1 if excl else 0)      # (C,C) decay exponent
        D = np.where(e >= 0, gamma ** np.maximum(e, 0), 0.0)
        carry = gamma ** (i + (0 if excl else 1))             # (C,)
        tail = gamma ** (C - 1 - i)                           # (C,) for the state update

        def step(S, blk):
            qb, kb, vb, sb = blk
            w = (1.0 - gamma) * sb[:, None] * vb              # (C,d) scaled values
            o = ((qb @ kb.T) * D) @ w + carry[:, None] * (qb @ S.T)
            S_new = gamma ** C * S + (w * tail[:, None]).T @ kb
            return S_new, o

        _, o = jax.lax.scan(step, np.zeros((d, d), dtype=q.dtype), (qc, kc, vc, sc))
        return o.reshape(n * C, d)[:L]

    def _apply_fastweight(self, x_seq, eps_seq):
        """The full role-2 branch: project -> gate -> recur -> read -> W_o.
        Returns (L, H), to be ADDED to the block output."""
        q, k, v = self._fw_project(x_seq, eps_seq)
        if self.fw_gate_mode == "off":
            s = np.ones(x_seq.shape[0], dtype=q.dtype)
        else:
            # 'const' pins kappa to a literal 0 so the gate is a learned
            # CONSTANT sigmoid(bias): identical parameter count to 'surprise',
            # identical free write-rate, differing only in z-dependence.  That
            # is what makes the ablation isolate the hypothesis and nothing else.
            kappa = 0.0 if self.fw_gate_mode == "const" else self.fw_kappa[0]
            s = jax.nn.sigmoid(kappa * self._surprise_z(eps_seq) + self.fw_bias[0])
        gamma = (jax.nn.sigmoid(self.fw_gamma_logit[0]) if self.fw_gamma_trainable
                 else np.asarray(self.fw_gamma_init, dtype=q.dtype))
        impl = {"seq": self._fw_seq, "scan": self._fw_scan,
                "chunk": self._fw_chunk}[self.fw_impl]
        return impl(q, k, v, s, gamma) @ self.fw_W_o                  # (L, H)

    def _apply_predictor_scan(self, input_sequence):
        """Run the predictor scan and produce x_hat(t).

        Forward-only mode: x_hat(t) = C_s @ s_fwd(t-1)  (causal prediction).
        Bidirectional mode: x_hat(t) = C_s @ s_fwd(t-1) + C_s2 @ s_bwd(t+1)
                            (causal + anti-causal predictions combined).

        Both directions use the SAME B_s_bar for input projection and the
        SAME Lambda_s_bar for state dynamics.  Only the READOUTS differ
        (matches how S5's main scan handles bidirectionality: shared B,
        separate C1/C2 readouts on concatenated state).

        Causal shifts prevent trivial identity prediction:
          Forward:  s_fwd_shifted[t] = s_fwd[t-1]  (never sees x(t))
          Backward: s_bwd_shifted[t] = s_bwd[t+1]  (never sees x(t))

        Args:
            input_sequence: (L, H) input
        Returns:
            x_hat: (L, H) predictions at H.  x_hat[0]=zeros in forward mode
                (no past state at t=0); x_hat[L-1] loses the backward
                contribution in bidirectional mode.
        """
        L = input_sequence.shape[0]
        # Predictor scan: s(t) = Lambda_s_bar * s(t-1) + B_s_bar @ x(t)
        Lambda_elements = self.Lambda_s_bar * np.ones((L, self.Lambda_s_bar.shape[0]))
        Bu_elements = jax.vmap(lambda u: self.B_s_bar @ u)(input_sequence)
        # B_s analog crossbar noise
        if self.noise_sigma > 0:
            Bu_elements = inject_analog_noise(
                Bu_elements, self.noise_sigma, self.make_rng('noise'))

        # Forward scan (always runs)
        _, s_fwd = jax.lax.associative_scan(binary_operator,
                                            (Lambda_elements, Bu_elements))
        if self.retention_sigma > 0:
            eta_pred = jax.random.normal(self.make_rng('noise'),
                                         Bu_elements.shape,
                                         dtype=Bu_elements.dtype) * self.retention_sigma
            _, noise_state = jax.lax.associative_scan(
                binary_operator, (Lambda_elements, eta_pred))
            s_fwd = s_fwd + noise_state
        if self.read_sigma > 0:
            s_fwd = s_fwd + jax.random.normal(
                self.make_rng('noise'), s_fwd.shape,
                dtype=s_fwd.dtype) * self.read_sigma
        # Causal shift: s_fwd_shifted[t] = s_fwd[t-1] for t>=1, zero for t=0
        zero_fwd = np.zeros_like(s_fwd[0:1])
        s_fwd_shifted = np.concatenate([zero_fwd, s_fwd[:-1]], axis=0)

        if self.bidir_predictor:
            # Backward scan
            _, s_bwd = jax.lax.associative_scan(
                binary_operator, (Lambda_elements, Bu_elements), reverse=True)
            if self.retention_sigma > 0:
                eta_pred_bwd = jax.random.normal(self.make_rng('noise'),
                                                 Bu_elements.shape,
                                                 dtype=Bu_elements.dtype) * self.retention_sigma
                _, noise_state_bwd = jax.lax.associative_scan(
                    binary_operator, (Lambda_elements, eta_pred_bwd), reverse=True)
                s_bwd = s_bwd + noise_state_bwd
            if self.read_sigma > 0:
                s_bwd = s_bwd + jax.random.normal(
                    self.make_rng('noise'), s_bwd.shape,
                    dtype=s_bwd.dtype) * self.read_sigma
            # Anti-causal shift: s_bwd_shifted[t] = s_bwd[t+1] for t<L-1,
            # zero for t=L-1.  Prevents predictor from cheating with x(t).
            zero_bwd = np.zeros_like(s_bwd[0:1])
            s_bwd_shifted = np.concatenate([s_bwd[1:], zero_bwd], axis=0)
            # Concat state, readout via concat(C_s, C_s2) (stored as C_s_tilde)
            s_read = np.concatenate([s_fwd_shifted, s_bwd_shifted], axis=-1)
        else:
            s_read = s_fwd_shifted

        # x_hat(t) = Re(C_s_tilde @ s_read(t))  (matches S5's read convention)
        if self.conj_sym:
            x_hat = jax.vmap(lambda x: 2 * (self.C_s_tilde @ x).real)(s_read)
        else:
            x_hat = jax.vmap(lambda x: (self.C_s_tilde @ x).real)(s_read)
        # C_s analog crossbar noise
        if self.noise_sigma > 0:
            x_hat = inject_analog_noise(
                x_hat, self.noise_sigma, self.make_rng('noise'))
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
        # Main B analog crossbar noise
        if self.noise_sigma > 0:
            Bu_x = inject_analog_noise(
                Bu_x, self.noise_sigma, self.make_rng('noise'))
        eps_write = jax.lax.stop_gradient(eps_seq) if self.gate_detach else eps_seq
        Bu_eps = jax.vmap(lambda u: self.W_eps_bar @ u)(eps_write)  # (L, local_P)
        # W_eps analog crossbar noise
        if self.noise_sigma > 0:
            Bu_eps = inject_analog_noise(
                Bu_eps, self.noise_sigma, self.make_rng('noise'))
        # v2 surprise gate: per-token signed modulation of the eps->W_eps write.
        # (kill-switch: when surprise_gate=False AND gate_detach=False this
        # block reduces to the original `Bu_eps = W_eps_bar @ eps_seq`.)
        if self.surprise_gate:
            Bu_eps = self._surprise_gate(eps_seq) * Bu_eps
        Bu_elements = Bu_x + Bu_eps                              # additive PC

        Lambda_elements = self.Lambda_bar * np.ones((L, self.Lambda_bar.shape[0]))

        # Forward scan
        _, xs_fwd = jax.lax.associative_scan(
            binary_operator, (Lambda_elements, Bu_elements))
        if self.retention_sigma > 0:
            eta_main = jax.random.normal(self.make_rng('noise'),
                                         Bu_elements.shape,
                                         dtype=Bu_elements.dtype) * self.retention_sigma
            _, noise_state = jax.lax.associative_scan(
                binary_operator, (Lambda_elements, eta_main))
            xs_fwd = xs_fwd + noise_state
        if self.read_sigma > 0:
            xs_fwd = xs_fwd + jax.random.normal(
                self.make_rng('noise'), xs_fwd.shape,
                dtype=xs_fwd.dtype) * self.read_sigma

        if self.bidirectional:
            _, xs_bwd = jax.lax.associative_scan(
                binary_operator, (Lambda_elements, Bu_elements), reverse=True)
            if self.retention_sigma > 0:
                eta_main_bwd = jax.random.normal(self.make_rng('noise'),
                                                 Bu_elements.shape,
                                                 dtype=Bu_elements.dtype) * self.retention_sigma
                _, noise_state_bwd = jax.lax.associative_scan(
                    binary_operator, (Lambda_elements, eta_main_bwd), reverse=True)
                xs_bwd = xs_bwd + noise_state_bwd
            if self.read_sigma > 0:
                xs_bwd = xs_bwd + jax.random.normal(
                    self.make_rng('noise'), xs_bwd.shape,
                    dtype=xs_bwd.dtype) * self.read_sigma
            xs = np.concatenate((xs_fwd, xs_bwd), axis=-1)
        else:
            xs = xs_fwd

        if self.conj_sym:
            ys = jax.vmap(lambda x: 2 * (self.C_tilde @ x).real)(xs)
        else:
            ys = jax.vmap(lambda x: (self.C_tilde @ x).real)(xs)
        # Main C crossbar noise
        if self.noise_sigma > 0:
            ys = inject_analog_noise(
                ys, self.noise_sigma, self.make_rng('noise'))
        return ys

    def __call__(self, input_sequence):
        """Forward pass.  Matches S5SSM signature exactly.

        Under noise_sigma > 0 / adc_bits > 0, injects noise at all analog
        compute sites: DAC in, predictor B_s/C_s crossbars, eps analog
        subtraction, main B/C crossbars, W_eps crossbar, ADC out.  Also
        applies ADC quantization at the block output.  Digital operations
        (gelu, out2, sigmoid, multiply) happen OUTSIDE this block and are
        NOT affected -- they get clean digital input.

        Args:
            input_sequence: (L, H) input sequence
        Returns:
            output: (L, H) — drop-in replacement for S5SSM's return.

        Intrinsic loss = mean(||eps||^2) is sown into Flax 'intermediates'
        collection so the train loop can aggregate across all blocks for
        logging and add lambda_pc * L_int to the task loss.
        """
        # ── 1) DAC in (digital -> analog): quantize digital input to
        # dac_bits, then add analog voltage noise from DAC nonidealities. ──
        x = quantize_adc(input_sequence, self.dac_bits)
        if self.noise_sigma > 0:
            x = inject_analog_noise(x, self.noise_sigma,
                                    self.make_rng('noise'))

        # ── 2) Predictor branch: x_hat(t) = C_s @ s(t-1) ──
        # (predictor scan internals also emit noise if enabled -- see method)
        x_hat = self._apply_predictor_scan(x)                          # (L, H)

        # ── 3) Surprise: eps = x - x_hat (analog subtraction) ──
        eps = x - x_hat
        if self.noise_sigma > 0:
            eps = inject_analog_noise(eps, self.noise_sigma,
                                      self.make_rng('noise'))

        # ── 4) Main scan with additive PC: u(t) = B @ x(t) + W_eps @ eps(t) ──
        ys = self._apply_main_scan_with_additive_pc(x, eps)            # (L, H)

        # ── 5) Feedthrough + Cluster-B fast weight: y = ys + D*x + W_o @ o ──
        Du = jax.vmap(lambda u: self.D * u)(x)                         # (L, H)
        # The fast weight is a SIBLING of the main scan, not a successor: it
        # consumes (x, eps) exactly as the main scan does, and neither consumes
        # the other, so the critical path stays
        #     predictor -> { main fwd || main bwd || fast weight } -> sum
        # i.e. this adds WIDTH, not DEPTH.  Folding it into Bu_elements instead
        # would have made it a predecessor and bought a third serial stage.
        # 0.0 when off => byte-identical to the v1 forward.
        o_fw = self._apply_fastweight(x, eps) if self.fast_weight else 0.0
        output = ys + Du + o_fw                                        # (L, H)

        # ── 6) ADC out (analog -> digital) ──
        if self.noise_sigma > 0:
            output = inject_analog_noise(output, self.noise_sigma,
                                         self.make_rng('noise'))
        output = quantize_adc(output, self.adc_bits)

        # ── Intrinsic loss: mean ||eps||^2 (used during training only) ──
        intrinsic_loss = np.mean(eps * eps)
        self.sow("intermediates", "intrinsic_loss", intrinsic_loss)

        return output


def init_MambinoSSM(H, P, Lambda_re_init, Lambda_im_init, V, Vinv,
                    C_init, discretization, dt_min, dt_max,
                    conj_sym, clip_eigs, bidirectional,
                    bidir_predictor=False,
                    noise_sigma=0.0, adc_bits=0, dac_bits=0,
                    dac_in_enabled=True, adc_out_enabled=True,
                    retention_sigma=0.0, read_sigma=0.0,
                    surprise_gate=False, gate_alpha=0.9, gate_range="signed",
                    gate_kappa_init=0.0, gate_bias_init=2.0, gate_detach=False,
                    fast_weight=False, fw_dim=8, fw_proj="shared",
                    fw_rule="hebb", fw_kq_source="x", fw_v_source="eps",
                    fw_impl="chunk", fw_chunk=64, fw_gate_mode="surprise",
                    fw_kappa_init=0.0, fw_bias_init=0.0, fw_gamma_init=0.95,
                    fw_gamma_trainable=True, fw_norm_qkv=True,
                    fw_out_init="zeros", fw_read="exclusive"):
    """Factory matching init_S5SSM signature exactly so MambinoSSM can
    be swapped in via a flag with no other changes.

    bidir_predictor (default False = original causal predictor).  When True,
    predictor scan runs both forward and backward with separate C_s, C_s2
    readouts (matches main scan's C1/C2 pattern).  Adds ~2K params per
    layer (C_s2) but enables predictor to use future context.
    """
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
                   bidirectional=bidirectional,
                   bidir_predictor=bidir_predictor,
                   noise_sigma=noise_sigma,
                   adc_bits=adc_bits,
                   dac_bits=dac_bits,
                   dac_in_enabled=dac_in_enabled,
                   adc_out_enabled=adc_out_enabled,
                   retention_sigma=retention_sigma,
                   read_sigma=read_sigma,
                   surprise_gate=surprise_gate,
                   gate_alpha=gate_alpha,
                   gate_range=gate_range,
                   gate_kappa_init=gate_kappa_init,
                   gate_bias_init=gate_bias_init,
                   gate_detach=gate_detach,
                   fast_weight=fast_weight,
                   fw_dim=fw_dim,
                   fw_proj=fw_proj,
                   fw_rule=fw_rule,
                   fw_kq_source=fw_kq_source,
                   fw_v_source=fw_v_source,
                   fw_impl=fw_impl,
                   fw_chunk=fw_chunk,
                   fw_gate_mode=fw_gate_mode,
                   fw_kappa_init=fw_kappa_init,
                   fw_bias_init=fw_bias_init,
                   fw_gamma_init=fw_gamma_init,
                   fw_gamma_trainable=fw_gamma_trainable,
                   fw_norm_qkv=fw_norm_qkv,
                   fw_out_init=fw_out_init,
                   fw_read=fw_read)
