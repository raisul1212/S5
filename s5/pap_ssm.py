"""Pure Adaptive Predictor SSM (PAP) — a drop-in replacement for S5SSM in SequenceLayer.

Undivided pure error-based predictor (spec: pure_adaptive_predictor_spec.md). Per token,
on the layer input u (H-dim), with real diagonal state h (P-dim):

    x_hat = C h                         # predict the layer input from state
    eps   = u - x_hat                   # prediction error
    z     = (||eps|| - mu)/sigma        # trailing-EMA-standardized surprise (causal)
    g     = sigmoid(kappa*z + b)        # ANCHOR 1 surprise gate  (pap_gate=False => g=1)
    h     = a (.) h + g*(B eps)         # surprise-GATED error integration
    y     = C_out * rmsnorm(h) + D (.) u

Because g depends on eps (hence h_{t-1}), this is a NONLINEAR recurrence with prediction
feedback -> a SEQUENTIAL lax.scan (no S5 parallel prefix). pap_gate=False gives g=1: the
error-integrator is then S5-CLASS *in spirit* (h~=(a-BC)h+B u) but NOT literally an S5 -- a is a
REAL positive diagonal (no oscillatory modes), and the state clip + per-token RMSNorm readout are
nonlinearities; it also carries a separate C_out (+H*P params/layer). Treat pap_gate=False as an
ablation of anchor 1 WITHIN the PAP family; the true S5 baseline is the separate MODEL=s5 arm.

Stability (see pap_prototype.py): B,C init SMALL (BC~O(1) at 1/sqrt(H) diverges); state clip;
RMSNorm readout. Causal (h_t depends on u_{<=t}; x_hat_t reads h_{t-1}); drops into the causal
LM path unchanged.  ANCHOR 2 (error-driven inference-time fast-weight) is applied at eval, not here.
"""
from functools import partial
import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import normal


class PAPSSM(nn.Module):
    H: int
    P: int
    pap_gate: bool = True          # anchor 1 on/off (False => linear error-integrator == S5)
    gate_alpha: float = 0.9        # surprise-EMA decay
    a_init: float = 2.2            # sigmoid(2.2) ~ 0.90 decay
    bc_scale: float = 0.5          # B,C init stddev = bc_scale/H  (keeps BC~O(bc_scale) -> stable)
    kappa_init: float = 2.0
    gate_bias_init: float = 0.0
    state_clip: float = 30.0
    # S5SSM-signature parity so SequenceLayer can pass these; PAP ignores them.
    step_rescale: float = 1.0
    dac_in_enabled: bool = True
    adc_out_enabled: bool = True

    def setup(self):
        P, H = self.P, self.H
        self.B = self.param("B", normal(stddev=self.bc_scale / H), (P, H))
        self.C = self.param("C", normal(stddev=self.bc_scale / H), (H, P))
        self.C_out = self.param("C_out", normal(stddev=1.0 / np.sqrt(P)), (H, P))
        self.a_raw = self.param("a_raw", lambda r, s: np.full(s, self.a_init), (P,))
        self.D = self.param("D", normal(stddev=1.0), (H,))
        self.pap_kappa = self.param("pap_kappa", lambda r, s: np.full(s, self.kappa_init), (1,))
        self.pap_bias = self.param("pap_bias", lambda r, s: np.full(s, self.gate_bias_init), (1,))

    def __call__(self, input_sequence):
        a = jax.nn.sigmoid(self.a_raw)
        al = self.gate_alpha
        gated = self.pap_gate

        def step(carry, u):
            h, m, v, cnt = carry
            xhat = self.C @ h                                   # (H,)
            eps = u - xhat
            n = np.sqrt(np.sum(eps * eps) + 1e-8)
            denom = np.where(cnt > 0, 1.0 - al ** cnt, 1.0)     # avoid 0/0 in the unselected where-branch
            mu = np.where(cnt > 0, m / denom, 0.0)              # trailing surprise stats (causal)
            var = np.where(cnt > 0, np.maximum(v / denom - mu * mu, 1e-6), 1.0)
            z = np.clip((n - mu) / np.sqrt(var), -6.0, 6.0)     # clip: 1e-6 var-floor vs ||eps||~O(10) else saturates/spikes
            g = jax.nn.sigmoid(self.pap_kappa[0] * z + self.pap_bias[0]) if gated else 1.0
            h = np.clip(a * h + g * (self.B @ eps), -self.state_clip, self.state_clip)
            m2 = al * m + (1.0 - al) * n
            v2 = al * v + (1.0 - al) * n * n
            h_out = h * jax.lax.rsqrt(np.mean(h * h) + 1e-6)    # RMSNorm readout (bounded logits)
            y = self.C_out @ h_out + self.D * u
            return (h, m2, v2, cnt + 1.0), y

        c0 = (np.zeros(self.P), 0.0, 0.0, 0.0)
        _, ys = jax.lax.scan(step, c0, input_sequence)          # (L, H)
        return ys


def init_PAPSSM(H, P, pap_gate=True, gate_alpha=0.9, **unused):
    """Mirror the init_S5SSM(...) -> partial(SSM, ...) convention so train.py can swap it in.
    PAP needs no HiPPO/Lambda/V; the extra S5 kwargs (Lambda_*_init, V, ...) are accepted+ignored."""
    return partial(PAPSSM, H=H, P=P, pap_gate=pap_gate, gate_alpha=gate_alpha)
