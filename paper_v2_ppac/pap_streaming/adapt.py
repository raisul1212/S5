"""Stage C, module 3 — ANCHOR 2: error-driven inference-time learning (the streaming eval).

Architecture-agnostic: operates only on the frozen backbone's per-position features + base logits.
A small readout fast-weight W_fw is updated at inference by the gradient of the self-supervised
next-token loss, chunk by chunk. ALWAYS-ON (no gate; |dW| proportional to error is intrinsic).

CAUSAL online protocol (score-then-learn): each chunk is SCORED with W_fw trained on PAST chunks
ONLY, THEN W_fw is updated on that chunk. So a chunk's NLL never sees its own targets first.
"""
import numpy as np
import jax
import jax.numpy as jnp
import optax
from functools import partial


def _rmsnorm(fc):
    return fc / (np.sqrt(np.mean(fc * fc, axis=-1, keepdims=True)) + 1e-6)


def _ce_readout(W, fcn, bc, tc):
    logits = bc + fcn @ W.T                     # fcn = RMSNorm'd features -> stable, scale-free readout
    return optax.softmax_cross_entropy_with_integer_labels(logits, tc).mean()


def _ce_bias(b, bc, tc):
    logits = bc + b[None, :]                     # bias-only: learns the L2 byte-MARGINAL (attribution control)
    return optax.softmax_cross_entropy_with_integer_labels(logits, tc).mean()


_gW = jax.jit(jax.grad(_ce_readout))
_gb = jax.jit(jax.grad(_ce_bias))


def stream_nll(feat, base, targets, adapt=True, eta=0.1, chunk=64, mode="readout", clip=2.0):
    """feat (L,H), base (L,V) from a FROZEN backbone; targets (L,) next-byte ids.
    mode='readout' -> fast-weight reads RMSNorm'd features (the anchor-2 adapter);
    mode='bias'    -> fast-weight is a per-class bias (learns the L2 marginal only) = the ATTRIBUTION
                      control (if bias-only recovers as much, the backbone features add nothing).
    CAUSAL score-then-learn; gradient clipped to `clip` (divergence guard). Returns (per-pos NLL, per-chunk ||dW||)."""
    L, H = feat.shape
    V = base.shape[1]
    featn = _rmsnorm(feat)
    bias_mode = (mode == "bias")
    W = np.zeros(V, np.float32) if bias_mode else np.zeros((V, H), np.float32)
    nll = np.zeros(L, np.float32)
    dW_norm = []
    for cs in range(0, L, chunk):
        sl = slice(cs, min(cs + chunk, L))
        fc, bc, tc = featn[sl], base[sl], targets[sl]
        logits = bc + (W[None, :] if bias_mode else fc @ W.T)
        m = logits.max(-1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(logits - m).sum(-1))
        nll[sl] = lse - logits[np.arange(len(tc)), tc]     # per-position CE (SCORE before update)
        if adapt:
            if bias_mode:
                g = np.asarray(_gb(jnp.asarray(W), jnp.asarray(bc), jnp.asarray(tc)))
            else:
                g = np.asarray(_gW(jnp.asarray(W), jnp.asarray(fc), jnp.asarray(bc), jnp.asarray(tc)))
            gn = float(np.linalg.norm(g))
            if clip and gn > clip:                          # divergence guard
                g = g * (clip / gn)
            W = W - eta * g
            dW_norm.append(float(np.linalg.norm(eta * g)))
        else:
            dW_norm.append(0.0)
    return nll, np.array(dW_norm)


def bpc(nll):
    return float(np.mean(nll) / np.log(2.0))
