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


def _ce_of_W(W, fcn, bc, tc):
    logits = bc + fcn @ W.T                     # fcn = RMSNorm'd features -> stable, scale-free readout
    return optax.softmax_cross_entropy_with_integer_labels(logits, tc).mean()


_gradW = jax.jit(jax.grad(_ce_of_W))


def stream_nll(feat, base, targets, adapt=True, eta=0.1, chunk=128):
    """feat (L,H), base (L,V) from a FROZEN backbone; targets (L,) next-byte ids.
    The fast-weight reads RMSNorm'd features (bounded scale -> stable across any backbone).
    Returns per-position NLL (nats). adapt=False => frozen readout (control). Also per-chunk ||dW||."""
    L, H = feat.shape
    V = base.shape[1]
    featn = _rmsnorm(feat)                                  # bound the fast-weight input scale
    W = np.zeros((V, H), np.float32)
    nll = np.zeros(L, np.float32)
    dW_norm = []
    for cs in range(0, L, chunk):
        sl = slice(cs, min(cs + chunk, L))
        fc, bc, tc = featn[sl], base[sl], targets[sl]
        logits = bc + fc @ W.T
        m = logits.max(-1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(logits - m).sum(-1))
        nll[sl] = lse - logits[np.arange(len(tc)), tc]     # per-position CE (SCORE before update)
        if adapt:
            g = np.asarray(_gradW(jnp.asarray(W), jnp.asarray(fc), jnp.asarray(bc), jnp.asarray(tc)))
            W = W - eta * g
            dW_norm.append(float(np.linalg.norm(eta * g)))
        else:
            dW_norm.append(0.0)
    return nll, np.array(dW_norm)


def bpc(nll):
    return float(np.mean(nll) / np.log(2.0))
