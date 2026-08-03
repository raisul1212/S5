"""Mambino-LM Stage-1 repo-integration tests (CPU, fast). Run BEFORE any cluster job.

Checks the load-bearing invariants of MambinoLMModel:
  1. builds, forward returns valid per-position log-probs of the right shape;
  2. the escalation is a clean ADDITIVE path: at alpha=0 the output is INDEPENDENT of
     the top/nudge params (the top off-switch is exact);
  3. CAUSALITY / no future leak: changing input token at position k leaves every output
     at positions < k bit-identical -- under BOTH the soft (train) and hard (eval) gate;
  4. the aux/ponder/esc terms are sown for the loss.
"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as onp
import jax, jax.numpy as jnp
from jax.scipy.linalg import block_diag
from s5.ssm import init_S5SSM
from s5.ssm_init import make_DPLR_HiPPO
from s5.seq_model import MambinoLMModel

# ---- build an S5 ssm_init_fn exactly like train.py ----
D, SSM, BLOCKS, V = 48, 32, 4, 32
L, STRIDE, NL, TOPL = 64, 4, 2, 1
block_size = SSM // BLOCKS
Lambda, _, B, Vmat, B_orig = make_DPLR_HiPPO(block_size)
block_size //= 2; ssm_size = SSM // 2
Lambda = Lambda[:block_size]; Vmat = Vmat[:, :block_size]; Vc = Vmat.conj().T
Lambda = (Lambda * onp.ones((BLOCKS, block_size))).ravel()
Vmat = block_diag(*([Vmat] * BLOCKS)); Vinv = block_diag(*([Vc] * BLOCKS))
ssm_init_fn = init_S5SSM(H=D, P=ssm_size, Lambda_re_init=Lambda.real, Lambda_im_init=Lambda.imag,
                         V=Vmat, Vinv=Vinv, C_init="trunc_standard_normal", discretization="zoh",
                         dt_min=0.001, dt_max=0.1, conj_sym=True, clip_eigs=False, bidirectional=False)

def make_model(training, alpha_override=-99.0):
    return MambinoLMModel(ssm=ssm_init_fn, d_output=V, d_model=D, n_layers=NL, activation="gelu",
                          batchnorm=False, prenorm=False, training=training, mlm_stride=STRIDE,
                          mlm_top_layers=TOPL, alpha_override=alpha_override)

key = jax.random.PRNGKey(0)
ids = jax.random.randint(key, (L,), 0, V)
x = jax.nn.one_hot(ids, V); its = jnp.ones((L,))
params = make_model(True).init(jax.random.PRNGKey(1), x, its)
nparam = sum(p.size for p in jax.tree_util.tree_leaves(params))

def fwd(training, alpha, p, xx):
    return make_model(training, alpha).apply(p, xx, its)

print("=" * 68); print("MAMBINO-LM STAGE-1 REPO TESTS"); print("=" * 68)
print(f"params: {nparam:,}  (D={D} ssm={ssm_size} bottom_L={NL} top_L={TOPL} stride={STRIDE})")

# 1. shape + valid log-probs
lp = fwd(False, -99.0, params, x)
assert lp.shape == (L, V), lp.shape
assert onp.allclose(onp.array(jax.scipy.special.logsumexp(lp, axis=-1)), 0.0, atol=1e-4)
print("[1] forward OK: shape (L,V), rows are valid log-probs (logsumexp=0)")

# 2. alpha=0 => output independent of top/nudge params (exact off-switch)
lp0 = fwd(False, 0.0, params, x)
pert = jax.tree_util.tree_map_with_path(
    lambda path, v: v + (0.7 if any(getattr(k, 'key', '') in ("top", "nudge_decoder") for k in path) else 0.0),
    params)
lp0_pert = fwd(False, 0.0, pert, x)
assert onp.allclose(onp.array(lp0), onp.array(lp0_pert), atol=1e-6), \
    f"alpha=0 leaked top params: max|d|={onp.max(onp.abs(onp.array(lp0)-onp.array(lp0_pert))):.2e}"
print("[2] alpha=0 off-switch EXACT: perturbing top+nudge params changes nothing")

# 3. causality: change x at position k -> outputs < k unchanged (soft AND hard gate)
for training, gate in [(True, "soft"), (False, "hard")]:
    k = 37
    ids2 = ids.at[k].set((int(ids[k]) + 5) % V)
    x2 = jax.nn.one_hot(ids2, V)
    a = onp.array(fwd(training, -99.0, params, x)); b = onp.array(fwd(training, -99.0, params, x2))
    md_before = float(onp.max(onp.abs(a[:k] - b[:k])))
    md_after = float(onp.max(onp.abs(a[k:] - b[k:])))
    assert md_before < 1e-6, f"[{gate}] FUTURE LEAK: outputs<k changed by {md_before:.2e}"
    print(f"[3] causality ({gate} gate): outputs<k unchanged (max|d|={md_before:.1e}); "
          f"outputs>=k do change ({md_after:.2e}) as expected")

# 4. sown terms present
_, mv = make_model(True).apply(params, x, its, mutable=["intermediates"])
keys = set()
def _walk(n):
    if isinstance(n, dict):
        for k, v in n.items():
            keys.add(k); _walk(v)
_walk(mv["intermediates"])
for want in ["mlm_aux", "mlm_ponder", "mlm_esc"]:
    assert want in keys, f"missing sown {want}: {keys}"
print("[4] sown for loss: mlm_aux / mlm_ponder / mlm_esc all present")

print("\nALL TESTS PASS  (causal, exact off-switch, loss terms wired)")
