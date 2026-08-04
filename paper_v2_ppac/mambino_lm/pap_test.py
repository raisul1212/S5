"""PAPSSM repo-integration test + iso-param sizing. CPU, fast. Run before the cluster T1."""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as onp, jax, jax.numpy as jnp
from s5.pap_ssm import init_PAPSSM
from s5.seq_model import LMModel

def make(P, gated, training, D, V, NL):
    return LMModel(ssm=init_PAPSSM(H=D, P=P, pap_gate=gated), d_output=V, d_model=D,
                   n_layers=NL, activation="gelu", batchnorm=False, prenorm=False, training=training)

# ---- correctness on small dims ----
D, V, L, NL = 128, 32, 64, 3
ids = jax.random.randint(jax.random.PRNGKey(0), (L,), 0, V)
x = jax.nn.one_hot(ids, V); its = jnp.ones((L,))
params = make(80, True, True, D, V, NL).init(jax.random.PRNGKey(1), x, its)
print("=" * 64); print("PAPSSM REPO TEST"); print("=" * 64)

lp = onp.array(make(80, True, False, D, V, NL).apply(params, x, its))
assert lp.shape == (L, V)
assert onp.allclose(onp.array(jax.scipy.special.logsumexp(jnp.array(lp), axis=-1)), 0.0, atol=1e-4)
print("[1] forward OK: valid per-position log-probs")

for gated, tag in [(True, "gated"), (False, "linear")]:
    p = make(80, gated, True, D, V, NL).init(jax.random.PRNGKey(1), x, its)
    k = 37; ids2 = ids.at[k].set((int(ids[k]) + 5) % V); x2 = jax.nn.one_hot(ids2, V)
    a = onp.array(make(80, gated, False, D, V, NL).apply(p, x, its))
    b = onp.array(make(80, gated, False, D, V, NL).apply(p, x2, its))
    md = float(onp.max(onp.abs(a[:k] - b[:k]))); mda = float(onp.max(onp.abs(a[k:] - b[k:])))
    assert md < 1e-6, f"[{tag}] FUTURE LEAK: {md:.2e}"
    print(f"[2] causality ({tag}): outputs<k unchanged ({md:.1e}); outputs>=k change ({mda:.1e}) OK")

# ---- iso-param sizing at real dims (V=256, D=256, P=80): find NL ~ 795K ----
print("\n[3] iso-param sizing (D=256, V=256, P=80) vs Pure S5 anchor 795,008:")
Dr, Vr = 256, 256
xr = jax.nn.one_hot(jax.random.randint(jax.random.PRNGKey(0), (32,), 0, Vr), Vr); itsr = jnp.ones((32,))
for nl in (8, 9, 10, 11, 12):
    pr = make(80, True, True, Dr, Vr, nl).init(jax.random.PRNGKey(1), xr, itsr)
    n = sum(pp.size for pp in jax.tree_util.tree_leaves(pr))
    print(f"    P=80  n_layers={nl:2d} : {n:,}")
print("\nPAP TEST PASS (forward valid, causal gated+linear, sizing printed)")
