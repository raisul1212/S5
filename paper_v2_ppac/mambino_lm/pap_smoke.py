"""PAP LM-path training smoke (CPU, synthetic). Gradients through the sequential scan + stability."""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from functools import partial
import numpy as onp, jax, jax.numpy as jnp
if not hasattr(jax, "tree_leaves"): jax.tree_leaves = jax.tree_util.tree_leaves
if not hasattr(jax, "tree_map"): jax.tree_map = jax.tree_util.tree_map
from s5.pap_ssm import init_PAPSSM
from s5.seq_model import BatchLMModel
from s5.train_helpers import create_train_state, lm_train_step

D, V, L, NL, BSZ, P = 128, 32, 96, 4, 24, 80
model_cls = partial(BatchLMModel, ssm=init_PAPSSM(H=D, P=P, pap_gate=True), d_output=V, d_model=D,
                    n_layers=NL, activation="gelu", batchnorm=False, prenorm=False)
state = create_train_state(model_cls, jax.random.PRNGKey(0), padded=False, retrieval=False,
                           in_dim=V, bsz=BSZ, seq_len=L, batchnorm=False, opt_config="standard",
                           ssm_lr=1e-3, lr=1e-3)
print(f"[*] PAP params: {sum(p.size for p in jax.tree_util.tree_leaves(state.params)):,}")

def toy(seed):
    r = onp.random.RandomState(seed); base = r.randint(0, V, size=(BSZ, 8))
    seq = onp.tile(base, (1, L // 8 + 1))[:, :L + 1]
    return jax.nn.one_hot(jnp.array(seq[:, :L]), V), jnp.array(seq[:, 1:L + 1]), jnp.ones((BSZ, L))

rng = jax.random.PRNGKey(1); first = last = None
for i in range(40):
    x, y, its = toy(100 + i); rng, dr = jax.random.split(rng)
    state, loss, task, intr, _ = lm_train_step(state, dr, x, y, its, model_cls(training=True), False, 0.0)
    if i == 0: first = float(task)
    last = float(task)
    if i % 10 == 0: print(f"    step {i:3d}  task {float(task):.4f}")
print(f"    step 39  task {last:.4f}")
assert onp.isfinite(first) and onp.isfinite(last), "non-finite loss (PAP diverged)"
assert last < first, f"task did not drop: {first:.3f} -> {last:.3f}"
print(f"\nPAP SMOKE PASS: task {first:.3f} -> {last:.3f} (down, finite, stable through the sequential scan)")
