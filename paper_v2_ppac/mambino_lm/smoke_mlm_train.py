"""Mambino-LM Stage-1 full-path TRAINING smoke (CPU, synthetic data, no enwik8).

Exercises the exact train_helpers path (create_train_state -> mlm_train_step ->
mlm_validate) to catch wiring bugs (sow collection, traced lambdas, vmap
intermediates axis, BPC/esc reporting) BEFORE the cluster.  A structured toy
stream (period-P repeats) is used so a few steps visibly lower the loss.
"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from functools import partial
import numpy as onp
import jax, jax.numpy as jnp
# Local JAX (0.10.2) removed these aliases the repo (written for 0.4.x, the Gilbreth
# env) still uses. Shim them so the EXISTING train_helpers runs locally for the smoke.
if not hasattr(jax, "tree_leaves"): jax.tree_leaves = jax.tree_util.tree_leaves
if not hasattr(jax, "tree_map"): jax.tree_map = jax.tree_util.tree_map
from jax.scipy.linalg import block_diag
from s5.ssm import init_S5SSM
from s5.ssm_init import make_DPLR_HiPPO
from s5.seq_model import BatchMambinoLMModel
from s5.train_helpers import create_train_state, mlm_train_step, mlm_validate

D, SSM, BLOCKS, V = 64, 64, 4, 32
L, STRIDE, NL, TOPL, BSZ = 128, 4, 3, 2, 16
block_size = SSM // BLOCKS
Lambda, _, B, Vmat, B_orig = make_DPLR_HiPPO(block_size)
block_size //= 2; ssm_size = SSM // 2
Lambda = Lambda[:block_size]; Vmat = Vmat[:, :block_size]; Vc = Vmat.conj().T
Lambda = (Lambda * onp.ones((BLOCKS, block_size))).ravel()
Vmat = block_diag(*([Vmat] * BLOCKS)); Vinv = block_diag(*([Vc] * BLOCKS))
ssm_init_fn = init_S5SSM(H=D, P=ssm_size, Lambda_re_init=Lambda.real, Lambda_im_init=Lambda.imag,
                         V=Vmat, Vinv=Vinv, C_init="trunc_standard_normal", discretization="zoh",
                         dt_min=0.001, dt_max=0.1, conj_sym=True, clip_eigs=False, bidirectional=False)
model_cls = partial(BatchMambinoLMModel, ssm=ssm_init_fn, d_output=V, d_model=D, n_layers=NL,
                    activation="gelu", batchnorm=False, prenorm=False, mlm_stride=STRIDE, mlm_top_layers=TOPL)

state = create_train_state(model_cls, jax.random.PRNGKey(0), padded=False, retrieval=False,
                           in_dim=V, bsz=BSZ, seq_len=L, batchnorm=False, opt_config="standard",
                           ssm_lr=1e-3, lr=1e-3)
nparam = sum(p.size for p in jax.tree_util.tree_leaves(state.params))
print(f"[*] Mambino-LM params: {nparam:,}")

def toy_batch(seed):
    r = onp.random.RandomState(seed)
    base = r.randint(0, V, size=(BSZ, 8))
    seq = onp.tile(base, (1, L // 8 + 1))[:, :L + 1]      # period-8 repeats (learnable structure)
    ids = seq[:, :L]; tgt = seq[:, 1:L + 1]
    return jax.nn.one_hot(jnp.array(ids), V), jnp.array(tgt), jnp.ones((BSZ, L))

print("[*] running 40 mlm_train_step ...")
rng = jax.random.PRNGKey(1)
first = last = None
for i in range(40):
    x, y, its = toy_batch(100 + i)
    rng, dr = jax.random.split(rng)
    lam_pond = jnp.float32(0.0 if i < 10 else 0.05 * min(1.0, (i - 10) / 20.0))
    state, loss, task, aux, pond, esc = mlm_train_step(
        state, dr, x, y, its, model_cls(training=True), False, jnp.float32(0.1), lam_pond)
    if i == 0: first = (float(loss), float(task), float(aux), float(pond), float(esc))
    last = (float(loss), float(task), float(aux), float(pond), float(esc))
    if i % 10 == 0:
        print(f"    step {i:3d}  loss {float(loss):.4f}  task {float(task):.4f}  "
              f"aux {float(aux):.4f}  ponder {float(pond):.3f}  esc {float(esc):.3f}")
print(f"    step 39  loss {last[0]:.4f}  task {last[1]:.4f}  aux {last[2]:.4f} ponder {last[3]:.3f} esc {last[4]:.3f}")

# a tiny "loader" of a few batches for mlm_validate
class L2:
    def __init__(self, n): self.n = n
    def __iter__(self):
        for i in range(self.n):
            x, y, its = toy_batch(9000 + i)
            yield (onp.array(jnp.argmax(x, -1)), onp.array(y))
class Wrap:  # prep_lm_batch calls .numpy(); wrap arrays
    def __init__(self, a): self._a = a
    def numpy(self): return self._a
class L3:
    def __init__(self, n): self.n = n
    def __iter__(self):
        for i in range(self.n):
            x, y, its = toy_batch(9000 + i)
            yield (Wrap(onp.array(jnp.argmax(x, -1))), Wrap(onp.array(y)))

bpc, esc = mlm_validate(state, model_cls, L3(3), L, V, batchnorm=False, max_batches=3)
print(f"[*] mlm_validate: BPC={bpc:.4f}  esc_rate={esc:.3f}")
assert onp.isfinite(first[0]) and onp.isfinite(last[0]), "non-finite loss"
assert last[1] < first[1], f"task loss did not drop: {first[1]:.3f} -> {last[1]:.3f}"
assert onp.isfinite(bpc) and 0.0 <= esc <= 1.0
print(f"\nSMOKE PASS: task {first[1]:.3f} -> {last[1]:.3f} (down), BPC finite, esc in [0,1], aux/ponder/esc live")
