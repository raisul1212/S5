"""Stage C, module 2 — load a trained backbone from its checkpoint and extract, over a byte
stream, the per-position pre-decoder FEATURES + base logits. Works for any arm (PAP or S5):
the arch is rebuilt from the checkpoint's saved args (meta.pkl), mirroring s5/train.py.

The FROZEN backbone features are extracted ONCE; anchor-2 adaptation (module 3) then adapts only
the readout on top of them (architecture-agnostic, cheap) — exactly the Stage-0 decomposition.
"""
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.serialization as fs

V = 256  # byte-level vocab


def _make_ssm_init(a):
    """Rebuild the ssm_init_fn from saved args, mirroring s5/train.py."""
    if a.get("use_pap", False):
        from s5.pap_ssm import init_PAPSSM
        pred = a.get("pap_predict")
        return init_PAPSSM(H=a["d_model"], P=a["ssm_size_base"],
                           pap_gate=a.get("pap_gate", True),   # mirror train.py default (meta always has the key)
                           pap_predict=(True if pred is None else pred),
                           gate_alpha=a.get("gate_alpha", 0.9))
    from s5.ssm import init_S5SSM
    from s5.ssm_init import make_DPLR_HiPPO
    from jax.scipy.linalg import block_diag
    ssm_size, blocks = a["ssm_size_base"], a["blocks"]
    block_size = ssm_size // blocks
    Lambda, _, B, Vm, _ = make_DPLR_HiPPO(block_size)
    if a.get("conj_sym", True):
        block_size //= 2
        ssm_size //= 2
    Lambda = Lambda[:block_size]; Vm = Vm[:, :block_size]; Vc = Vm.conj().T
    Lambda = (Lambda * np.ones((blocks, block_size))).ravel()
    Vm = block_diag(*([Vm] * blocks)); Vinv = block_diag(*([Vc] * blocks))
    return init_S5SSM(H=a["d_model"], P=ssm_size, Lambda_re_init=Lambda.real,
                      Lambda_im_init=Lambda.imag, V=Vm, Vinv=Vinv, C_init=a["C_init"],
                      discretization=a["discretization"], dt_min=a["dt_min"], dt_max=a["dt_max"],
                      conj_sym=a.get("conj_sym", True), clip_eigs=a.get("clip_eigs", False),
                      bidirectional=False, out_rmsnorm=a.get("s5_out_rmsnorm", False))


def build_model(a, training=False):
    """Build the UNBATCHED causal LMModel matching the saved config (params are batch-shared)."""
    from s5.seq_model import LMModel
    return LMModel(ssm=_make_ssm_init(a), d_output=V, d_model=a["d_model"], n_layers=a["n_layers"],
                   activation=a.get("activation_fn", "gelu"), batchnorm=False,
                   prenorm=a.get("prenorm", True), glu_rank=a.get("glu_rank", 0),
                   glu_structure=a.get("glu_structure", "dense"), training=training)


def load_backbone(msgpack_path, meta_path):
    """Returns (model, params, args). Rebuilds arch from meta, loads params via flax from_bytes."""
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    a = meta["args"]
    model = build_model(a, training=False)
    dummy = jax.nn.one_hot(jnp.zeros((16,), jnp.int32), V)
    template = model.init(jax.random.PRNGKey(0), dummy, jnp.ones((16,)))
    with open(msgpack_path, "rb") as f:
        params = fs.from_bytes({"params": template["params"]}, f.read())["params"]
    return model, params, a


def extract(model, params, ids):
    """Run the frozen backbone over ids (int64[L]); return (feat[L,H], base_logits[L,V]).
    feat = pre-decoder encoder output (sown); base = decoder(feat) raw logits (pre log-softmax)."""
    x = jax.nn.one_hot(jnp.asarray(ids), V)
    _, mv = model.apply({"params": params}, x, jnp.ones((len(ids),)), mutable=["intermediates"])
    feat = mv["intermediates"]["pre_decode"][0]                      # (L, H)
    base = nn.Dense(V).apply({"params": params["decoder"]}, feat)    # (L, V) raw logits
    return np.asarray(feat), np.asarray(base)
