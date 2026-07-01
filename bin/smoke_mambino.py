"""Smoke test: construct MambinoSSM, run a single forward pass, verify
shapes and that intrinsic_loss is sown correctly.  Also exercises the
full S5 SequenceLayer + StackedEncoderModel wrapper with MambinoSSM in
place of S5SSM.

Run on cluster (jax CPU is enough):
    python -m bin.smoke_mambino
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as np
import numpy as onp
from jax.scipy.linalg import block_diag

from s5.mambino_ssm import init_MambinoSSM, MambinoSSM
from s5.ssm import init_S5SSM
from s5.ssm_init import make_DPLR_HiPPO
from s5.seq_model import BatchClassificationModel


def setup_hippo(ssm_size_base=16, blocks=8, conj_sym=True):
    """Mirror s5/train.py:train() HiPPO setup exactly."""
    block_size = ssm_size_base // blocks
    Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)
    if conj_sym:
        block_size = block_size // 2
        ssm_size = ssm_size_base // 2
    else:
        ssm_size = ssm_size_base
    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T
    Lambda = (Lambda * onp.ones((blocks, block_size))).ravel()
    V = block_diag(*([V] * blocks))
    Vinv = block_diag(*([Vc] * blocks))
    return Lambda, V, Vinv, ssm_size


def test_mambino_ssm_unit():
    """Test MambinoSSM module in isolation: construct, init, forward."""
    print("=" * 60)
    print("TEST 1: MambinoSSM unit test")
    print("=" * 60)
    H = 128
    Lambda, V, Vinv, P = setup_hippo(ssm_size_base=16, blocks=8, conj_sym=True)
    print(f"P (after conj_sym halving) = {P}")
    print(f"Lambda shape = {Lambda.shape}")
    print(f"V shape = {V.shape}")
    print(f"Vinv shape = {Vinv.shape}")

    ssm = MambinoSSM(
        H=H, P=P,
        Lambda_re_init=Lambda.real, Lambda_im_init=Lambda.imag,
        V=V, Vinv=Vinv,
        C_init="lecun_normal",
        discretization="zoh",
        dt_min=0.001, dt_max=0.1,
        conj_sym=True, clip_eigs=False,
        bidirectional=True,
    )

    key = jax.random.PRNGKey(0)
    L = 64  # short for smoke
    dummy_input = np.zeros((L, H))
    print(f"Input shape: {dummy_input.shape}")

    # Init with sow collecting intermediates
    variables = ssm.init(key, dummy_input)
    params = variables["params"]

    # Count params
    total = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"\nMambinoSSM single-block params (real count): {total:,}")
    print("Param tree:")
    for path, val in jax.tree_util.tree_flatten_with_path(params)[0]:
        path_str = "/".join(str(p.key) if hasattr(p, "key") else str(p) for p in path)
        print(f"  {path_str}: shape={tuple(val.shape)}  size={val.size:,}")

    # Forward pass
    output, mod_vars = ssm.apply({"params": params}, dummy_input,
                                  mutable=["intermediates"])
    print(f"\nOutput shape: {output.shape} (expected ({L}, {H}))")
    assert output.shape == (L, H), f"Output shape mismatch: {output.shape}"

    # Check intrinsic_loss was sown
    intrinsics = mod_vars.get("intermediates", {}).get("intrinsic_loss", None)
    print(f"intrinsic_loss sown: {intrinsics}")
    assert intrinsics is not None, "intrinsic_loss was not sown!"
    print(f"  -> mean ||eps||^2 = {float(intrinsics[0]):.6f}")

    print("\nTEST 1 PASS")


def test_full_classification_model():
    """Test full BatchClassificationModel with MambinoSSM."""
    print("\n" + "=" * 60)
    print("TEST 2: Full BatchClassificationModel with MambinoSSM")
    print("=" * 60)
    H = 128
    Lambda, V, Vinv, P = setup_hippo(ssm_size_base=16, blocks=8, conj_sym=True)

    ssm_init_fn = init_MambinoSSM(
        H=H, P=P,
        Lambda_re_init=Lambda.real, Lambda_im_init=Lambda.imag,
        V=V, Vinv=Vinv,
        C_init="lecun_normal", discretization="zoh",
        dt_min=0.001, dt_max=0.1,
        conj_sym=True, clip_eigs=False, bidirectional=True,
    )

    n_layers = 8
    n_classes = 10
    in_dim = 20
    bsz = 2
    L = 64

    model = BatchClassificationModel(
        ssm=ssm_init_fn, d_output=n_classes, d_model=H,
        n_layers=n_layers, padded=True, activation="half_glu2",
        dropout=0.0, training=True, mode="pool",
        prenorm=False, batchnorm=True, bn_momentum=0.9, step_rescale=1.0,
    )

    key = jax.random.PRNGKey(0)
    dropout_key = jax.random.PRNGKey(1)
    # Matches what prep_batch produces after the one-hot conversion:
    #   data shape (bsz, L, in_dim) float, lengths (bsz,) float
    dummy_data = np.zeros((bsz, L, in_dim), dtype=np.float32)
    dummy_len = np.full((bsz,), float(L))
    integration_timesteps = np.ones((bsz, L))

    print("Initializing full model (may take ~10s due to JAX compilation)...")
    variables = model.init(
        {"params": key, "dropout": dropout_key},
        (dummy_data, dummy_len),
        integration_timesteps,
    )
    params = variables["params"]
    batch_stats = variables.get("batch_stats", {})

    total = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"\nFull model total trainable params: {total:,}")
    print(f"(expected ~238K for the lean Mambino-in-S5 plan)")

    # Forward pass with intermediates collection
    print("\nRunning forward pass...")
    logits, mod_vars = model.apply(
        {"params": params, "batch_stats": batch_stats},
        (dummy_data, dummy_len),
        integration_timesteps,
        rngs={"dropout": dropout_key},
        mutable=["intermediates", "batch_stats"],
    )

    print(f"Logits shape: {logits.shape} (expected ({bsz}, {n_classes}))")
    assert logits.shape == (bsz, n_classes), f"Logits shape mismatch: {logits.shape}"

    # Verify intermediates contains per-block intrinsic losses
    # Path: {'encoder' or 'BatchSequenceLayer_*': {'intrinsic_loss': (...)}}
    def count_intrinsic_leaves(d, path=""):
        n = 0
        if isinstance(d, dict):
            for k, v in d.items():
                if k == "intrinsic_loss":
                    n += 1
                    arr = v[0] if isinstance(v, tuple) else v
                    print(f"  intrinsic at {path}: mean={float(np.mean(arr)):.6f}")
                else:
                    n += count_intrinsic_leaves(v, path + "/" + k)
        return n

    intermediates = mod_vars.get("intermediates", {})
    n_intrinsics = count_intrinsic_leaves(intermediates)
    print(f"\nTotal intrinsic_loss leaves found: {n_intrinsics}")
    print(f"(expected {n_layers} = one per SequenceLayer)")
    assert n_intrinsics == n_layers, (
        f"Expected {n_layers} intrinsic_loss sown, got {n_intrinsics}")

    print("\nTEST 2 PASS")


def test_aggregator_function():
    """Test _sum_intrinsic_losses helper from train_helpers."""
    print("\n" + "=" * 60)
    print("TEST 3: _sum_intrinsic_losses aggregator")
    print("=" * 60)
    from s5.train_helpers import _sum_intrinsic_losses

    # Simulate intermediates pytree as Flax produces it
    intermediates = {
        "encoder": {
            "layers_0": {"intrinsic_loss": (np.array(1.0),)},
            "layers_1": {"intrinsic_loss": (np.array(2.0),)},
            "layers_2": {"intrinsic_loss": (np.array(3.0),)},
        }
    }
    total = _sum_intrinsic_losses(intermediates)
    print(f"Sum of [1.0, 2.0, 3.0] = {float(total)} (expected 6.0)")
    assert abs(float(total) - 6.0) < 1e-5

    # Empty (vanilla S5 case)
    total_empty = _sum_intrinsic_losses({})
    print(f"Sum of empty dict = {float(total_empty)} (expected 0.0)")
    assert float(total_empty) == 0.0

    print("\nTEST 3 PASS")


def test_save_checkpoint():
    """Smoke test save_checkpoint helper."""
    print("\n" + "=" * 60)
    print("TEST 4: save_checkpoint helper")
    print("=" * 60)
    from s5.train_helpers import save_checkpoint
    import tempfile

    # Build a minimal stand-in for state.  save_checkpoint just needs
    # .params, .opt_state, .step (and .batch_stats if batchnorm=True).
    class DummyState:
        def __init__(self):
            self.params = {"w": np.zeros((3, 3))}
            self.opt_state = {"adam_m": np.zeros((3, 3))}
            self.step = 42
            self.batch_stats = {"bn_mean": np.zeros((3,))}

    state = DummyState()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "best.pkl")
        save_checkpoint(state, path, epoch=5, test_acc=0.62, test_loss=1.23,
                        args_dict={"foo": "bar"}, batchnorm=True)
        assert os.path.isfile(path), f"checkpoint not created at {path}"
        import pickle
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        assert ckpt["epoch"] == 5
        assert abs(ckpt["test_acc"] - 0.62) < 1e-9
        assert ckpt["args"]["foo"] == "bar"
        assert "batch_stats" in ckpt
        print(f"Checkpoint round-trip OK; keys: {list(ckpt.keys())}")

    print("\nTEST 4 PASS")


if __name__ == "__main__":
    test_aggregator_function()
    test_save_checkpoint()
    test_mambino_ssm_unit()
    test_full_classification_model()
    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)
