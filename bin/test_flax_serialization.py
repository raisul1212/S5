"""Test flax.serialization round-trip for the Corner 1 checkpoint.

Steps:
  1. Load the original best.pkl via pickle (extract params + batch_stats)
  2. Round-trip through flax.serialization.to_bytes / from_bytes
  3. Load result into a fresh state via state.replace
  4. Call train_helpers.validate() and compare to saved test_acc

If validate returns 0.6155, flax.serialization is the fix.
If it still gives 0.08, the issue is elsewhere.
"""
import argparse
import pickle
import sys
import os
from functools import partial

import jax
import jax.numpy as np
from jax import random
from jax.scipy.linalg import block_diag
import flax.serialization

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s5.dataloading import Datasets
from s5.ssm import init_S5SSM
from s5.ssm_init import make_DPLR_HiPPO
from s5.seq_model import BatchClassificationModel
from s5.train_helpers import create_train_state, validate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", type=str, required=True)
    args = p.parse_args()

    print(f"Loading {args.ckpt_path}")
    with open(args.ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    saved_acc = ckpt["test_acc"]
    print(f"saved test_acc: {saved_acc:.4f}")

    # ── Build ssm_init_fn + model_cls (identical to training) ───────
    ssm_size_base = 16
    blocks = 8
    ssm_size = ssm_size_base
    block_size = int(ssm_size / blocks)
    Lambda, _, _, V, _ = make_DPLR_HiPPO(block_size)
    block_size = block_size // 2
    ssm_size = ssm_size // 2
    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T
    Lambda = (Lambda * np.ones((blocks, block_size))).ravel()
    V = block_diag(*([V] * blocks))
    Vinv = block_diag(*([Vc] * blocks))
    ssm_init_fn = init_S5SSM(
        H=128, P=ssm_size,
        Lambda_re_init=Lambda.real, Lambda_im_init=Lambda.imag,
        V=V, Vinv=Vinv,
        C_init="lecun_normal", discretization="zoh",
        dt_min=0.001, dt_max=0.1,
        conj_sym=True, clip_eigs=False, bidirectional=True,
        noise_sigma=0.0, adc_bits=0,
    )
    model_cls = partial(
        BatchClassificationModel,
        ssm=ssm_init_fn,
        d_output=10, d_model=128, n_layers=8,
        padded=True, activation="half_glu2", dropout=0.0,
        mode="pool", prenorm=False, batchnorm=True, bn_momentum=0.9,
        glu_rank=0,
    )

    # Dataset
    create_dataset_fn = Datasets["listops-classification"]
    _, valloader, testloader, _, _, seq_len, in_dim, _ = \
        create_dataset_fn("./raw_datasets", seed=6554595, bsz=50)

    # Fresh state as template
    init_rng = random.PRNGKey(6554595)
    state = create_train_state(model_cls, init_rng,
                                padded=True, retrieval=False,
                                in_dim=in_dim, bsz=50, seq_len=seq_len,
                                weight_decay=0.04, batchnorm=True,
                                opt_config="BfastandCdecay",
                                ssm_lr=0.001, lr=0.003, dt_global=False)

    # ── Method A: pickle load, put directly (like sanity_v2) ────────
    print("\n=== METHOD A: pickle -> state.replace ===")
    state_a = state.replace(params=ckpt["params"],
                            batch_stats=ckpt["batch_stats"])
    _, test_acc_a = validate(state_a, model_cls, testloader,
                             seq_len, in_dim, batchnorm=True)
    print(f"  test_acc: {test_acc_a:.4f}  saved: {saved_acc:.4f}  "
          f"delta: {abs(test_acc_a - saved_acc):.4f}")

    # ── Method B: pickle -> flax msgpack -> from_bytes -> state.replace ──
    print("\n=== METHOD B: pickle -> flax.serialization roundtrip -> state.replace ===")
    # Serialize the loaded params + batch_stats
    payload = {"params": ckpt["params"], "batch_stats": ckpt["batch_stats"]}
    msgpack_bytes = flax.serialization.to_bytes(payload)
    print(f"  serialized to {len(msgpack_bytes)} bytes")

    # Deserialize back using fresh state's structure as template
    template = {"params": state.params, "batch_stats": state.batch_stats}
    restored = flax.serialization.from_bytes(template, msgpack_bytes)
    state_b = state.replace(params=restored["params"],
                            batch_stats=restored["batch_stats"])
    _, test_acc_b = validate(state_b, model_cls, testloader,
                             seq_len, in_dim, batchnorm=True)
    print(f"  test_acc: {test_acc_b:.4f}  saved: {saved_acc:.4f}  "
          f"delta: {abs(test_acc_b - saved_acc):.4f}")

    # ── Method C: use flax.serialization.from_state_dict on template ──
    print("\n=== METHOD C: flax.serialization.from_state_dict ===")
    restored_c = flax.serialization.from_state_dict(
        {"params": state.params, "batch_stats": state.batch_stats},
        {"params": ckpt["params"], "batch_stats": ckpt["batch_stats"]})
    state_c = state.replace(params=restored_c["params"],
                            batch_stats=restored_c["batch_stats"])
    _, test_acc_c = validate(state_c, model_cls, testloader,
                             seq_len, in_dim, batchnorm=True)
    print(f"  test_acc: {test_acc_c:.4f}  saved: {saved_acc:.4f}  "
          f"delta: {abs(test_acc_c - saved_acc):.4f}")

    print("\n=== SUMMARY ===")
    print(f"Method A (pickle direct):          {test_acc_a:.4f}")
    print(f"Method B (msgpack roundtrip):      {test_acc_b:.4f}")
    print(f"Method C (from_state_dict):        {test_acc_c:.4f}")
    print(f"Saved:                             {saved_acc:.4f}")
    if any(abs(a - saved_acc) < 0.01 for a in [test_acc_a, test_acc_b, test_acc_c]):
        print("AT LEAST ONE METHOD WORKS - use it going forward")
    else:
        print("ALL METHODS FAIL - bug is not in load mechanism")


if __name__ == "__main__":
    main()
