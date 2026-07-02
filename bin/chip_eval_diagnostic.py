"""Diagnostic script to find the ACTUAL bug in the eval path.
No assumptions -- verify each step by inspection.

Sequence of tests:
  1. Load ckpt, inspect keys + first param values
  2. Fresh-init a model, inspect param structure
  3. Compare loaded vs fresh param TREE STRUCTURES leaf-by-leaf
  4. Replace state.params/batch_stats, verify replace actually took effect
  5. Run 1 forward pass, print logits range
  6. Compute test accuracy on 1 batch

If any step reveals a mismatch, we know exactly where the bug is.
"""
import argparse
import pickle
import sys
import os
from functools import partial

import jax
import jax.numpy as np
from jax import random, tree_util
from jax.scipy.linalg import block_diag

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s5.dataloading import Datasets
from s5.ssm import init_S5SSM
from s5.ssm_init import make_DPLR_HiPPO
from s5.seq_model import BatchClassificationModel
from s5.train_helpers import create_train_state, prep_batch, compute_accuracy


def leaf_stats(arr):
    """Return summary stats for a leaf array."""
    a = np.asarray(arr)
    return f"shape={tuple(a.shape)} dtype={a.dtype} mean={float(a.mean()):.4f} std={float(a.std()):.4f} min={float(a.min()):.4f} max={float(a.max()):.4f}"


def flatten_tree_paths(tree, prefix=""):
    """Yield (path_string, leaf) for every leaf in a nested pytree."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from flatten_tree_paths(v, prefix + "/" + k if prefix else k)
    elif hasattr(tree, "shape"):
        yield (prefix, tree)
    else:
        yield (prefix, tree)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", type=str, required=True)
    args = p.parse_args()

    # ─── Step 1: Load ckpt, inspect keys + values ────────────────
    print("=" * 70)
    print("STEP 1: Load checkpoint, inspect")
    print("=" * 70)
    with open(args.ckpt_path, "rb") as f:
        ckpt = pickle.load(f)

    print(f"ckpt keys: {list(ckpt.keys())}")
    print(f"saved test_acc: {ckpt['test_acc']}")
    print(f"epoch: {ckpt['epoch']}")

    print(f"\nparams tree paths (first 15):")
    leaves = list(flatten_tree_paths(ckpt["params"]))
    for path, leaf in leaves[:15]:
        print(f"  {path}: {leaf_stats(leaf)}")
    print(f"...total {len(leaves)} param leaves")

    print(f"\nbatch_stats tree paths (first 10):")
    bs_leaves = list(flatten_tree_paths(ckpt["batch_stats"]))
    for path, leaf in bs_leaves[:10]:
        print(f"  {path}: {leaf_stats(leaf)}")
    print(f"...total {len(bs_leaves)} batch_stats leaves")

    # ─── Step 2: Build fresh state ───────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: Build fresh state, inspect")
    print("=" * 70)
    # Dataset
    create_dataset_fn = Datasets["listops-classification"]
    _, valloader, testloader, _, _, seq_len, in_dim, _ = \
        create_dataset_fn("./raw_datasets", seed=6554595, bsz=50)

    # SSM init
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

    init_rng = random.PRNGKey(6554595)
    state = create_train_state(model_cls, init_rng,
                                padded=True, retrieval=False,
                                in_dim=in_dim, bsz=50, seq_len=seq_len,
                                weight_decay=0.04, batchnorm=True,
                                opt_config="BfastandCdecay",
                                ssm_lr=0.001, lr=0.003, dt_global=False)

    print(f"fresh state type: {type(state).__name__}")
    print(f"fresh state fields: {list(state.__dataclass_fields__.keys())}")

    fresh_leaves = list(flatten_tree_paths(state.params))
    print(f"\nfresh params tree paths (first 15):")
    for path, leaf in fresh_leaves[:15]:
        print(f"  {path}: {leaf_stats(leaf)}")
    print(f"...total {len(fresh_leaves)} fresh param leaves")

    # ─── Step 3: Compare tree structures ─────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3: Compare loaded vs fresh param TREE STRUCTURES")
    print("=" * 70)
    loaded_paths = set(p for p, _ in flatten_tree_paths(ckpt["params"]))
    fresh_paths = set(p for p, _ in flatten_tree_paths(state.params))
    only_loaded = loaded_paths - fresh_paths
    only_fresh = fresh_paths - loaded_paths
    print(f"paths in loaded but not fresh: {len(only_loaded)}")
    for p in sorted(only_loaded)[:10]:
        print(f"  {p}")
    print(f"paths in fresh but not loaded: {len(only_fresh)}")
    for p in sorted(only_fresh)[:10]:
        print(f"  {p}")
    common = loaded_paths & fresh_paths
    print(f"common paths: {len(common)}")

    if only_loaded or only_fresh:
        print("!!! TREE STRUCTURE MISMATCH -- CANNOT MATCH LOADED TO FRESH !!!")

    # Check shapes at common paths
    loaded_dict = {p: l for p, l in flatten_tree_paths(ckpt["params"])}
    fresh_dict = {p: l for p, l in flatten_tree_paths(state.params)}
    shape_mismatches = []
    for p in common:
        if np.asarray(loaded_dict[p]).shape != np.asarray(fresh_dict[p]).shape:
            shape_mismatches.append((p, np.asarray(loaded_dict[p]).shape, np.asarray(fresh_dict[p]).shape))
    print(f"shape mismatches on common paths: {len(shape_mismatches)}")
    for p, ls, fs in shape_mismatches[:5]:
        print(f"  {p}: loaded={ls}, fresh={fs}")

    # ─── Step 4: Replace and verify replacement worked ───────────
    print("\n" + "=" * 70)
    print("STEP 4: Replace params + batch_stats, verify replacement")
    print("=" * 70)
    # Pick a sample leaf to check before/after
    sample_path = sorted(common)[0]
    print(f"Sample path: {sample_path}")
    before_fresh = np.asarray(fresh_dict[sample_path])
    before_loaded = np.asarray(loaded_dict[sample_path])
    print(f"BEFORE fresh: {leaf_stats(before_fresh)}")
    print(f"BEFORE loaded: {leaf_stats(before_loaded)}")
    print(f"BEFORE fresh == loaded? {bool(np.allclose(before_fresh, before_loaded, atol=1e-6))}")

    new_state = state.replace(params=ckpt["params"],
                              batch_stats=ckpt["batch_stats"])

    # Now inspect new_state.params[sample_path]
    new_dict = {p: l for p, l in flatten_tree_paths(new_state.params)}
    after_new = np.asarray(new_dict[sample_path])
    print(f"AFTER new_state: {leaf_stats(after_new)}")
    print(f"AFTER new_state == loaded? {bool(np.allclose(after_new, before_loaded, atol=1e-6))}")
    print(f"AFTER new_state == fresh? {bool(np.allclose(after_new, before_fresh, atol=1e-6))}")

    # Also verify batch_stats replaced
    new_bs = list(flatten_tree_paths(new_state.batch_stats))[0]
    loaded_bs = list(flatten_tree_paths(ckpt["batch_stats"]))[0]
    print(f"\nSample batch_stat: {new_bs[0]}")
    print(f"new_state batch_stat[0]: {leaf_stats(new_bs[1])}")
    print(f"loaded batch_stat[0]:    {leaf_stats(loaded_bs[1])}")
    print(f"batch_stats equal? {bool(np.allclose(np.asarray(new_bs[1]), np.asarray(loaded_bs[1]), atol=1e-6))}")

    # ─── Step 5: Run 1 forward pass ──────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5: Run 1 forward pass, inspect logits")
    print("=" * 70)
    # Grab 1 batch from testloader
    for batch in testloader:
        inputs, labels, integration_times = prep_batch(batch, seq_len, in_dim)
        break

    print(f"inputs shape: {inputs.shape}, dtype: {inputs.dtype}")
    print(f"labels shape: {labels.shape}, first 5: {labels[:5]}")

    # Apply with LOADED params + batch_stats via new_state
    model = model_cls(training=False)
    logits_loaded = model.apply(
        {"params": new_state.params, "batch_stats": new_state.batch_stats},
        inputs, integration_times)
    print(f"logits (loaded weights): shape={logits_loaded.shape}")
    print(f"  {leaf_stats(logits_loaded)}")
    predictions = np.argmax(logits_loaded, axis=-1)
    print(f"  first 10 predictions: {predictions[:10]}")
    print(f"  first 10 labels:      {labels[:10]}")
    correct = int(np.sum(predictions == labels))
    print(f"  correct: {correct}/{len(labels)} = {correct/len(labels):.3f}")

    # Compare with FRESH state (random init) for control
    logits_fresh = model.apply(
        {"params": state.params, "batch_stats": state.batch_stats},
        inputs, integration_times)
    print(f"\nlogits (fresh random init): {leaf_stats(logits_fresh)}")
    correct_f = int(np.sum(np.argmax(logits_fresh, axis=-1) == labels))
    print(f"  correct: {correct_f}/{len(labels)} = {correct_f/len(labels):.3f}")

    # If loaded gives same acc as fresh, params aren't being loaded
    if correct / len(labels) < 0.15 and correct_f / len(labels) < 0.15:
        print("\n!!! BOTH loaded and fresh give random accuracy !!!")
        print("!!! Either params load isn't working OR the model itself is broken !!!")


if __name__ == "__main__":
    main()
