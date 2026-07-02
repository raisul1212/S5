"""Compat test: does the old S5SSM (without noise_sigma/adc_bits fields)
load Corner 1's ckpt correctly?  If YES, my chip-module changes are
the bug.  If NO, the bug is elsewhere.

Uses the same code as train_helpers.validate() but constructs S5SSM
using ONLY the original signature -- passes noise_sigma=0.0 and
adc_bits=0 explicitly then verifies loaded weights give correct
forward pass.

Also tests bypassing state.replace() entirely and just building
variables dict directly.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s5.dataloading import Datasets
from s5.ssm import S5SSM
from s5.ssm_init import make_DPLR_HiPPO
from s5.seq_model import BatchClassificationModel
from s5.train_helpers import prep_batch, compute_accuracy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", type=str, required=True)
    args = p.parse_args()

    print("Loading ckpt...")
    with open(args.ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    print(f"saved test_acc: {ckpt['test_acc']}")

    # Build SSM using DIRECT S5SSM (not via init_S5SSM factory)
    # This ensures we use the class exactly as it exists in our code.
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

    ssm_init_fn = partial(
        S5SSM,
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

    # Grab 1 batch
    for batch in testloader:
        inputs, labels, integration_times = prep_batch(batch, seq_len, in_dim)
        break

    # DIRECT apply: build variables from ckpt, apply model
    print("\n=== Direct model.apply with ckpt params + batch_stats ===")
    model = model_cls(training=False)
    variables = {"params": ckpt["params"], "batch_stats": ckpt["batch_stats"]}
    logits = model.apply(variables, inputs, integration_times)
    print(f"logits shape={logits.shape} mean={float(np.mean(logits)):.2f} "
          f"std={float(np.std(logits)):.2f} min={float(np.min(logits)):.2f} "
          f"max={float(np.max(logits)):.2f}")
    preds = np.argmax(logits, axis=-1)
    correct = int(np.sum(preds == labels))
    print(f"correct: {correct}/{len(labels)} = {correct/len(labels):.4f}")

    # Full test loop
    print("\n=== Full test loader with ckpt params + batch_stats ===")
    total_correct, total_count = 0, 0
    for batch in testloader:
        inputs, labels, integration_times = prep_batch(batch, seq_len, in_dim)
        logits = model.apply(variables, inputs, integration_times)
        preds = np.argmax(logits, axis=-1)
        total_correct += int(np.sum(preds == labels))
        total_count += len(labels)
    acc = total_correct / total_count
    print(f"Full test acc: {acc:.4f} (saved: {ckpt['test_acc']:.4f}, "
          f"delta: {abs(acc - ckpt['test_acc']):.4f})")
    if abs(acc - ckpt['test_acc']) < 0.01:
        print("PASS: matches saved test_acc")
    else:
        print("FAIL: does not match saved test_acc")


if __name__ == "__main__":
    main()
