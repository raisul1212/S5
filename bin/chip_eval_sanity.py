"""Sanity checks for the chip-analysis infrastructure BEFORE running
the full sweep.

Tests:
  1. Fast path (sigma=0, bits=0) reproduces the training's saved
     test_acc from Corner 1's best.pkl.  If this fails, we broke
     the module's forward pass somewhere.
  2. Under high noise (sigma=0.10), output changes vs baseline
     (proves noise IS being injected -- not silently no-op'd).
  3. Under very high noise (sigma=0.20), accuracy DROPS below
     baseline (proves noise INFLUENCES the model, not just adds
     to unused paths).
  4. Fast path deterministic: same input twice gives same output.
  5. Noise path stochastic: same input twice gives DIFFERENT output.

If any of these fails, we have a bug.  If all pass, run the sweep.

Usage:
  python bin/chip_eval_sanity.py --ckpt_path=checkpoints/pure_s5_11170658/best.pkl
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
from s5.ssm import init_S5SSM
from s5.ssm_init import make_DPLR_HiPPO
from s5.mambino_ssm import init_MambinoSSM
from s5.seq_model import BatchClassificationModel
from s5.train_helpers import prep_batch, compute_accuracy


def build_ssm_init(args, use_mambino, glu_rank, sigma, bits):
    ssm_size = args.ssm_size_base
    block_size = int(ssm_size / args.blocks)
    Lambda, _, _, V, _ = make_DPLR_HiPPO(block_size)
    if args.conj_sym:
        block_size = block_size // 2
        ssm_size = ssm_size // 2
    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T
    Lambda = (Lambda * np.ones((args.blocks, block_size))).ravel()
    V = block_diag(*([V] * args.blocks))
    Vinv = block_diag(*([Vc] * args.blocks))

    if use_mambino:
        return init_MambinoSSM(
            H=args.d_model, P=ssm_size,
            Lambda_re_init=Lambda.real, Lambda_im_init=Lambda.imag,
            V=V, Vinv=Vinv,
            C_init="lecun_normal", discretization="zoh",
            dt_min=0.001, dt_max=0.1,
            conj_sym=True, clip_eigs=False,
            bidirectional=True, bidir_predictor=False,
            noise_sigma=sigma, adc_bits=bits,
        )
    else:
        return init_S5SSM(
            H=args.d_model, P=ssm_size,
            Lambda_re_init=Lambda.real, Lambda_im_init=Lambda.imag,
            V=V, Vinv=Vinv,
            C_init="lecun_normal", discretization="zoh",
            dt_min=0.001, dt_max=0.1,
            conj_sym=True, clip_eigs=False,
            bidirectional=True,
            noise_sigma=sigma, adc_bits=bits,
        )


def build_model(args, ssm_init_fn, glu_rank):
    return partial(
        BatchClassificationModel,
        ssm=ssm_init_fn,
        d_output=10, d_model=args.d_model, n_layers=args.n_layers,
        padded=True, activation="half_glu2", dropout=0.0,
        mode="pool", prenorm=False, batchnorm=True, bn_momentum=0.9,
        glu_rank=glu_rank,
    )


def apply_model(model_cls, params, batch_stats, inputs, integration_times,
                noise_key, needs_noise_rng=True):
    """Apply the model.  Only pass 'noise' RNG when the modules will
    actually consume it (sigma > 0 or bits > 0).  Passing an unused
    RNG through nn.vmap can silently disturb the forward pass in some
    Flax versions.  For fast-path (sigma=0, bits=0), match training's
    eval path exactly: no rngs at all.
    """
    model = model_cls(training=False)
    variables = {"params": params}
    if batch_stats is not None:
        variables["batch_stats"] = batch_stats
    if needs_noise_rng:
        logits = model.apply(variables, inputs, integration_times,
                             rngs={"noise": noise_key})
    else:
        logits = model.apply(variables, inputs, integration_times)
    return logits


def eval_batches(model_cls, params, batch_stats, loader, seq_len, in_dim,
                 noise_key, max_batches=None, needs_noise_rng=True):
    correct, count = 0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        inputs, labels, integration_times = prep_batch(batch, seq_len, in_dim)
        noise_key, sub = random.split(noise_key)
        logits = apply_model(model_cls, params, batch_stats,
                             inputs, integration_times, sub,
                             needs_noise_rng=needs_noise_rng)
        acc = compute_accuracy(logits, labels)
        correct += int(np.sum(acc))
        count += int(acc.shape[0])
    return correct / count if count > 0 else 0.0, noise_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--use_mambino_ssm", type=lambda x: x.lower() == "true",
                        default=False)
    parser.add_argument("--glu_rank", type=int, default=0)
    parser.add_argument("--dir_name", type=str, default="./raw_datasets")
    parser.add_argument("--bsz", type=int, default=50)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--ssm_size_base", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--conj_sym", type=lambda x: x.lower() == "true",
                        default=True)
    parser.add_argument("--jax_seed", type=int, default=6554595)
    args = parser.parse_args()

    print(f"[sanity] Loading {args.ckpt_path}")
    with open(args.ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    saved_acc = ckpt.get("test_acc")
    print(f"[sanity] Checkpoint saved test_acc: {saved_acc:.4f}")
    print(f"[sanity] use_mambino_ssm={args.use_mambino_ssm}, glu_rank={args.glu_rank}")

    # Dataset -- use SAME batch size as training so BN running stats match.
    create_dataset_fn = Datasets["listops-classification"]
    _, valloader, testloader, _, _, seq_len, in_dim, _ = \
        create_dataset_fn(args.dir_name, seed=args.jax_seed, bsz=args.bsz)

    params = ckpt["params"]
    batch_stats = ckpt.get("batch_stats")

    # ═════════════════════ TEST 1: Fast path baseline ═════════════════════
    print("\n" + "="*60)
    print("TEST 1: Fast path (sigma=0, bits=0) reproduces saved test_acc")
    print("="*60)
    ssm_init = build_ssm_init(args, args.use_mambino_ssm, args.glu_rank, 0.0, 0)
    model_cls = build_model(args, ssm_init, args.glu_rank)
    key = random.PRNGKey(0)
    # Fast path: sigma=0, bits=0 -> no noise RNG needed.  Match training's
    # validate() exactly (no rngs passed).
    baseline_test, _ = eval_batches(model_cls, params, batch_stats,
                                     testloader, seq_len, in_dim, key,
                                     needs_noise_rng=False)
    print(f"[TEST 1] baseline test_acc: {baseline_test:.4f}")
    print(f"[TEST 1] saved test_acc:    {saved_acc:.4f}")
    delta1 = abs(baseline_test - saved_acc)
    ok1 = delta1 < 0.01
    print(f"[TEST 1] delta: {delta1:.4f}  {'PASS' if ok1 else 'FAIL (must be <0.01)'}")

    # ═════════════════════ TEST 2: Noise IS injected ═════════════════════
    print("\n" + "="*60)
    print("TEST 2: Same input under noise gives DIFFERENT output (2 runs)")
    print("="*60)
    ssm_init_n = build_ssm_init(args, args.use_mambino_ssm, args.glu_rank, 0.10, 0)
    model_cls_n = build_model(args, ssm_init_n, args.glu_rank)
    # Grab 1 batch
    for batch in testloader:
        inputs, labels, integration_times = prep_batch(batch, seq_len, in_dim)
        break
    key_a = random.PRNGKey(42)
    key_b = random.PRNGKey(99)
    logits_a = apply_model(model_cls_n, params, batch_stats,
                            inputs, integration_times, key_a)
    logits_b = apply_model(model_cls_n, params, batch_stats,
                            inputs, integration_times, key_b)
    max_diff = float(np.max(np.abs(logits_a - logits_b)))
    print(f"[TEST 2] max |logits_a - logits_b| = {max_diff:.4f}")
    ok2 = max_diff > 1e-4
    print(f"[TEST 2] {'PASS -- noise IS being injected' if ok2 else 'FAIL -- noise appears to be no-op'}")

    # ═════════════════════ TEST 3: Fast path IS deterministic ═════════════
    print("\n" + "="*60)
    print("TEST 3: Fast path (sigma=0) with different keys gives IDENTICAL output")
    print("="*60)
    logits_c = apply_model(model_cls, params, batch_stats,
                            inputs, integration_times, key_a,
                            needs_noise_rng=False)
    logits_d = apply_model(model_cls, params, batch_stats,
                            inputs, integration_times, key_b,
                            needs_noise_rng=False)
    max_diff2 = float(np.max(np.abs(logits_c - logits_d)))
    print(f"[TEST 3] max |logits_c - logits_d| = {max_diff2:.6e}")
    ok3 = max_diff2 < 1e-4
    print(f"[TEST 3] {'PASS -- fast path is deterministic' if ok3 else 'FAIL -- unexpected stochasticity'}")

    # ═════════════════════ TEST 4: High noise DEGRADES accuracy ═════════
    print("\n" + "="*60)
    print("TEST 4: High noise (sigma=0.20) reduces accuracy from baseline")
    print("="*60)
    ssm_init_h = build_ssm_init(args, args.use_mambino_ssm, args.glu_rank, 0.20, 0)
    model_cls_h = build_model(args, ssm_init_h, args.glu_rank)
    key = random.PRNGKey(7)
    noisy_acc, _ = eval_batches(model_cls_h, params, batch_stats,
                                 testloader, seq_len, in_dim, key,
                                 max_batches=10)
    print(f"[TEST 4] noisy test_acc (10 batches, sigma=0.20): {noisy_acc:.4f}")
    print(f"[TEST 4] baseline (full loader, sigma=0):          {baseline_test:.4f}")
    ok4 = noisy_acc < baseline_test
    print(f"[TEST 4] {'PASS -- noise degrades accuracy' if ok4 else 'FAIL -- noise did NOT reduce accuracy'}")

    # ═════════════════════ TEST 5: ADC quantization works ══════════════
    print("\n" + "="*60)
    print("TEST 5: ADC quantization at bits=4 gives different output vs no-quant")
    print("="*60)
    ssm_init_q = build_ssm_init(args, args.use_mambino_ssm, args.glu_rank, 0.0, 4)
    model_cls_q = build_model(args, ssm_init_q, args.glu_rank)
    logits_q = apply_model(model_cls_q, params, batch_stats,
                            inputs, integration_times, key_a)
    max_diff3 = float(np.max(np.abs(logits_q - logits_c)))
    print(f"[TEST 5] max |logits_bits4 - logits_bits0| = {max_diff3:.4f}")
    ok5 = max_diff3 > 1e-4
    print(f"[TEST 5] {'PASS -- ADC quant IS applied' if ok5 else 'FAIL -- ADC quant appears to be no-op'}")

    # ═════════════════════ Summary ═════════════════════════════════════
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    tests = [("Fast path baseline reproduces saved acc", ok1),
             ("Noise injection alters output", ok2),
             ("Fast path is deterministic", ok3),
             ("High noise degrades accuracy", ok4),
             ("ADC quantization alters output", ok5)]
    for name, ok in tests:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    all_ok = all(ok for _, ok in tests)
    print(f"\n{'ALL PASS -- ready for full sweep' if all_ok else 'FAILURES -- do not run full sweep yet'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
