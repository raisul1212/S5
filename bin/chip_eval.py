"""Chip-analysis evaluation: load a best.pkl checkpoint, sweep analog
noise sigma + ADC bit depth, report LRA-ListOps test accuracy under each
condition.

The point: prove Mambino's analog-scalability claim.  Mambino has ~9
analog compute sites per layer (predictor + main SSM + W_eps) vs pure
S5's ~5 (main SSM alone).  Under signal-proportional Gaussian noise
injected ONLY at these analog sites (digital gate compute remains
noise-free), can Mambino maintain accuracy comparable to pure S5
despite the 1.8x more analog exposure?

Usage:
  python bin/chip_eval.py \
      --ckpt_path=checkpoints/pure_s5_11170658/best.pkl \
      --use_mambino_ssm=False \
      --output=chip_eval_pure_s5.csv

  python bin/chip_eval.py \
      --ckpt_path=checkpoints/mambino_iso_p8_11171006/best.pkl \
      --use_mambino_ssm=True \
      --glu_rank=40 \
      --output=chip_eval_mambino_iso.csv

CSV output: sigma, adc_bits, val_acc, test_acc
"""
import argparse
import csv
import pickle
import sys
import os
from functools import partial

import jax
import jax.numpy as np
from jax import random
from jax.scipy.linalg import block_diag

# Import from the s5 package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s5.dataloading import Datasets
from s5.ssm import init_S5SSM
from s5.ssm_init import make_DPLR_HiPPO
from s5.mambino_ssm import init_MambinoSSM
from s5.seq_model import BatchClassificationModel
from s5.train_helpers import prep_batch, compute_accuracy, cross_entropy_loss


def build_model(args, ssm_init_fn):
    """Construct the BatchClassificationModel matching the training config."""
    return partial(
        BatchClassificationModel,
        ssm=ssm_init_fn,
        d_output=10,               # LRA-ListOps 10 classes
        d_model=args.d_model,
        n_layers=args.n_layers,
        padded=True,               # ListOps is padded
        activation=args.activation_fn,
        dropout=0.0,               # eval only
        mode="pool",
        prenorm=False,
        batchnorm=args.batchnorm,
        bn_momentum=0.9,
        glu_rank=args.glu_rank,
    )


def make_ssm_init(args, noise_sigma, adc_bits):
    """Build the SSM initializer with the given chip-analysis knobs."""
    ssm_size = args.ssm_size_base
    block_size = int(ssm_size / args.blocks)
    Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)
    if args.conj_sym:
        block_size = block_size // 2
        ssm_size = ssm_size // 2
    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T
    Lambda = (Lambda * np.ones((args.blocks, block_size))).ravel()
    V = block_diag(*([V] * args.blocks))
    Vinv = block_diag(*([Vc] * args.blocks))

    if args.use_mambino_ssm:
        return init_MambinoSSM(
            H=args.d_model, P=ssm_size,
            Lambda_re_init=Lambda.real, Lambda_im_init=Lambda.imag,
            V=V, Vinv=Vinv,
            C_init=args.C_init, discretization=args.discretization,
            dt_min=args.dt_min, dt_max=args.dt_max,
            conj_sym=args.conj_sym, clip_eigs=args.clip_eigs,
            bidirectional=args.bidirectional,
            bidir_predictor=False,
            noise_sigma=noise_sigma, adc_bits=adc_bits,
        )
    else:
        return init_S5SSM(
            H=args.d_model, P=ssm_size,
            Lambda_re_init=Lambda.real, Lambda_im_init=Lambda.imag,
            V=V, Vinv=Vinv,
            C_init=args.C_init, discretization=args.discretization,
            dt_min=args.dt_min, dt_max=args.dt_max,
            conj_sym=args.conj_sym, clip_eigs=args.clip_eigs,
            bidirectional=args.bidirectional,
            noise_sigma=noise_sigma, adc_bits=adc_bits,
        )


def eval_loader(state_params, batch_stats, model, loader, seq_len, in_dim,
                batchnorm, noise_key):
    """Evaluate on the full loader, returning mean accuracy.

    Noise RNG: we pass a rng under key 'noise' so the modules'
    make_rng('noise') calls produce fresh keys per site per batch.
    """
    variables = {"params": state_params}
    if batchnorm:
        variables["batch_stats"] = batch_stats
    total_correct = 0
    total_count = 0
    for batch_idx, batch in enumerate(loader):
        inputs, labels, integration_times = prep_batch(batch, seq_len, in_dim)
        noise_key, sub = random.split(noise_key)
        logits = model.apply(
            variables, inputs, integration_times,
            rngs={"noise": sub},
        )
        acc = compute_accuracy(logits, labels)
        total_correct += int(np.sum(acc))
        total_count += int(acc.shape[0])
    return total_correct / total_count, noise_key


def run_condition(args, ckpt, sigma, bits, seed):
    """Load model, apply the chip knobs, evaluate val + test."""
    # Build datasets
    create_dataset_fn = Datasets[args.dataset]
    trainloader, valloader, testloader, aux, n_classes, seq_len, in_dim, train_size = \
        create_dataset_fn(args.dir_name, seed=args.jax_seed, bsz=args.bsz)

    # Build model with chip knobs
    ssm_init_fn = make_ssm_init(args, noise_sigma=sigma, adc_bits=bits)
    model_cls = build_model(args, ssm_init_fn)
    model = model_cls(training=False)

    # Eval
    noise_key = random.PRNGKey(seed)
    val_acc, noise_key = eval_loader(
        ckpt["params"], ckpt.get("batch_stats"), model, valloader,
        seq_len, in_dim, args.batchnorm, noise_key)
    test_acc, noise_key = eval_loader(
        ckpt["params"], ckpt.get("batch_stats"), model, testloader,
        seq_len, in_dim, args.batchnorm, noise_key)
    return val_acc, test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--use_mambino_ssm", type=lambda x: x.lower() == "true",
                        default=False)
    parser.add_argument("--glu_rank", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="listops-classification")
    parser.add_argument("--dir_name", type=str, default="./raw_datasets")
    parser.add_argument("--bsz", type=int, default=50)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--ssm_size_base", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--C_init", type=str, default="lecun_normal")
    parser.add_argument("--activation_fn", type=str, default="half_glu2")
    parser.add_argument("--batchnorm", type=lambda x: x.lower() == "true",
                        default=True)
    parser.add_argument("--bidirectional", type=lambda x: x.lower() == "true",
                        default=True)
    parser.add_argument("--conj_sym", type=lambda x: x.lower() == "true",
                        default=True)
    parser.add_argument("--clip_eigs", type=lambda x: x.lower() == "true",
                        default=False)
    parser.add_argument("--discretization", type=str, default="zoh")
    parser.add_argument("--dt_min", type=float, default=0.001)
    parser.add_argument("--dt_max", type=float, default=0.1)
    parser.add_argument("--jax_seed", type=int, default=6554595)
    parser.add_argument("--eval_seed", type=int, default=0)
    parser.add_argument("--sigma_sweep", type=str,
                        default="0,0.01,0.02,0.05,0.08,0.12")
    parser.add_argument("--bits_sweep", type=str,
                        default="0,4,5,6,7,8")  # 0 = no quant, baseline
    args = parser.parse_args()

    print(f"[chip_eval] loading checkpoint {args.ckpt_path}")
    with open(args.ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    print(f"[chip_eval] checkpoint epoch={ckpt.get('epoch')}, "
          f"test_acc={ckpt.get('test_acc'):.4f}")

    sigmas = [float(s) for s in args.sigma_sweep.split(",")]
    bits_list = [int(b) for b in args.bits_sweep.split(",")]

    print(f"[chip_eval] sigma sweep: {sigmas}")
    print(f"[chip_eval] bits sweep:  {bits_list}")

    # Baseline first: sigma=0, bits=0 (no noise, no quant)
    print(f"\n[chip_eval] === BASELINE (sigma=0, bits=0) ===")
    val0, test0 = run_condition(args, ckpt, 0.0, 0, args.eval_seed)
    print(f"[chip_eval] baseline: val={val0:.4f} test={test0:.4f}")

    rows = [{"phase": "baseline", "sigma": 0.0, "bits": 0,
             "val_acc": val0, "test_acc": test0}]

    # Noise sweep at bits=0 (no quantization, only noise)
    print(f"\n[chip_eval] === NOISE SWEEP (adc_bits=0) ===")
    for sigma in sigmas:
        if sigma == 0:
            continue  # already have baseline
        val_a, test_a = run_condition(args, ckpt, sigma, 0, args.eval_seed)
        print(f"[chip_eval] sigma={sigma:.3f}  val={val_a:.4f}  test={test_a:.4f}")
        rows.append({"phase": "noise_sweep", "sigma": sigma, "bits": 0,
                     "val_acc": val_a, "test_acc": test_a})

    # ADC bit sweep at sigma=0 (no noise, only quantization)
    print(f"\n[chip_eval] === ADC QUANT SWEEP (sigma=0) ===")
    for bits in bits_list:
        if bits == 0:
            continue
        val_a, test_a = run_condition(args, ckpt, 0.0, bits, args.eval_seed)
        print(f"[chip_eval] bits={bits}  val={val_a:.4f}  test={test_a:.4f}")
        rows.append({"phase": "adc_sweep", "sigma": 0.0, "bits": bits,
                     "val_acc": val_a, "test_acc": test_a})

    # Combined chip-realistic: sigma=0.02 + ADC bits
    print(f"\n[chip_eval] === COMBINED (sigma=0.02 + bit sweep) ===")
    for bits in bits_list:
        if bits == 0:
            continue
        val_a, test_a = run_condition(args, ckpt, 0.02, bits, args.eval_seed)
        print(f"[chip_eval] sigma=0.02 bits={bits}  val={val_a:.4f}  test={test_a:.4f}")
        rows.append({"phase": "combined", "sigma": 0.02, "bits": bits,
                     "val_acc": val_a, "test_acc": test_a})

    # Write CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\n[chip_eval] wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
