"""Offline chip sweep on a saved msgpack checkpoint.

Loads best.msgpack (or a specified checkpoint prefix), rebuilds the model
with the correct chip knobs per (sigma, bits) cell, runs validate() on
val+test, writes CSV.

No training. No side effects on the checkpoint.  Safe to re-run.

Usage:
    python chip_sweep.py \
        --ckpt_prefix checkpoints/chip_pure_s5_11181830/best \
        --use_mambino_ssm=False \
        --activation_fn=half_glu2 \
        --glu_rank=0 \
        --dataset=listops-classification \
        --d_model=128 --ssm_size_base=16 --n_layers=8 --blocks=8 \
        --bidirectional=True --C_init=lecun_normal --batchnorm=True \
        --conj_sym=True --sigmas=0,0.005,0.01,0.02,0.05,0.08 \
        --bits=0,4,5,6,7,8 \
        --csv chip_pure_s5_sweep.csv
"""
import argparse
import csv
import os
from functools import partial

from jax import random
import jax.numpy as np
from jax.scipy.linalg import block_diag

from s5.utils.util import str2bool
from s5.train_helpers import (
    create_train_state, validate, load_checkpoint_msgpack,
)
from s5.dataloading import Datasets
from s5.seq_model import BatchClassificationModel, RetrievalModel
from s5.ssm import init_S5SSM
from s5.ssm_init import make_DPLR_HiPPO
from s5.mambino_ssm import init_MambinoSSM


def build_ssm_init_fn(args, ssm_size, block_size, Lambda, V, Vinv,
                      sigma, bits):
    """Build SSM init fn for a sweep cell.  DAC and ADC precision are
    driven from a single `bits` value: they share the analog<->digital
    boundary between adjacent layers on the physical chip, so specifying
    them independently would explore unrealistic configurations (e.g.
    6-bit ADC feeding a 3-bit DAC).  Sweep stays 6x6 (sigma x bits)."""
    if args.use_mambino_ssm:
        return init_MambinoSSM(
            H=args.d_model, P=ssm_size,
            Lambda_re_init=Lambda.real, Lambda_im_init=Lambda.imag,
            V=V, Vinv=Vinv,
            C_init=args.C_init, discretization=args.discretization,
            dt_min=args.dt_min, dt_max=args.dt_max,
            conj_sym=args.conj_sym, clip_eigs=args.clip_eigs,
            bidirectional=args.bidirectional,
            bidir_predictor=args.bidir_predictor,
            noise_sigma=sigma, adc_bits=bits, dac_bits=bits,
        )
    return init_S5SSM(
        H=args.d_model, P=ssm_size,
        Lambda_re_init=Lambda.real, Lambda_im_init=Lambda.imag,
        V=V, Vinv=Vinv,
        C_init=args.C_init, discretization=args.discretization,
        dt_min=args.dt_min, dt_max=args.dt_max,
        conj_sym=args.conj_sym, clip_eigs=args.clip_eigs,
        bidirectional=args.bidirectional,
        noise_sigma=sigma, adc_bits=bits, dac_bits=bits,
    )


def build_model_cls(args, ssm_init_fn, n_classes, padded, retrieval):
    if retrieval:
        return partial(
            RetrievalModel,
            ssm=ssm_init_fn, d_output=n_classes, d_model=args.d_model,
            n_layers=args.n_layers, padded=padded,
            activation=args.activation_fn, dropout=args.p_dropout,
            prenorm=args.prenorm, batchnorm=args.batchnorm,
            bn_momentum=args.bn_momentum,
        )
    return partial(
        BatchClassificationModel,
        ssm=ssm_init_fn, d_output=n_classes, d_model=args.d_model,
        n_layers=args.n_layers, padded=padded,
        activation=args.activation_fn, dropout=args.p_dropout,
        mode=args.mode, prenorm=args.prenorm, batchnorm=args.batchnorm,
        bn_momentum=args.bn_momentum,
        glu_rank=args.glu_rank,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_prefix", type=str, required=True,
                   help="Path prefix; loads <prefix>.msgpack + <prefix>.meta.pkl")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--dir_name", type=str, default="./cache_dir")
    p.add_argument("--sigmas", type=str, required=True)
    p.add_argument("--bits", type=str, required=True)
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--use_mambino_ssm", type=str2bool, default=False)
    p.add_argument("--bidir_predictor", type=str2bool, default=False)
    # model geometry (must match training)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--ssm_size_base", type=int, default=16)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--C_init", type=str, default="lecun_normal")
    p.add_argument("--discretization", type=str, default="zoh")
    p.add_argument("--mode", type=str, default="pool")
    p.add_argument("--activation_fn", type=str, default="half_glu2")
    p.add_argument("--conj_sym", type=str2bool, default=True)
    p.add_argument("--clip_eigs", type=str2bool, default=False)
    p.add_argument("--bidirectional", type=str2bool, default=True)
    p.add_argument("--dt_min", type=float, default=0.001)
    p.add_argument("--dt_max", type=float, default=0.1)
    p.add_argument("--prenorm", type=str2bool, default=True)
    p.add_argument("--batchnorm", type=str2bool, default=True)
    p.add_argument("--bn_momentum", type=float, default=0.95)
    p.add_argument("--bsz", type=int, default=50)
    p.add_argument("--p_dropout", type=float, default=0.0)
    p.add_argument("--glu_rank", type=int, default=0)
    p.add_argument("--jax_seed", type=int, default=6554595)
    # optimizer plumbing (only needed for create_train_state template)
    p.add_argument("--ssm_lr_base", type=float, default=1e-3)
    p.add_argument("--lr_factor", type=float, default=3.0)
    p.add_argument("--weight_decay", type=float, default=0.04)
    p.add_argument("--opt_config", type=str, default="BfastandCdecay")
    p.add_argument("--dt_global", type=str2bool, default=False)
    args = p.parse_args()

    sigmas = [float(x) for x in args.sigmas.split(",")]
    bits_list = [int(x) for x in args.bits.split(",")]

    padded = args.dataset in ("imdb-classification",
                              "listops-classification",
                              "aan-classification")
    retrieval = args.dataset == "aan-classification"

    key = random.PRNGKey(args.jax_seed)
    init_rng, _ = random.split(key)

    # dataset
    create_dataset_fn = Datasets[args.dataset]
    trainloader, valloader, testloader, aux, n_classes, seq_len, in_dim, tsize \
        = create_dataset_fn(args.dir_name, seed=args.jax_seed, bsz=args.bsz)
    print(f"[chip_sweep] dataset={args.dataset} seq_len={seq_len} "
          f"in_dim={in_dim} n_classes={n_classes}")

    # SSM eigendecomp (identical to train.py)
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

    # Build clean model_cls + template state so we can load msgpack
    clean_ssm = build_ssm_init_fn(args, ssm_size, block_size,
                                  Lambda, V, Vinv, sigma=0.0, bits=0)
    clean_model_cls = build_model_cls(args, clean_ssm, n_classes,
                                      padded, retrieval)
    template_state = create_train_state(
        clean_model_cls, init_rng, padded, retrieval,
        in_dim=in_dim, bsz=args.bsz, seq_len=seq_len,
        weight_decay=args.weight_decay, batchnorm=args.batchnorm,
        opt_config=args.opt_config,
        ssm_lr=args.ssm_lr_base, lr=args.ssm_lr_base * args.lr_factor,
        dt_global=args.dt_global,
    )

    state, meta = load_checkpoint_msgpack(args.ckpt_prefix, template_state)
    print(f"[chip_sweep] loaded {args.ckpt_prefix}.msgpack  meta={meta}")

    # Verify clean baseline
    _, v_acc0 = validate(state, clean_model_cls, valloader,
                         seq_len, in_dim, args.batchnorm)
    _, t_acc0 = validate(state, clean_model_cls, testloader,
                         seq_len, in_dim, args.batchnorm)
    print(f"[chip_sweep] clean baseline val={v_acc0:.4f}  test={t_acc0:.4f}")

    # Sweep
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma", "bits", "val_acc", "test_acc",
                    "val_loss", "test_loss"])
        for sigma in sigmas:
            for bits in bits_list:
                n_ssm = build_ssm_init_fn(args, ssm_size, block_size,
                                          Lambda, V, Vinv, sigma, bits)
                n_cls = build_model_cls(args, n_ssm, n_classes,
                                        padded, retrieval)
                # Always pass a valid rng.  The SSM's non-fast path calls
                # self.make_rng('noise') unconditionally (before checking
                # sigma), and JIT caching across different (sigma, bits)
                # traces can otherwise leave stale None-rng compilations.
                # Unused rng is harmless.
                nrs = 42
                v_loss, v_acc = validate(state, n_cls, valloader,
                                         seq_len, in_dim, args.batchnorm,
                                         noise_rng_seed=nrs)
                t_loss, t_acc = validate(state, n_cls, testloader,
                                         seq_len, in_dim, args.batchnorm,
                                         noise_rng_seed=nrs)
                print(f"[chip_sweep] sigma={sigma:.4f} bits={bits} "
                      f"val={v_acc:.4f} test={t_acc:.4f}")
                w.writerow([sigma, bits, f"{v_acc:.4f}", f"{t_acc:.4f}",
                            f"{v_loss:.4f}", f"{t_loss:.4f}"])
    print(f"[chip_sweep] wrote {args.csv}")


if __name__ == "__main__":
    main()
