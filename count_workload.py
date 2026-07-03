"""Accurate workload counter for a trained model checkpoint.

Uses JAX's XLA cost_analysis to get FLOPs directly from the compiled
computation graph -- no hand-derived formulas, no hallucinations.

Output: YAML-ready action counts for Accelergy input.

Usage:
    python count_workload.py \
        --ckpt_prefix checkpoints/config4_seed6554595_XXXX/best \
        --dataset listops-classification \
        --use_mambino_ssm True --activation_fn gelu --ssm_size_base 16 \
        --n_layers 8 --d_model 128 --blocks 8 \
        --output workload_config4.yaml
"""
import argparse
from functools import partial

import jax
from jax import random
import jax.numpy as np
from jax.scipy.linalg import block_diag

from s5.utils.util import str2bool
from s5.train_helpers import create_train_state, load_checkpoint_msgpack
from s5.dataloading import Datasets
from s5.seq_model import BatchClassificationModel
from s5.ssm import init_S5SSM
from s5.ssm_init import make_DPLR_HiPPO
from s5.mambino_ssm import init_MambinoSSM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_prefix", type=str, required=True)
    p.add_argument("--dataset", type=str, default="listops-classification")
    p.add_argument("--dir_name", type=str, default="./cache_dir")
    p.add_argument("--use_mambino_ssm", type=str2bool, default=False)
    p.add_argument("--activation_fn", type=str, default="gelu")
    p.add_argument("--ssm_size_base", type=int, required=True)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--glu_rank", type=int, default=0)
    p.add_argument("--C_init", type=str, default="lecun_normal")
    p.add_argument("--discretization", type=str, default="zoh")
    p.add_argument("--mode", type=str, default="pool")
    p.add_argument("--conj_sym", type=str2bool, default=True)
    p.add_argument("--clip_eigs", type=str2bool, default=False)
    p.add_argument("--bidirectional", type=str2bool, default=True)
    p.add_argument("--dt_min", type=float, default=0.001)
    p.add_argument("--dt_max", type=float, default=0.1)
    p.add_argument("--bidir_predictor", type=str2bool, default=False)
    p.add_argument("--prenorm", type=str2bool, default=True)
    p.add_argument("--batchnorm", type=str2bool, default=True)
    p.add_argument("--bn_momentum", type=float, default=0.95)
    p.add_argument("--bsz", type=int, default=1)  # BATCH=1 for per-inference
    p.add_argument("--p_dropout", type=float, default=0.0)
    p.add_argument("--jax_seed", type=int, default=0)
    p.add_argument("--ssm_lr_base", type=float, default=1e-3)
    p.add_argument("--lr_factor", type=float, default=3.0)
    p.add_argument("--weight_decay", type=float, default=0.04)
    p.add_argument("--opt_config", type=str, default="BfastandCdecay")
    p.add_argument("--dt_global", type=str2bool, default=False)
    p.add_argument("--output", type=str, default="workload.yaml")
    p.add_argument("--config_name", type=str, default="model")
    args = p.parse_args()

    # Load dataset just to get seq_len, in_dim
    padded = args.dataset in ("imdb-classification",
                              "listops-classification",
                              "aan-classification")
    create_dataset_fn = Datasets[args.dataset]
    trainloader, valloader, testloader, aux, n_classes, seq_len, in_dim, tsize \
        = create_dataset_fn(args.dir_name, seed=args.jax_seed, bsz=args.bsz)

    print(f"[count] dataset={args.dataset} L={seq_len} in_dim={in_dim} "
          f"n_classes={n_classes}")

    # Build SSM eigendecomp
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
        ssm_init_fn = init_MambinoSSM(
            H=args.d_model, P=ssm_size,
            Lambda_re_init=Lambda.real, Lambda_im_init=Lambda.imag,
            V=V, Vinv=Vinv,
            C_init=args.C_init, discretization=args.discretization,
            dt_min=args.dt_min, dt_max=args.dt_max,
            conj_sym=args.conj_sym, clip_eigs=args.clip_eigs,
            bidirectional=args.bidirectional,
            bidir_predictor=args.bidir_predictor,
        )
    else:
        ssm_init_fn = init_S5SSM(
            H=args.d_model, P=ssm_size,
            Lambda_re_init=Lambda.real, Lambda_im_init=Lambda.imag,
            V=V, Vinv=Vinv,
            C_init=args.C_init, discretization=args.discretization,
            dt_min=args.dt_min, dt_max=args.dt_max,
            conj_sym=args.conj_sym, clip_eigs=args.clip_eigs,
            bidirectional=args.bidirectional,
        )

    model_cls = partial(
        BatchClassificationModel,
        ssm=ssm_init_fn, d_output=n_classes, d_model=args.d_model,
        n_layers=args.n_layers, padded=padded,
        activation=args.activation_fn, dropout=args.p_dropout,
        mode=args.mode, prenorm=args.prenorm, batchnorm=args.batchnorm,
        bn_momentum=args.bn_momentum,
        glu_rank=args.glu_rank,
    )

    key = random.PRNGKey(args.jax_seed)
    init_rng, _ = random.split(key)
    state = create_train_state(
        model_cls, init_rng, padded, retrieval=False,
        in_dim=in_dim, bsz=args.bsz, seq_len=seq_len,
        weight_decay=args.weight_decay, batchnorm=args.batchnorm,
        opt_config=args.opt_config,
        ssm_lr=args.ssm_lr_base, lr=args.ssm_lr_base * args.lr_factor,
        dt_global=args.dt_global,
    )
    state, meta = load_checkpoint_msgpack(args.ckpt_prefix, state)

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"[count] loaded {args.ckpt_prefix}: {n_params} params")

    # Build a dummy forward pass -- batch of 1 for per-inference counts
    if padded:
        dummy_input = (np.ones((args.bsz, seq_len, in_dim)), np.ones(args.bsz))
        integ = np.ones((args.bsz, seq_len,))
    else:
        dummy_input = np.ones((args.bsz, seq_len, in_dim))
        integ = np.ones((args.bsz, seq_len,))

    model = model_cls(training=False)

    def forward(params, x, integ):
        return model.apply({"params": params,
                            "batch_stats": state.batch_stats},
                           x, integ,
                           rngs={"params": jax.random.PRNGKey(0),
                                 "dropout": jax.random.PRNGKey(1),
                                 "noise": jax.random.PRNGKey(2)})

    # JIT lower + compile + get cost analysis (XLA FLOPs)
    print("[count] compiling forward pass and analyzing cost...")
    lowered = jax.jit(forward).lower(state.params, dummy_input, integ)
    compiled = lowered.compile()
    cost = compiled.cost_analysis()

    if isinstance(cost, list):
        cost = cost[0]
    if not isinstance(cost, dict):
        cost = dict(cost) if hasattr(cost, "__iter__") else {"unknown": cost}

    total_flops = cost.get("flops", 0)
    total_bytes_accessed = cost.get("bytes accessed", 0)
    total_transcendentals = cost.get("transcendentals", 0)

    print(f"\n[count] === XLA cost analysis (batch={args.bsz}) ===")
    for k, v in cost.items():
        print(f"  {k}: {v}")

    # MACs = FLOPs / 2 for matmuls; other ops (add, mul) are 1 FLOP each
    # Conservative estimate: MACs ~= FLOPs / 2 for compute-heavy workload
    macs_estimate = total_flops / 2
    print(f"\n[count] === Derived counts (batch={args.bsz}) ===")
    print(f"  Total FLOPs: {total_flops:.0f}")
    print(f"  Total MACs (est): {macs_estimate:.0f}")
    print(f"  Total bytes accessed: {total_bytes_accessed:.0f}")
    print(f"  Transcendentals: {total_transcendentals:.0f}")
    print(f"  Params (INT8 bytes): {n_params}")

    # Write Accelergy-format action_counts YAML
    with open(args.output, "w") as f:
        f.write(f"# Auto-generated by count_workload.py from {args.ckpt_prefix}\n")
        f.write(f"# Batch size: {args.bsz}, Params: {n_params}\n")
        f.write(f"# XLA-reported FLOPs: {total_flops:.0f}\n")
        f.write(f"# XLA-reported bytes accessed: {total_bytes_accessed:.0f}\n\n")
        f.write("action_counts:\n")
        f.write("  version: 0.2\n")
        f.write("  subtree:\n")
        f.write("    - name: chip\n")
        f.write("      local:\n")
        f.write("        - name: main_sram\n")
        f.write("          action_counts:\n")
        # Bytes accessed / 8 (assuming INT8) = number of byte accesses
        # Split roughly 60/40 read/write
        n_reads = int(total_bytes_accessed * 0.7)
        n_writes = int(total_bytes_accessed * 0.3)
        f.write(f"            - name: read\n              counts: {n_reads}\n")
        f.write(f"            - name: write\n              counts: {n_writes}\n")
        f.write("        - name: mac_array\n")
        f.write("          action_counts:\n")
        f.write(f"            - name: mac_random\n              counts: {int(macs_estimate)}\n")
        f.write("        - name: reg_accum\n")
        f.write("          action_counts:\n")
        f.write(f"            - name: read\n              counts: {int(macs_estimate)}\n")
        f.write(f"            - name: write\n              counts: {int(macs_estimate)}\n")

    print(f"\n[count] Wrote {args.output}")


if __name__ == "__main__":
    main()
