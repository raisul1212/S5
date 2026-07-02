"""Minimal sanity: use train_helpers.validate() directly on the loaded
checkpoint.  If this gives 0.6155, the eval path works.  Then we know
any deviation is a bug in the chip_eval script's custom apply loop.

Reconstructs the train state from the checkpoint by building a fresh
state via create_train_state(), then replacing params + batch_stats
with the checkpoint values.  Calls validate() which is the SAME code
path used at every training epoch to produce test_acc.
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
from s5.train_helpers import create_train_state, validate


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
    parser.add_argument("--jax_seed", type=int, default=6554595)
    args = parser.parse_args()

    print(f"[v2] loading {args.ckpt_path}")
    with open(args.ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    saved_acc = ckpt.get("test_acc")
    print(f"[v2] checkpoint saved test_acc: {saved_acc:.4f}")
    print(f"[v2] ckpt keys: {list(ckpt.keys())}")
    print(f"[v2] params root keys: {list(ckpt['params'].keys())}")
    if "batch_stats" in ckpt:
        print(f"[v2] batch_stats root keys: {list(ckpt['batch_stats'].keys())}")
    else:
        print(f"[v2] NO batch_stats in checkpoint!")

    # Build SSM init
    ssm_size = args.ssm_size_base
    block_size = int(ssm_size / args.blocks)
    Lambda, _, _, V, _ = make_DPLR_HiPPO(block_size)
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
            C_init="lecun_normal", discretization="zoh",
            dt_min=0.001, dt_max=0.1,
            conj_sym=True, clip_eigs=False,
            bidirectional=True, bidir_predictor=False,
            noise_sigma=0.0, adc_bits=0,
        )
    else:
        ssm_init_fn = init_S5SSM(
            H=args.d_model, P=ssm_size,
            Lambda_re_init=Lambda.real, Lambda_im_init=Lambda.imag,
            V=V, Vinv=Vinv,
            C_init="lecun_normal", discretization="zoh",
            dt_min=0.001, dt_max=0.1,
            conj_sym=True, clip_eigs=False,
            bidirectional=True,
            noise_sigma=0.0, adc_bits=0,
        )

    # Build model_cls MATCHING training
    model_cls = partial(
        BatchClassificationModel,
        ssm=ssm_init_fn,
        d_output=10, d_model=args.d_model, n_layers=args.n_layers,
        padded=True, activation="half_glu2", dropout=0.0,
        mode="pool", prenorm=False, batchnorm=True, bn_momentum=0.9,
        glu_rank=args.glu_rank,
    )

    # Dataset
    create_dataset_fn = Datasets["listops-classification"]
    _, valloader, testloader, _, _, seq_len, in_dim, _ = \
        create_dataset_fn(args.dir_name, seed=args.jax_seed, bsz=args.bsz)

    # Create fresh state (random init) then replace params + batch_stats
    print(f"[v2] Building fresh train state to swap in loaded weights...")
    init_rng = random.PRNGKey(args.jax_seed)
    state = create_train_state(model_cls, init_rng,
                                padded=True, retrieval=False,
                                in_dim=in_dim, bsz=args.bsz, seq_len=seq_len,
                                weight_decay=0.04, batchnorm=True,
                                opt_config="BfastandCdecay",
                                ssm_lr=0.001, lr=0.003, dt_global=False)

    print(f"[v2] fresh state params root: {list(state.params.keys())}")
    print(f"[v2] fresh state batch_stats root: {list(state.batch_stats.keys())}")

    # Replace with loaded
    state = state.replace(params=ckpt["params"],
                          batch_stats=ckpt["batch_stats"])
    print(f"[v2] Swapped in loaded params + batch_stats")

    # Run validate
    print(f"[v2] Running validate() on test loader...")
    test_loss, test_acc = validate(state, model_cls, testloader,
                                   seq_len, in_dim, batchnorm=True)
    print(f"[v2] test_acc via validate(): {test_acc:.4f}")
    print(f"[v2] saved test_acc:          {saved_acc:.4f}")
    print(f"[v2] delta: {abs(test_acc - saved_acc):.4f}")
    if abs(test_acc - saved_acc) < 0.01:
        print(f"[v2] PASS -- validate() reproduces saved test_acc")
        sys.exit(0)
    else:
        print(f"[v2] FAIL -- something is broken in the eval path or ckpt")
        sys.exit(1)


if __name__ == "__main__":
    main()
