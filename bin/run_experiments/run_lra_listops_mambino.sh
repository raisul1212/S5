#!/bin/bash
# Mambino-SSM plugged into S5's LRA-ListOps pipeline.
#
# Identical to bin/run_experiments/run_lra_listops.sh EXCEPT for the
# --use_mambino_ssm flag (which routes ssm_init to init_MambinoSSM
# instead of init_S5SSM) and --lambda_pc / --ckpt_dir.
#
# All other knobs mirror S5's proven ListOps recipe exactly:
#   - char-level LRA-standard preprocessing (--dataset=listops-classification)
#   - HiPPO-N block-diag init (blocks=8, ssm_size_base=16 -> P=8 after conj_sym)
#   - bidirectional MAIN scan; predictor scan forward-only (Mambino native)
#   - BfastandCdecay optimizer (extended to tag Lambda_s_*/log_step_s)
#   - half_glu2 + BatchNorm + masked-mean pool (S5 wrapper unchanged)
#
# --lambda_pc=0.0 matches the 0.4965 Mambino hero setting: predictor
# branch is architecturally active (W_eps feedback lives), but no
# separate intrinsic-loss gradient.  L_int is still computed and
# logged (via self.sow('intermediates', 'intrinsic_loss', ...)).
#
# Checkpoints saved to --ckpt_dir/best.pkl (on every val-acc improve)
# and --ckpt_dir/final.pkl (every epoch, overwrites) so runs can be
# reloaded / extended if needed.

CKPT_DIR="${CKPT_DIR:-./checkpoints/lra_listops_mambino}"
mkdir -p "$CKPT_DIR"

python run_train.py \
    --use_mambino_ssm=True \
    --lambda_pc=0.0 \
    --ckpt_dir="$CKPT_DIR" \
    \
    --C_init=lecun_normal --activation_fn=half_glu2 --batchnorm=True \
    --bidirectional=True --blocks=8 --bsz=50 --d_model=128 \
    --dataset=listops-classification \
    --epochs=40 --jax_seed=6554595 --lr_factor=3 --n_layers=8 \
    --opt_config=BfastandCdecay \
    --p_dropout=0 --ssm_lr_base=0.001 --ssm_size_base=16 \
    --warmup_end=1 --weight_decay=0.04
