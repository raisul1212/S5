#!/bin/bash -l
# ============================================================================
# S7-PROTOCOL sweep: what do our architectures do with a REAL training budget?
#
# Every Mambino/S5 number we own was trained for 40 epochs at p_dropout=0.
# S7 reports 63.77% on LRA-ListOps using 200 epochs, dropout 0.235 and a
# 10-epoch warmup -- and compares that against S5's CITED 62.15%, which came
# from a 188,490-param model trained for 40 epochs with no dropout.  Their
# released repo (TaylanSoydan/S7) contains NO S5 baseline config: all 14 model
# configs are input_dependent=true, i.e. S7 itself.  So the published gap is
# uncontrolled for both capacity (~1.08 M vs 188 K) and budget (200 vs 40).
#
# This script closes that confound on OUR side.  It changes ONLY the training
# protocol -- architecture, optimizer, lr and directionality are untouched:
#     epochs      40   -> 200
#     p_dropout   0    -> 0.235   (S7's value)
#     warmup_end  1    -> 10      (S7's 10 warmup epochs)
# Cosine annealing was ALREADY on (--cosine_anneal defaults True), so that part
# of S7's recipe we have always had.
#
# DELIBERATELY NOT MATCHED (and why):
#   ssm_base_lr   S7 uses 1e-5 (x lr_factor 3); we keep our tuned 1e-3.  Moving
#                 lr AND epochs together would confound the very thing we are
#                 trying to isolate.  Worth a follow-up sweep on its own.
#   bidirectional S7 is UNIDIRECTIONAL on ListOps; we stay bidirectional.  That
#                 is architecture, not protocol.
#   prenorm/depth/width  ditto.
#
# CONFIGS (all at d_model=128, blocks=8, n_layers=8, ssm_size_base=16):
#   CONFIG=corner1    Pure S5, half_glu2, full-rank gate   188,490 params
#                     == the S5 authors' OWN ListOps config, bit-for-bit
#   CONFIG=clusterA   Config 4 (Mambino gelu) + surprise gate      105,754
#   CONFIG=clusterB   Cluster A + fast weight (STACKED, arm 8)     123,698
#
# 40-epoch references (8-seed val-selected): corner1 0.6089 | clusterA 0.6050
#   | config4 gateless 0.5993.  Under the single-seed best-epoch convention
#   prior work uses, corner1 is 0.6235 vs S5's published 0.6215 -- i.e. we
#   already reproduce and slightly exceed it at identical params AND budget.
#
# Usage:
#   for C in corner1 clusterA clusterB; do
#     for S in 6554595 42; do
#       CONFIG=$C SEED=$S sbatch bin/gilbreth_lra_listops_s7protocol.sh
#     done
#   done
# ~10-15 h per run at 200 epochs.  a30 MaxTime is UNLIMITED; the 12h in the
# other launchers was self-imposed.
# ============================================================================
#SBATCH --job-name=s5-s7proto
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=30:00:00
#SBATCH --output=results/slurm/s7proto_%j.out
#SBATCH --error=results/slurm/s7proto_%j.err

source /etc/profile.d/modules.sh 2>/dev/null || true
source /etc/profile.d/lmod.sh 2>/dev/null || true
source ~/.bashrc 2>/dev/null || true
module purge
module load external
module load anaconda/2024.10-py312
module load cuda/13.1.0

CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate /scratch/gilbreth/raisul/envs/s5m

export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PYTHONUTF8=1
cd $SLURM_SUBMIT_DIR
mkdir -p results/slurm

CONFIG=${CONFIG:-clusterB}
SEED=${SEED:-6554595}
EPOCHS=${EPOCHS:-200}
DROPOUT=${DROPOUT:-0.235}
WARMUP=${WARMUP:-10}

case "$CONFIG" in
  corner1)
    ARCH="--use_mambino_ssm=False --activation_fn=half_glu2 --glu_rank=0" ;;
  clusterA)
    ARCH="--use_mambino_ssm=True --activation_fn=gelu --lambda_pc=0.0 \
          --surprise_gate=True --gate_range=signed --gate_alpha=0.9 \
          --gate_kappa_init=0.0 --gate_bias_init=2.0" ;;
  clusterB)
    ARCH="--use_mambino_ssm=True --activation_fn=gelu --lambda_pc=0.0 \
          --surprise_gate=True --gate_range=signed --gate_alpha=0.9 \
          --gate_kappa_init=0.0 --gate_bias_init=2.0 \
          --fast_weight=True --fw_dim=8 --fw_proj=shared --fw_rule=hebb \
          --fw_kq_source=x --fw_v_source=eps --fw_impl=chunk --fw_chunk=64 \
          --fw_gate_mode=surprise --fw_kappa_init=0.0 --fw_bias_init=0.0 \
          --fw_gamma_init=0.95 --fw_gamma_trainable=True --fw_norm_qkv=True \
          --fw_out_init=zeros --fw_read=exclusive" ;;
  *) echo "unknown CONFIG=$CONFIG (corner1|clusterA|clusterB)"; exit 2 ;;
esac

JOB=$SLURM_JOB_ID
SHA=$(git rev-parse HEAD)
CKPT_DIR="./checkpoints/s7proto_${CONFIG}_e${EPOCHS}_do${DROPOUT}_s${SEED}_${JOB}"
mkdir -p "$CKPT_DIR"

echo "S7-PROTOCOL | CONFIG=$CONFIG epochs=$EPOCHS dropout=$DROPOUT warmup=$WARMUP"
echo "LRA-ListOps JOB=$JOB SHA=$SHA seed=$SEED ckpt=$CKPT_DIR"

python -u run_train.py \
    $ARCH \
    --ckpt_dir="$CKPT_DIR" \
    \
    --C_init=lecun_normal --batchnorm=True \
    --bidirectional=True --blocks=8 --bsz=50 --d_model=128 \
    --dataset=listops-classification \
    --epochs=$EPOCHS --jax_seed=$SEED --lr_factor=3 --n_layers=8 \
    --opt_config=BfastandCdecay \
    --p_dropout=$DROPOUT --ssm_lr_base=0.001 --ssm_size_base=16 \
    --warmup_end=$WARMUP --weight_decay=0.04 --cosine_anneal=True \
    \
    2>&1 | tee "$CKPT_DIR/run.log"

echo "Done $(date)"
