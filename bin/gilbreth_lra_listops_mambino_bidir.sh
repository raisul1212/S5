#!/bin/bash -l
# Gilbreth SLURM wrapper for Mambino-in-S5 with SYMMETRIC BIDIRECTIONAL
# predictor + main scan on LRA-ListOps.
#
# Deltas vs the standard Mambino-in-S5 run (11169698, full-rank half_glu2):
#   --bidir_predictor=True
#
# What changes architecturally:
#   Before (11169698): forward-only predictor, bidirectional main scan.
#                      Backward main consumes ε computed from forward predictor
#                      only -- asymmetric between direction and information flow.
#   Now:               predictor is bidirectional too.  Forward reads s_fwd(t-1)
#                      via C_s (causal past), backward reads s_bwd(t+1) via C_s2
#                      (anti-causal future).  Both shifts prevent trivial x(t)
#                      identity prediction.  ε = x - (x_hat_fwd + x_hat_bwd)
#                      carries "residual after full-context prediction."
#
# Param cost: +2,048/layer for new C_s2, total 254,218 params (+6.9% over
# standard Mambino-in-S5's 237,834).  Not iso-param -- this is the RICH
# variant that shows the ceiling of what the predictor pathway can do
# when given full context during training.
#
# All other args match 11169698 exactly (seed 6554595, epochs=40,
# bidirectional=True for main, BfastandCdecay, batchnorm, etc).
#SBATCH --job-name=s5-mamb-bidir
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/s5_mambino_bidir_listops_%j.out
#SBATCH --error=results/slurm/s5_mambino_bidir_listops_%j.err

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

JOB=$SLURM_JOB_ID
SHA=$(git rev-parse HEAD)
CKPT_DIR="./checkpoints/mambino_bidir_${JOB}"
mkdir -p "$CKPT_DIR"

echo "Bidirectional-predictor Mambino-in-S5 LRA-ListOps (JOB=$JOB, SHA=$SHA)"
echo "Design: forward-only predictor -> symmetric bidirectional predictor"
echo "Checkpoints: $CKPT_DIR"

python -u run_train.py \
    --use_mambino_ssm=True \
    --lambda_pc=0.0 \
    --bidir_predictor=True \
    --ckpt_dir="$CKPT_DIR" \
    \
    --C_init=lecun_normal --activation_fn=half_glu2 --batchnorm=True \
    --bidirectional=True --blocks=8 --bsz=50 --d_model=128 \
    --dataset=listops-classification \
    --epochs=40 --jax_seed=6554595 --lr_factor=3 --n_layers=8 \
    --opt_config=BfastandCdecay \
    --p_dropout=0 --ssm_lr_base=0.001 --ssm_size_base=16 \
    --warmup_end=1 --weight_decay=0.04 \
    \
    2>&1 | tee "$CKPT_DIR/run.log"

echo "Done $(date)"
