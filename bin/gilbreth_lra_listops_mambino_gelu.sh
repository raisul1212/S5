#!/bin/bash -l
# Mambino-in-S5 P=8 with activation=gelu -- NO half_glu2 gate at all.
# Tests whether Mambino's predictor pathway alone is sufficient
# for LRA-ListOps without the block-level gating.
#
# Expected params: ~106K (main SSM + predictor + W_eps + BN + encoder + head).
# This is the "minimum viable Mambino": SSM + predictor + gelu.
#
# Matched to gilbreth_lra_listops_pure_s5_gelu.sh -- same seed,
# same activation, same everything except Mambino's predictor branch
# is present.  Direct test of "predictor alone vs gate alone".
#SBATCH --job-name=s5-mamb-gelu
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/s5_mambino_gelu_listops_%j.out
#SBATCH --error=results/slurm/s5_mambino_gelu_listops_%j.err

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
CKPT_DIR="./checkpoints/mambino_gelu_${JOB}"
mkdir -p "$CKPT_DIR"

echo "Mambino-in-S5 P=8 activation=gelu (no gate) LRA-ListOps (JOB=$JOB, SHA=$SHA)"
echo "Checkpoints: $CKPT_DIR"

python -u run_train.py \
    --use_mambino_ssm=True \
    --lambda_pc=0.0 \
    --ckpt_dir="$CKPT_DIR" \
    \
    --C_init=lecun_normal --activation_fn=gelu --batchnorm=True \
    --bidirectional=True --blocks=8 --bsz=50 --d_model=128 \
    --dataset=listops-classification \
    --epochs=40 --jax_seed=6554595 --lr_factor=3 --n_layers=8 \
    --opt_config=BfastandCdecay \
    --p_dropout=0 --ssm_lr_base=0.001 --ssm_size_base=16 \
    --warmup_end=1 --weight_decay=0.04 \
    \
    2>&1 | tee "$CKPT_DIR/run.log"

echo "Done $(date)"
