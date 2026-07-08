#!/bin/bash -l
# lambda_pc ablation sweep on Config 4 (Mambino gelu 106K).
# Tests whether the predictive-coding loss coefficient matters for
# accuracy, or if the two-branch architecture works just as well with
# lambda_pc = 0 (predictor trained only via main-task backprop).
#
# Usage:
#   sbatch bin/gilbreth_lambda_pc_sweep.sh <LAMBDA_PC> [SEED]
#
# Single-seed sweep values (user recommendation, 0.1 = failure ceiling):
#   0.0001, 0.001, 0.01, 0.1
# (lambda_pc = 0 is Config 4's default; already covered by 8-seed Config 4 sweep.)
#SBATCH --job-name=lpc-sweep
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=results/slurm/lpc_sweep_%j.out
#SBATCH --error=results/slurm/lpc_sweep_%j.err

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
export WANDB_MODE=disabled
cd $SLURM_SUBMIT_DIR
mkdir -p results/slurm

LPC=${1:?lambda_pc required as first argument}
SEED=${2:-6554595}
JOB=$SLURM_JOB_ID
SHA=$(git rev-parse HEAD)
# Sanitize LPC for use in filename (replace . with p)
LPC_TAG=$(echo "$LPC" | tr '.' 'p')
CKPT_DIR="./checkpoints/config4_lpc${LPC_TAG}_seed${SEED}_${JOB}"
mkdir -p "$CKPT_DIR"

echo "Config 4 + lambda_pc=$LPC seed=$SEED (JOB=$JOB, SHA=$SHA)"

python -u run_train.py \
    --use_mambino_ssm=True \
    --lambda_pc=$LPC \
    --ckpt_dir="$CKPT_DIR" \
    \
    --C_init=lecun_normal --activation_fn=gelu --batchnorm=True \
    --bidirectional=True --blocks=8 --bsz=50 --d_model=128 \
    --dataset=listops-classification --dt_min=0.001 --dt_max=0.1 \
    --epochs=40 --jax_seed=${SEED} --lr_factor=3 --n_layers=8 \
    --opt_config=BfastandCdecay \
    --p_dropout=0 --ssm_lr_base=0.001 --ssm_size_base=16 \
    --warmup_end=1 --weight_decay=0.04 \
    2>&1 | tee "$CKPT_DIR/run.log"

echo "Done $(date)"
