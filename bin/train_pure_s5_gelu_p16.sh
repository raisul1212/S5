#!/bin/bash -l
# Pure S5 gelu P=16 (~106K) -- iso-params with Config 4 (Mambino gelu P=8).
# Core iso-both comparison for the paper:
#   Pure S5 gelu (P=16, 16 states, ~106K params)
#     vs
#   Mambino gelu (P=8 main + P=8 predictor = 16 total states, 106K params)
# Same total state modes, same params, only architectural difference is
# the predictor branch (predictive SSM).
#SBATCH --job-name=config5
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=results/slurm/config5_%j.out
#SBATCH --error=results/slurm/config5_%j.err

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
export WANDB_MODE=disabled  # bypass wandb service startup (see run.log for reason)
cd $SLURM_SUBMIT_DIR
mkdir -p results/slurm

SEED=${1:-6554595}
JOB=$SLURM_JOB_ID
CKPT_DIR="./checkpoints/config5_seed${SEED}_${JOB}"
mkdir -p "$CKPT_DIR"

echo "Config 5 Pure S5 gelu P=16 (~106K) seed=$SEED JOB=$JOB SHA=$(git rev-parse HEAD)"

python -u run_train.py \
    --ckpt_dir="$CKPT_DIR" \
    \
    --C_init=lecun_normal --activation_fn=gelu --batchnorm=True \
    --bidirectional=True --blocks=8 --bsz=50 --d_model=128 \
    --dataset=listops-classification --dt_min=0.001 --dt_max=0.1 \
    --epochs=40 --jax_seed=${SEED} --lr_factor=3 --n_layers=8 \
    --opt_config=BfastandCdecay \
    --p_dropout=0 --ssm_lr_base=0.001 --ssm_size_base=32 \
    --warmup_end=1 --weight_decay=0.04 \
    2>&1 | tee "$CKPT_DIR/run.log"

echo "Done $(date)"
