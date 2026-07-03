#!/bin/bash -l
# Pure S5 gelu P=16 (~106K) -- iso-params with Config 4 (Mambino gelu P=8).
# Core iso-both comparison for the paper:
#   Pure S5 gelu (P=16, 16 states, ~106K params)
#     vs
#   Mambino gelu (P=8 main + P=8 predictor = 16 total states, 106K params)
# Same total state modes, same params, only architectural difference is
# the predictor branch (predictive SSM).
#SBATCH --job-name=tr-pure-s5-gelu-p16
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/tr_pure_s5_gelu_p16_%j.out
#SBATCH --error=results/slurm/tr_pure_s5_gelu_p16_%j.err

source /etc/profile.d/modules.sh 2>/dev/null || true
source ~/.bashrc 2>/dev/null || true
module purge
module load external anaconda/2024.10-py312 cuda/13.1.0
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
conda activate /scratch/gilbreth/raisul/envs/s5m

export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
cd $SLURM_SUBMIT_DIR
mkdir -p results/slurm

JOB=$SLURM_JOB_ID
CKPT_DIR="./checkpoints/train_pure_s5_gelu_p16_${JOB}"
mkdir -p "$CKPT_DIR"

echo "Pure S5 gelu P=16 (~106K) -- JOB=$JOB SHA=$(git rev-parse HEAD)"

python -u run_train.py \
    --ckpt_dir="$CKPT_DIR" \
    \
    --C_init=lecun_normal --activation_fn=gelu --batchnorm=True \
    --bidirectional=True --blocks=8 --bsz=50 --d_model=128 \
    --dataset=listops-classification \
    --epochs=40 --jax_seed=6554595 --lr_factor=3 --n_layers=8 \
    --opt_config=BfastandCdecay \
    --p_dropout=0 --ssm_lr_base=0.001 --ssm_size_base=32 \
    --warmup_end=1 --weight_decay=0.04 \
    2>&1 | tee "$CKPT_DIR/run.log"

echo "Done $(date)"
