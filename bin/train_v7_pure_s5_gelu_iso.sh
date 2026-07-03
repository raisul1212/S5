#!/bin/bash -l
# Config 8: Pure S5 gelu at P=32 (~106K), iso-param with Mambino gelu.
# Fills the iso-param gelu comparison mirroring Corner 1 vs Corner 3'.
# Pure S5 with gelu (no gate), but higher effective P (32 vs baseline 8)
# to reach Mambino gelu's 106K without a gate to burn params on.
#SBATCH --job-name=tr-v7-pure-s5-gelu-iso
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/tr_v7_pure_s5_gelu_iso_%j.out
#SBATCH --error=results/slurm/tr_v7_pure_s5_gelu_iso_%j.err

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
CKPT_DIR="./checkpoints/train_v7_pure_s5_gelu_iso_${JOB}"
mkdir -p "$CKPT_DIR"

echo "Config 8: Pure S5 gelu P=32 iso-106K -- JOB=$JOB SHA=$(git rev-parse HEAD)"

python -u run_train.py \
    --ckpt_dir="$CKPT_DIR" \
    \
    --C_init=lecun_normal --activation_fn=gelu --batchnorm=True \
    --bidirectional=True --blocks=8 --bsz=50 --d_model=128 \
    --dataset=listops-classification \
    --epochs=40 --jax_seed=6554595 --lr_factor=3 --n_layers=8 \
    --opt_config=BfastandCdecay \
    --p_dropout=0 --ssm_lr_base=0.001 --ssm_size_base=64 \
    --warmup_end=1 --weight_decay=0.04 \
    2>&1 | tee "$CKPT_DIR/run.log"

echo "Done $(date)"
