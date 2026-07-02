#!/bin/bash -l
# Pure S5 P=8 with glu_rank=20 -- mid-decay sweep point.
# Expected params: ~98K.  Fills the sweep gap between r=40 (139K) and
# r=10 (78K), matching the ranks already queued for Mambino (r=20 at 148K).
# Enables the matched-rank Pure S5 vs Mambino comparison at r=20.
#SBATCH --job-name=s5-pure-r20
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/s5_pure_r20_listops_%j.out
#SBATCH --error=results/slurm/s5_pure_r20_listops_%j.err

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
CKPT_DIR="./checkpoints/pure_s5_r20_${JOB}"
mkdir -p "$CKPT_DIR"

echo "Pure S5 P=8 glu_rank=20 LRA-ListOps sweep (JOB=$JOB, SHA=$SHA)"
echo "Checkpoints: $CKPT_DIR"

python -u run_train.py \
    --ckpt_dir="$CKPT_DIR" \
    --glu_rank=20 \
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
