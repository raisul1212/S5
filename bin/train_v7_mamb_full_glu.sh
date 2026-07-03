#!/bin/bash -l
# Config 5: Mambino P=8 r=0 (full GLU, higher-param ~238K).
# Higher-param Mambino at same P=8 as Corner 3', but without the r=40
# GLU compression -- tests whether GLU capacity helps at param cost.
#SBATCH --job-name=tr-v7-mamb-full-glu
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/tr_v7_mamb_full_glu_%j.out
#SBATCH --error=results/slurm/tr_v7_mamb_full_glu_%j.err

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
CKPT_DIR="./checkpoints/train_v7_mamb_full_glu_${JOB}"
mkdir -p "$CKPT_DIR"

echo "Config 5: Mambino P=8 r=0 (full GLU) -- JOB=$JOB SHA=$(git rev-parse HEAD)"

python -u run_train.py \
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
    --warmup_end=1 --weight_decay=0.04 \
    2>&1 | tee "$CKPT_DIR/run.log"

echo "Done $(date)"
