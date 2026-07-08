#!/bin/bash -l
# Corner 2 ablation config.
# Trains Pure S5 P=16 half_glu2 with rank-40 gate. Designed to be
# ISO-PARAMS with Corner 1 and Corner 3' at 188K, and ISO-STATE with
# Corner 3' (Mambino main+predictor total state ≈ Pure S5 P=16 state).
#
# Isolates: predictor mechanism vs just state-size increase.
#   Corner 1: Pure S5 P=8, r=full (small state, big gate)
#   Corner 2: Pure S5 P=16, r=40 (big state, small gate)
#   Corner 3': Mambino P=8, r=40 (Mambino main+pred state ≈ P=16, small gate)
#
# If Corner 2 ≈ Corner 3' → predictor is just "more state" in disguise
# If Corner 3' > Corner 2 → predictor mechanism is architecturally distinct
#SBATCH --job-name=corner2
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=results/slurm/corner2_%j.out
#SBATCH --error=results/slurm/corner2_%j.err

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

SEED=${1:-6554595}
JOB=$SLURM_JOB_ID
SHA=$(git rev-parse HEAD)
CKPT_DIR="./checkpoints/corner2_seed${SEED}_${JOB}"
mkdir -p "$CKPT_DIR"

echo "Corner 2 (Pure S5 half_glu2 P=16 r=40) seed=$SEED (JOB=$JOB, SHA=$SHA)"

python -u run_train.py \
    --ckpt_dir="$CKPT_DIR" \
    \
    --glu_rank=40 \
    --C_init=lecun_normal --activation_fn=half_glu2 --batchnorm=True \
    --bidirectional=True --blocks=8 --bsz=50 --d_model=128 \
    --dataset=listops-classification --dt_min=0.001 --dt_max=0.1 \
    --epochs=40 --jax_seed=${SEED} --lr_factor=3 --n_layers=8 \
    --opt_config=BfastandCdecay \
    --p_dropout=0 --ssm_lr_base=0.001 --ssm_size_base=32 \
    --warmup_end=1 --weight_decay=0.04 \
    2>&1 | tee "$CKPT_DIR/run.log"

echo "Done $(date)"
