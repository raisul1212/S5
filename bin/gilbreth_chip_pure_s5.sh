#!/bin/bash -l
# Corner 1 config with in-session chip eval sweep.
# Trains Pure S5 P=8 full-gate (188K params) then runs analog noise + ADC
# quantization sweep at end using LIVE state (bypasses broken reload path).
#SBATCH --job-name=corner1
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=results/slurm/corner1_%j.out
#SBATCH --error=results/slurm/corner1_%j.err

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
SHA=$(git rev-parse HEAD)
CKPT_DIR="./checkpoints/corner1_seed${SEED}_${JOB}"
mkdir -p "$CKPT_DIR"

echo "Corner 1 (Pure S5 half_glu2 P=8) seed=$SEED (JOB=$JOB, SHA=$SHA)"

python -u run_train.py \
    --ckpt_dir="$CKPT_DIR" \
    \
    --C_init=lecun_normal --activation_fn=half_glu2 --batchnorm=True \
    --bidirectional=True --blocks=8 --bsz=50 --d_model=128 \
    --dataset=listops-classification --dt_min=0.001 --dt_max=0.1 \
    --epochs=40 --jax_seed=${SEED} --lr_factor=3 --n_layers=8 \
    --opt_config=BfastandCdecay \
    --p_dropout=0 --ssm_lr_base=0.001 --ssm_size_base=16 \
    --warmup_end=1 --weight_decay=0.04 \
    2>&1 | tee "$CKPT_DIR/run.log"

echo "Done $(date)"
