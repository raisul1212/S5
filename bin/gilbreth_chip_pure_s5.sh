#!/bin/bash -l
# Corner 1 config with in-session chip eval sweep.
# Trains Pure S5 P=8 full-gate (188K params) then runs analog noise + ADC
# quantization sweep at end using LIVE state (bypasses broken reload path).
#SBATCH --job-name=chip-pure-s5
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/chip_pure_s5_%j.out
#SBATCH --error=results/slurm/chip_pure_s5_%j.err

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
CKPT_DIR="./checkpoints/chip_pure_s5_${JOB}"
mkdir -p "$CKPT_DIR"

echo "Corner 1 + chip eval (JOB=$JOB, SHA=$SHA)"

python -u run_train.py \
    --ckpt_dir="$CKPT_DIR" \
    --chip_eval_sigmas=0,0.005,0.01,0.02,0.05,0.08 \
    --chip_eval_bits=0,4,5,6,7,8 \
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
