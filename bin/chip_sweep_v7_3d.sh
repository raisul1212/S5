#!/bin/bash -l
# v7: full 3D sweep -- sigma x bits x crossings_every.
# 7 sigmas (extended to 10% for dithering-regime coverage)
# x 9 bits (0,4,5,6,7,8,10,12,14 -- 14 is effectively FP32 per quantize_adc)
# x 4 crossings_every (1,2,4,8) = 252 cells per config, ~1-1.5 hr per config.
#
# The 3 axes fill the design space:
#   - sigma x bits at crossings_every=1: baseline noise/precision interaction
#   - crossings x bits at fixed sigma: iso-ADC-area tradeoff
#   - crossings x sigma at fixed bits: dithering hypothesis
#
# Usage: sbatch -J <name> bin/chip_sweep_v7_3d.sh <ckpt_dir> <extras>
#
#SBATCH --job-name=cs-v7
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=results/slurm/cs_v7_%j.out
#SBATCH --error=results/slurm/cs_v7_%j.err

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

CKPT_DIR="$1"; shift

echo "chip sweep v7_3d: $CKPT_DIR  extras: $@"
echo "sha: $(git rev-parse HEAD)"

python -u chip_sweep.py \
    --ckpt_prefix="$CKPT_DIR/best" \
    --csv="$CKPT_DIR/sweep_v7.csv" \
    --dataset=listops-classification \
    --sigmas=0,0.02,0.05,0.08,0.10,0.12,0.15 \
    --bits=0,4,5,6,7,8,10,12,14 \
    --crossings=1,2,4,8 \
    --weight_sigmas=0,0.02,0.05 \
    --mc_seeds=0,1,2 \
    --d_model=128 --ssm_size_base=16 --n_layers=8 --blocks=8 \
    --bidirectional=True --C_init=lecun_normal --batchnorm=True \
    --conj_sym=True --bsz=50 \
    "$@" \
    2>&1 | tee "$CKPT_DIR/sweep_v7.log"

echo "Done $(date)"
