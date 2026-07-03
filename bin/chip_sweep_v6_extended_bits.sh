#!/bin/bash -l
# v6: bit sweep extended to 14 to find where accuracy converges to
# within 1% of the FP32 baseline.  Bits 10, 12, 14 added.  Diagnostic
# prints removed from ssm.py.
#
# Sweep: 6 sigmas x 9 bits = 54 cells per config.
#
#SBATCH --job-name=cs-v6-bits
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/cs_v6_%j.out
#SBATCH --error=results/slurm/cs_v6_%j.err

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

echo "chip sweep v6_extended_bits: $CKPT_DIR  extras: $@"
echo "sha: $(git rev-parse HEAD)"

python -u chip_sweep.py \
    --ckpt_prefix="$CKPT_DIR/best" \
    --csv="$CKPT_DIR/sweep_v6.csv" \
    --dataset=listops-classification \
    --sigmas=0,0.005,0.01,0.02,0.05,0.08 \
    --bits=0,4,5,6,7,8,10,12,14 \
    --d_model=128 --ssm_size_base=16 --n_layers=8 --blocks=8 \
    --bidirectional=True --C_init=lecun_normal --batchnorm=True \
    --conj_sym=True --bsz=50 \
    "$@" \
    2>&1 | tee "$CKPT_DIR/sweep_v6.log"

echo "Done $(date)"
