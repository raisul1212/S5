#!/bin/bash -l
# v3: fix InvalidRngError by registering "noise" rng at model.init()
# time.  create_train_state now includes noise_rng in init's rngs dict,
# AND chip_sweep.py builds the template state with a noise-AWARE
# model_cls so init tracing actually calls make_rng('noise').
#
# Rename tag: v3_init_noise_rng.  Search for this in logs to identify
# runs launched with this specific fix.
#
# Usage: sbatch bin/chip_sweep_v3_init_noise_rng.sh <ckpt_dir> <extras>
#
#SBATCH --job-name=cs-v3-rng
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm/cs_v3_%j.out
#SBATCH --error=results/slurm/cs_v3_%j.err

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

CKPT_DIR="$1"
shift

echo "chip sweep v3_init_noise_rng: $CKPT_DIR  extras: $@"
echo "sha: $(git rev-parse HEAD)"

python -u chip_sweep.py \
    --ckpt_prefix="$CKPT_DIR/best" \
    --csv="$CKPT_DIR/sweep_v3.csv" \
    --dataset=listops-classification \
    --sigmas=0,0.005,0.01,0.02,0.05,0.08 \
    --bits=0,4,5,6,7,8 \
    --d_model=128 --ssm_size_base=16 --n_layers=8 --blocks=8 \
    --bidirectional=True --C_init=lecun_normal --batchnorm=True \
    --conj_sym=True --bsz=50 \
    "$@" \
    2>&1 | tee "$CKPT_DIR/sweep_v3.log"

echo "Done $(date)"
