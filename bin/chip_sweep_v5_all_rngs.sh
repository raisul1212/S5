#!/bin/bash -l
# v5: pass all 3 rngs (params/dropout/noise) at apply, matching
# nn.vmap's split_rngs.  Diagnostic prints in ssm.py remain active.
#
#SBATCH --job-name=cs-v5-rngs
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm/cs_v5_%j.out
#SBATCH --error=results/slurm/cs_v5_%j.err

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

echo "chip sweep v5_all_rngs: $CKPT_DIR  extras: $@"
echo "sha: $(git rev-parse HEAD)"

python -u chip_sweep.py \
    --ckpt_prefix="$CKPT_DIR/best" \
    --csv="$CKPT_DIR/sweep_v5.csv" \
    --dataset=listops-classification \
    --sigmas=0,0.005,0.01,0.02,0.05,0.08 \
    --bits=0,4,5,6,7,8 \
    --d_model=128 --ssm_size_base=16 --n_layers=8 --blocks=8 \
    --bidirectional=True --C_init=lecun_normal --batchnorm=True \
    --conj_sym=True --bsz=50 \
    "$@" \
    2>&1 | tee "$CKPT_DIR/sweep_v5.log"

echo "Done $(date)"
