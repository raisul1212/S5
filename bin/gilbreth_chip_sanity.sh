#!/bin/bash -l
# Gilbreth SLURM wrapper for the chip-analysis SANITY CHECK.
# Runs 5 sanity tests on the Pure S5 checkpoint (Corner 1):
#   1) fast path reproduces saved test_acc
#   2) noise injection alters output
#   3) fast path is deterministic
#   4) high noise degrades accuracy
#   5) ADC quant alters output
#
# Only if all 5 PASS should we run the full sweep.
#SBATCH --job-name=s5-chip-sanity
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=results/slurm/chip_sanity_%j.out
#SBATCH --error=results/slurm/chip_sanity_%j.err

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
echo "Chip-analysis SANITY CHECK (JOB=$JOB, SHA=$SHA)"

# ── Test on Corner 1 (Pure S5) first
echo
echo "==================================================================="
echo "SANITY: Corner 1 (Pure S5 P=8 full, 188,490 params)"
echo "==================================================================="
python -u bin/chip_eval_sanity.py \
    --ckpt_path=checkpoints/pure_s5_11170658/best.pkl \
    --use_mambino_ssm=False \
    --glu_rank=0 \
    --bsz=50 --d_model=128 --n_layers=8 --ssm_size_base=16 --blocks=8

rc1=$?
echo "Corner 1 sanity exit code: $rc1"

# ── Only if Corner 1 passed, test Corner 3'
if [ $rc1 -eq 0 ]; then
    echo
    echo "==================================================================="
    echo "SANITY: Corner 3' (Mambino P=8 r=40, 188,682 params)"
    echo "==================================================================="
    python -u bin/chip_eval_sanity.py \
        --ckpt_path=checkpoints/mambino_iso_p8_11171006/best.pkl \
        --use_mambino_ssm=True \
        --glu_rank=40 \
        --bsz=50 --d_model=128 --n_layers=8 --ssm_size_base=16 --blocks=8
    rc2=$?
    echo "Corner 3' sanity exit code: $rc2"
else
    echo "Corner 1 sanity FAILED -- skipping Corner 3' test"
fi

echo
echo "Done $(date)"
