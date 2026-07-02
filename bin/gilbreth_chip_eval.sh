#!/bin/bash -l
# Gilbreth SLURM wrapper for the chip-analysis evaluation:
# Load best.pkl from Corner 1 (Pure S5, 188,490 params, val-sel 0.6155)
# and Corner 3' (Mambino iso, 188,682 params, val-sel 0.6140).
# For each, sweep analog noise sigma + ADC bit depth, report val + test.
#
# The paper's chip-substrate section is built from this output.
#SBATCH --job-name=s5-chip-eval
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=results/slurm/chip_eval_%j.out
#SBATCH --error=results/slurm/chip_eval_%j.err

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
mkdir -p results/slurm results/chip_eval

JOB=$SLURM_JOB_ID
SHA=$(git rev-parse HEAD)
OUT_DIR="results/chip_eval/${JOB}"
mkdir -p "$OUT_DIR"

echo "Chip-eval sweep (JOB=$JOB, SHA=$SHA)"
echo "Output dir: $OUT_DIR"

# Corner 1: Pure S5 P=8 r=full, blocks=8
echo
echo "=========================================="
echo "Corner 1: Pure S5 (188,490 params)"
echo "=========================================="
python -u bin/chip_eval.py \
    --ckpt_path=checkpoints/pure_s5_11170658/best.pkl \
    --output="$OUT_DIR/corner1_pure_s5.csv" \
    --use_mambino_ssm=False \
    --glu_rank=0 \
    --dataset=listops-classification --bsz=50 --d_model=128 \
    --n_layers=8 --ssm_size_base=16 --blocks=8 \
    --sigma_sweep=0,0.01,0.02,0.05,0.08,0.12 \
    --bits_sweep=0,4,5,6,7,8 \
    2>&1 | tee "$OUT_DIR/corner1_pure_s5.log"

# Corner 3': Mambino P=8 r=40, blocks=8
echo
echo "=========================================="
echo "Corner 3': Mambino iso (188,682 params)"
echo "=========================================="
python -u bin/chip_eval.py \
    --ckpt_path=checkpoints/mambino_iso_p8_11171006/best.pkl \
    --output="$OUT_DIR/corner3p_mambino_iso.csv" \
    --use_mambino_ssm=True \
    --glu_rank=40 \
    --dataset=listops-classification --bsz=50 --d_model=128 \
    --n_layers=8 --ssm_size_base=16 --blocks=8 \
    --sigma_sweep=0,0.01,0.02,0.05,0.08,0.12 \
    --bits_sweep=0,4,5,6,7,8 \
    2>&1 | tee "$OUT_DIR/corner3p_mambino_iso.log"

echo
echo "Done $(date). CSVs in $OUT_DIR/"
