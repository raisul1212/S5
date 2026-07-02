#!/bin/bash -l
# Minimal sanity check using train_helpers.validate() directly.
# If this gives 0.6155 on Corner 1, our eval path works and any deviation
# in chip_eval.py is a bug in the custom apply loop.
#SBATCH --job-name=s5-chip-san2
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=results/slurm/chip_sanity_v2_%j.out
#SBATCH --error=results/slurm/chip_sanity_v2_%j.err

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

echo "Sanity v2: use train_helpers.validate() on Corner 1 ckpt"
python -u bin/chip_eval_sanity_v2.py \
    --ckpt_path=checkpoints/pure_s5_11170658/best.pkl \
    --use_mambino_ssm=False --glu_rank=0

echo "Done $(date)"
