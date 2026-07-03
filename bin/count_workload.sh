#!/bin/bash -l
# Quick XLA cost analysis for a checkpoint -- verifies workload counts
# for Accelergy PPAC input.
#SBATCH --job-name=count
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/count_%j.out
#SBATCH --error=results/slurm/count_%j.err

source /etc/profile.d/modules.sh 2>/dev/null || true
source ~/.bashrc 2>/dev/null || true
module purge
module load external anaconda/2024.10-py312 cuda/13.1.0
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
conda activate /scratch/gilbreth/raisul/envs/s5m

export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
cd $SLURM_SUBMIT_DIR
mkdir -p results/slurm

echo "=== Config 4 Mambino gelu (checkpoint 11181833) ==="
python -u count_workload.py \
    --ckpt_prefix=checkpoints/chip_mamb_gelu_11181833/best \
    --use_mambino_ssm=True \
    --activation_fn=gelu \
    --ssm_size_base=16 \
    --output=workload_config4.yaml \
    --config_name=config4_mambino

echo ""
echo "=== Config 5 Pure S5 gelu P=16 (checkpoint 11187128) ==="
python -u count_workload.py \
    --ckpt_prefix=checkpoints/train_pure_s5_gelu_p16_11187128/best \
    --use_mambino_ssm=False \
    --activation_fn=gelu \
    --ssm_size_base=32 \
    --output=workload_config5.yaml \
    --config_name=config5_pures5

echo ""
echo "=== Config 3 Pure S5 gelu P=8 (checkpoint 11181832) ==="
python -u count_workload.py \
    --ckpt_prefix=checkpoints/chip_pure_gelu_11181832/best \
    --use_mambino_ssm=False \
    --activation_fn=gelu \
    --ssm_size_base=16 \
    --output=workload_config3.yaml \
    --config_name=config3_pures5

echo "Done $(date)"
