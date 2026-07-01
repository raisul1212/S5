#!/bin/bash -l
# Gilbreth SLURM wrapper for MambinoSSM plugged into S5's ListOps pipeline.
#
# Same S5 recipe (char-level, HiPPO, BfastandCdecay, bidirectional main
# scan) but with MambinoSSM replacing S5SSM.  Target: match or beat
# S5's 62.15% at ~238K params.
#
# Data: expects LRA-standard listops dump at $LRA_DATA_ROOT (default
# ./raw_datasets/lra_release/lra_release/listops-1000).  User must
# have downloaded via ./bin/download_lra.sh once.
#SBATCH --job-name=s5-mambino-listops
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/s5_mambino_listops_%j.out
#SBATCH --error=results/slurm/s5_mambino_listops_%j.err

source /etc/profile.d/modules.sh 2>/dev/null || true
source /etc/profile.d/lmod.sh 2>/dev/null || true
source ~/.bashrc 2>/dev/null || true
module purge
module load external
module load anaconda/2024.10-py312
module load cuda/13.1.0

# Use the dedicated s5m env on scratch (torch 2.4.1+cu121, jax 0.4.30 [cuda12],
# flax 0.8.5, cudnn 9 shared by torch and jax).  Built by
# /scratch/gilbreth/raisul/install_s5m_v2.sh; env has ~3GB deps that don't
# fit in the 25GB /home quota, hence living on scratch.  torchtext is
# absent from this env by design (no 0.18+ wheel matches torch 2.4);
# s5/dataloaders/lra.py falls back to an inlined _build_vocab_from_iterator
# equivalent when _HAS_TORCHTEXT is False.
CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate /scratch/gilbreth/raisul/envs/s5m

export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PYTHONUTF8=1
cd $SLURM_SUBMIT_DIR
mkdir -p results/slurm

JOB=$SLURM_JOB_ID
SHA=$(git rev-parse HEAD)
CKPT_DIR="./checkpoints/mambino_s5_${JOB}"
mkdir -p "$CKPT_DIR"

echo "S5-with-MambinoSSM LRA-ListOps (JOB=$JOB, SHA=$SHA)"
echo "Checkpoints: $CKPT_DIR"

python -u run_train.py \
    --use_mambino_ssm=True \
    --lambda_pc=0.0 \
    --ckpt_dir="$CKPT_DIR" \
    \
    --C_init=lecun_normal --activation_fn=half_glu2 --batchnorm=True \
    --bidirectional=True --blocks=8 --bsz=50 --d_model=128 \
    --dataset=listops-classification \
    --epochs=40 --jax_seed=6554595 --lr_factor=3 --n_layers=8 \
    --opt_config=BfastandCdecay \
    --p_dropout=0 --ssm_lr_base=0.001 --ssm_size_base=16 \
    --warmup_end=1 --weight_decay=0.04 \
    \
    2>&1 | tee "$CKPT_DIR/run.log"

echo "Done $(date)"
