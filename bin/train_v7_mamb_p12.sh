#!/bin/bash -l
# Config 7: Mambino P=16 r=24 iso-param 188K, blocks=8.
# Bigger SSM state (P=16 vs Corner 3' P=8) traded for sparser GLU
# (r=24 vs r=40).  Tests whether Mambino benefits from a richer SSM
# state at the cost of gate capacity.
#
# ssm_size_base=32 with blocks=8 -> block_size=4 -> conj_sym halves
# to 2 -> effective P per block = 2, blocks*block_size = 16 = P.
# blocks=8 preserved (same HiPPO structure as Corner 3').  Delta from
# P=8 to P=16 is ~34K (SSM state + predictor scaling); compensated by
# r=40 -> r=24 saving ~34K.  Note: was P=12 blocks=6 in earlier draft;
# switched to P=16 blocks=8 for consistent HiPPO with Corner 3'.
#SBATCH --job-name=tr-v7-mamb-p16
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/tr_v7_mamb_p12_%j.out
#SBATCH --error=results/slurm/tr_v7_mamb_p12_%j.err

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
CKPT_DIR="./checkpoints/train_v7_mamb_p16_${JOB}"
mkdir -p "$CKPT_DIR"

echo "Config 7: Mambino P=16 r=24 iso-188K blocks=8 L=8 -- JOB=$JOB SHA=$(git rev-parse HEAD)"

python -u run_train.py \
    --use_mambino_ssm=True \
    --lambda_pc=0.0 \
    --glu_rank=24 \
    --ckpt_dir="$CKPT_DIR" \
    \
    --C_init=lecun_normal --activation_fn=half_glu2 --batchnorm=True \
    --bidirectional=True --blocks=8 --bsz=50 --d_model=128 \
    --dataset=listops-classification \
    --epochs=40 --jax_seed=6554595 --lr_factor=3 --n_layers=8 \
    --opt_config=BfastandCdecay \
    --p_dropout=0 --ssm_lr_base=0.001 --ssm_size_base=32 \
    --warmup_end=1 --weight_decay=0.04 \
    2>&1 | tee "$CKPT_DIR/run.log"

echo "Done $(date)"
