#!/bin/bash -l
# Gilbreth SLURM wrapper for iso-param Mambino-in-S5 on LRA-ListOps.
#
# Iso-param variant: sparsify half_glu2 to redirect params into a
# richer SSM state.  Design goal: match pure S5's 188K within +5%
# while making the SSM state 50% larger.
#
# Deltas vs the standard Mambino-in-S5 run (11169698):
#   ssm_size_base:  16 -> 24  (P=8 complex -> P=12 complex; 16 -> 24 real state)
#   blocks:          8 -> 6   (block_size 4, halved by conj_sym to 2; 6*2=12=P)
#   glu_rank:        -  -> 20 (half_glu2 out2 becomes rank-20 factorization
#                              instead of full 128 -- gate params drop from
#                              16,512/layer to 5,248/layer)
#
# All other args match 11169698 exactly (seed 6554595, epochs=40,
# BfastandCdecay, batchnorm, bidirectional, etc).
#
# Expected params: ~197K (+4.6% over pure S5's 188K, -17% below full-gate Mambino)
#
# Hypothesis: richer state (24 real vs 16 real) can compensate for a
# sparser gate (rank 20 vs full 128) and match pure S5's accuracy.  If yes,
# proves the Mambino predictor pathway is genuinely load-bearing at
# iso-param, not just budget-filling.
#SBATCH --job-name=s5-mamb-iso
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/s5_mambino_iso_listops_%j.out
#SBATCH --error=results/slurm/s5_mambino_iso_listops_%j.err

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
CKPT_DIR="./checkpoints/mambino_iso_${JOB}"
mkdir -p "$CKPT_DIR"

echo "Iso-param Mambino-in-S5 LRA-ListOps (JOB=$JOB, SHA=$SHA)"
echo "Design: P=12 (ssm_size_base=24), blocks=6, glu_rank=20"
echo "Checkpoints: $CKPT_DIR"

python -u run_train.py \
    --use_mambino_ssm=True \
    --lambda_pc=0.0 \
    --ckpt_dir="$CKPT_DIR" \
    --glu_rank=20 \
    \
    --C_init=lecun_normal --activation_fn=half_glu2 --batchnorm=True \
    --bidirectional=True --blocks=6 --bsz=50 --d_model=128 \
    --dataset=listops-classification \
    --epochs=40 --jax_seed=6554595 --lr_factor=3 --n_layers=8 \
    --opt_config=BfastandCdecay \
    --p_dropout=0 --ssm_lr_base=0.001 --ssm_size_base=24 \
    --warmup_end=1 --weight_decay=0.04 \
    \
    2>&1 | tee "$CKPT_DIR/run.log"

echo "Done $(date)"
