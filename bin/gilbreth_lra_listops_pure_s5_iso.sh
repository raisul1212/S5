#!/bin/bash -l
# Gilbreth SLURM wrapper for PURE S5 at iso-param (~197K) as the fourth
# corner of the paper's capacity-allocation ablation.
#
# Purpose: isolate the effect of SSM state richness ALONE (P=12 vs P=8)
# without the Mambino predictor branch.  This is the control that answers:
# "at iso-param, does the Mambino predictor branch add value beyond what
# richer SSM state already provides?"
#
# 4-corner design (all at seed 6554595, 40 epochs, otherwise identical):
#   Corner 1 - Pure S5 P=8 full gate   (188K, published baseline)
#              -> gilbreth_lra_listops_pure_s5.sh (11170658)
#   Corner 2 - Pure S5 P=12 r=56       (~197K, THIS RUN)
#              -> gilbreth_lra_listops_pure_s5_iso.sh
#   Corner 3 - Mambino-iso P=12 r=20   (~197K)
#              -> gilbreth_lra_listops_mambino_iso.sh (11170737)
#   Corner 4 - Mambino-full P=8 full   (238K, extra params via predictor)
#              -> gilbreth_lra_listops_mambino.sh (11169698)
#
# Pairwise decompositions:
#   1 -> 2: does state-richness alone help? (pure state-vs-gate tradeoff)
#   2 -> 3: does the predictor branch add value at iso-param, ISOLATED?
#   1 -> 3: full architectural swap at iso-param (published state-heavy comparison)
#   1 -> 4: extra params via predictor (non-iso; naive Mambino win)
#
# Deltas vs the standard pure S5 run (11170658):
#   ssm_size_base:  16 -> 24  (P=8 complex -> P=12 complex)
#   blocks:          8 -> 6   (block_size=4, halved to 2; 6*2=12)
#   glu_rank:        -  -> 56 (half_glu2 rank 56/128 -- matches iso-param
#                              target of ~197K)
#
# Expected params: ~196.8K = within 0.2% of Mambino-iso's 197K.
#
# All other args match 11170658 exactly (seed 6554595, epochs=40, etc).
#SBATCH --job-name=s5-pure-iso
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/s5_pure_iso_listops_%j.out
#SBATCH --error=results/slurm/s5_pure_iso_listops_%j.err

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
CKPT_DIR="./checkpoints/pure_s5_iso_${JOB}"
mkdir -p "$CKPT_DIR"

echo "Pure S5 iso-param P=12 r=56 LRA-ListOps (JOB=$JOB, SHA=$SHA)"
echo "Design: fourth corner of the capacity-allocation ablation"
echo "Checkpoints: $CKPT_DIR"

python -u run_train.py \
    --ckpt_dir="$CKPT_DIR" \
    --glu_rank=56 \
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
