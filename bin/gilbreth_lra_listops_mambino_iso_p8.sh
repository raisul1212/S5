#!/bin/bash -l
# Gilbreth SLURM wrapper for CLEAN iso-param Mambino-in-S5 on LRA-ListOps.
#
# Cleaner iso-param design than the confounded P=12 r=20 variant (11170737,
# which changed both state size AND gate rank simultaneously).  Here we
# change ONLY the gate rank -- everything else matches pure S5 exactly,
# with the extra params from the predictor branch offset by a sparser gate.
#
# Deltas vs pure S5 baseline (11170658, 188K):
#   --use_mambino_ssm=True       (adds predictor + W_eps: +49,344 params)
#   --glu_rank=40                (out2 rank drops from full 128 to 40:
#                                 -49,152 params, offsetting predictor cost)
#
# Everything else IDENTICAL to pure S5: P=8, blocks=8, seed 6554595,
# epochs=40, all hyperparams.  This is the CLEANEST possible
# "predictor value at iso-param" test.
#
# Expected params: ~188,682 (within 0.1% of pure S5's 188,490).
#
# 5-corner paper design:
#   Pure S5 P=8 full        188K (11170658, DONE: 61.55% val-sel / 62.50% peak)
#   Pure S5 P=12 r=56       197K (11170875, queued: state richness alone)
#   Mambino P=12 r=20       197K (11170737, running: confounded 2-var swap)
#   Mambino P=8 r=40        189K (THIS run: CLEAN predictor swap at iso-param)
#   Mambino full P=8 r=full 238K (11169698, DONE: 61.70% val-sel / 62.45% peak)
#   Mambino bidir predictor 254K (11170831, queued: ceiling with future context)
#
# The Mambino P=8 r=40 (this run) is the clean predictor-value test:
# same P as pure S5 P=8, only gate rank differs, so any Δ vs pure S5 is
# causally the predictor branch contribution at iso-param.
#SBATCH --job-name=s5-mamb-iso-p8
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/s5_mambino_iso_p8_listops_%j.out
#SBATCH --error=results/slurm/s5_mambino_iso_p8_listops_%j.err

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
CKPT_DIR="./checkpoints/mambino_iso_p8_${JOB}"
mkdir -p "$CKPT_DIR"

echo "Clean iso-param Mambino-in-S5 P=8 r=40 LRA-ListOps (JOB=$JOB, SHA=$SHA)"
echo "Design: predictor swap only -- same P as pure S5, only gate rank differs"
echo "Checkpoints: $CKPT_DIR"

python -u run_train.py \
    --use_mambino_ssm=True \
    --lambda_pc=0.0 \
    --ckpt_dir="$CKPT_DIR" \
    --glu_rank=40 \
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
