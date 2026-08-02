#!/bin/bash -l
# ============================================================================
# Two scale-up directions for Mambino-G, both at 40 epochs / dropout 0 so they
# sit directly against the existing 8-seed references.
#
#   CONFIG=G_P16   "Mambino-G-P16"   204,442 params  (+8.46% vs Pure S5)
#                  gelu, NO output gate, ssm_size_base=32
#                  -> P=16 MAIN + P=16 PREDICTOR states (2x Mambino-G everywhere)
#                  Tests whether Mambino-G scales through STATE.  Note this is
#                  NOT iso-param with Pure S5 (188,490) -- the state knob moves
#                  in quanta of 12,336 params, so the nearest options are
#                  155,098 (P=12) and 204,442 (P=16); 188,490 is between them.
#                  Named G_P16 rather than "G-iso" for that reason.
#                  RISK: Corner 2 is the cautionary precedent -- Pure S5 with P
#                  doubled 8->16 went BACKWARDS (0.5991 vs 0.6089, p=0.005).
#                  This asks whether Mambino behaves the same way.
#
#   CONFIG=M3_G    "Mambino-3-G"     188,698 params  (+0.11% vs Pure S5)
#                  half_glu2 rank-40 gate + SWG, ssm_size_base=16
#                  = Mambino-3 (Corner 3') plus the 16-param surprise gate.
#                  TRUE iso-param (paper convention: Corner 3' is +0.10%).
#                  Mambino-3 alone already beats Pure S5: 0.6138 vs 0.6089,
#                  +0.49 pp, p=0.022 at n=8.  The gate bought +0.57 pp on the
#                  gelu host; this asks whether it also pays on the gated host.
#                  CHIP TOLL: the r=40 gate saves ZERO net silicon under the
#                  relaxed-util rule (master doc 6d -- both configs' gates land
#                  at 4,096 PE), which is why Mambino-3 is 19.76 mm2 against
#                  Mambino-G's 16.01.  Accuracy anchor, not efficiency hero.
#
# References on these 8 seeds: Pure S5 0.6089 | Mambino-3 0.6138 |
#   Mambino-G 0.6050 | Mambino-0 0.5993 | S5-gateless 0.5917
#
# Deploy:
#   for C in G_P16 M3_G; do
#     for S in 6554595 42 12345 271828 314159 1 2 3; do
#       CONFIG=$C SEED=$S sbatch --job-name=$C.e40d0.$S \
#         bin/gilbreth_lra_listops_isoscale.sh
#     done
#   done
# ============================================================================
#SBATCH --job-name=s5-isoscale
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/isoscale_%j.out
#SBATCH --error=results/slurm/isoscale_%j.err

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

CONFIG=${CONFIG:-M3_G}
SEED=${SEED:-6554595}
EPOCHS=${EPOCHS:-40}

case "$CONFIG" in
  G_P16) ARCH="--activation_fn=gelu      --glu_rank=0  --ssm_size_base=32" ;;
  M3_G)  ARCH="--activation_fn=half_glu2 --glu_rank=40 --ssm_size_base=16" ;;
  *) echo "unknown CONFIG=$CONFIG (G_P16|M3_G)"; exit 2 ;;
esac

JOB=$SLURM_JOB_ID
SHA=$(git rev-parse HEAD)
CKPT_DIR="./checkpoints/isoscale_${CONFIG}_s${SEED}_${JOB}"
mkdir -p "$CKPT_DIR"
echo "ISOSCALE | CONFIG=$CONFIG  $ARCH  seed=$SEED epochs=$EPOCHS"
echo "JOB=$JOB SHA=$SHA ckpt=$CKPT_DIR"

python -u run_train.py \
    --use_mambino_ssm=True \
    --surprise_gate=True --gate_range=signed --gate_alpha=0.9 \
    --gate_kappa_init=0.0 --gate_bias_init=2.0 --lambda_pc=0.0 \
    $ARCH \
    --ckpt_dir="$CKPT_DIR" \
    \
    --C_init=lecun_normal --batchnorm=True \
    --bidirectional=True --blocks=8 --bsz=50 --d_model=128 \
    --dataset=listops-classification \
    --epochs=$EPOCHS --jax_seed=$SEED --lr_factor=3 --n_layers=8 \
    --opt_config=BfastandCdecay \
    --p_dropout=0 --ssm_lr_base=0.001 \
    --warmup_end=1 --weight_decay=0.04 \
    \
    2>&1 | tee "$CKPT_DIR/run.log"

echo "Done $(date)"
