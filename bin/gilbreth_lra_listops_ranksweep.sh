#!/bin/bash -l
# ============================================================================
# LRA-ListOps GATE-RANK SWEEP.  Accuracy and parameters as a function of output
# gate rank, for BOTH families at fixed P=8.  Design-space curve for DATE 2027.
#
# WHY. The v1 paper compares two points (no gate vs dense gate). This maps the
# whole curve and asks where, if anywhere, the predictor and the gate cross as
# ways to spend parameters.
#
# Gate cost per layer, H=128:  none 0 | rank r  2rH+H | dense H^2+H = 16,512
# At r = H/2 = 64 the factorisation costs exactly what dense costs, so r < 64
# is the only region where low-rank saves anything. That break-even is analytic,
# not empirical, and bounds the useful sweep.
#
#  rank    gate/layer    S5 total   Mambino total
#  none             0      56,394        105,738   <- Mambino none = Mambino-0
#  8            2,176      73,802        123,146
#  16           4,224      90,186        139,530
#  32           8,320     122,954        172,298
#  40          10,368     139,338        188,682   <- Mambino r=40 = Mambino-3
#  dense       16,512     188,490        237,834   <- S5 dense = S5-Dense
#
# THREE ISO-PARAMETER CROSS-FAMILY PAIRS fall out, which are the load-bearing
# comparisons (does a predictor or a bigger gate buy more at matched budget?):
#     Mambino r=8  123,146  ==  S5 r=32   122,954   (+0.16%)
#     Mambino r=16 139,530  ==  S5 r=40   139,338   (+0.14%)
#     Mambino r=40 188,682  ==  S5 dense  188,490   (+0.10%)
#
# Everything else matches the frozen v1 ListOps recipe exactly, so these points
# sit on the same axes as the n=16 results: 40 epochs, p_dropout=0, warmup_end=1,
# lr_factor=3, ssm_lr_base=0.001, weight_decay=0.04, bsz=50, d_model=128,
# n_layers=8, blocks=8, ssm_size_base=16 (P=8), bidirectional, batchnorm,
# BfastandCdecay, lambda_pc=0, NO surprise gate (it is null at n=16).
#
# Deploy (8 seeds per point; endpoints already have n=16 and are skipped):
#   for F in S5 Mambino; do for R in none 8 16 32 40; do
#     for S in 6554595 42 12345 271828 314159 1 2 3; do
#       [ "$F" = "S5" ] && [ "$R" = "dense" ] && continue
#       [ "$F" = "Mambino" ] && [ "$R" = "none" ] && continue
#       FAMILY=$F RANK=$R SEED=$S sbatch --job-name=rs.$F.$R.$S \
#         bin/gilbreth_lra_listops_ranksweep.sh
#   done; done; done
# ============================================================================
#SBATCH --job-name=s5-ranksweep
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/rs_%j.out
#SBATCH --error=results/slurm/rs_%j.err

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

FAMILY=${FAMILY:-Mambino}
RANK=${RANK:-40}
SEED=${SEED:-6554595}
EPOCHS=${EPOCHS:-40}

case "$FAMILY" in
  S5)      MAMB="--use_mambino_ssm=False" ;;
  Mambino) MAMB="--use_mambino_ssm=True --surprise_gate=False --lambda_pc=0.0" ;;
  *) echo "unknown FAMILY=$FAMILY (S5|Mambino)"; exit 2 ;;
esac

# rank -> activation + glu_rank.  glu_rank=0 with half_glu2 means the DENSE gate;
# "none" means no gate at all, i.e. a parameter-free GELU.
case "$RANK" in
  none)  GATEA="--activation_fn=gelu      --glu_rank=0" ;;
  dense) GATEA="--activation_fn=half_glu2 --glu_rank=0" ;;
  ''|*[!0-9]*) echo "RANK must be none|dense|<positive int>, got '$RANK'"; exit 2 ;;
  *)     [ "$RANK" -lt 1 ] && { echo "RANK must be >= 1"; exit 2; }
         GATEA="--activation_fn=half_glu2 --glu_rank=$RANK" ;;
esac

JOB=$SLURM_JOB_ID
SHA=$(git rev-parse HEAD)
CKPT_DIR="./checkpoints/rs_${FAMILY}_r${RANK}_s${SEED}_${JOB}"
mkdir -p "$CKPT_DIR"
echo "RANKSWEEP | FAMILY=$FAMILY RANK=$RANK seed=$SEED epochs=$EPOCHS"
echo "  $MAMB $GATEA"
echo "JOB=$JOB SHA=$SHA ckpt=$CKPT_DIR"

# v1 ListOps recipe below this line, unchanged. Only $MAMB/$GATEA vary.
python -u run_train.py \
    $MAMB $GATEA \
    --ckpt_dir="$CKPT_DIR" \
    \
    --C_init=lecun_normal --batchnorm=True \
    --bidirectional=True --blocks=8 --bsz=50 --d_model=128 \
    --dataset=listops-classification \
    --epochs=$EPOCHS --jax_seed=$SEED --lr_factor=3 --n_layers=8 \
    --opt_config=BfastandCdecay \
    --p_dropout=0 --ssm_lr_base=0.001 --ssm_size_base=16 \
    --warmup_end=1 --weight_decay=0.04 \
    \
    2>&1 | tee "$CKPT_DIR/run.log"

echo "Done $(date)"
