#!/bin/bash -l
# ============================================================================
# v1 paper: extend all four configurations from n=8 to n=16 matched seeds.
#
# WHY. Both mechanism claims are under-powered at n=8 and neither clears
# alpha=0.05 two-tailed:
#     Claim 1  predictor      Mambino-0 - S5-0      +0.76 pp  7/8  t=2.20  p=0.064
#     Claim 2  surprise gate  Mambino-G - Mambino-0 +0.58 pp  6/8  t=1.84  p~0.11
# In the retired 5-config paper these were supporting evidence and S5-LowRank
# carried the attribution at p=0.005. In the 4-config paper they ARE the paper.
# t scales as sqrt(n), so at n=16 (if the effects hold) Claim 1 reaches t~3.11
# (p~0.007) and Claim 2 t~2.60 (p~0.020). One day of cluster time moves both
# central claims from suggestive to established.
#
# PRE-REGISTERED SEEDS. The eight new seeds are 4 5 6 7 8 9 10 11 -- the next
# integers after the existing set's 1, 2, 3. Fixed by rule and committed BEFORE
# any of these runs completed, so the choice cannot be contingent on results.
# Existing eight: 6554595 42 12345 271828 314159 1 2 3.
#
# All four configs are extended, not just the pair under test, so the protocol
# stays uniform at one n across the paper.
#
# Everything else matches the frozen recipe exactly: 40 epochs, p_dropout=0,
# warmup_end=1, lr_factor=3, ssm_lr_base=0.001, weight_decay=0.04, bsz=50,
# d_model=128, n_layers=8, blocks=8, bidirectional, batchnorm, BfastandCdecay.
#
# Deploy:
#   for C in S5-0 Mambino-0 Mambino-G S5-Dense; do
#     for S in 4 5 6 7 8 9 10 11; do
#       CONFIG=$C SEED=$S sbatch --job-name=$C.e40d0.$S \
#         bin/gilbreth_lra_listops_v1_seeds.sh
#     done
#   done
# ============================================================================
#SBATCH --job-name=s5-v1seed
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/v1seed_%j.out
#SBATCH --error=results/slurm/v1seed_%j.err

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

CONFIG=${CONFIG:-Mambino-G}
SEED=${SEED:-4}
EPOCHS=${EPOCHS:-40}

# Names are canonical per mambino-paper/CLAUDE.md section 1.
case "$CONFIG" in
  S5-0)       ARCH="--use_mambino_ssm=False --activation_fn=gelu      --glu_rank=0 --ssm_size_base=32"
              GATE="" ;;
  Mambino-0)  ARCH="--use_mambino_ssm=True  --activation_fn=gelu      --glu_rank=0 --ssm_size_base=16"
              GATE="--surprise_gate=False --lambda_pc=0.0" ;;
  Mambino-G)  ARCH="--use_mambino_ssm=True  --activation_fn=gelu      --glu_rank=0 --ssm_size_base=16"
              GATE="--surprise_gate=True --gate_range=signed --gate_alpha=0.9 --gate_kappa_init=0.0 --gate_bias_init=2.0 --lambda_pc=0.0" ;;
  S5-Dense)   ARCH="--use_mambino_ssm=False --activation_fn=half_glu2 --glu_rank=0 --ssm_size_base=16"
              GATE="" ;;
  *) echo "unknown CONFIG=$CONFIG (S5-0|Mambino-0|Mambino-G|S5-Dense)"; exit 2 ;;
esac

JOB=$SLURM_JOB_ID
SHA=$(git rev-parse HEAD)
CKPT_DIR="./checkpoints/v1seed_${CONFIG}_s${SEED}_${JOB}"
mkdir -p "$CKPT_DIR"
echo "V1-SEEDS | CONFIG=$CONFIG seed=$SEED epochs=$EPOCHS"
echo "  $ARCH $GATE"
echo "JOB=$JOB SHA=$SHA ckpt=$CKPT_DIR"

python -u run_train.py \
    $ARCH $GATE \
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
