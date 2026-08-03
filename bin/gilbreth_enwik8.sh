#!/bin/bash -l
# ============================================================================
# enwik8 byte-level char-LM (causal) — Mambino 2.0 vs gateless vs Pure S5.
# Parametrized by env vars so ONE script covers Stage 0 (calib) / 1 (ablation)
# / 2 (scaled). Causal LM: bidirectional=False, batchnorm=False (LayerNorm),
# step-based warmup->cosine schedule over --lm_steps.
#
#   MODEL=s5|mambino|mambino2p0   (default s5)
#   GATE_RANGE=signed|unsigned    (mambino2p0 only, default signed)
#   SEED D_MODEL N_LAYERS SSM_SIZE L BSZ LM_STEPS EVAL_EVERY WARMUP SSM_LR LR_FACTOR
#
# Examples:
#   MODEL=s5 D_MODEL=256 N_LAYERS=8 SSM_SIZE=64 L=1024 LM_STEPS=6000 EVAL_EVERY=1000 \
#     sbatch bin/gilbreth_enwik8.sh                              # Stage 0 calib
#   for M in s5 mambino mambino2p0; do MODEL=$M LM_STEPS=60000 sbatch bin/gilbreth_enwik8.sh; done
# ============================================================================
#SBATCH --job-name=enwik8-lm
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/enwik8_%j.out
#SBATCH --error=results/slurm/enwik8_%j.err

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

MODEL=${MODEL:-s5}
GATE_RANGE=${GATE_RANGE:-signed}
SEED=${SEED:-6554595}
D_MODEL=${D_MODEL:-256}
N_LAYERS=${N_LAYERS:-8}
SSM_SIZE=${SSM_SIZE:-64}
L=${L:-1024}
BSZ=${BSZ:-32}
LM_STEPS=${LM_STEPS:-60000}
EVAL_EVERY=${EVAL_EVERY:-2000}
WARMUP=${WARMUP:-1000}
SSM_LR=${SSM_LR:-0.001}
LR_FACTOR=${LR_FACTOR:-4}
EVAL_BATCHES=${EVAL_BATCHES:-60}
LAMBDA_PC=${LAMBDA_PC:-0.0}
ACTIVATION=${ACTIVATION:-gelu}    # gelu | half_glu2 (corner1/corner3' use half_glu2)
GLU_RANK=${GLU_RANK:-0}           # 0 = full GLU; 40 = low-rank (corner3')
# -- Mambino-LM (Stage 1) knobs (MODEL=mambinolm) --
MLM_STRIDE=${MLM_STRIDE:-4}       # top ticks once per s tokens (L must be divisible by s)
MLM_TOP_LAYERS=${MLM_TOP_LAYERS:-2}
MLM_LAMBDA_AUX=${MLM_LAMBDA_AUX:-0.1}
MLM_LAMBDA_POND=${MLM_LAMBDA_POND:-0.05}
MLM_WARMUP_FRAC=${MLM_WARMUP_FRAC:-0.15}
MLM_KAPPA=${MLM_KAPPA:-4.0}
EPOCHS=$(( LM_STEPS / EVAL_EVERY ))
[ $EPOCHS -lt 1 ] && EPOCHS=1

case $MODEL in
  s5)         MFLAGS="--use_mambino_ssm=False"; TAG="s5" ;;
  mambino)    MFLAGS="--use_mambino_ssm=True --surprise_gate=False"; TAG="mambino" ;;
  mambino2p0) MFLAGS="--use_mambino_ssm=True --surprise_gate=True --gate_range=$GATE_RANGE --gate_kappa_init=0.0 --gate_bias_init=2.0 --gate_alpha=0.9"; TAG="mambino2p0_$GATE_RANGE" ;;
  mambinolm)  MFLAGS="--use_mambino_ssm=False --mambino_lm=2level --mlm_stride=$MLM_STRIDE --mlm_top_layers=$MLM_TOP_LAYERS --mlm_lambda_aux=$MLM_LAMBDA_AUX --mlm_lambda_pond=$MLM_LAMBDA_POND --mlm_warmup_frac=$MLM_WARMUP_FRAC --mlm_kappa_init=$MLM_KAPPA"; TAG="mambinolm_s${MLM_STRIDE}_top${MLM_TOP_LAYERS}" ;;
  *) echo "unknown MODEL=$MODEL"; exit 1 ;;
esac

JOB=$SLURM_JOB_ID
CKPT="./checkpoints/enwik8_${TAG}_${ACTIVATION}r${GLU_RANK}_d${D_MODEL}_L${N_LAYERS}_p${SSM_SIZE}_ctx${L}_s${SEED}_${JOB}"
mkdir -p "$CKPT"
echo "enwik8 char-LM | MODEL=$MODEL TAG=$TAG | d=$D_MODEL L=$N_LAYERS state=$SSM_SIZE ctx=$L bsz=$BSZ"
echo "steps=$LM_STEPS warmup=$WARMUP eval_every=$EVAL_EVERY epochs=$EPOCHS ssm_lr=$SSM_LR lrf=$LR_FACTOR seed=$SEED"
echo "JOB=$JOB  SHA=$(git rev-parse HEAD)  ckpt=$CKPT"

python -u run_train.py \
    --task=lm --dataset=enwik8-lm --dir_name=./cache_dir \
    --lm_seqlen=$L --lm_steps=$LM_STEPS --lm_warmup_steps=$WARMUP \
    --lm_max_steps=$EVAL_EVERY --lm_eval_batches=$EVAL_BATCHES \
    $MFLAGS \
    --C_init=lecun_normal --activation_fn=$ACTIVATION --glu_rank=$GLU_RANK --batchnorm=False --bidirectional=False \
    --blocks=8 --bsz=$BSZ --d_model=$D_MODEL --n_layers=$N_LAYERS --ssm_size_base=$SSM_SIZE \
    --epochs=$EPOCHS --jax_seed=$SEED --lr_factor=$LR_FACTOR --ssm_lr_base=$SSM_LR \
    --opt_config=BfastandCdecay --p_dropout=0 --warmup_end=1 --weight_decay=0.05 \
    --lambda_pc=$LAMBDA_PC --ckpt_dir="$CKPT" \
    2>&1 | tee "$CKPT/run.log"

echo "Done $(date)"
