#!/bin/bash -l
# ============================================================================
# Cluster A: Mambino-gelu (GATELESS) + v2 SURPRISE GATE on the eps->W_eps write.
#   g(t) = tanh(kappa*z + bias),  z = running-EMA-normalized ||eps||
#   u(t) = B_bar@x(t) + g(t)*(W_eps_bar@eps(t))
# Adds ~16 params total (2 scalars/layer).  Isolates the surprise gate on the
# gateless host (no half_glu2 output-GLU confound).
#
# BASELINE (this exact recipe, gate OFF = gilbreth_lra_listops_mambino_gelu.sh):
#   ~0.599 val-sel.  S5 P=8 = 0.616.  Mambino half_glu2 r40 = 0.617 (+10,240 params).
#   QUESTION: does the ~16-param signed gate lift gelu-Mambino toward/over S5
#   and the rank-40 output gate?
#
# Ablation grid via env vars (defaults = headline signed run):
#   GATE_RANGE   signed|unsigned  (default signed)   -- tests whether the signed
#                                                        ERASE (pop) matters vs gain-only
#   GATE_DETACH  True|False       (default False)    -- True = calibration fix
#                                                        (predictor trained by L_int only)
#   LAMBDA_PC    0.0|0.1          (default 0.0)       -- 0.1 = more residual-like eps
# THE experiment (detach=False, lambda_pc=0 throughout -- 2 runs):
#   for R in signed unsigned; do GATE_RANGE=$R sbatch bin/gilbreth_lra_listops_mambino_gelu_sgate.sh; done
# gate_detach is NOT run: it needs lambda_pc>0 to avoid a frozen predictor,
# but lambda_pc>0 empirically makes the predictor over-optimize PREDICTION at
# the TASK's expense (why the recipe uses lambda_pc=0).  The task-shaped
# surprise (lambda_pc=0) is exactly what carries the push/pop signal; the
# EMA gate rides on it and needs no calibrated residual.
# ============================================================================
#SBATCH --job-name=s5-mamb-sgate
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/s5_mambino_gelu_sgate_listops_%j.out
#SBATCH --error=results/slurm/s5_mambino_gelu_sgate_listops_%j.err

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

GATE_RANGE=${GATE_RANGE:-signed}
GATE_DETACH=${GATE_DETACH:-False}
LAMBDA_PC=${LAMBDA_PC:-0.0}

JOB=$SLURM_JOB_ID
SHA=$(git rev-parse HEAD)
CKPT_DIR="./checkpoints/mambino_gelu_sgate_${GATE_RANGE}_d${GATE_DETACH}_lpc${LAMBDA_PC}_${JOB}"
mkdir -p "$CKPT_DIR"

echo "Mambino-gelu + SURPRISE GATE (range=$GATE_RANGE detach=$GATE_DETACH lambda_pc=$LAMBDA_PC)"
echo "LRA-ListOps  JOB=$JOB  SHA=$SHA  ckpt=$CKPT_DIR"

python -u run_train.py \
    --use_mambino_ssm=True \
    --surprise_gate=True \
    --gate_range=$GATE_RANGE \
    --gate_detach=$GATE_DETACH \
    --gate_alpha=0.9 \
    --gate_kappa_init=0.0 \
    --gate_bias_init=2.0 \
    --lambda_pc=$LAMBDA_PC \
    --ckpt_dir="$CKPT_DIR" \
    \
    --C_init=lecun_normal --activation_fn=gelu --batchnorm=True \
    --bidirectional=True --blocks=8 --bsz=50 --d_model=128 \
    --dataset=listops-classification \
    --epochs=40 --jax_seed=6554595 --lr_factor=3 --n_layers=8 \
    --opt_config=BfastandCdecay \
    --p_dropout=0 --ssm_lr_base=0.001 --ssm_size_base=16 \
    --warmup_end=1 --weight_decay=0.04 \
    \
    2>&1 | tee "$CKPT_DIR/run.log"

echo "Done $(date)"
