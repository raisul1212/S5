#!/bin/bash -l
# ============================================================================
# Cluster B: Mambino-gelu + surprise-gated FAST WEIGHT (role 2 = inference-time
# learning).  Parallel additive branch, sibling of the main scan:
#     k,q = norm(W_k x), norm(W_q x)      shared address space
#     v   = norm(W_v eps)                 SIGNED content (the first moment)
#     s   = sigmoid(kappa*z + bias)       UNSIGNED write strength
#     M_t = gamma*M_{t-1} + (1-gamma)*s*(v k^T)     d x d STATE, not params
#     o_t = M_{t-1} q_t                   exclusive read (read before write)
#     y  += W_o o_t
#
# Default d=8 shared => +17,944 params (123,682 total, -34.4% vs Corner 1's
# 188,490 = the S5 authors' OWN ListOps config) and +22.5% MACs.
#
# TARGET (convention-independent): close ~0.4 pp.
#   Cluster A (mambino2p0) 0.6050 val-sel / 0.6175 best-seed-best-epoch
#   Corner 1 (Pure S5 188K) 0.6089 val-sel / 0.6235 best-seed-best-epoch
#   S5 published            0.6215 (single-seed best-epoch, inferred)
#
# ARMS via env vars.  Run 2 and 3 FIRST and read the ablation before spending
# on anything else -- arm 3 is the load-bearing control.
#   ARM=surprise   arm 2  the role-2 hypothesis                    (headline)
#   ARM=const      arm 3  kappa pinned to 0, bias learned -- SAME param count,
#                         SAME free write rate, differs ONLY in z-dependence
#   ARM=nomem      arm 4  gamma frozen at 0: gated LOCAL FEATURE, no memory.
#                         If this matches arm 2 there is no inference-time
#                         LEARNING, only a gated instantaneous path.
#   ARM=contentx   arm 6  fw_v_source=x: does surprise-as-CONTENT earn its
#                         place, or is the gate alone enough?
#   ARM=lean       arm 7  d=4 (+8.1% params)
#   ARM=wide       arm 9  d=16 (+36.8%, still -23% vs Corner 1) -- only if
#                         arm 2 shows a large effect
#   ARM=stacked    arm 8  fast weight ON TOP of the Cluster-A gate
#
# The full 8-seed sweep for the load-bearing pair:
#   for A in surprise const; do
#     for S in 6554595 42 12345 271828 314159 1 2 3; do
#       ARM=$A SEED=$S sbatch bin/gilbreth_lra_listops_mambino_gelu_fw.sh
#     done
#   done
# Analyse PAIRED across matched seeds (never as two independent means): paired
# gives ~0.9 power at n=8 for a 0.8 pp effect, unpaired only ~0.6.  Effects
# below ~0.5 pp are pre-registered as INCONCLUSIVE, not null.
# ============================================================================
#SBATCH --job-name=s5-mamb-fw
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/s5_mambino_gelu_fw_listops_%j.out
#SBATCH --error=results/slurm/s5_mambino_gelu_fw_listops_%j.err

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

ARM=${ARM:-surprise}
SEED=${SEED:-6554595}          # v1's 8 seeds: 6554595 42 12345 271828 314159 1 2 3
EPOCHS=${EPOCHS:-40}           # EPOCHS=1 for a smoke run

# arm defaults (headline), then per-arm overrides
FW_DIM=8; FW_GATE_MODE=surprise; FW_V_SOURCE=eps
FW_GAMMA_INIT=0.95; FW_GAMMA_TRAINABLE=True; SURPRISE_GATE=False
case "$ARM" in
  surprise) ;;
  const)    FW_GATE_MODE=const ;;
  nomem)    FW_GAMMA_INIT=0.0; FW_GAMMA_TRAINABLE=False ;;
  contentx) FW_V_SOURCE=x ;;
  lean)     FW_DIM=4 ;;
  wide)     FW_DIM=16 ;;
  stacked)  SURPRISE_GATE=True ;;
  *) echo "unknown ARM=$ARM"; exit 2 ;;
esac

JOB=$SLURM_JOB_ID
SHA=$(git rev-parse HEAD)
CKPT_DIR="./checkpoints/mambino_gelu_fw_${ARM}_d${FW_DIM}_s${SEED}_${JOB}"
mkdir -p "$CKPT_DIR"

echo "Cluster B fast weight | ARM=$ARM d=$FW_DIM gate=$FW_GATE_MODE v_src=$FW_V_SOURCE"
echo "  gamma_init=$FW_GAMMA_INIT trainable=$FW_GAMMA_TRAINABLE cluster_A_gate=$SURPRISE_GATE"
echo "LRA-ListOps  JOB=$JOB  SHA=$SHA  seed=$SEED  ckpt=$CKPT_DIR"

python -u run_train.py \
    --use_mambino_ssm=True \
    --fast_weight=True \
    --fw_dim=$FW_DIM \
    --fw_proj=shared \
    --fw_rule=hebb \
    --fw_kq_source=x \
    --fw_v_source=$FW_V_SOURCE \
    --fw_impl=chunk \
    --fw_chunk=64 \
    --fw_gate_mode=$FW_GATE_MODE \
    --fw_kappa_init=0.0 \
    --fw_bias_init=0.0 \
    --fw_gamma_init=$FW_GAMMA_INIT \
    --fw_gamma_trainable=$FW_GAMMA_TRAINABLE \
    --fw_norm_qkv=True \
    --fw_out_init=zeros \
    --fw_read=exclusive \
    --surprise_gate=$SURPRISE_GATE \
    --gate_range=signed --gate_alpha=0.9 --gate_kappa_init=0.0 --gate_bias_init=2.0 \
    --lambda_pc=0.0 \
    --ckpt_dir="$CKPT_DIR" \
    \
    --C_init=lecun_normal --activation_fn=gelu --batchnorm=True \
    --bidirectional=True --blocks=8 --bsz=50 --d_model=128 \
    --dataset=listops-classification \
    --epochs=$EPOCHS --jax_seed=$SEED --lr_factor=3 --n_layers=8 \
    --opt_config=BfastandCdecay \
    --p_dropout=0 --ssm_lr_base=0.001 --ssm_size_base=16 \
    --warmup_end=1 --weight_decay=0.04 \
    \
    2>&1 | tee "$CKPT_DIR/run.log"

echo "Done $(date)"
