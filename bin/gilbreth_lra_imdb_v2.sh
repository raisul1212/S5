#!/bin/bash -l
# ============================================================================
# LRA-Text (IMDB) — the paper's SECOND task. Four configurations, eight seeds.
#
# Protocol, metrics and claims are PRE-REGISTERED in
#   paper_v2_ppac/memos/imdb_preregistration.md   (committed 2026-08-04,
#   before any IMDB run was launched). Read it before changing anything here.
#
# WHY IMDB and not sCIFAR/Pathfinder/AAN. The mechanism is parameter-favourable
# only when H > 6P (predictor costs 6PH, the gate it displaces costs H^2).
# Across the six configs the S5 authors chose, ONLY ListOps clears it:
#   ListOps H/P=16.0 | IMDB 2.7 | sCIFAR 2.7 | Pathfinder 1.5 | AAN 1.0 | Path-X 1.0
# So no second LRA task sits inside the favourable regime, and IMDB is picked on
# cost (35 epochs vs 200-250) and on being the same single-sequence code path.
# The chip result is EXPECTED to be weaker here; that is the point. It turns the
# H > 6P design rule from a derivation into a measurement on both sides of the
# threshold.
#
# CONFIGS (params asserted, not estimated — see the memo's table):
#   S5-Dense   P=96  ssm=192 blocks=12 half_glu2   1,321,154   (S5's own config)
#   S5-0       P=138 ssm=276 blocks=6  gelu        1,314,230
#   Mambino-0  P=69  ssm=138 blocks=3  gelu+pred   1,314,230   (== S5-0 exactly)
#   Mambino-G  P=69  ssm=138 blocks=3  +gate       1,314,242
#
# BLOCKS. train.py does block_size = ssm_size_base/blocks, then halves it under
# conjugate symmetry and tiles blocks x block_size. So ssm_size_base/blocks must
# be EVEN or the halving truncates: 276/12 = 23 -> 23//2 = 11 -> 11*12 = 132
# state entries where 138 are needed, and the run dies with
#   TypeError: mul got incompatible shapes for broadcasting: (132,), (138,)
# blocks=6 and blocks=3 both give block_size=46 -> 23 per block. Parameter counts
# are UNAFFECTED by blocks (it only tiles the HiPPO init), so the pre-registered
# totals are unchanged.
#
# VAL SPLIT. S5 ships no val split for IMDB; their loader puts the TEST set in
# the validation role ("Use test set as val set, as done in the LRA paper",
# s5/dataloaders/lra.py:139), so their reported checkpoint is test-selected.
# We pass --val_split=0.1 to select on genuinely held-out data, matching the
# ListOps protocol. --val_split defaults to 0.0 everywhere else, which
# reproduces the released behaviour byte-for-byte.
#
# Deploy (pilot the mechanism claim FIRST — 6 runs, not 32):
#   for C in S5-0 Mambino-0; do
#     for S in 6554595 42 12345; do
#       CONFIG=$C SEED=$S sbatch --job-name=imdb.$C.$S bin/gilbreth_lra_imdb_v2.sh
#     done
#   done
# Then, only if the sign matches ListOps, the full set:
#   for C in S5-0 Mambino-0 Mambino-G S5-Dense; do
#     for S in 6554595 42 12345 271828 314159 1 2 3; do ... done
#   done
# ============================================================================
#SBATCH --job-name=s5-imdb
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=results/slurm/imdb_%j.out
#SBATCH --error=results/slurm/imdb_%j.err

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

CONFIG=${CONFIG:-Mambino-0}
SEED=${SEED:-6554595}
EPOCHS=${EPOCHS:-35}
VAL_SPLIT=${VAL_SPLIT:-0.1}

# Names are canonical per mambino-paper/CLAUDE.md section 1.
case "$CONFIG" in
  S5-0)       ARCH="--use_mambino_ssm=False --activation_fn=gelu      --glu_rank=0 --ssm_size_base=276 --blocks=6"
              GATE="" ;;
  Mambino-0)  ARCH="--use_mambino_ssm=True  --activation_fn=gelu      --glu_rank=0 --ssm_size_base=138 --blocks=3"
              GATE="--surprise_gate=False --lambda_pc=0.0" ;;
  Mambino-G)  ARCH="--use_mambino_ssm=True  --activation_fn=gelu      --glu_rank=0 --ssm_size_base=138 --blocks=3"
              GATE="--surprise_gate=True --gate_range=signed --gate_alpha=0.9 --gate_kappa_init=0.0 --gate_bias_init=2.0 --lambda_pc=0.0" ;;
  S5-Dense)   ARCH="--use_mambino_ssm=False --activation_fn=half_glu2 --glu_rank=0 --ssm_size_base=192 --blocks=12"
              GATE="" ;;

  # ---- REDUCED-BUDGET ARM (suffix -R). Tests a DIFFERENT claim from C1-C3: ----
  # parameter reduction, not mechanism-at-fixed-parameters. Here the gate's
  # budget is BANKED rather than reallocated, giving 926,402 params = 70.1% of
  # S5-Dense (-29.9%). Mambino-0-R at P=48 is still exactly iso-parameter with
  # S5-0-R at P=96, by the same identity (predictor at P == doubling P).
  # block_size = 16 throughout, matching S5-Dense's own blocking.
  # This is the arm where the chip result should be strongest: the 64x64 gate
  # array is deleted and the saving is kept, instead of being spent on a larger
  # state as in the iso-parameter arm above.
  #   S5-0-R       P=96 ssm=192 blocks=12 gelu           926,402
  #   Mambino-0-R  P=48 ssm=96  blocks=6  gelu+pred      926,402  (== S5-0-R)
  #   Mambino-G-R  P=48 ssm=96  blocks=6  +gate          926,414
  S5-0-R)     ARCH="--use_mambino_ssm=False --activation_fn=gelu      --glu_rank=0 --ssm_size_base=192 --blocks=12"
              GATE="" ;;
  Mambino-0-R) ARCH="--use_mambino_ssm=True --activation_fn=gelu      --glu_rank=0 --ssm_size_base=96  --blocks=6"
              GATE="--surprise_gate=False --lambda_pc=0.0" ;;
  Mambino-G-R) ARCH="--use_mambino_ssm=True --activation_fn=gelu      --glu_rank=0 --ssm_size_base=96  --blocks=6"
              GATE="--surprise_gate=True --gate_range=signed --gate_alpha=0.9 --gate_kappa_init=0.0 --gate_bias_init=2.0 --lambda_pc=0.0" ;;

  *) echo "unknown CONFIG=$CONFIG (S5-0|Mambino-0|Mambino-G|S5-Dense|S5-0-R|Mambino-0-R|Mambino-G-R)"; exit 2 ;;
esac

JOB=$SLURM_JOB_ID
SHA=$(git rev-parse HEAD)
CKPT_DIR="./checkpoints/imdb_${CONFIG}_s${SEED}_${JOB}"
mkdir -p "$CKPT_DIR"
echo "IMDB-V2 | CONFIG=$CONFIG seed=$SEED epochs=$EPOCHS val_split=$VAL_SPLIT"
echo "  $ARCH $GATE"
echo "JOB=$JOB SHA=$SHA ckpt=$CKPT_DIR"

# S5's own IMDB recipe below this line, unchanged. Only $ARCH/$GATE vary.
python -u run_train.py \
    $ARCH $GATE \
    --ckpt_dir="$CKPT_DIR" \
    --val_split=$VAL_SPLIT \
    \
    --C_init=lecun_normal --batchnorm=True \
    --bidirectional=True --bsz=50 --d_model=256 \
    --dataset=imdb-classification --dt_global=True \
    --epochs=$EPOCHS --jax_seed=$SEED --lr_factor=4 --n_layers=6 \
    --opt_config=standard \
    --p_dropout=0.1 --ssm_lr_base=0.001 \
    --warmup_end=0 --weight_decay=0.07 \
    \
    2>&1 | tee "$CKPT_DIR/run.log"

echo "Done $(date)"
