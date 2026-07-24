#!/bin/bash -l
# v2 gate PILOT: Corner 3' slot (Mambino half_glu2 P=8, lambda_pc=0.0), one gate arm.
# All arms are iso-slot; only the output gate differs, so any accuracy delta is the gate.
#   Usage: sbatch bin/gilbreth_v2_pilot.sh <arm> <seed>
#   arm in {dense_r40, dense_r32, monarch, blockdiag}
#SBATCH --job-name=v2pilot
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=results/slurm/v2pilot_%j.out
#SBATCH --error=results/slurm/v2pilot_%j.err

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
export WANDB_MODE=disabled
cd $SLURM_SUBMIT_DIR
mkdir -p results/slurm

ARM=${1:-dense_r40}
SEED=${2:-42}
JOB=$SLURM_JOB_ID
SHA=$(git rev-parse HEAD)

# Gate arm -> flags.  glu_rank>0 is the "gate-on" trigger; for monarch/blockdiag the
# structured op's size is set by its own knob (heads=3 -> 9,216; blocks=2 -> 8,192).
case "$ARM" in
  dense_r40) GATE="--glu_structure=dense --glu_rank=40" ;;
  dense_r32) GATE="--glu_structure=dense --glu_rank=32" ;;
  monarch)   GATE="--glu_structure=monarch --glu_rank=40 --glu_monarch_heads=3" ;;
  blockdiag) GATE="--glu_structure=blockdiag --glu_rank=40 --glu_blockdiag_blocks=2" ;;
  monarch_16_8)      GATE="--glu_structure=monarch --glu_rank=40 --glu_monarch_heads=3 --glu_monarch_b=16" ;;
  monarch_16_8_res4) GATE="--glu_structure=monarch --glu_rank=40 --glu_monarch_heads=3 --glu_monarch_b=16 --glu_monarch_residual_rank=4" ;;
  *) echo "unknown arm '$ARM'"; exit 1 ;;
esac

CKPT_DIR="./checkpoints/v2pilot_${ARM}_seed${SEED}_${JOB}"
mkdir -p "$CKPT_DIR"
echo "v2 pilot arm=$ARM seed=$SEED (JOB=$JOB SHA=$SHA) gate=[$GATE]"

python -u run_train.py \
    --use_mambino_ssm=True \
    --lambda_pc=0.0 \
    --ckpt_dir="$CKPT_DIR" \
    $GATE \
    --C_init=lecun_normal --activation_fn=half_glu2 --batchnorm=True \
    --bidirectional=True --blocks=8 --bsz=50 --d_model=128 \
    --dataset=listops-classification --dt_min=0.001 --dt_max=0.1 \
    --epochs=40 --jax_seed=${SEED} --lr_factor=3 --n_layers=8 \
    --opt_config=BfastandCdecay \
    --p_dropout=0 --ssm_lr_base=0.001 --ssm_size_base=16 \
    --warmup_end=1 --weight_decay=0.04 \
    2>&1 | tee "$CKPT_DIR/run.log"

echo "Done $(date)"
