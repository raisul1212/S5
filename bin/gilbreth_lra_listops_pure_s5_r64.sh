#!/bin/bash -l
# Pure S5 P=8 with glu_rank=64 factorized -- rank-constraint control.
# Params: ~188K (SAME as Pure S5 full-rank via nn.Dense).  Only difference
# from Corner 1 is: factorized gate Dense(128,64) -> Dense(64,128), rank
# constrained to <=64 instead of full <=128.
#
# Isolates whether the Pure S5 r=40 (0.6285) anomaly comes from:
#   (a) fewer params (regularization from param reduction) OR
#   (b) rank constraint (implicit regularizer at same param count).
#
# If r=64 factorized lands near Pure S5 full's 0.6155 -> rank constraint
# alone doesn't help; the r=40 win is from param reduction.
# If r=64 factorized lands near Pure S5 r=40's 0.6285 -> rank constraint
# IS the regularizer; win transfers at iso-param.
#SBATCH --job-name=s5-pure-r64
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/s5_pure_r64_listops_%j.out
#SBATCH --error=results/slurm/s5_pure_r64_listops_%j.err

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
CKPT_DIR="./checkpoints/pure_s5_r64_${JOB}"
mkdir -p "$CKPT_DIR"

echo "Pure S5 P=8 glu_rank=64 factorized LRA-ListOps (JOB=$JOB, SHA=$SHA)"
echo "Checkpoints: $CKPT_DIR"

python -u run_train.py \
    --ckpt_dir="$CKPT_DIR" \
    --glu_rank=64 \
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
