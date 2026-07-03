#!/bin/bash -l
# Smoke test 3: weight_sigma (STATIC device variation) with MC.
# Realistic well-designed 22nm chip range: 0.01 - 0.03.
# Baseline (weight=0) must match FP32.  3 MC seeds per weight value
# gives std across chip instances.
#SBATCH --job-name=smoke-weight
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/smoke_wgt_%j.out
#SBATCH --error=results/slurm/smoke_wgt_%j.err

source /etc/profile.d/modules.sh 2>/dev/null || true
source ~/.bashrc 2>/dev/null || true
module purge
module load external anaconda/2024.10-py312 cuda/13.1.0
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
conda activate /scratch/gilbreth/raisul/envs/s5m

export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
cd $SLURM_SUBMIT_DIR
mkdir -p results/slurm

CKPT=checkpoints/chip_mamb_iso_11181831
echo "smoke_weight on $CKPT -- sha=$(git rev-parse HEAD)"

python -u chip_sweep.py \
    --ckpt_prefix="$CKPT/best" \
    --csv="$CKPT/smoke_weight.csv" \
    --dataset=listops-classification \
    --sigmas=0 --bits=0 --crossings=1 \
    --weight_sigmas=0,0.01,0.02,0.03 \
    --mc_seeds=0,1,2 \
    --retention_sigmas=0 --read_sigmas=0 --digital_bits_list=0 \
    --d_model=128 --ssm_size_base=16 --n_layers=8 --blocks=8 \
    --bidirectional=True --C_init=lecun_normal --batchnorm=True \
    --conj_sym=True --bsz=50 \
    --activation_fn=half_glu2 --use_mambino_ssm=True --glu_rank=40 \
    2>&1 | tee "$CKPT/smoke_weight.log"

echo "Done $(date)"
