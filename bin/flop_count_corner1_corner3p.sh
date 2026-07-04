#!/bin/bash -l
#SBATCH --job-name=flop-corner
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/flop_corner_%j.out
#SBATCH --error=results/slurm/flop_corner_%j.err

source /etc/profile.d/modules.sh 2>/dev/null || true
source ~/.bashrc 2>/dev/null || true
module purge
module load external anaconda/2024.10-py312 cuda/13.1.0
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
conda activate /scratch/gilbreth/raisul/envs/s5m

export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
cd $SLURM_SUBMIT_DIR
mkdir -p results/slurm

# Corner 1: Pure S5 half_glu2, ssm_size_base=16, bidirectional, batchnorm
# Corner 3p: Same as Corner 1 but --use_mambino_ssm=True
for CFG in "corner1:corner1_seed42_11187391:False:16:half_glu2" \
           "corner3p:chip_mamb_iso_11181831:True:16:half_glu2"; do
  NAME=$(echo $CFG | cut -d: -f1)
  DIR=$(echo $CFG | cut -d: -f2)
  MAMB=$(echo $CFG | cut -d: -f3)
  SSMBASE=$(echo $CFG | cut -d: -f4)
  ACT=$(echo $CFG | cut -d: -f5)
  echo ""
  echo "############################################################"
  echo "# $NAME  (ckpt=$DIR  mambino=$MAMB  P=$SSMBASE  act=$ACT)"
  echo "############################################################"
  python -u flop_counter_jaxpr.py \
    --ckpt_prefix=checkpoints/${DIR}/best \
    --use_mambino_ssm=$MAMB \
    --activation_fn=$ACT \
    --ssm_size_base=$SSMBASE \
    --bidirectional=True \
    --batchnorm=True \
    --config_name=$NAME
done

echo "Done $(date)"
