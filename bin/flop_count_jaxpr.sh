#!/bin/bash -l
# JAXPR-based FLOP counting on 3 gelu configs -- walks actual compiled
# computation graph.  No hand-derived formulas.
#SBATCH --job-name=flop-jaxpr
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/flop_jaxpr_%j.out
#SBATCH --error=results/slurm/flop_jaxpr_%j.err

source /etc/profile.d/modules.sh 2>/dev/null || true
source ~/.bashrc 2>/dev/null || true
module purge
module load external anaconda/2024.10-py312 cuda/13.1.0
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
conda activate /scratch/gilbreth/raisul/envs/s5m

export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
cd $SLURM_SUBMIT_DIR
mkdir -p results/slurm

for CFG in "config3:chip_pure_gelu_11181832:False:16" \
           "config4:chip_mamb_gelu_11181833:True:16" \
           "config5:train_pure_s5_gelu_p16_11187128:False:32"; do
  NAME=$(echo $CFG | cut -d: -f1)
  DIR=$(echo $CFG | cut -d: -f2)
  MAMB=$(echo $CFG | cut -d: -f3)
  SSMBASE=$(echo $CFG | cut -d: -f4)
  echo ""
  echo "############################################################"
  echo "# $NAME  (ckpt=$DIR  mambino=$MAMB  ssm_size_base=$SSMBASE)"
  echo "############################################################"
  python -u flop_counter_jaxpr.py \
    --ckpt_prefix=checkpoints/${DIR}/best \
    --use_mambino_ssm=$MAMB \
    --activation_fn=gelu \
    --ssm_size_base=$SSMBASE \
    --config_name=$NAME
done

echo "Done $(date)"
