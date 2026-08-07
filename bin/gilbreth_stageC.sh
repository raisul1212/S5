#!/bin/bash -l
# ============================================================================
# Stage C: cross-lingual streaming recovery (anchor-2 inference-time learning).
# Runs the EN->L2->EN eval for PAP (frozen / +adapt / +bias-control) and S5
# (frozen / +adapt), for L2 in {fr, ru}. Short GPU job (eval only).
#   PAP_JOB / S5_JOB override the checkpoint job ids; SEG/CHUNK/ETA tunables.
# ============================================================================
#SBATCH --job-name=stageC
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=results/slurm/stageC_%j.out
#SBATCH --error=results/slurm/stageC_%j.err

source /etc/profile.d/modules.sh 2>/dev/null || true
source ~/.bashrc 2>/dev/null || true
module purge; module load external; module load anaconda/2024.10-py312; module load cuda/13.1.0
CONDA_BASE=$(conda info --base 2>/dev/null); source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate /scratch/gilbreth/raisul/envs/s5m
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PYTHONUTF8=1
cd $SLURM_SUBMIT_DIR; mkdir -p results/slurm results/stageC

PAP=$(ls -d checkpoints/enwik8_pap_linear_gelur0_*_11456877 2>/dev/null | head -1)/best
S5=$(ls -d checkpoints/enwik8_s5_gelur0_*_11456878 2>/dev/null | head -1)/best
SEG=${SEG:-20000}; CHUNK=${CHUNK:-64}; ETA=${ETA:-0.1}
echo "PAP=$PAP  S5=$S5  SEG=$SEG CHUNK=$CHUNK ETA=$ETA  SHA=$(git rev-parse --short HEAD)"

for L2 in fr ru; do
  echo "==================== L2=$L2 ===================="
  python -u paper_v2_ppac/pap_streaming/run_stageC.py \
    --pap_msgpack ${PAP}.msgpack --pap_meta ${PAP}.meta.pkl \
    --s5_msgpack ${S5}.msgpack --s5_meta ${S5}.meta.pkl \
    --eng cache_dir/enwik8_data/enwik8 --l2 cache_dir/l2_${L2}.txt \
    --seg_bytes $SEG --chunk $CHUNK --eta $ETA --l2_label ${L2^^} \
    --out results/stageC/stageC_${L2}.npz 2>&1
done
echo "Done $(date)"
