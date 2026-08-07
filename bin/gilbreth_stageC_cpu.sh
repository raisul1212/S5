#!/bin/bash -l
# ============================================================================
# Stage C on CPU (no GPU) -- avoids the gres/hp_a100 concurrency limit that is
# blocking the GPU queue. Eval only (frozen backbone + cheap readout adaptation).
#   PAP sequential scan is slower on CPU but this is a one-pass eval; ~30-50 min.
# ============================================================================
#SBATCH --job-name=stageC_cpu
#SBATCH --partition=a30
#SBATCH --qos=standby
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/stageC_cpu_%j.out
#SBATCH --error=results/slurm/stageC_cpu_%j.err

source /etc/profile.d/modules.sh 2>/dev/null || true
source ~/.bashrc 2>/dev/null || true
module purge; module load external; module load anaconda/2024.10-py312
CONDA_BASE=$(conda info --base 2>/dev/null); source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate /scratch/gilbreth/raisul/envs/s5m
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PYTHONUTF8=1
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd $SLURM_SUBMIT_DIR; mkdir -p results/slurm results/stageC

PAP=$(ls -d checkpoints/enwik8_pap_linear_gelur0_*_11456877 2>/dev/null | head -1)/best
S5=$(ls -d checkpoints/enwik8_s5_gelur0_*_11456878 2>/dev/null | head -1)/best
SEG=${SEG:-20000}; CHUNK=${CHUNK:-64}; ETA=${ETA:-0.1}
echo "CPU StageC | PAP=$PAP  S5=$S5  SEG=$SEG CHUNK=$CHUNK ETA=$ETA  SHA=$(git rev-parse --short HEAD)"

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
