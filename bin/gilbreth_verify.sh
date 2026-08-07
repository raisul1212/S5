#!/bin/bash -l
# Stage-C rigorous verification as a GPU job (fast: PAP scan is ~seconds on GPU vs minutes on
# CPU). Standby QOS (submit with --qos=standby --time<4h). Writes to shared /home (robust vs the
# login-node round-robin). SEG/N/ETAS overridable.
#SBATCH --job-name=verify
#SBATCH --partition=a30
#SBATCH --account=raisul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=results/slurm/verify_%j.out
#SBATCH --error=results/slurm/verify_%j.err

source /etc/profile.d/modules.sh 2>/dev/null || true
source ~/.bashrc 2>/dev/null || true
module purge; module load external; module load anaconda/2024.10-py312; module load cuda/13.1.0
CONDA_BASE=$(conda info --base 2>/dev/null); source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate /scratch/gilbreth/raisul/envs/s5m
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PYTHONUTF8=1
cd $SLURM_SUBMIT_DIR; mkdir -p results/stageC results/slurm

PAP=$(ls -d checkpoints/enwik8_pap_linear_gelur0_*_11456877 2>/dev/null | head -1)/best
S5=$(ls -d checkpoints/enwik8_s5_gelur0_*_11456878 2>/dev/null | head -1)/best
SEG=${SEG:-20000}; N=${N:-5}; ETAS=${ETAS:-0.03,0.1,0.3}
echo "GPU verify | PAP=$PAP S5=$S5 SEG=$SEG N=$N SHA=$(git rev-parse --short HEAD) start=$(date)"

python -u paper_v2_ppac/pap_streaming/run_verify.py \
  --pap_msgpack ${PAP}.msgpack --pap_meta ${PAP}.meta.pkl \
  --s5_msgpack ${S5}.msgpack --s5_meta ${S5}.meta.pkl \
  --eng cache_dir/enwik8_data/enwik8 --l2_fr cache_dir/l2_fr.txt --l2_ru cache_dir/l2_ru.txt \
  --seg $SEG --n $N --chunk 64 --etas $ETAS --out results/stageC/verify.npz 2>&1
echo "end=$(date)"
