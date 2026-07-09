# Mambino: Predictive-Coding-Augmented State-Space Models for Efficient Long-Range Reasoning on Constrained Hardware

**Reference implementation and reproducibility bundle for the Mambino paper.**

- **Paper:** *Mambino: Predictive-Coding-Augmented State-Space Models for Efficient Long-Range Reasoning on Constrained Hardware*
- **Author:** Raisul Islam
- **Affiliation:** Purdue University
- **Trained checkpoints + logs:** [PURR DOI 10.4231/9ADT-WP13](https://doi.org/10.4231/9ADT-WP13)
- **Code repo (this):** [github.com/raisul1212/S5](https://github.com/raisul1212/S5) — tag `mambino-paper-v2-public`
- **Contact:** raisul@alumni.stanford.edu

---

## What this repo lets you do

1. **Rebuild the exact model architectures** (Config 4, Config 5, Corner 1, Corner 2, Corner 3′) from source.
2. **Retrain from scratch** on LRA-ListOps and reproduce the paper's accuracy numbers within seed-noise (±0.3 pp).
3. **Download a trained checkpoint** from PURR and extract its workload with a hard invariant check.
4. **Run the v2 chip-PPAC pipeline** (Accelergy + Timeloop) and reproduce the paper's per-config chip energy numbers.
5. **Verify every paper number** end-to-end from training code → trained checkpoint → JAXPR-extracted workload → chip PPAC.

---

## Quick reproduction — the 10-minute path

If you just want to verify **one paper number** without retraining:

```bash
# 1. Clone + install
git clone --branch mambino-paper-v2-public https://github.com/raisul1212/S5.git mambino
cd mambino
pip install -r requirements_gpu.txt
pip install -e .

# 2. Download a checkpoint from PURR
#    (Corner 1 seed 42 = the paper's 188K Pure S5 baseline)
wget https://doi.org/10.4231/9ADT-WP13  # follow to download page
# extract to checkpoints/corner1_seed42_11187522/best.{pkl,meta.pkl,msgpack}

# 3. Extract JAXPR workload with hard invariant check
python extract_workload_from_jaxpr.py \
    --ckpt_prefix checkpoints/corner1_seed42_11187522/best \
    --config_name corner1 \
    --out_dir     paper_v2_ppac/workloads/ \
    --dir_name    ./cache_dir \
    --dataset     listops-classification \
    --use_mambino_ssm False --ssm_size_base 16 --activation_fn half_glu2 \
    --glu_rank 0 --bidirectional True --batchnorm True --blocks 8 \
    --n_layers 8 --d_model 128
# Expected output:
#   [extract] dot_general: walker=882,903,552 emitter=882,903,552 (OK); ...
#   [extract] HARD INVARIANT PASSED — workload matches walker exactly.

# 4. Run the v2 chip PPAC pipeline (canonical GEMM verification)
cd paper_v2_ppac/chip
accelergy arch_accelergy.yaml components/*.yaml dummy_16x16_actions.yaml -o out_accelergy
timeloop-model arch_mambino.cfg
# Expected output:  Utilization = 1.00 | pJ/Compute = 5.613 | Cycles: 8192 | Energy: 11.77 uJ
```

If both steps produce the expected output, the pipeline is working on your machine.

---

## Full reproduction — retraining from scratch

```bash
# Retrain Corner 1 (Pure S5 baseline, 188K, seed=42)
python run_train.py \
    --ckpt_dir="checkpoints/corner1_repro" \
    --glu_rank=0 --C_init=lecun_normal --activation_fn=half_glu2 \
    --batchnorm=True --bidirectional=True --blocks=8 --bsz=50 \
    --d_model=128 --dataset=listops-classification \
    --dt_min=0.001 --dt_max=0.1 --epochs=40 --jax_seed=42 \
    --lr_factor=3 --n_layers=8 --opt_config=BfastandCdecay \
    --p_dropout=0 --ssm_lr_base=0.001 --ssm_size_base=16 \
    --warmup_end=1 --weight_decay=0.04

# Expected wall-clock: ~2 hours on a single NVIDIA A30 or equivalent
# Expected result: test@peakval within 0.615 ± 0.005 (paper reports 0.620 for seed=42)
```

Sbatch scripts for all 5 configs at all 8 seeds are in `bin/`.

---

## What's in this repo

| Directory | Purpose |
|---|---|
| `s5/` | Model architecture (`mambino_ssm.py`, `ssm.py`, `layers.py`, ...) |
| `bin/` | SLURM sbatch scripts, one per paper config |
| `flop_counter_jaxpr.py` | JAXPR walker; counts primitives from the compiled forward pass |
| `extract_workload_from_jaxpr.py` | Workload extractor; enforces hard invariant |
| `paper_v2_ppac/workloads/` | Extracted per-config workload files (4 files × 5 configs) |
| `paper_v2_ppac/chip/` | v2 chip PPAC pipeline (Accelergy + Timeloop configs) |
| `paper_v2_ppac/chip/pipeline_diagram.html` | Detailed methods figure (open in browser) |
| `accelergy/LOCKED_master_ppac_document.md` | Master doc with all reported numbers, citations, and provenance |
| `accelergy/DATA_AVAILABILITY_STATEMENT.md` | PURR DOI + reproduction instructions |
| `accelergy/mambino_environment.md` | Exact software + hardware environment for reported numbers |
| `paper/figure_data/` | CSV data behind every paper figure |

---

## The methodology invariant

Every chip-PPAC workload used in the paper is **extracted directly from the compiled JAXPR of the trained model's forward pass**. There are no hand-authored SCALE-Sim CSVs, no hand-authored Timeloop problem YAMLs, and no hand-crafted Accelergy action counts describing what we *think* the model computes.

The extractor enforces a hard invariant on every emit:

```
sum(emitted.dot_general_FLOPs) == JAXPR walker.total_dot_general
sum(emitted.elementwise_FLOPs) == JAXPR walker.total_elementwise
```

If either fails, the script exits non-zero and the workload is not usable.

This means: **any paper number → tool output → explicit mapping → workload YAML → JAXPR walker → training code → checkpoint** is bit-for-bit traceable. Reviewer challenges 5 years from now can be answered by rerunning the pipeline on the PURR-archived checkpoint.

See `paper_v2_ppac/chip/pipeline_diagram.html` for the full end-to-end diagram.

---

## Software environment

- Python 3.11.15
- JAX 0.4.30 + jaxlib 0.4.30
- Flax 0.8.5
- NumPy 1.26.4
- NVIDIA CUDA 13.1 (any recent version compatible with jaxlib 0.4.30)
- Accelergy 0.4 + CACTI plug-in + NeuroSim plug-in
- Timeloop (mainline branch, built from source — see `paper_v2_ppac/chip/README.md`)
- SCALE-Sim v2.0.2 (`pip install scalesim==2.0.2`)

Full spec + pip freeze: `accelergy/mambino_environment.md`, `accelergy/mambino_pip_freeze.txt`.

---

## Hardware used to produce the paper's numbers

- Purdue University Gilbreth cluster, `a30` SLURM partition
- Per SLURM job: NVIDIA A30 24 GB HBM, 4 CPU cores, 16 GB host RAM
- Wall-clock per training run: 1.5–2.5 h for 40 epochs on LRA-ListOps (L=2048)
- Total training compute across the 8-seed × 5-config sweep + λ_pc ablation:
  approximately 90 GPU-hours

---

## Citing this work

```bibtex
@inproceedings{islam2026mambino,
  author    = {Islam, Raisul},
  title     = {Mambino: Predictive-Coding-Augmented State-Space Models for Efficient Long-Range Reasoning on Constrained Hardware},
  booktitle = {[TBD conference proceedings]},
  year      = {2026}
}
```

If you use the checkpoints or extracted workloads, please also cite the PURR deposit:

```bibtex
@dataset{islam2026purr,
  author    = {Islam, Raisul},
  title     = {Mambino: trained checkpoints, run logs, and per-run manifest (v1)},
  publisher = {Purdue University Research Repository},
  year      = {2026},
  doi       = {10.4231/9ADT-WP13}
}
```

---

## Licenses

- **Code:** MIT License (see `LICENSE`)
- **Trained checkpoints and logs** (PURR deposit): CC-BY-4.0
- **Paper text and figures:** subject to the publication venue's copyright

## Acknowledgments

Built on Simplified State Space Layers for Sequence Modeling
([S5, Smith et al. 2023](https://openreview.net/forum?id=Ai8Hw3AXqks)) — this repo
is a fork with the Mambino architecture added.

Chip PPAC uses [Timeloop](https://github.com/NVlabs/timeloop) (NVIDIA / MIT),
[Accelergy](https://github.com/Accelergy-Project/accelergy) (MIT Sze/Emer),
[SCALE-Sim](https://github.com/scalesim-project/scale-sim-v2) (Georgia Tech Krishna).
