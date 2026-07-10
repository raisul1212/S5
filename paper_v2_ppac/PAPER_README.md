# Mambino: Predictive-Coding-Augmented State-Space Models for Efficient Long-Range Reasoning on Constrained Hardware

**Reference implementation and reproducibility bundle for the Mambino paper.**

- **Paper**: Islam et al., *Mambino: Predictive-Coding-Augmented State-Space Models for Efficient Long-Range Reasoning on Constrained Hardware*, IEEE TNNLS (in submission)
- **Author**: Raisul Islam (raisul@alumni.stanford.edu)
- **Affiliation**: Purdue University
- **Trained checkpoints + logs**: PURR DOI (populate at submission)
- **Code repo**: [github.com/raisul1212/S5](https://github.com/raisul1212/S5) — tag `mambino-paper-v2-public`

---

## What this repo lets you do

1. **Rebuild the exact model architectures** (Config 4, Config 5, Corner 1, Corner 2, Corner 3′) from source.
2. **Retrain from scratch** on LRA-ListOps and reproduce the paper's accuracy numbers within seed-noise (±0.3 pp).
3. **Download a trained checkpoint** from PURR and extract its workload with a hard invariant check.
4. **Run the v4 multi-array chip-PPAC pipeline** and reproduce the paper's chip numbers bit-for-bit.
5. **Cross-validate cycle counts against SCALE-Sim v2** on rectangular arrays.
6. **Verify every paper number** end-to-end from training code → trained checkpoint → JAXPR-extracted workload → chip PPAC.

---

## Quick reproduction — the 10-minute path

Verify one paper number without retraining:

```bash
# 1. Clone + install (2 min)
git clone --branch mambino-paper-v2-public https://github.com/raisul1212/S5.git mambino
cd mambino
pip install -r requirements_cpu.txt          # or requirements_gpu.txt

# 2. Run the v4 chip PPAC simulator (30 sec)
python paper_v2_ppac/chip/multi_array_ppac_v4.py

# 3. Compare against the paper's headline numbers
python -c "
import json
d = json.load(open('paper_v2_ppac/chip/multi_array_sweep_v4.json'))
c1 = min(d['corner1'],  key=lambda r: r['area_x_latency'])
c3 = min(d['corner3p'], key=lambda r: r['area_x_latency'])
print(f'Corner 1 ATP-optimal:  {c1[\"total_area_mm2\"]:.2f} mm², {c1[\"power_mW\"]:.0f} mW, acc {c1[\"acc\"]:.4f}')
print(f'Corner 3\\' ATP-optimal: {c3[\"total_area_mm2\"]:.2f} mm², {c3[\"power_mW\"]:.0f} mW, acc {c3[\"acc\"]:.4f}')
print(f'Delta: area {100*(c3[\"total_area_mm2\"]-c1[\"total_area_mm2\"])/c1[\"total_area_mm2\"]:+.1f}%, '
      f'power {100*(c3[\"power_mW\"]-c1[\"power_mW\"])/c1[\"power_mW\"]:+.1f}%, '
      f'accuracy {100*(c3[\"acc\"]-c1[\"acc\"]):+.2f} pp')
"
```

Expected output:
```
Corner 1 ATP-optimal:  32.09 mm², 1875 mW, acc 0.6089
Corner 3' ATP-optimal: 19.76 mm², 816 mW,  acc 0.6138
Delta: area -38.4%, power -56.5%, accuracy +0.49 pp
```

These are the paper's headline chip numbers.

---

## Full retraining path (2–3 hours per config on 1× A30 GPU)

To retrain all five configs (Config 4, Config 5, Corner 1, Corner 2, Corner 3′):

```bash
# Set up dataset (~2 GB, needs ~1 hour download)
python -m s5.dataloaders.lra --data_dir ./cache_dir --task listops

# Train Corner 3' (Mambino low-rank, the paper's hero) at seed=42
python run_train.py \
    --dir_name ./cache_dir --use_mambino_ssm True --activation_fn half_glu2 \
    --ssm_size_base 16 --glu_rank 40 --lambda_pc 0.0 \
    --n_layers 8 --d_model 128 --blocks 8 --bidirectional True \
    --dataset listops-classification --epochs 40 --bsz 50 --jax_seed 42 \
    --dt_min 0.001 --dt_max 0.1 --C_init lecun_normal \
    --opt_config BfastandCdecay --p_dropout 0 --ssm_lr_base 0.001 \
    --lr_factor 3 --warmup_end 1 --weight_decay 0.04 \
    --wandb_project mambino_reproduction --wandb_entity <your_wandb_entity>
```

Full 8-seed sweep sbatch scripts are in `bin/`. For all five configs' training args, see
[`accelergy/LOCKED_master_ppac_document.md`](accelergy/LOCKED_master_ppac_document.md) §2.

---

## v4 chip PPAC methodology (2 min read)

The paper's chip PPAC is a **multi-array direct-instrumentation simulator** at
22 nm INT8. Each unique GEMM shape in a config's workload gets its own dedicated
weight-stationary systolic array; blocks execute serially per inference.

**Design rule per block**: smallest {8, 16, 32, 64}² rectangle with 100% array
utilization whose total cycles fit under a per-config bottleneck target `T`.
Sweeping `T` traces each config's Area × Latency Product Pareto.

**Applied corrections** (documented in code as H1–H4 and F1–F3):
- **H1** fill/drain per weight-tile boundary
- **H2** operand-role detection (smaller matrix is stationary weight)
- **H3** elementwise ops charge activation-SRAM memory traffic
- **H4** partial-sum spill accounting
- **F1** M-chunking so psum residency fits activation SRAM (preserves no-DRAM claim)
- **F2** wc_rep no longer double-counted in picker
- **F3** elementwise structural ops (reshape/broadcast) skip memory unless materializing

**Silicon includes**: PE arrays, weight/activation/state SRAM, NoC (15% of PE area),
control logic (5% of PE area). NoC + control dynamic energy = 10% of MAC energy.

Full methodology in [`accelergy/LOCKED_master_ppac_document.md`](accelergy/LOCKED_master_ppac_document.md) §6.

---

## Verifying the JAXPR workload invariant

Every chip number in the paper is anchored to a specific trained checkpoint via a JAXPR
walker with a hard invariant:

```bash
# Download a checkpoint from PURR (e.g., Corner 3' seed=42)
# Then extract its workload
python extract_workload_from_jaxpr.py \
    --ckpt_prefix ./checkpoints/corner3p_seed42/best \
    --config_name corner3p \
    --out_dir ./verify_workload/

# Verify hard invariant
python -c "
import yaml
m = yaml.safe_load(open('./verify_workload/workload_corner3p_manifest.yaml'))
walker = m['jaxpr_walker_totals']
gemms = yaml.safe_load(open('./verify_workload/workload_corner3p_gemms.yaml'))
elems = yaml.safe_load(open('./verify_workload/workload_corner3p_elemwise.yaml'))
emitted_dot = sum(g['total_real_macs'] * 2 for g in gemms['gemms'])
walker_dot = walker['dot_general']
assert emitted_dot == walker_dot, f'INVARIANT FAILED: {emitted_dot} != {walker_dot}'
print(f'Hard invariant PASSED: {emitted_dot:,} dot_general FLOPs emitted == walker total')
"
```

If the invariant fails, the chip PPAC number is not trustworthy — abort and file an issue.

---

## SCALE-Sim v2 cross-validation

Cycle model validation on 10 shape × rectangle combinations from Corner 1 and Corner 3′ ATP-optimal
block assignments:

```bash
# Requires scalesim-v2 installed: pip install scalesim
python paper_v2_ppac/chip/run_scalesim_multi_array_validation.py
```

On B-stationary shapes (5/10) the analytical cycle model agrees with SCALE-Sim within
0.75–1.03× (mean 0.91×); on clean-divisibility shapes (3/10) within ±3%. A-stationary
shapes (5/10) cannot be directly validated against SCALE-Sim v2 (no A-stationary WS mode);
they use the same fill/drain formula by symmetry with B-stationary.

See [`paper_v2_ppac/chip/scalesim_multi_array_validation/validation_summary.csv`](paper_v2_ppac/chip/scalesim_multi_array_validation/validation_summary.csv).

---

## Repository layout

```
mambino/
├── LICENSE                                (MIT)
├── PAPER_README.md                        (this file)
├── CITATION.cff                           (BibTeX for the paper)
├── setup.py
├── requirements_cpu.txt, requirements_gpu.txt
│
├── s5/                                    (training code)
│   ├── ssm.py                             (Pure S5)
│   ├── mambino_ssm.py                     (Mambino predictor branch)
│   ├── layers.py                          (SequenceLayer w/ half_glu2 + rank-r gate)
│   ├── seq_model.py                       (BatchClassificationModel)
│   └── ...
│
├── run_train.py                           (training entry point)
├── flop_counter_jaxpr.py                  (JAXPR walker)
├── extract_workload_from_jaxpr.py         (workload extractor + hard invariant)
│
├── bin/                                   (6 sbatch scripts, one per config)
│
├── paper_v2_ppac/                         (v4 chip PPAC pipeline)
│   ├── README.md
│   ├── PAPER_TNNLS_OUTLINE.md             (paper structure)
│   ├── PUBLIC_RELEASE_PLAN.md             (this release's provenance)
│   ├── workloads/                         (20 JAXPR-verified files)
│   └── chip/
│       ├── multi_array_ppac_v4.py         ← the paper's ground-truth simulator
│       ├── multi_array_sweep_v4.json      ← canonical output
│       ├── build_chip_specs.py
│       ├── gen_ert_all_variants.sh
│       ├── run_scalesim_multi_array_validation.py
│       ├── out_accelergy_{tier}_{size}/   (12 Accelergy runs = ERT + ART)
│       ├── components/                    (Accelergy compound classes)
│       ├── scalesim_multi_array_validation/
│       └── pipeline_diagram.html          (methods figure)
│
├── accelergy/
│   ├── LOCKED_master_ppac_document.md     (the paper's methodology reference)
│   ├── DATA_AVAILABILITY_STATEMENT.md
│   ├── mambino_environment.md
│   ├── mambino_pip_freeze.txt
│   └── primitive_component_libs/mambino_primitives.lib.yaml
│
└── paper/figure_data/                     (figure CSVs)
```

---

## Environment

Frozen at repo tag `mambino-paper-v2-public`:

- Python 3.11.15
- JAX 0.4.30, jaxlib 0.4.30
- Flax 0.8.5, chex 0.1.86, einops 0.8.2
- NumPy 1.26.4
- Accelergy 0.4 + CACTI 7 + NeuroSim (for regenerating ERTs)
- SCALE-Sim v2 (for cycle cross-validation)

Full env in `accelergy/mambino_environment.md` and `accelergy/mambino_pip_freeze.txt`.

---

## Citing the paper

```bibtex
@article{islam2026mambino,
  author  = {Raisul Islam},
  title   = {{Mambino}: Predictive-Coding-Augmented State-Space Models for Efficient
             Long-Range Reasoning on Constrained Hardware},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  year    = {2026},
  note    = {In submission},
}
```

---

## License

MIT for the code (see `LICENSE`). Pre-trained checkpoints on PURR are CC-BY-4.0.

---

## Contact + support

- Issues: [github.com/raisul1212/S5/issues](https://github.com/raisul1212/S5/issues)
- Email: raisul@alumni.stanford.edu
