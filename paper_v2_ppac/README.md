# Mambino paper — v2 chip PPAC (work in progress)

This directory holds the **v2 chip PPAC pipeline** that replaces the v7.1
"1 SRAM read per MAC" Accelergy setup with a cross-validated multi-tool pipeline:

- **SCALE-Sim v2** — cycle-accurate systolic array simulation, memory bandwidth
- **Timeloop** — dataflow mapping search + per-component energy attribution
- **Accelergy 0.4** — per-access energy backend (CACTI + NeuroSim), shared with both

## Methodology rule (locked 2026-07-08)

**Every PPAC workload MUST be extracted directly from the compiled JAXPR of the
trained model's forward pass.** No hand-authored SCALE-Sim CSVs, no hand-authored
Timeloop problem YAMLs. The extractor (`../extract_workload_from_jaxpr.py`) walks
the JAXPR of an actual checkpoint and emits per-op workload records with a
**hard invariant check**:

```
sum(emitted_dot_general_FLOPs) == JAXPR_walker.total_dot_general_FLOPs
sum(emitted_elementwise_FLOPs) == JAXPR_walker.total_elementwise_FLOPs
```

Any deviation → extractor exits with non-zero status. Any workload file in this
directory has already passed the invariant (see `manifest.yaml` per-config).

## Locked chip target (2026-07-08)

- Edge inference class, ~1 mm² die target
- **64×64 systolic MAC array**, weight-stationary, INT8, 22 nm, 1 GHz
- 256 KB weight SRAM, 128 KB activation SRAM, 64 KB state SRAM
- **No off-chip DRAM** (fully on-chip at all 5 configs' 105K–188K parameter sizes)

## Directory contents

- **`workloads/`** — 5 configs × 4 files = 20 files
  - `workload_<cfg>_gemms.csv` — SCALE-Sim `-i gemm` format, one row per `dot_general`
  - `workload_<cfg>_gemms.yaml` — Timeloop-friendly deduped unique GEMM shapes with repeat counts
  - `workload_<cfg>_elemwise.yaml` — elementwise ops grouped by (primitive, shape, dtype), tagged with `energy_class` (trivial / moderate / transcendental / reduction / structural)
  - `workload_<cfg>_manifest.yaml` — provenance (checkpoint path, git SHA, training args, walker totals, hard invariant check result)

## Per-config summary (extracted 2026-07-09)

| Config | dot_general FLOPs | dot_general/2 real MACs | elementwise FLOPs | Invariant |
|---|---:|---:|---:|:---:|
| Config 4 (Mambino gelu 106K) | 614,468,096 | 307,234,048 | 67,627,100 | ✓ |
| Config 5 (Pure S5 gelu 106K) | 681,576,960 | 340,788,480 | 63,631,580 | ✓ |
| Corner 1 (Pure S5 half_glu2 188K) | 882,903,552 | 441,451,776 | 65,766,556 | ✓ |
| Corner 2 (Pure S5 P=16 r=40 188K) | 1,017,121,280 | 508,560,640 | 76,214,492 | ✓ |
| Corner 3′ (Mambino r=40 188K) | 950,012,416 | 475,006,208 | 80,210,012 | ✓ |

dot_general/2 real-MAC values match the LOCKED master doc §4 exactly.

## Elementwise energy_class semantics

Downstream PPAC tools should model each class differently:

| Class | Examples | Typical cost | Chip modeling |
|---|---|---|---|
| **trivial** | `add`, `sub`, `neg`, `abs` | ~1 real FLOP/elem | Folds into MAC array pipeline (no separate energy line) |
| **moderate** | `mul`, `div`, `pow`, `integer_pow` | ~1–3 real FLOPs/elem | Small ALU adjacent to MAC array |
| **transcendental** | `exp`, `log`, `tanh`, `logistic`, `rsqrt`, `sqrt`, `erf` | ~4–10 real FLOPs/elem | LUT or Taylor expansion, ~5–10× MAC cost |
| **reduction** | `reduce_sum`, `reduce_max`, `reduce_prod` | ~N adds for N-element reduction | Chain of adds, memory-bound |
| **structural** | `reshape`, `transpose`, `broadcast_in_dim`, `slice`, `gather`, `lt`, `max`, etc. | 0 FLOPs | Data movement only; costs bytes through buffers, no compute |

## Reproducibility

To rebuild any workload file:

```bash
cd ~/dev/ssm-baselines/S5
python extract_workload_from_jaxpr.py \
    --ckpt_prefix checkpoints/<ckpt>/best \
    --config_name <cfg> \
    --out_dir     paper_v2_ppac/workloads/ \
    --dir_name    <path-to-cache-dir> \
    --dataset     listops-classification \
    --use_mambino_ssm <True|False> \
    --ssm_size_base <16|32> \
    --activation_fn <gelu|half_glu2> \
    --glu_rank <0|40> \
    --bidirectional True --batchnorm True \
    --blocks 8 --n_layers 8 --d_model 128
```

The hard invariant check will PASS if the walker + extractor agree bit-for-bit.

## Status

- ✅ Phase 1 — Tooling install (Timeloop, SCALE-Sim v2, Accelergy all working on Gilbreth)
- ✅ Phase 1a — Canonical GEMM cross-check (Timeloop 1.92 μJ vs SCALE-Sim 1271 cyc — both agree on MACs)
- ✅ Phase 1.5 — Extractor built + hard invariant enforced
- ✅ Phase 2 — All 5 configs' workloads extracted (this directory)
- ⏳ Phase 3 — Anchor validation against Simba or Eyeriss v2 published PPAC
- ⏳ Phase 4 — Run all 5 configs through Timeloop + SCALE-Sim + Accelergy; cross-tool agreement table
- ⏳ Phase 5 — Rewrite master doc §6/§7/§8 + Fig 4 + Discussion P3; tag `mambino-paper-v2`
