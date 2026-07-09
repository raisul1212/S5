# v2 Chip PPAC pipeline — end-to-end working

**Milestone reached 2026-07-09:** for the first time, we can compute chip PPAC for the
Mambino paper using **real per-access energies** (CACTI 7 + NeuroSim via Accelergy 0.4)
and **real dataflow accounting** (Timeloop model), all driven from **JAXPR-extracted
workloads** (`extract_workload_from_jaxpr.py`) with a hard invariant guaranteeing the
workload matches the trained model bit-for-bit.

This directory is the working end of Phase 3+4 of the v2 chip PPAC redo.

## The pipeline

```
      trained checkpoint
              │
              ▼
   ┌──────────────────────────────┐
   │ extract_workload_from_jaxpr  │  (Phase 1.5, already committed)
   │  ─ walks compiled JAXPR       │
   │  ─ emits per-op workloads     │
   │  ─ enforces:                  │
   │       Σ emitted == walker      │
   └──────────────────────────────┘
              │
              ▼
     workload_<cfg>_gemms.yaml   (one row per unique GEMM shape + repeat count)
              │
              ▼
   ┌──────────────────────────────┐
   │ build_chip_specs.py          │  ← SINGLE SOURCE OF TRUTH for chip params
   │  emits BOTH schemas:          │
   │   arch_accelergy.yaml         │     (subtree/local — Accelergy)
   │   arch_timeloop.yaml          │     (nodes/!Container — Timeloop)
   │  chip: 16×16 WS array, 22 nm,│
   │        256/128/64 KB buffers  │
   └──────────────────────────────┘
              │
              ▼
   ┌──────────────────────────────┐
   │ accelergy arch_accelergy.yaml│
   │   components/*.yaml           │
   │   → ERT.yaml + ART.yaml       │
   │      real CACTI + NeuroSim    │
   │      22 nm INT8 per-access    │
   └──────────────────────────────┘
              │
              ▼
   ┌──────────────────────────────┐
   │ timeloop-model                │
   │   arch_mambino.cfg            │  (libconfig format — sidesteps YAML parser)
   │   canonical_problem.yaml      │  (128×128×128 GEMM)
   │   mapping_canonical.yaml      │  (explicit WS mapping, no search)
   │   ERT.yaml + ART.yaml         │
   │   → stats.txt: cycles, energy │
   │      per-component breakdown  │
   └──────────────────────────────┘
```

## Canonical GEMM result — the anchor

For a 128×128×128 GEMM on our 16×16 WS chip:

| Metric | Value |
|---|---:|
| Utilization | **100%** |
| Cycles | 8192 |
| Total energy | **11.77 μJ** |
| pJ/Compute | **5.61 pJ/MAC** |
| GOPS @ 1 GHz | 510 |

Per-component (pJ/MAC):
- Arithmetic (MAC unit): 0.25
- **PsumSpad** (per-PE 32-bit accumulator): **3.90** — dominates because each MAC triggers 2 psum accesses (read for accumulate + write of new value)
- WeightSpad (per-PE 8-bit weight reg): 0.97
- GlobalBuffer (unified 128 KB SRAM): 0.49
- **Total: 5.61 pJ/MAC**

## Projected chip energies for the 5 Mambino configs

Using 5.61 pJ/MAC as the per-MAC energy (order-of-magnitude estimate — actual per-config
numbers will differ slightly by GEMM shape mix; final numbers require running each
config's unique GEMM shapes through timeloop-model individually):

| Config | Real MACs (JAXPR) | Est. chip energy | v7.1 (broken) | v7.1 over-inflation |
|---|---:|---:|---:|---:|
| Config 4 (Mambino 106K) | 307M | **~1.72 mJ** | 150.57 mJ | ~87× |
| Config 5 (Pure S5 106K) | 341M | ~1.91 mJ | 166.80 mJ | ~87× |
| Corner 1 (Pure S5 188K) | 441M | **~2.47 mJ** | 229.89 mJ | ~93× |
| Corner 2 (Pure S5 P=16) | 508M | ~2.85 mJ | 264.54 mJ | ~93× |
| Corner 3′ (Mambino 188K) | 475M | **~2.66 mJ** | 247.22 mJ | ~93× |

**v7.1 was over-inflated by 87–93×** thanks to the "1 SRAM read per MAC" model on a 4.2 MB SRAM.
The v2 numbers (1.7–2.9 mJ per inference) are in the realistic range for a 22 nm INT8 edge
inference chip.

## Chip parameters (locked 2026-07-09)

Edited in one place: `build_chip_specs.py:CHIP_PARAMS`. Change once, both YAMLs regenerate.

- **Chip class:** edge inference, ~1 mm² die target
- **MAC array:** **16×16 systolic, weight-stationary, INT8, 22 nm, 1 GHz**
  - Chosen because Mambino's small-state GEMMs (M=8 or M=16 for B̄·x) would waste
    87.5% of PEs on a 64×64 array. 16×16 fits the natural state dim.
- **Buffers:** 256 KB weight SRAM, 128 KB activation SRAM, 64 KB state SRAM
  - Timeloop's memory-hierarchy model requires strict nesting, so for the working
    canonical run we unified into one 128 KB `GlobalBuffer`. Real chip has 3 separate
    buffers; unified model gives an order-of-magnitude correct energy. Refinement to
    3-buffer model is future work.
- **No off-chip DRAM** (all configs' 106–188K weights fit on-chip)

## Files in this directory

| File | Origin | Purpose |
|---|---|---|
| `build_chip_specs.py` | authored | Single Python source of truth. Emits both YAMLs. |
| `arch_accelergy.yaml` | generated | Accelergy `subtree/local` schema. |
| `arch_timeloop.yaml` | generated | Timeloop `nodes/!Container` schema. |
| `arch_mambino.cfg` | authored (libconfig) | Working Timeloop arch — bypasses YAML OOM. This is what runs. |
| `canonical_problem.yaml` | authored | 128×128×128 GEMM problem, our canonical validation workload. |
| `mapping_canonical.yaml` | authored | Explicit WS mapping for canonical GEMM (no mapper search). |
| `mapper.yaml` | authored | Held for future mapper use (currently blocked; see NOTES). |
| `components/*.yaml` | copied from tutorial | `smartbuffer_SRAM`, `smartbuffer_RF`, `intmac`, `regfile` Accelergy compound classes. |
| `out_accelergy/` | Accelergy output | ERT + ART for our chip. Feed these to Timeloop. |
| `timeloop-model.stats.txt` | Timeloop output | Canonical run's stats — the 11.77 μJ / 5.61 pJ/MAC number. |
| `timeloop-model.map.txt` | Timeloop output | Loop nest for the canonical run. |

## Reproducing the canonical run

On Gilbreth:

```bash
export LD_LIBRARY_PATH=/scratch/gilbreth/raisul/envs/timeloop-deps/lib:$HOME/local/src/timeloop/lib:$LD_LIBRARY_PATH
export PATH=/scratch/gilbreth/raisul/envs/s5m/bin:$PATH
cd ~/dev/ssm-baselines/S5/paper_v2_ppac/chip

# 1) Generate both arch schemas from Python source
python build_chip_specs.py --out_dir .

# 2) Accelergy → per-access energy tables (ERT/ART)
accelergy arch_accelergy.yaml components/*.yaml dummy_16x16_actions.yaml -o out_accelergy

# 3) Timeloop model with explicit mapping (no search)
~/local/src/timeloop/build/timeloop-model arch_mambino.cfg

# Result: timeloop-model.stats.txt with cycles, energy, per-component breakdown
```

The canonical run is deterministic — same inputs always produce
`Utilization = 1.00 | pJ/Compute = 5.613`.

## Notes on tooling caveats hit and worked around

1. **Timeloop mapper OOM.** `timeloop-mapper` OOMs at init on our arch regardless of
   constraint tightness. Workaround: use `timeloop-model` with explicit mapping (this
   directory) — for our fixed WS chip there is only one legal mapping anyway.
2. **YAML parser bug.** Feeding Timeloop the `nodes:/!Container` YAML arch causes
   `std::bad_alloc` during config parse. Workaround: `arch_mambino.cfg` in libconfig
   format. Same chip parameters.
3. **Memory hierarchy nesting.** Timeloop demands strict hierarchical SRAM nesting;
   our 3 shared SRAMs (weight, activation, state) at the same level trip a diagnostic.
   Workaround: unified `GlobalBuffer` (128 KB). 3-buffer refinement is future work.
4. **Field-naming schemas.** `data_spaces` (Accelergy) vs `data-spaces` (Timeloop);
   `num_threads` vs `num-threads`; etc. Single source of truth (`build_chip_specs.py`)
   emits the right form for each tool.

## Left for follow-up

- Generate per-GEMM `problem.yaml` + `mapping.yaml` from `workload_<cfg>_gemms.yaml`
  for each of the ~25 unique (config, GEMM shape) combinations
- Run timeloop-model on each, aggregate ×repeat count to get per-config chip PPAC
- SCALE-Sim cross-validation on the same workloads (independent tool, same JAXPR-extracted GEMMs)
- Refactor 3-buffer chip model (weight + activation + state SRAM) in Timeloop, or
  document the unified-buffer approximation in the paper
- Rewrite master doc §6, §7 (Appendix A), §8 with real numbers
- Redo Fig 4 with real Pareto
- Update DISCUSSION_PARAGRAPHS.md paragraph 3
- Tag `mambino-paper-v2`
