# Public open-source release plan — Mambino paper (v2, 2026-07-10)

**Status: READY TO EXECUTE.** Updated for v4 chip PPAC pipeline (H1–H4 + F1–F3 + NoC/control).

## Target venue

**IEEE TNNLS** (Transactions on Neural Networks and Learning Systems, Q1, IF ≈10.4).
See [`PAPER_TNNLS_OUTLINE.md`](PAPER_TNNLS_OUTLINE.md) for the paper structure and
submission checklist. This release bundle is the code+data supplement cited by the paper.

## Goal

Publish a curated, minimal, self-contained code bundle alongside the paper so any
reviewer or reader can:

1. Rebuild the exact model architecture from the code (`s5/`, `run_train.py`)
2. Re-extract workloads from a downloaded checkpoint via
   `extract_workload_from_jaxpr.py` and verify the hard invariant passes
3. Regenerate the v4 chip PPAC pipeline end-to-end via
   `paper_v2_ppac/chip/multi_array_ppac_v4.py` and match
   `multi_array_sweep_v4.json` bit-for-bit
4. Cross-validate cycle counts against SCALE-Sim v2 via
   `run_scalesim_multi_array_validation.py`

While NOT exposing:

- Exploratory dead-end code paths / stale sbatch scripts
- Development artifacts (wandb configs, cache dirs, dev-only checkpoints)
- Companion analog-state-cell IP (deferred to companion paper)
- Cluster-specific paths and account-scoped configs

## Two-tag strategy

| Tag | Repo state | Purpose |
|---|---|---|
| **`mambino-paper-v1`** (already exists) | Full research repo, all history | Internal snapshot; for collaborators + auditors |
| **`mambino-paper-v2-public`** (TO CREATE) | Curated subset, dedicated branch | The public code URL cited in the paper |

The public tag lives on a separate branch (`mambino-paper-v2-public-branch`) so that
the file-removal history is auditable — anyone can `git log mambino-paper-v1..mambino-paper-v2-public` and see exactly what was excluded and why.

## Files to KEEP in the public release

### Training code (`s5/`)

Keep:
- `s5/ssm.py` (Pure S5 SSM implementation)
- `s5/mambino_ssm.py` (Mambino architecture)
- `s5/layers.py` (SequenceLayer with half_glu2 + rank-r gate)
- `s5/seq_model.py` (BatchClassificationModel)
- `s5/ssm_init.py` (HiPPO init)
- `s5/train_helpers.py`
- `s5/dataloading.py`
- `s5/dataloaders/__init__.py` + `s5/dataloaders/lra.py` (for LRA-ListOps)
- `s5/utils/util.py` (str2bool etc.)
- `run_train.py`

Remove:
- `s5/dataloaders/audio.py` if present (not used for our results)
- Any exploratory files with `_v0[3-6]` or `_dspc_*` prefixes if they exist

### Sbatch scripts (`bin/`)

Keep only the 5 sbatch scripts backing the paper's 5 configs plus the λ_pc sweep:
- `bin/gilbreth_chip_mambino_gelu.sh` (Config 4)
- `bin/train_pure_s5_gelu_p16.sh` (Config 5)
- `bin/gilbreth_chip_pure_s5.sh` (Corner 1)
- `bin/gilbreth_chip_pure_s5_p16_r40.sh` (Corner 2)
- `bin/gilbreth_chip_mambino_iso.sh` (Corner 3′)
- `bin/gilbreth_lambda_pc_sweep.sh` (§5f ablation)

Remove:
- All other sbatch scripts in `bin/` (exploration, dev, cluster tuning, etc.)

### Workload extraction (root)

Keep:
- `flop_counter_jaxpr.py`
- `extract_workload_from_jaxpr.py`

### v2 chip PPAC pipeline (`paper_v2_ppac/`) — v4 authenticated

Keep from `paper_v2_ppac/`:

**Workloads** (JAXPR-verified, hard-invariant-checked):
- `paper_v2_ppac/workloads/workload_{cfg}_gemms.yaml` × 5 configs
- `paper_v2_ppac/workloads/workload_{cfg}_elemwise.yaml` × 5
- `paper_v2_ppac/workloads/workload_{cfg}_manifest.yaml` × 5 (provenance + hard invariant)

**v4 simulator (the paper's ground truth)**:
- `paper_v2_ppac/chip/multi_array_ppac_v4.py` — direct-instrumentation simulator
  with H1–H4 + F1–F3 + NoC/control (the code cited by the paper)
- `paper_v2_ppac/chip/multi_array_sweep_v4.json` — canonical output (bit-for-bit
  reproducible)
- `paper_v2_ppac/chip/build_chip_specs.py` — chip spec generator
- `paper_v2_ppac/chip/gen_ert_all_variants.sh` — Accelergy driver (3 SRAM tiers ×
  4 array sizes = 12 ERT runs)

**Per-config ERTs** (Accelergy + CACTI + NeuroSim outputs):
- `paper_v2_ppac/chip/out_accelergy_320KB_{8,16,32,64}x{8,16,32,64}/` (Config 4/5)
- `paper_v2_ppac/chip/out_accelergy_384KB_*/` (Corner 2/3′)
- `paper_v2_ppac/chip/out_accelergy_512KB_*/` (Corner 1)
- `paper_v2_ppac/chip/arch_accelergy_{tier}_{size}.yaml` (12 spec YAMLs)
- `paper_v2_ppac/chip/components/*.yaml` — Accelergy compound classes

**SCALE-Sim v2 cross-validation**:
- `paper_v2_ppac/chip/run_scalesim_multi_array_validation.py` — script
- `paper_v2_ppac/chip/scalesim_multi_array_validation/validation_summary.csv` —
  10-shape B/A-stationary agreement table

**Docs**:
- `paper_v2_ppac/PAPER_TNNLS_OUTLINE.md` — paper structure + submission checklist
- `paper_v2_ppac/README.md` — pipeline overview
- `paper_v2_ppac/chip/README.md` — v4 simulator quick-start
- `paper_v2_ppac/chip/pipeline_diagram.html` — methods figure

**Remove from paper_v2_ppac/chip/** (superseded intermediate work):
- `direct_ppac.py` — v3 single-array simulator (superseded by v4 multi-array)
- `run_all_configs_ppac.py` — v2 Timeloop-model driver (never authenticated,
  Fable found dataflow bug)
- `run_scalesim_sweep.py`, `run_scalesim_crosscheck.py` — v2 SCALE-Sim on
  16×16 only (superseded by multi-array validation)
- `per_gemm_ppac.json`, `per_config_ppac.json`, `per_gemm_ppac_v3.json`,
  `per_config_ppac_v3.json`, `sweep_summary.csv`, `sweep_summary_v3.csv`,
  `ppac_all_methods.csv`, `multi_array_sweep.json` — v2/v3 intermediate outputs
- `arch_timeloop_{8,16,32,64}x*.yaml` — v3 Timeloop specs (superseded by
  arch_accelergy per-tier per-size YAMLs)
- `gen_fig_csvs.py`, `gen_fig_csvs_v3.py` — v2/v3 figure exporters (replace
  with a single v4 figure exporter when the paper's figures are drafted)
- `scalesim_sweep/`, `scalesim_crosscheck/`, `scalesim_config_*x*.cfg`,
  `scalesim_sweep_*.csv`, `scalesim_gemms.csv`, `scalesim_vs_timeloop_16x16.csv` —
  v2 SCALE-Sim intermediate

### Paper docs (`accelergy/`)

Keep:
- `accelergy/LOCKED_master_ppac_document.md` (v8 when finalized; v7.1 as historical)
- `accelergy/DATA_AVAILABILITY_STATEMENT.md`
- `accelergy/DISCUSSION_PARAGRAPHS.md`
- `accelergy/mambino_environment.md`
- `accelergy/mambino_pip_freeze.txt`
- `accelergy/mambino_jax_env.txt`
- `accelergy/primitive_component_libs/mambino_primitives.lib.yaml`

Remove:
- `accelergy/LOCKED_digital_ppac_results.md` (v7.1 superseded — subsumed by master doc)
- `accelergy/LOCKED_mixed_signal_reference.md` (v7.1 superseded)
- `accelergy/v4_*.yaml`, `v5_*.yaml` (early Accelergy work, replaced by v6)
- `accelergy/v6_*.yaml` (v7.1 hand-authored action counts — superseded by v2 pipeline;
  keep as tarball attached to PURR deposit for historical audit)
- `accelergy/STUDENT_STATE_CELL_MEMO.md` + `STUDENT_STATE_CELL_FOLLOWUP.md` (companion
  analog state-cell paper IP)

### Figure data (`paper/figure_data/`)

Keep the accuracy CSVs (fig3_*, fig7_lambda_pc_ablation.csv, fig_external_sota.csv).
Remove or clearly deprecate the chip data CSVs (fig4_acc_per_mJ.csv,
fig5_energy_breakdown.csv) — those had v7.1 numbers. Replace with v2 numbers when
Phase 5 completes.

### Top-level

Keep:
- `LICENSE`
- `README.md` (replace with the new `PAPER_README.md` — landing page for reviewers)
- `requirements_gpu.txt`, `requirements_cpu.txt`, `setup.py`
- `CITATION.cff` (add if not present, with the paper citation)

Remove:
- Any legacy exploratory scripts at the top level (`chip_ppac*.py`, `chip_sweep.py`,
  `count_workload.py`, `test_flop_walker_sanity.py`, various CSVs)
- `.git` internal dev artifacts (dotfiles that shouldn't ship)
- Anything with cluster-specific paths hardcoded

## What to EXCLUDE — summarized

1. Exploratory sbatch scripts (`bin/*` other than the 6 listed above)
2. `chip_ppac*.py` (v7.1's superseded methodology; keep the master doc's historical
   reference to it, delete the code)
3. `accelergy/v4_*`, `v5_*`, `v6_*` YAMLs (v7.1's action count model — superseded)
4. `STUDENT_STATE_CELL_*` (companion paper IP)
5. Cluster-scoped configs (wandb, cache dirs, scratch paths)
6. Dev-only smoke scripts, sanity tests, one-off notebooks

## Post-cleanup structure (target)

```
mambino-paper-public/
├── LICENSE                                (MIT or Apache-2.0)
├── PAPER_README.md                        (landing page: install → reproduce)
├── CITATION.cff                           (BibTeX for the paper)
├── setup.py
├── requirements_gpu.txt
├── requirements_cpu.txt
│
├── s5/                                    (training code)
│   ├── ssm.py
│   ├── mambino_ssm.py
│   ├── layers.py
│   ├── seq_model.py
│   ├── ssm_init.py
│   ├── train_helpers.py
│   ├── dataloading.py
│   ├── dataloaders/
│   └── utils/
│
├── run_train.py                           (training entry point)
├── flop_counter_jaxpr.py                  (JAXPR walker)
├── extract_workload_from_jaxpr.py         (workload extractor with hard invariant)
│
├── bin/                                   (6 sbatch scripts, one per config)
│
├── paper_v2_ppac/                         (v2 chip PPAC pipeline)
│   ├── README.md
│   ├── workloads/                         (20 files: 4 per config × 5 configs)
│   └── chip/
│       ├── README.md
│       ├── pipeline_diagram.html          (methods supplementary)
│       ├── build_chip_specs.py
│       ├── arch_mambino.cfg
│       ├── arch_accelergy.yaml
│       ├── arch_timeloop.yaml
│       ├── canonical_problem.yaml
│       ├── mapping_canonical.yaml
│       ├── mapper.yaml
│       ├── components/
│       ├── out_accelergy/                 (reference ERT/ART)
│       ├── timeloop-model.stats.txt       (reference canonical output)
│       └── timeloop-model.map.txt
│
├── accelergy/                             (paper docs + env freeze)
│   ├── LOCKED_master_ppac_document.md
│   ├── DATA_AVAILABILITY_STATEMENT.md
│   ├── DISCUSSION_PARAGRAPHS.md
│   ├── mambino_environment.md
│   ├── mambino_pip_freeze.txt
│   ├── mambino_jax_env.txt
│   └── primitive_component_libs/mambino_primitives.lib.yaml
│
└── paper/                                 (figure data + source SVG)
    ├── figure_data/
    │   ├── fig3_accuracy_per_seed.csv
    │   ├── fig3_accuracy_aggregate.csv
    │   ├── fig3_pairwise_ttests.csv
    │   ├── fig7_lambda_pc_ablation.csv
    │   ├── fig_external_sota.csv
    │   ├── fig4_acc_per_mJ.csv            (v2 numbers, from Phase 5)
    │   ├── fig5_energy_breakdown.csv      (v2 numbers, from Phase 5)
    │   └── README.md
    ├── figures_1_and_2.html               (Fig 1 architecture, Fig 2 chip)
    ├── figure_pure_s5.html
    └── fig3_draft.html
```

## Execution steps (once approved)

```bash
# 1. Create clean public branch from mambino-ssm
git checkout mambino-ssm
git checkout -b mambino-paper-v2-public-branch

# 2. Delete excluded files in clean commits (one per category)
git rm bin/mambino_bidir_*.sh bin/mambino_dspc_*.sh ...
git commit -m "public prep: remove exploratory sbatch scripts"

git rm accelergy/v[456]_*.yaml
git commit -m "public prep: remove v7.1 hand-authored Accelergy action counts (superseded by v2)"

git rm chip_ppac.py chip_ppac_v[234]_*.csv chip_sweep.py count_workload.py test_flop_walker_sanity.py flop_counter.py
git commit -m "public prep: remove superseded v7.1 chip-PPAC scripts"

git rm accelergy/STUDENT_STATE_CELL_*.md accelergy/LOCKED_digital_ppac_results.md accelergy/LOCKED_mixed_signal_reference.md
git commit -m "public prep: exclude companion state-cell IP + superseded LOCKED docs"

# 3. Write top-level PAPER_README.md
git add PAPER_README.md
git commit -m "public prep: add landing README with reproduce instructions"

# 4. Push branch, tag
git push origin mambino-paper-v2-public-branch
git tag -a mambino-paper-v2-public -m "Public release: minimal reproducible bundle for the Mambino paper"
git push origin mambino-paper-v2-public
```

## Approval gate

Reviewer (you): read this document, and either
- Approve the list as-is; or
- Mark specific files/directories with KEEP / REMOVE / MOVE annotations; or
- Add categories to reconsider.

Once approved, the execution steps above run in ~10 minutes.
