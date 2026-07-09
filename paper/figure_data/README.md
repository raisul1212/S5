# Mambino paper — figure data (CSVs)

Every number in these CSVs traces back to
[`../../accelergy/LOCKED_master_ppac_document.md`](../../accelergy/LOCKED_master_ppac_document.md)
and the Accelergy output YAMLs (`../../accelergy/v6_out_digital_*/energy_estimation.yaml`).
Generated at git tag `mambino-paper-v1`.

> ## ⚠ CHIP-DATA CSVs ARE SUPERSEDED
>
> The following CSVs contain chip PPAC numbers that are **under revision** — do not use
> in the paper submission:
> - `fig4_acc_per_mJ.csv` (energy_mJ column and acc_per_mJ columns)
> - `fig5_energy_breakdown.csv` (per-component energy)
>
> The chip PPAC pipeline is being redone with SCALE-Sim v2 + Timeloop + Accelergy for a
> defined edge inference architecture (64×64 systolic, weight-stationary, INT8;
> 256/128/64 KB on-chip buffers, no off-chip DRAM). New CSVs will land at git tag
> `mambino-paper-v2`.
>
> **Accuracy CSVs (`fig3_*.csv`, `fig7_lambda_pc_ablation.csv`, `fig_external_sota.csv`)
> are UNAFFECTED and remain valid.**

## Files

| File | Rows | Purpose |
|---|---:|---|
| [`fig3_accuracy_per_seed.csv`](fig3_accuracy_per_seed.csv) | 40 | raw n=8 per-seed test@peakval + test_max, one row per (config, seed) — 5 configs × 8 seeds. Backs §5d exactly. |
| [`fig3_accuracy_aggregate.csv`](fig3_accuracy_aggregate.csv) | 5 | aggregate stats per config: mean, std, sem, 95% CI half-width, min, median, max — for both test@peakval and test_max. Backs §5a. |
| [`fig3_pairwise_ttests.csv`](fig3_pairwise_ttests.csv) | 4 | paired-t significance annotations for Fig 3 bar overlays. Backs §5b. |
| [`fig4_acc_per_mJ.csv`](fig4_acc_per_mJ.csv) | 10 | chip-efficiency Pareto — (energy_mJ, mean_acc) for each config × topology. 5 configs × {pipelined, sequential}. Backs §8a and §8b. |
| [`fig5_energy_breakdown.csv`](fig5_energy_breakdown.csv) | 10 | per-component energy in mJ from Accelergy YAMLs: weight_sram, activation_sram, state_sram, mac_array, reg_accum, total. 5 configs × 2 topologies. |
| [`fig7_lambda_pc_ablation.csv`](fig7_lambda_pc_ablation.csv) | 5 | λ_pc single-seed sweep on Config 4 seed=42. Backs §5f. |
| [`fig_external_sota.csv`](fig_external_sota.csv) | 3 | S7 63.77%, Mega 63.14%, S5 62.15% — external SOTA reference lines / dashed markers on accuracy plots. Backs §5g. |

## Suggested Fig 3 layout using these CSVs

Two-panel bar chart:

```python
import pandas as pd
agg = pd.read_csv("fig3_accuracy_aggregate.csv")
seed = pd.read_csv("fig3_accuracy_per_seed.csv")
ttests = pd.read_csv("fig3_pairwise_ttests.csv")

# Panel A (106K): Config 4 vs Config 5
# Panel B (188K): Corner 3p vs Corner 1 vs Corner 2

# Bar height = mean_peakval, error bar = sem_peakval (or ci95_hw_peakval)
# Overlay per-seed points from seed
# Annotate pairs from ttests with marker column
```

## Suggested Fig 4 layout

Scatter (energy_mJ, mean_test_peakval), two panels (pipelined, sequential).
Config 4 should sit alone on the Pareto frontier in both.

```python
import pandas as pd
df = pd.read_csv("fig4_acc_per_mJ.csv")
pipe = df[df.topology == "pipelined"]
seq  = df[df.topology == "sequential"]
```

## Suggested Fig 5 layout

Stacked bar per config — 5 stacks (one per component). Pick topology per subplot.

## Suggested Fig 7 layout

Line + points: log10(λ_pc + 1e-6) on x, test@peakval on y. Or bar with baseline
dashed reference at λ_pc=0.

## Units and conventions

- **Accuracies** are fractional (0.6138), not percentages. Multiply by 100 for %.
- **Energies** are in mJ per single LRA-ListOps L=2048 inference at INT8, 22 nm, 1 GHz.
- **Areas** are in mm².
- **acc_per_mJ_x1e3** column is `(mean accuracy) / (energy in mJ) × 1000` — matches §8 tables.
- **Params** column is trainable-parameter count from `count_params` on the JAX model,
  as reported at run-log line "Trainable Parameters: N".
- **Seed order** in per-seed rows: `[6554595, 42, 12345, 271828, 314159, 1, 2, 3]`.
