# `ppac_detailed_4corners.csv` — detailed chip PPAC, 4 configurations

Regenerate with `python paper_v2_ppac/make_ppac_detailed_csv.py`. Every value is
produced by **executing** the PPAC model, not transcribed. Writes to this folder
and to the Overleaf `figure_data_v1/`.

Five rows: the four configurations, plus a second row for Mambino-G under the
concurrent-predictor schedule.

## The two power columns are NOT on the same basis

This is the single thing to get right when plotting.

| column | basis | safe to compare against |
|---|---|---|
| `avg_mW` | `energy_full / latency` — **the paper's convention** | anything |
| `peak_datapath_mW` | GEMM arrays **only** | other `peak_datapath_mW` values |
| `peak_chip_lo_mW` / `peak_chip_hi_mW` | complete-chip band | `avg_mW` |

`peak_datapath_mW` is **below** `avg_mW` on four of five rows. That is not an
error and not physically impossible — the v5 peak timeline covers the GEMM
arrays only, while elementwise, state-SRAM and leakage power (49–57% of total
energy) sit off the timeline (`multi_array_ppac_v5.py:19-25`). The complete-chip
band brackets it as `[array_peak, array_peak + background_power]`.

**Never divide `peak_datapath_mW` by another config's `avg_mW`, and never mix the
two in one chart.** The superseded `fig3_ppac.csv` / `fig4_efficiency.csv` export
`peak_mW` and `acc_per_W_peak` as their only power columns; drawing IPW from
`acc_per_W_peak` **inverts the Mambino-0 vs Mambino-G ranking** relative to the
paper's `tab:threeaxis` (1.1937 < 1.2052 there, 0.701 > 0.678 in the table).
Use `IPW_acc_per_W_avg` from this file instead. Paper `CLAUDE.md` §3 is explicit:
*"Never write 'peak power'."*

## Column notes

- `throughput_ips` — pipelined steady state, `1 GHz / max(per-block cycles)`.
  **Not** `1000/latency`. The two differ because blocks pipeline across chunks.
- `acc` — `test@peakval`, mean over the 8 matched seeds, re-derived from the
  Gilbreth run logs. The generator asserts the model's baked-in accuracy matches
  the log-derived value, so a drift in either fails loudly.
- `acc_per_mJ` — latency-free by construction (`acc / energy`). A config does not
  lead this axis by being fast; it leads by spending less energy.
- `IPW_acc_per_W_avg` — `acc / (avg_mW/1000)`, average-power basis.
- Concurrent row — `PEs` (+512) and `throughput_ips` are derived, not returned by
  `barrier_point`; the dedicated second 8×64 array collapses the
  `8x2048x128_complex_real` block's rep by ×16/24, after which
  `128x2048x16_complex_complex` becomes the bottleneck at 146,048 cycles.
  The collapse **is** paid for (+512 PEs, +1.843 mm², +364.6 mW peak).

## `honest=True` is load-bearing

The generator passes `honest=True` throughout. This disables the v4
`wall_clock_rep` 24→16 predictor-overlap freebie, which grants free throughput on
an array already at 100% utilization. Do not regenerate any row with
`honest=False`.

## Values are full precision

Rounding happens at the `.tex` layer, not here. Note `tab:ppac`'s S5-0 power cell
(1,784 mW) was computed from *displayed rounded* energy and latency; the model
gives **1,787.7 mW**, which is what this file carries.
