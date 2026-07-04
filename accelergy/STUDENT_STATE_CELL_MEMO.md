# Memo: State-Cell Characterization for Paper-Defensible Mixed-Signal PPAC

**To:** [student]
**From:** Raisul Islam
**Date:** 2026-07-03 (rev 2)
**Re:** SPICE + measured data needed to replace the NeuroSim RRAM reference
with our actual analog RC state-cell numbers

## 1. Context

The Mambino paper reports two PPAC surfaces at 22nm:

1. **Digital PPAC** — LOCKED (Accelergy 0.4 + CACTI + NeuroSim intadder).
   Reports for pipelined + sequential chip topologies.
2. **Mixed-signal PPAC** — currently modeled as a **generic RRAM crossbar
   reference** (NeuroSim PIM plugin, `nvmexplorer_RRAM.cell`). Reports at
   both chip topologies. **This is NOT our chip.**

**Our chip is a true analog RC state-cell** — a capacitor-based analog
integrator array with on-chip temporal dynamics, NOT a current-summing
crossbar. The mixed-signal reference above is a bystander comparison, not a
claim about our substrate.

**One sentence for reviewers:** the analog RC state cell will beat RRAM
crossbar on write energy (no filament switching) and on retention-vs-refresh-
overhead when tuned for μs-scale RC — write energy comparison is the primary
axis where our substrate wins, so make sure the numbers below let us produce
a defensible write-energy plot against nvmexplorer_RRAM.cell's 0.6 pJ.

To publish paper-defensible mixed-signal PPAC for our actual substrate, we
need SPICE-verified and measured characterization of the state cell. This
memo lists exactly what.

## 2. What we need — the minimum viable set

For each entry, give: (a) the number, (b) how you got it (SPICE / measured /
scaled), (c) 22nm-relevance (native tech, or Dennard/reduced-Vdd scaling
from another node).

### 2a. Cell electrical characterization

| Parameter | Symbol | Target range (guess) | Priority |
|---|---|---|---|
| Storage capacitance | C_state | 1-10 fF | HIGH |
| **RC time constant** | **τ = R·C** | **10 μs - 1 ms** | **HIGH** |
| Read-path transconductance | g_m,read | 10-100 μS | HIGH |
| Retention time @ 1% signal loss | T_ret | **≥ 10 μs min, ≥ 100 μs target** | HIGH |
| Refresh period target | T_refresh | 5 × T_ret | HIGH |
| Read noise (input-referred RMS) | v_n,read | ≤ **2 mV** (= LSB/4 at INT8 ±1V) | HIGH |
| Write settling time (target 0.1% error) | t_wr | 1-10 ns | HIGH |
| Read access time (small-signal) | t_rd | 1-5 ns | HIGH |
| **Integrator INL over ±1 V input** | **INL(V_in)** | **≤ 1% FSR full range** | **HIGH** |
| **Row wiring parasitic cap (per unit length)** | **C_par_row** | **0.1-1 fF/μm** | **HIGH** |
| **Column wiring parasitic cap (per unit length)** | **C_par_col** | **0.1-1 fF/μm** | **HIGH** |
| **Array-scale parasitic total (P × L × 8 × 3 cells)** | **C_par_total** | **computed** | **HIGH** |
| DC leakage per cell | I_leak | 1 pA - 1 nA | MED |
| PVT sensitivity of T_ret across TT/SS/FF, -40/25/85/125 °C | ΔT_ret/T_ret | ± range | MED |

**Why the tightened T_ret / v_n_read numbers:** one LRA-ListOps scan at
L=2048 tokens × 1 GHz = 2 μs per layer. Bidirectional adds 2×, 8 layers
sequenced adds 8×. So worst-case retention need before refresh is ~30 μs.
Anything under 10 μs forces refresh every scan → refresh energy dominates.
For read noise: INT8 ±1V = LSB ≈ 8 mV. Noise must be < LSB/4 = 2 mV for
< 1% ENOB loss.

**Why INL matters:** SSM dynamics assume linear time-invariant (LTI) state
transition. If g_m compresses at rails or leakage is voltage-dependent,
Lambda / A_bar mismatches training and the whole recurrent chain drifts.
INL(V_in) curve tells us the LTI-valid input range.

**Why wiring parasitics matter:** at P=8, L=2048, 8 layers, 3 trajectories
= 393K cells per array. A cell holding 1 fF of storage cap next to 100+ pF
of wiring parasitic is dominated by wiring — the RC dynamics get set by the
routing, not the cell. Ask for both single-cell and array-total.

### 2b. Cell energy characterization (SPICE, transient sim at 22nm)

| Parameter | Symbol | Target range | Priority |
|---|---|---|---|
| Energy per write (full-swing @ Vdd) | E_wr | 1-100 fJ | HIGH |
| Energy per read (integrated over t_rd) | E_rd | 0.1-10 fJ | HIGH |
| Energy per refresh cycle | E_refresh | 1-10 fJ | HIGH |
| Static leakage power | P_leak | 0.1-100 fW/cell | HIGH |
| Energy per row activation (full row) | E_row | scales as rows × E_wr | MED |
| Energy per column integration | E_col | scales as cols × E_rd | MED |

### 2c. Cell physical/area characterization

| Parameter | Symbol | Target | Priority |
|---|---|---|---|
| Cell area at 22nm layout | A_cell | 10-500 μm² | HIGH |
| Cell aspect ratio (row/col) | AR | 1-4 | MED |
| Peripheral overhead (drivers, refresh) per K-cell array | A_periph | HIGH |
| Interconnect density limit (cells per mm² incl. routing) | rho | HIGH |

### 2d. Peripheral analog blocks (SPICE)

| Block | Params needed |
|---|---|
| Refresh driver (per row) | Energy per refresh, area |
| Charge amplifier / column readout | Bandwidth, energy per read, area |
| ADC at column output | Resolution used, energy/sample, area (or reuse Murmann-based estimator) |
| DAC for input (if voltage inputs) | Resolution, energy/sample, area |

### 2e. Failure modes and yield

| Item | What we need |
|---|---|
| Retention failure vs T_ret distribution | Cell-level Monte Carlo (100+ samples) |
| Row-write failure vs t_wr | Timing margin analysis |
| Signal drift over 1 s (worst case) | Long-transient SPICE |
| **Cell mismatch histogram (write→read consistency)** | **Mismatch Monte Carlo — PROMOTED TO P2, see §5** |
| Yield vs cell area | Model like Wong Sci Rep 2015 |

### 2f. Two-topology array rollups (feeds §7 of master PPAC doc directly)

The chip PPAC is reported at two topologies (see main paper §5). For each,
give the total cells, total leakage power, and refresh energy per full scan
so both columns land cleanly. Compute from your single-cell data — no new
SPICE needed:

| Rollup | Formula | Priority |
|---|---|---|
| Pipelined-array cell count | local_P × L × n_layers × n_traj = 16 × 2048 × 8 × 3 = 786,432 (Config 4) | HIGH |
| Pipelined-array leakage power | total_cells × P_leak | HIGH |
| Pipelined-array refresh energy per full scan | total_cells × E_refresh × (scan_time / T_refresh) | HIGH |
| Sequential-array cell count | local_P × 1 layer × n_traj = 16 × 1 × 3 = 48 (Config 4) | HIGH |
| Sequential-array leakage power | as above with sequential cell count | HIGH |
| Sequential-array refresh energy per full scan | as above (dominated by re-load traffic) | HIGH |

**Numbers to plug in for each Mambino config:**
- Config 4: local_P=16, n_traj=3 → pipelined 786,432 cells, sequential 48 cells
- Corner 3': local_P=16, n_traj=3 → pipelined 786,432 cells, sequential 48 cells

Without §2f, your data lands in only one column of the paper's PPAC table.

## 3. Format we need it in

**Two artifacts:**

### 3a. NVMExplorer-style `.cell` file (drop-in for NeuroSim)

Format:

```
-MemCellType: analog_rc_state
-CellArea (F^2): [number, F = 22nm feature size]
-CellAspectRatio: [number]
-AccessType: CMOS
-AccessCMOSWidth (F): [number]
-ReadMode: [current | voltage | charge]
-ReadVoltage (V): [number]
-ReadPower (uW): [number]
-ReadEnergy (fJ): [number, this is the KEY number]
-WriteMode: [voltage | current]
-WriteVoltage (V): [number]
-WritePulse (ns): [number]
-WriteEnergy (fJ): [number, KEY]
-RefreshMode: [periodic]
-RefreshPeriod (ns): [number]
-RefreshEnergy (fJ): [number, KEY]
-LeakageCurrent (pA): [number]
-Capacitance (fF): [number, storage cap]
-RetentionTime (ns): [number to 1% signal loss]
```

We drop this into `~/accelergy_setup/accelergy-neurosim-plug-in/cells/`
and reference it from the arch YAML as `cell_config: 'our_state_cell'`.

### 3b. CSV of full sweep (for uncertainty bands in the paper)

One row per (voltage, temperature, cell_area) point. Columns: E_wr, E_rd,
E_refresh, T_ret, A_cell, source (SPICE/measured), notes. Aim for ~30 rows
covering:

- Vdd: 0.6, 0.7, 0.8, 0.9 V
- Temperature: **-40, 25, 85, 125 °C** (aligned with §2a PVT list)
- Cell area: 3-4 different sizings

We use this for Pareto envelopes and worst-case reporting.

## 4. Where to run this

**SPICE:** BSIM4 22nm PTM models. Or if we have a specific 22nm PDK access,
use that. Cadence Spectre or ngspice both work. Use TT/SS/FF corners.

**Layout / area:** Cadence Virtuoso if we have PDK access; otherwise
Klayout with a 22nm process design kit. If neither, estimate from
Ni VLSI 2019 gain-cell area with cap-size scaling.

**Measured (nice-to-have):** if any test chip data exists, cross-check the
SPICE numbers against it.

## 5. Timeline ask

**Priority 1 (this week if possible):** the .cell drop-in (§3a) + the two
2f array rollups (§2f). Just the single-point estimate at nominal Vdd/T/cell
size. This unblocks a paper-defensible mixed-signal PPAC line for both
topology columns in the paper's main table.

**Priority 2 (next 2 weeks):**
- The CSV sweep (§3b). Uncertainty envelopes for the supplementary figure.
- **The cell mismatch Monte Carlo histogram (§2e).** *Promoted from P3.*
  Rationale: our noise-sweep story (see paper §6) assumes σ_read from
  intrinsic noise only. If mismatch inflates the effective σ_read by 2-3×,
  the whole downstream noise-tolerance narrative changes. We need this data
  *before* submission crunch, not during.

**Priority 3 (before submission):** any additional Monte Carlo / yield /
refresh model that lets us defend against reviewer "what about drift,
retention, mismatch" questions beyond the P2 items.

## 6. What we'll do with it

Once we have §3a + §2f, we drop the `.cell` file into NeuroSim's cell
library, change the arch YAML from `cell_config: 'nvmexplorer_RRAM'` to
`cell_config: 'our_state_cell'`, plug §2f rollups into the Accelergy state
SRAM component energy, re-run Accelergy for both pipelined and sequential
topologies, and get paper-defensible energy/area/latency for our substrate
on the LRA-ListOps workload.

This becomes the paper's headline mixed-signal number: the analog-RC
state-cell PPAC on a real reasoning workload, showing whatever competitive
picture the numbers actually support, at both throughput-optimized and
latency-optimized chip variants.

## 7. Non-goals

- We do NOT need to characterize the full digital periphery (adders,
  registers, ADC, SRAM) — Accelergy/NeuroSim/CACTI cover those with published
  models.
- We do NOT need training PPAC — the chip is inference-only.
- We do NOT need cross-technology-node data — 22nm is the target.

## 8. Contact / questions

Ping me for: how to format the .cell file, help running Accelergy, questions
about how the number will be used in the paper.

---

**Paper section this feeds:** §5 "Chip-level performance" — mixed-signal
substrate table (both pipelined + sequential columns). The digital PPAC
numbers are already locked ([`LOCKED_master_ppac_document.md`](LOCKED_master_ppac_document.md)).
