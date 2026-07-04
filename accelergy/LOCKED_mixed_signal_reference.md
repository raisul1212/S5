# MIXED-SIGNAL PPAC -- NEUROSIM PIM CROSSBAR REFERENCE (2026-07-03)

## !!! CRITICAL FRAMING !!!

This IS NOT the paper's actual substrate. The paper's substrate is a TRUE
ANALOG RC STATE CELL -- a capacitor-based analog integrator with on-chip
temporal dynamics. That's a fundamentally different topology from an analog
crossbar; state-cell PPAC requires SPICE + measured characterization from the
circuit team (see STUDENT_STATE_CELL_MEMO.md).

The numbers below model a GENERIC 22nm ANALOG CROSSBAR CHIP (RRAM current-
summing PIM array) using NeuroSim primitives. Interpret them as a
peer-reviewed reference point of "what a mainstream analog crossbar chip would
look like on the same workload."

## Config 4 Mambino gelu 106K -- Mixed-signal PIM Reference

Methodology: Accelergy 0.4 + NeuroSim PIM plugin (Chen, Peng, Yu IEDM 2018)
+ CACTI SRAM (Wilton/Jouppi/Balasubramonian TACO 2017).
Crossbar: 128x128 tiles, all cols active, cell = nvmexplorer_RRAM reference.

### Energy per inference (pJ)

| Component | Energy | % of total |
|---|---:|---:|
| pim_row_drivers | 115,526 | <0.01% |
| pim_col_drivers | 4 | <0.01% |
| pim_cells | 209,714 | 0.01% |
| pim_adc | 1,203,499 | 0.08% |
| activation_sram (CACTI) | 1,207,681,139 | **76.6%** |
| state_sram (CACTI) | 339,090,604 | 21.5% |
| acc_adder | 45,523 | <0.01% |
| reg_accum | 27,852,777 | 1.77% |
| **TOTAL** | **1,576,198,789 pJ = 1.58 mJ** | 100% |

### Area (μm²)

| Component | Area |
|---|---:|
| pim_row_drivers | 219 |
| pim_col_drivers | 202 |
| pim_cells (one tile) | 0.03 |
| pim_adc | 401,079 |
| activation_sram | 4,353,200 |
| state_sram | 1,281,930 |
| acc_adder | 22 |
| reg_accum | 40 |
| **TOTAL** | **6,036,712 μm² = 6.04 mm²** |

## Digital vs Mixed-signal PIM (Config 4 Mambino)

| Metric | Digital (worst-case 1r/MAC) | Mixed-signal PIM (batched) | Ratio |
|---|---:|---:|---:|
| Energy | 167.26 mJ | 1.58 mJ | **106× lower** |
| Area | 5.72 mm² | 6.04 mm² | +5.6% larger |

**Note on the 106× energy ratio:** Most of this comes from (a) weights stored
on-array (zero per-MAC weight SRAM fetch), (b) analog current-summing being
~13,000× cheaper than digital MAC per operation, and (c) the digital model
using worst-case 1-read-per-MAC. If digital used line-batched reads (128 MACs
per SRAM line access), digital would drop to ~1.3 mJ and the crossbar
advantage would be ~1.2× not 106×.

## Position for paper

The Config 4 Mambino architecture achieves competitive digital and analog
PPAC on the same workload. Analog PIM crossbar reference gives a lower bound
of ~1.58 mJ per inference at ~6 mm² area. The paper's true analog RC
state-cell substrate is a distinct chip topology whose PPAC will be published
separately once the student's SPICE + measured data are integrated.
