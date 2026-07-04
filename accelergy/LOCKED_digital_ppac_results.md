# LOCKED DIGITAL PPAC RESULTS (2026-07-03)

Two independently-derived methodologies both find **Mambino wins ~8-10% over Pure S5 at
iso-params 106K**. Use these as the paper's digital PPAC numbers.

## Methodology 1 — Accelergy v0.4 (CACTI + NeuroSim)

- SRAM (all 3): CACTI plugin (Wu, Sze, Emer RSP 2019 — Accelergy framework)
- MAC array: NeuroSim intadder as INT8 MAC approximation
- Register file: NeuroSim flip_flop, n_bits=32
- Technology: 22 nm, 1 GHz clock

| Component | Config 4 Mambino | Config 5 Pure S5 | Ratio |
|---|---:|---:|---:|
| weight_sram | 12.31 mJ | 13.45 mJ | 91.5% |
| activation_sram | 154.58 mJ | 168.71 mJ | 91.6% |
| state_sram | 0.34 mJ | 0.96 mJ | 35.5% |
| mac_array | 2.93 μJ | 3.20 μJ | 91.6% |
| reg_accum | 27.85 μJ | 30.43 μJ | 91.5% |
| **Energy TOTAL** | **167.26 mJ** | **183.14 mJ** | **91.3%** |
| **Area TOTAL** | **5.72 mm²** | **6.38 mm²** | **89.7%** |

**Note on absolute energy scale.** These action counts model 1 SRAM read per MAC
(worst-case, no line-buffering). A real chip would read a 128-byte line and reuse
across 128 MACs, dropping absolute energy by ~100×. The RATIO Mambino/Pure S5 is
invariant to this batching factor -- both configs would drop by the same factor.

## Methodology 2 — chip_ppac.py v4 (Horowitz + Chen + Wong)

- INT8 MAC (22nm): Horowitz ISSCC 2014 scaled → 0.4 pJ/MAC
- SRAM 8-bit read (22nm): Chen Eyeriss ISSCC 2016 → 0.05 pJ
- SRAM cell area (22nm 6T): Wong Sci Rep 2015 → 0.081 μm²/bit
- FLOP counts: JAXPR-verified (399M/682M/745M for Configs 3/4/5)

| Metric | Config 4 Mambino | Config 5 Pure S5 | Ratio |
|---|---:|---:|---:|
| Energy | 450.7 nJ | 492.8 nJ | 91.5% |
| Latency | 333.1 μs | 363.9 μs | 91.5% |
| Area | 4.81 mm² | 5.22 mm² | 92.1% |

## Cross-methodology agreement

Both frameworks report the same Mambino/Pure S5 ratio: **~8.5% energy win, ~10% area win**.

## Paper-ready claim

At iso-params (106K), the Mambino architecture achieves **8.5% lower digital chip
energy and 10% lower area** than the Pure S5 baseline of comparable capacity, using
Accelergy 0.4 with CACTI (SRAM) and NeuroSim (MAC, register file) plugins at 22nm.

Combined with the +2.25 pp accuracy advantage on LRA-ListOps (0.6055 vs 0.5830 gelu),
this is a Pareto-dominant point.
