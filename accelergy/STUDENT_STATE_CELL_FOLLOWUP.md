# Follow-up: Round 2 of State-Cell Characterization

**To:** [student]
**From:** Raisul Islam
**Date:** 2026-07-06
**Re:** Five additional SPICE measurements needed to close out the mixed-signal
PPAC section of the paper.

## Context

Your first round of characterization was excellent — write energy vs `Cstore`,
retention vs `Cstore`, refresh energy transient, and the two candidate design
points are all clean and well-documented. Thank you.

Based on that work, we've picked **Point A (`Cstore=100 fF, VTAU=0V,
M_tau L=10μm`)** as the canonical design point for the paper's mixed-signal
PPAC. Rationale:

- Point A's `T_ret_1% = 72.48 μs` gives a **36× margin** over the L=2048 scan
  time (2 μs at 1 GHz).
- Point B's extra retention headroom (107 μs) buys only ~1.5× more margin
  for a **+52% write-energy cost**, so it's not worth the trade for our
  workload.
- Write energy at Point A (`98.9 fJ`) sits right at the edge of the memo's
  original `1–100 fJ` target.

## Q1 — Tech node: switch to open-source PDKs for Round 2

Round 1 was done on TSMC 180 nm PDK — that data is fine for internal design
guidance (Point A is our locked design point) but the specific numbers
cannot be published due to foundry NDA. For a defensible paper we need
characterization at an OPEN-SOURCE PDK whose results we can publish
directly.

**Round-2 measurements must be at BOTH of the following nodes:**

### FreePDK45 (NCSU, 45 nm bulk CMOS, Apache 2.0)

- URL: https://eda.ncsu.edu/freepdk/freepdk45/
- BSIM4 bulk-MOSFET models — direct analog to your existing 180 nm design
- Voltage rails at 45 nm: Vdd ≈ 1.0 V core, 1.8 V I/O — matches what you
  already used in Round 1, so the 3T + 1C topology transfers directly
- Same tool flow as your existing setup (Cadence Virtuoso + Spectre)
- Zero redesign required — just re-simulate at 45 nm

### FreePDK15 (NCSU, 15 nm FinFET, Apache 2.0)

- URL: https://eda.ncsu.edu/freepdk15/
- FinFET devices — 15 nm PTM-MG HP model
- **Requires topology re-sizing:** W/L → fin count. Preserve the 3T +
  MOM-cap architecture but expect device sizing to change. Also:
  - Vdd at 15 nm FinFET ≈ 0.7-0.8 V — lower rails, potentially better
    retention headroom
  - Sub-threshold leakage at FinFET is lower per equivalent-W device than
    bulk, so retention should IMPROVE at 15 nm (opposite of the 180 nm →
    22 nm bulk scaling concern)
  - MOM cap density is higher at 15 nm — cell area should shrink
- Timeline: FinFET analog sizing may take an extra week; if it's a burden
  and #0 / #0b results at FreePDK45 already show good numbers, you can
  reduce FreePDK15 to a subset of the priority items (see below).

### Why both nodes

Two open-source data points bracket the 22 nm chip target (45 nm above,
15 nm below). This lets the paper's mixed-signal PPAC section:

- Cite ACTUAL measured numbers at TWO nodes rather than a single-node
  measurement plus a hand-wavy scaling justification
- Show the scaling trend with real data → reviewers can verify our 22 nm
  interpolation instead of trusting a theoretical scaling factor
- Validate the cell topology as portable across bulk and FinFET
  device families

**Nothing from Round 1 (TSMC 180 nm) will appear in the paper. It remains
an internal-only design-space exploration result.**

## Q2 — Refresh methodology (still needs clarification)

Your reported refresh energy of `1.23 fJ` at `Cstore = 50 fF` is 40× smaller
than the write energy at the same capacitance (`50.23 fJ`). That's
suspicious. Please clarify: does the 1.23 fJ measurement account only for
the ΔV_leaked top-up charge, or does it include the full write-driver + WWL
overhead? For paper defensibility we need refresh to be measured on the
same basis as write (full switch + driver cycle), even if that means it
comes out closer to write energy.

## Five new SPICE measurements needed (prioritized)

Priority ordering below reflects "biggest blocker for Accelergy modeling"
first. Items #0 and #0b are new criticality — they gate everything else.
Please try to get these back within one to two weeks.

### Priority #0 — Read circuit noise floor (MOST CRITICAL)

Measure and report:

- **v_n,read,RMS**: input-referred RMS noise of your intended readout
  circuit, band-limited to the SSM sample rate (1 GHz → ~500 MHz effective
  bandwidth).
- **V_offset,read**: DC systematic offset from Monte Carlo (mean-of-mean and
  worst-case at 3σ).

If you haven't picked a readout topology yet, run this for both:
- source-follower + comparator
- current-integrator + comparator

Target values (from memo rev2 §2a):
- `v_n,read,RMS ≤ 2 mV` (= LSB/4 at INT8 with ±1 V full-scale)
- `V_offset` should not eat more than 1 LSB (~8 mV) after calibration

**Why this is #0:** without this, we cannot verify Point A is viable. If
the readout floor is above ~10 mV RMS, no choice of `Cstore` rescues the
signal — the readout becomes the binding constraint, not the storage cell.
Increasing `Cstore` would then buy us nothing but write energy.

**Format expected:** two numbers (one topology minimum), corner range
across TT/FF/SS, with noise-vs-bandwidth plot if available.

### Priority #0b — I_leak(V) curve on M_tau (CRITICAL)

Your Round 1 retention `T_ret` was measured at some fixed V(h). Sub-threshold
leakage in M_tau scales roughly as `exp(qV/kT)`, so leakage at signal
amplitude near Vdd could be 10-100× worse than at signal amplitudes near
mid-rail.

Measure and report I_leak through M_tau at:

- V(h) = 0.1 V, 0.3 V, 0.5 V, 0.7 V, 0.9 V (5 points minimum)
- VTAU = 0 V
- M_tau geometry: 220 nm / 10 μm (as in Point A)
- Temperature: 25 °C nominal; add 85 °C if easy

**Why this is #0b:** if I_leak at V(h)=0.9V is > 5× the number implied by
your reported T_ret, then the effective retention at working amplitude
is significantly worse than 72 μs — and Point A might not be viable
without a bigger `Cstore`.

**Format expected:** table of 5 I_leak values or an I-V plot exported as
CSV.

### Priority #1 — Read energy E_rd (per read)

The paper's Accelergy modeling for mixed-signal PPAC needs an energy
number per read access. Please measure:

- Cell + readout combined energy per single read cycle at the recommended
  design point (Point A, `Cstore = 100 fF`, `VTAU = 0 V`).
- Assume typical mid-rail V(h) ≈ 0.5 V for the "nominal" number, plus one
  extreme (V(h) = 0.9 V) for worst case.

Expected range: **0.1 – 1 fJ per read** for a source-follower or current-
integrator readout of a 100 fF node. Anything higher deserves a note about
readout topology.

**Format expected:** two energy numbers (nominal + worst), same reporting
format as write energy in Round 1.

### Priority #2 — A_cell (Cell area)

Layout area of one full state cell at `Cstore = 100 fF` (and 150 fF if easy,
for cross-check).

**What we need:**
- Total cell area in μm² (including transistors, cap, contacts, local routing).
- Aspect ratio (row-pitch × col-pitch) so we can estimate array packing
  efficiency.
- Note whether MOM cap dominates area vs the three transistors.

For 22nm at 100 fF MOM cap, **expect ~15-30 μm² per cell**. If it's much
larger (>50 μm²), flag why — likely a MIM cap or a larger routing overhead
that's worth understanding.

**Format expected:** μm² + aspect ratio + one sentence on what dominates.

### Priority #3 — P_leak per cell + refresh at 100 fF

Two related items:

- **P_leak per cell:** static DC leakage power at nominal Vdd,
  V(h) = 0.5 V, VTAU = 0 V, room temp. Report in fW or pW.

  Expected: **~0.1 – 10 pW per cell** at 22nm LP. Feeds into standby
  power estimate for the chip.

- **E_refresh at Cstore = 100 fF** (and 150 fF if easy): remeasure using the
  same methodology as write (full driver cycle, not just ΔV top-up).
  See Q2 above for clarification of what "full refresh" means. Report on
  same basis as write energy.

**Why together:** these two numbers together determine whether refresh
dominates or leakage dominates at long-idle standby. We need both for a
clean chip-level power model.

**Format expected:** two energy/power numbers.

## What we DON'T need in this round

To keep the ask focused, defer these to a future round:

- Full v_n,read vs bandwidth curve (single-point number is enough for now)
- Monte Carlo mismatch histogram
- Full PVT sensitivity sweep
- Wiring parasitics C_par_row/col
- Integrator INL vs V_in curve
- τ = R·C explicit measurement
- Array-scale rollups (§2f of Round-1 memo)

We'll pick these up in Round 3 once the Round-2 items above are in.

## Timeline ask (revised for dual-node characterization)

### Week 1 — FreePDK45 sanity check + gating measurements

Start with FreePDK45 because it's a direct port of your existing 180 nm
Cadence flow. Get **Priority #0 (read noise floor) and #0b (I_leak(V)
curve) at FreePDK45** back within one week. Even back-of-envelope SPICE
numbers with clear caveats are more useful than polished reports that
arrive too late.

### Week 2 — FreePDK45 remaining items + FreePDK15 setup

Complete Priority #1 – #3 at FreePDK45. In parallel, begin FreePDK15
device sizing (fin count) and re-simulate the write-energy vs Cstore
sweep at FreePDK15 to establish the two-node design point.

### Week 3 — FreePDK15 full Round-2 measurements

Repeat the 5 priority items at FreePDK15. If time is tight, prioritize
**Priority #0, #0b, and #1** at FreePDK15 — the aggregate energy numbers
that let us bracket 22 nm interpolation. #2 (A_cell) and #3 (P_leak) at
FreePDK15 are useful but the FreePDK45 numbers already cover the paper's
core PPAC claims.

**If FreePDK15 turns out to be more than 2-3 weeks of additional work,
tell us right away** and we'll fall back to FreePDK45 only + scaling
methodology for 22 nm (still publishable, just weaker story).

## What we'll do with your Round-2 data

The moment #0 and #0b arrive we'll know whether Point A ships as-is or
whether we need to escalate to Point B (or larger). Once we have that
call, we'll drop the E_rd + A_cell + P_leak values into an
NVMExplorer-style `.cell` file, run Accelergy on both pipelined and
sequential chip topologies, and get paper-defensible mixed-signal PPAC
numbers.

## Contact

Ping me any time with questions about the readout circuit spec (I can
help with the intended topology) or on how the numbers will be used.

---

**Priority order recap:** #0 (read noise floor) → #0b (I_leak(V)) → #1 (E_rd)
→ #2 (A_cell) → #3 (P_leak + refresh at 100 fF).

**Also please confirm:** tech node (Q1) and refresh methodology basis (Q2).
