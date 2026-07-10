# Mambino — TNNLS submission outline (v1, 2026-07-10)

**Target venue:** IEEE Transactions on Neural Networks and Learning Systems (TNNLS)
**Impact factor:** ≈10.4 (Q1 in Artificial Intelligence)
**Format:** Regular Paper (12–15 published pages, single-column double-spaced manuscript)

**Title:** *Mambino: Predictive-Coding-Augmented State-Space Models for Efficient Long-Range Reasoning on Constrained Hardware*

**Corresponding author:** Raisul Islam ([raisul@alumni.stanford.edu](mailto:raisul@alumni.stanford.edu))

**Suggested keywords / Index Terms:** state-space models, predictive coding, sequence
modeling, edge inference, systolic arrays, INT8 quantization, low-rank factorization,
chip PPAC, LRA benchmark.

---

## Abstract (200–250 words)

State-space models (SSMs) offer a chip-friendly alternative to attention for long-range
sequence reasoning, but the highest-accuracy variants (S7, Mega) sacrifice on-chip
efficiency by introducing selective/dynamic parameters that require expensive per-timestep
compute. We introduce **Mambino**, a state-space model that grafts a *predictor* branch
running one timestep behind the main scan onto S5's stable diagonal substrate, and a
*low-rank gate* factorization that reduces the dominant matmul silicon by 4× without
accuracy loss. The predictor's *s(t−1)* shift enables its computation to overlap with the
main scan on chip, and its dedicated SSM state absorbs the representational load that
would otherwise require doubling the main SSM state dimension.

We evaluate Mambino under a JAXPR-verified multi-array chip PPAC methodology at 22 nm
INT8, with per-config activation SRAM tiers sized to eliminate off-chip DRAM traffic.
Under Area × Latency Product-optimal chip designs, Mambino at 188 K parameters achieves
+0.49 pp accuracy on LRA-ListOps versus the published S5 baseline while using **37% less
die area** (17.84 mm² vs 28.36 mm²) and **56% less peak power** (800 mW vs 1,829 mW),
trading +31% inference energy and 3× per-sample latency. Mechanism attribution via a
matched-parameters low-rank-gate baseline without predictor shows that naive
low-rank-plus-P-doubling *fails on both silicon and accuracy* (+7% area, −0.98 pp acc),
isolating the predictor branch as the load-bearing mechanism. Mambino's Pareto profile —
lower peak power at higher energy per operation, slower per query — mirrors biological
predictive-coding computation.

---

## I. Introduction (1.5 pp)

- **Motivation.** Edge sequence-inference chips (mobile SoC NPUs, wearable AI, always-on
  sensor systems) face a chip-friendliness gap: attention is O(N²) and uses expensive
  key-value caches; SSMs achieve competitive accuracy at O(N) with regular matmul
  workloads, but the state-of-the-art SSM variants (S7 at 63.8% on LRA-ListOps, Mega at
  63.1%) achieve their accuracy via selective/dynamic scans that undo the SSM chip
  advantage.
- **Gap.** No published SSM combines (a) SOTA-close accuracy on LRA reasoning with (b) a
  chip design that is unambiguously smaller/lower-power than the S5 baseline.
- **This paper.** We propose Mambino, a predictive-coding-augmented S5 variant, and
  demonstrate that at 188 K parameters it delivers +0.49 pp accuracy on LRA-ListOps *and*
  −37% die area *and* −56% peak power vs published S5 at 22 nm INT8. The efficiency wins
  come from a specific mechanism — a predictor branch running at *s(t−1)* — which we
  isolate empirically via a matched-parameters ablation.
- **Contributions:**
  1. **Architecture**: Mambino — a predictor-augmented state-space model that unlocks
     low-rank gate factorization without accuracy loss.
  2. **Mechanism attribution**: a Pure-S5-with-low-rank-gate-and-P=16 baseline (Corner 2)
     proves the predictor branch (not the low-rank factorization) is the load-bearing
     mechanism; naive cheap-outs fail on both silicon and accuracy.
  3. **Chip PPAC methodology**: a JAXPR-verified multi-array direct-instrumentation
     simulator with hard workload–model invariant, per-config activation SRAM tiers, and
     documented fixes for four common oversights (fill/drain, operand-role assignment,
     elementwise memory traffic, partial-sum spill).
  4. **Empirical wins**: −37% die area, −56% peak power, +0.49 pp accuracy vs published S5
     at 22 nm INT8.
  5. **Brain-alignment framing**: Mambino's Pareto profile (lower peak power at higher
     energy per operation, slower per query) is qualitatively consistent with biological
     predictive-coding computation.
- **Paper organization.**

---

## II. Related Work (1.5 pp)

- **State-space models.** S4 (Gu et al. 2022), S5 (Smith et al. ICLR 2023), Mamba/S6 (Gu &
  Dao 2023), S7 (Han et al. 2024), Hyena, Griffin, Zamba. Emphasize S5's diagonal SSM as
  the chip-friendly baseline and S7/Mega/S6-selective-scan as SOTA but chip-hostile.
- **Predictive coding.** Rao & Ballard (1999) hierarchical predictive coding; Friston's
  free-energy principle; deep predictive coding networks (Lotter et al. 2017); why
  predictive coding is naturally low-power in biology.
- **Low-rank neural components.** LoRA (Hu et al. 2022), MoRA, adapters, factorized
  Dense; when low-rank helps and when it doesn't (bottleneck vs full rank).
- **Chip PPAC methodology.** Accelergy (Wu et al. 2019), Timeloop (Parashar et al. 2019),
  CACTI 7 (Balasubramonian et al. 2017), NeuroSim (Chen et al. 2018), SCALE-Sim v2
  (Samajdar et al. 2020). Position our multi-array direct-instrumentation approach as a
  complement to Timeloop's mapper search.
- **Hardware-aware neural network design.** Cite Eyeriss (Chen et al. ISSCC 2016), TPU
  papers, edge NPU deployment literature (MLPerf Tiny). Mambino sits in the hardware-first
  design line but grounded in a specific mechanism claim.

---

## III. The Mambino Architecture (2 pp)

### III.A. S5 base and its chip weakness

Brief recap of S5's diagonal SSM parallel scan and the half_glu2 gate. Highlight that the
dense H×H gate is the dominant matmul on chip (K = N = H = 128 → 64×64 systolic array =
4,096 PEs = the largest single silicon block).

### III.B. Mambino's two innovations

**Low-rank gate factorization (r = 40)**: replace Dense(H, H) with Dense(H, r) → Dense(r,
H). Reduces gate parameters from 16,384 to 10,240 (−38%) while preserving H-dim output for
element-wise gating. Same mathematical structure as LoRA adapters but applied
end-to-end.

**Predictor branch**: at each timestep, run a *second* SSM (state dimension P_pred = P_main
= 8, complex diagonal) using the previous timestep's state s(t−1) as input. Because the
predictor operates one timestep behind, its scan is fully independent from the current
timestep's main scan and can be scheduled in parallel *on the same physical array* — no
additional silicon.

Formally the layer computation is:

```
s_t^main       = A_bar s_{t-1}^main    + B_bar x_t
s_t^pred       = A_pred s_{t-1}^pred   + B_pred x_t   (uses s(t-1) inputs)
y_t^main       = C_main s_t^main
y_t^pred       = C_pred s_t^pred                       ← extra readout
y_t            = (y_t^main + y_t^pred) ⊙ σ(V (U x_t))  ← low-rank gate
                                        \_________/ 
                                          rank-40 gate
output_t       = LayerNorm(y_t)
```

### III.C. Predictive coding interpretation

Frame `y_t^pred` as the network's prediction of x_t based on state history, and `x_t −
y_t^pred` as the prediction error being routed through the main SSM. This is
architecturally a Rao–Ballard hierarchical predictive coding block, embedded as a per-layer
Bayesian correction. Cite predictive-coding literature and connect to biological
motivation.

### III.D. Parameter accounting

Config 4 (Mambino gelu 106 K, P = 8), Corner 3' (Mambino half_glu2 r = 40 188 K, P = 8),
with dominant blocks and shape justification. Match Corner 3' to Corner 1 (Pure S5 dense
half_glu2 188 K) to enable iso-parameter comparison.

---

## IV. Chip PPAC Methodology (3 pp)

### IV.A. Design assumptions

Target: 22 nm INT8 systolic accelerator, 1 GHz clock, no off-chip DRAM, weight-stationary
per-block dedicated arrays. Per-config chip tier: activation SRAM sized to fit worst-case
matmul footprint (Config 4/5: 320 KB, Corner 2/3': 384 KB, Corner 1: 512 KB). Weight SRAM
256 KB across all configs; state SRAM 64 KB.

### IV.B. JAXPR-verified workload extraction

Extract per-config workload from the compiled JAXPR of the trained checkpoint's forward
pass. Emit per-GEMM records (M, N, K, dtype, per_mac_flops, repeat, total_real_macs) and
per-elemwise records (op class totals). **Hard invariant**: sum of emitted FLOPs equals
JAXPR-walker total FLOPs bit-for-bit. Any deviation aborts the extraction. This anchors
every chip number to a specific trained checkpoint SHA.

### IV.C. Multi-array chip design rule

For each unique GEMM shape, dedicate a weight-stationary systolic array sized under
**max-utilization + no-over-provisioning**:

> Pick the smallest standard {8, 16, 32, 64}² rectangle whose (a) spatial fanout achieves
> 100% array utilization on the shape, and (b) whose total cycles fit under a per-config
> bottleneck target `T`.

Sweep `T` from 8 k to 2 M cycles to trace each config's Area × Latency Product Pareto and
report the ATP-optimal design point. Where no standard array achieves 100% utilization
(Corner 2 and Corner 3' K = 40 low-rank gate blocks), relax the rule to ≥62.5% utilization;
Corner 1 has no K = 40 shape and is unaffected.

### IV.D. Cycle model with fill/drain (H1)

Per-tile cycles = *stream_cyc* + (*ay* + *ax* − 2), where *stream_cyc* is the
streaming-dim length. Depth-1 weight scratchpad forces preload per tile boundary. Total
per-instance cycles = per-tile × outer-tile-count × dtype_scale. This corrects a common
under-count where systolic pipeline fill/drain is ignored.

### IV.E. Operand-role assignment (H2)

For each GEMM, the smaller matrix is the parameter (stationary weight) and the larger is
the streaming activation. For SSM matmuls (small M ≈ P, large N = L), this makes the small
`B̄` matrix stationary — not the L = 2048 activation sequence. Correctly routes traffic
between weight and activation SRAMs.

### IV.F. Elementwise memory traffic (H3)

Per scalar op = 2 activation-SRAM reads + 1 write. Compute cost by op class (trivial 1×,
moderate 3×, transcendental 10×, reduction 2× MAC energy; structural addressing ops treated
as 30% materializing). Fable-audited sensitivity to Horowitz-anchored coefficients keeps
total energy within ±6% and preserves all comparison directions.

### IV.G. Partial-sum spill and M-chunking (H4 + F1)

When outer accumulation-K tile count exceeds 1, running M × N INT32 partial sums spill
through activation SRAM. When the resident psum footprint exceeds 75% of activation SRAM,
the block M-chunks so each chunk's psum fits — preserving the no-DRAM invariant.

### IV.H. Predictor s(t − 1) overlap

Mambino's predictor SSM matmul (same shape as main SSM's `B̄ · x`) shares the main SSM
array via the *s(t − 1)* shift. For latency, the main SSM matmul's 24 JAXPR instances
collapse to 16 wall-clock instances. For energy, all 24 instances are charged in full —
overlap hides time, not joules.

### IV.I. Area model

PE: 3,000 μm² per INT8 MAC + weight/psum scratchpads at 22 nm. SRAM: 0.7 Mb/mm² at 22 nm HD
SRAM density. Per-tier ERTs generated by CACTI 7 (SRAMs) + NeuroSim (MAC + spads).

### IV.J. Fable-audited validation

Every H1–H4 finding and the three follow-up F1–F3 corrections applied and re-audited by an
independent Claude-Fable agent. Full audit trail archived with the paper's supplementary
material. Remaining known limitations (F4–F6, <1% impact) documented in Appendix.

---

## V. Experimental Setup (1 pp)

### V.A. Benchmark
LRA-ListOps, sequence length 2 048, 10-class classification. Chosen for its combination of
(a) reasoning complexity that exposes accuracy differences and (b) sequence length
representative of long-context edge inference.

### V.B. Training protocol
40 epochs, Adam with the S5 opt_config, bf16 mixed-precision, 8-seed sweep for accuracy
statistics (seeds: 6554595, 42, 12345, 271828, 314159, 1, 2, 3). Best validation-epoch
checkpoint reported.

### V.C. Configurations (five iso-parameter and iso-family comparisons)
| Config | Family | Activation | glu_rank | ssm_size_base (P) | Params |
|---|---|---|---|---|---|
| Config 4 | Mambino | gelu | — | 16 (P = 8) | 106 K |
| Config 5 | Pure S5 | gelu | — | 32 (P = 16) | 106 K |
| Corner 1 | Pure S5 | half_glu2 | 0 (dense) | 16 (P = 8) | 188 K |
| Corner 2 | Pure S5 | half_glu2 | 40 (low-rank) | 32 (P = 16) | 189 K |
| Corner 3' | Mambino | half_glu2 | 40 (low-rank) | 16 (P = 8) | 189 K |

The Corner 1 / Corner 2 / Corner 3' triple is the mechanism-attribution set: Corner 1 is
the reference; Corner 2 tests "cheap out gate rank + bump P" naive strategy; Corner 3'
tests Mambino's predictor-plus-low-rank-gate.

### V.D. Chip target
22 nm INT8 WS systolic, 1 GHz, no off-chip DRAM, edge inference class.

### V.E. Baselines
- Published S5 accuracy (Smith et al. 2023): 62.15%.
- External SOTA context: S7 63.77%, Mega 63.14% (cited but not run — both are chip-hostile
  under our design rule).

---

## VI. Results (3–4 pp)

### VI.A. Accuracy
Full 8-seed table with means and paired significance tests (t-test on matched seeds). Key
findings:
- **Corner 3' vs Corner 1**: +0.49 pp (p = 0.022, matched-seed paired t)
- **Corner 3' vs Corner 2**: +1.47 pp (p = 6.5 × 10⁻⁵, matched-seed paired t)
- **Config 4 vs Config 5**: +0.76 pp
- **Corner 3'** ≈ published S5 SOTA vintage but at −37% area

### VI.B. Mechanism attribution — the Corner 2 finding
Table showing Corner 1 vs Corner 2 vs Corner 3' on accuracy AND chip cost. Emphasize
Corner 2 is Pareto-dominated: it loses accuracy AND it costs more silicon (+7% vs Corner
1). This isolates the predictor branch as the load-bearing mechanism — low-rank gate alone
is a *double loss*.

### VI.C. Chip PPAC (v4)
Reproduce master doc §6a table. Highlight:
- **Area**: Corner 3' 17.84 mm² vs Corner 1 28.36 mm² (−37%)
- **Peak power**: 800 mW vs 1,829 mW (−56%)
- **Energy per inference**: 721 μJ vs 549 μJ (+31%, honest cost)
- **Latency**: 0.902 ms vs 0.300 ms (3× slower, honest cost)

### VI.D. Area and energy breakdown
- Area breakdown (§6b): where the −37% savings come from (PE: −48%; SRAM: −15%).
- Energy breakdown (§6c): elemwise dominates energy at 43–57% across all configs. MAC
  compute is only 21–26%. This is the qualitative surprise vs classical NN-chip intuition.

### VI.E. Design Pareto curve
Corner 3' Pareto: sweep the bottleneck target and trace area vs latency. Show the
ATP-optimal point, the matched-throughput point (target = 131 k cycles → 6,336 PE, 27.25
mm², 6,966 /s, essentially Corner 1's throughput), and the low-power/slow extreme.
Reader-facing message: Mambino is Pareto-favorable on die-area axis for the full
throughput range 5.6 – 7.0 k /s.

### VI.F. Accuracy per unit chip cost (§8)
Three-axis efficiency table: acc/mJ, acc/mm², acc/mW.
- Mambino wins acc/mm² and acc/mW at both 106 K and 188 K.
- Pure S5 wins acc/mJ (energy per inference) — reported honestly.

### VI.G. Sensitivity analysis
Horowitz-anchored coefficient substitution: ≤6% total energy shift across all configs,
directions preserved. Fable audit findings summarized. Uncertainty band 30-36% for the
Corner 3' vs Corner 1 energy gap.

---

## VII. Discussion (1 pp)

### VII.A. Brain-aligned Pareto
Mambino's chip profile — lower peak power at higher energy per operation, slower per
query, higher accuracy — is qualitatively consistent with biological predictive-coding
computation. Brains use ∼20 W distributed across cortex (low peak power), consume more
metabolic energy per computation than optimized digital, and are often slower per decision
than dedicated compute — yet excel on reasoning tasks. Mambino's predictor branch
implements a Rao-Ballard hierarchical predictive-coding step and inherits the same
tradeoff shape at the chip level.

### VII.B. When to use Mambino
- **Edge inference** (battery-limited, thermal-limited, die-area-cost-limited): Mambino
  wins on area, power, accuracy — the three metrics that customers use.
- **Datacenter batched inference** (throughput-per-watt-limited): Pure S5 wins on
  energy-per-inference and throughput. Mambino would need architectural changes (e.g.,
  weight-preload pipelining, deeper spads) to close the energy gap.

### VII.C. Comparison to SOTA
Cite S7 63.8% (chip-hostile due to selective scan), Mega 63.1% (attention hybrid,
quadratic in sequence length). Mambino at 61.4% is 2.4 pp below S7 but at ≈17.8 mm² on
22 nm INT8 vs S7's unmeasured but structurally much larger chip footprint. Position as
**SOTA-close at chip-realistic silicon** rather than SOTA-chasing.

### VII.D. Limitations
- Single task (LRA-ListOps) — future work: full LRA-6 suite, LM benchmark, MQAR.
- Single chip target (22 nm INT8 WS systolic) — future work: 7 nm, other dataflows.
- Elemwise coefficients from literature averages, not per-op ISSCC-measured — sensitivity
  band bounds impact but per-op RTL synthesis (Path 3 in supplementary materials) is
  future work for a chip-venue follow-up.
- No operator-fusion assumption — real fused GLU/activation chains would reduce SRAM
  round-trips; the paper's chip numbers are a conservative upper bound.

---

## VIII. Conclusion (0.5 pp)

Mambino demonstrates that predictive-coding-augmented state-space models achieve
edge-inference-class chip efficiency (−37% area, −56% peak power) while gaining accuracy
(+0.49 pp on LRA-ListOps at 188 K). The load-bearing mechanism is the s(t − 1)-shifted
predictor branch — not the low-rank gate factorization on its own, which we show is a
double loss on both silicon and accuracy without the predictor. The chip cost profile —
higher energy per inference, slower per query — mirrors biological predictive-coding
computation. Mambino's methodology, PPAC code, JAXPR-verified workloads, and per-config
Accelergy ERTs are open-sourced under `paper_v2_ppac/`, with a 10-minute reproduction path
documented in the accompanying README.

---

## References (∼30 entries)

Grouped by topic: SSMs (10) · Predictive coding (5) · Low-rank / LoRA (4) · Chip PPAC
tools (5) · Chip venues context (4) · JAXPR / JAX (2).

---

## Appendix A. Chip PPAC methodology limitations

Fable-audited F4–F6 residuals: (F4) unary transcendentals charged 2 reads instead of 1
(≤10 μJ total impact); (F5) head 1 × 10 × 128 role misassigned by size heuristic (≤0.01 μJ);
(F6) fill/drain omits weight-preload overlap (<0.2% cycles). None affect any published
comparison direction.

## Appendix B. Per-config detailed block assignments

Full per-block table for each config at its ATP-optimal design (block-shape, array,
utilization, cycles, silicon, energy per block).

## Appendix C. Reproduction protocol

- Clone `github.com/raisul1212/S5` at tag `mambino-paper-v2-public`.
- Download PURR-deposited checkpoints.
- Run `python paper_v2_ppac/chip/multi_array_ppac_v4.py`.
- Verify output against archived `multi_array_sweep_v4.json`.

---

## Estimated page budget (TNNLS format)

| Section | Pages |
|---|---:|
| Abstract + Index Terms | 0.5 |
| I. Introduction | 1.5 |
| II. Related Work | 1.5 |
| III. Architecture | 2.0 |
| IV. Chip PPAC Methodology | 3.0 |
| V. Setup | 0.7 |
| VI. Results (5 subsections + tables + figures) | 3.5 |
| VII. Discussion | 1.0 |
| VIII. Conclusion | 0.5 |
| References | 1.0 |
| Appendices | 1.0 |
| **Total** | **~16 pp** |

Well within TNNLS Regular Paper limits.

---

## Figures needed (candidates)

1. **Mambino architecture diagram** — the predictor branch, its s(t − 1) shift, the
   low-rank gate factorization, and how they compose per layer. Publication-critical.
2. **Chip PPAC pipeline diagram** — JAXPR → workload YAMLs → multi-array chip → ATP-optimal
   sweep. (`paper_v2_ppac/chip/pipeline_diagram.html` already drafted.)
3. **Fig 4 candidate**: Area vs Accuracy Pareto (five configs, annotated with the Corner
   1/2/3' mechanism-attribution triangle). Publication-critical.
4. **Fig 5 candidate**: Energy per-component stacked bars showing elemwise dominance.
5. **Fig 6 candidate**: Corner 3' design Pareto sweep (area vs latency, ATP-optimal point
   marked).
6. Optional: chip block diagram — the multi-array topology with dedicated arrays per shape.

The published chip-PPAC dashboard artifact
(`~/AppData/Local/Temp/.../mambino_chip_ppac.html`) already contains publication-ready
versions of Figs 3-6. Extract as PDF/PNG for the manuscript.

---

## Submission checklist before TNNLS submission

- [ ] Prose written for Sections I–VIII.
- [ ] All references formatted to IEEE style.
- [ ] Figures at ≥300 DPI PNG or vector SVG/PDF; captions self-contained.
- [ ] Tables in IEEE two-column final layout.
- [ ] Supplementary material: workload YAMLs, ERTs, `multi_array_ppac_v4.py`,
      `multi_array_sweep_v4.json`, master doc §6 excerpt.
- [ ] Code deposit: `github.com/raisul1212/S5` at `mambino-paper-v2-public` tag.
- [ ] Data deposit: PURR archive with checkpoints + run logs.
- [ ] Author contribution statement + funding acknowledgement + conflict-of-interest.
- [ ] IEEE submission cover letter emphasizing hardware-aware neural architecture fit.

---

## Post-acceptance roadmap

- Follow-up chip-venue paper (JSSC / ISCA / MICRO) building on Path 3 open-source RTL
  synthesis (OpenROAD + Nangate 45 nm → 22 nm scaled). Adds Verilog + synthesis reports
  as gold-standard per-op numbers.
- Companion state-cell paper (Nature Electronics track) covers mixed-signal PIM
  implementation as complementary hardware substrate.
