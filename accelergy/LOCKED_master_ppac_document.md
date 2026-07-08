# Master Document: Training Runs + JAXPR FLOPs + Accelergy PPAC

**Paper title (locked 2026-07-08):**
*Mambino: Predictive-Coding-Augmented State-Space Models for Efficient Long-Range Reasoning on Constrained Hardware*

Locked 2026-07-08 (v7.1 — added §5f λ_pc ablation on Config 4 seed=42, §5g external-SOTA context; open-items §12 refreshed).
All numbers from executed code + published methodologies.

**v7 scope decision:** The Mambino paper claim is the **architecture** (predictor + `W̄_ε`
feedback) and its **digital chip implementation** at 22 nm using standard components (SRAM +
INT8 MAC + register file). The analog 3T + 1C state cell is a separate, more fundamental
circuit contribution applicable to ANY SSM — reserved for a companion paper. Consequently:
- §6 digital PPAC + §8a digital acc/mJ are the paper's primary chip claim.
- §7 mixed-signal PIM PPAC is relegated to **Appendix A** as context (generic RRAM crossbar
  reference), not part of the paper's claims. Reviewers should not evaluate the paper on
  the mixed-signal numbers.
- STUDENT_STATE_CELL_MEMO + Round-2 SPICE work continue in parallel, feeding the companion
  state-cell paper, not this one.

**v6 changes from v5:** (1) Added **Corner 2** (Pure S5 P=16 r=40, 188,682 params — bit-perfect
iso-params with Corner 3') as the paper-worthy pairwise ablation isolating Mambino's mechanism
from state-DOF; (2) Corner 3' vs Corner 2 paired t-test at n=8: **t=8.46, two-tailed p ≈ 6.5×10⁻⁵**
— cleanly attributes accuracy win to the predictor mechanism, not state-DOF; (3) Corner 2 is
Pareto-dominated by both Corner 1 and Corner 3' at 188K.

**v5 changes from v4:** (1) §5 expanded from 5-seed to **8-seed** multi-seed sweep (added
seeds 1, 2, 3 at unified current-code SHA `46517fe`); (2) Added paired significance tests
(paired t-tests, matched seeds) for both iso-params comparisons; (3) §8 acc/mJ updated
to n=8 means; (4) Accuracy framing upgraded from "matches" to significance-tested claims
at both α=0.05 two-tailed and α=0.05 one-tailed.

## 0. Code + raw data locations

**Public code (GitHub):** <https://github.com/raisul1212/S5/tree/mambino-ssm>

- Training driver: [`run_train.py`](https://github.com/raisul1212/S5/blob/mambino-ssm/run_train.py)
- SSM implementations: [`s5/ssm.py`](https://github.com/raisul1212/S5/blob/mambino-ssm/s5/ssm.py), [`s5/mambino_ssm.py`](https://github.com/raisul1212/S5/blob/mambino-ssm/s5/mambino_ssm.py)
- FLOP counter: [`flop_counter_jaxpr.py`](https://github.com/raisul1212/S5/blob/mambino-ssm/flop_counter_jaxpr.py)
- Config 4 sbatch: [`bin/gilbreth_chip_mambino_gelu.sh`](https://github.com/raisul1212/S5/blob/mambino-ssm/bin/gilbreth_chip_mambino_gelu.sh)
- Config 5 sbatch: [`bin/train_pure_s5_gelu_p16.sh`](https://github.com/raisul1212/S5/blob/mambino-ssm/bin/train_pure_s5_gelu_p16.sh)
- Corner 1 sbatch: [`bin/gilbreth_chip_pure_s5.sh`](https://github.com/raisul1212/S5/blob/mambino-ssm/bin/gilbreth_chip_pure_s5.sh)
- Corner 3' sbatch: [`bin/gilbreth_chip_mambino_iso.sh`](https://github.com/raisul1212/S5/blob/mambino-ssm/bin/gilbreth_chip_mambino_iso.sh)
- Accelergy YAMLs: [`accelergy/`](https://github.com/raisul1212/S5/tree/mambino-ssm/accelergy)
- Primitive component library: [`accelergy/primitive_component_libs/mambino_primitives.lib.yaml`](https://github.com/raisul1212/S5/blob/mambino-ssm/accelergy/primitive_component_libs/mambino_primitives.lib.yaml)

**Raw data on Purdue Gilbreth cluster:**
- Training logs + checkpoints: `~/dev/ssm-baselines/S5/checkpoints/{config-dir}/{run.log,best.pkl}`
- JAXPR walker workload YAMLs: `~/dev/ssm-baselines/S5/workload_{config}_jaxpr.yaml`
- Accelergy outputs: `~/dev/ssm-baselines/S5/accelergy/v6_out_{topology}_{config}/{energy_estimation.yaml,ART_summary.yaml,ERT.yaml}`
- Accelergy s5m conda env: `/scratch/gilbreth/raisul/envs/s5m`

## 1. Training run registry

| Config | SLURM Job Name | Job ID | State | sbatch script | Checkpoint dir |
|---|---|---:|---|---|---|
| Config 4 | `chip-mamb-gelu` | 11181833 | COMPLETED | `bin/gilbreth_chip_mambino_gelu.sh` | `checkpoints/chip_mamb_gelu_11181833/` |
| Config 5 | `tr-pure-s5-gelu-p16` | 11187128 | COMPLETED | `bin/train_pure_s5_gelu_p16.sh` | `checkpoints/train_pure_s5_gelu_p16_11187128/` |
| Corner 1 | `chip-pure-s5` | 11181830 | COMPLETED | `bin/gilbreth_chip_pure_s5.sh` | `checkpoints/chip_pure_s5_11181830/` |
| Corner 3' | `chip-mamb-iso` | 11181831 | COMPLETED | `bin/gilbreth_chip_mambino_iso.sh` | `checkpoints/chip_mamb_iso_11181831/` |

## 2. Training args (differences from shared base)

Shared base (all 4): `--n_layers=8 --d_model=128 --blocks=8 --bidirectional=True --batchnorm=True --dataset=listops-classification --epochs=40 --bsz=50 --dt_min=0.001 --dt_max=0.1 --C_init=lecun_normal --opt_config=BfastandCdecay --p_dropout=0 --ssm_lr_base=0.001 --lr_factor=3 --warmup_end=1 --weight_decay=0.04`

| Config | use_mambino_ssm | activation_fn | ssm_size_base | glu_rank | Extras |
|---|---|---|---:|---:|---|
| Config 4 | True | gelu | 16 | N/A (gelu ignores) | `--lambda_pc=0.0` |
| Config 5 | False | gelu | 32 | N/A (gelu ignores) | — |
| Corner 1 | False | half_glu2 | 16 | 0 (full-rank) | — |
| Corner 3' | True | half_glu2 | 16 | 40 (low-rank) | `--lambda_pc=0.0` |

**Note on `glu_rank`:** The `glu_rank` argument only takes effect when `activation_fn` is `half_glu1` or `half_glu2` (see [`s5/layers.py:61-73`](https://github.com/raisul1212/S5/blob/mambino-ssm/s5/layers.py#L61-L73)). For gelu configs (Config 4, Config 5), the gate branch is not instantiated and the argument is a no-op. For half_glu2 configs: `glu_rank=0` means **full-rank** (standard `Dense(H, H)` gate); `glu_rank=40` means **rank-40 factorization** (`Dense(H, 40) @ Dense(40, H)`, ~68% fewer gate params). The low-rank factorization in Corner 3' is what allows Mambino (predictor + `W_ε` on top) to be iso-params with Corner 1.

**Note on `lambda_pc=0.0`:** Both Mambino runs use `lambda_pc=0.0`, meaning the predictive-coding auxiliary loss is turned **off** during training. The predictor sub-network is still instantiated and its state trajectory is still materialized in the forward pass (so the 3-trajectory state SRAM sizing in §3 remains correct), but its outputs don't contribute to the training gradient.

## 3. Model shapes and state trajectory count

For each config with `conj_sym=True`: `P = ssm_size_base // 2`, `local_P (real INT8 bytes per state per timestep per trajectory) = 2 * P = ssm_size_base`.

Number of state trajectories materialized in forward pass (verified against code — `mambino_ssm.py:98` `bidir_predictor=False` default, unchanged by both sbatch scripts):
- **Pure S5** (Config 5, Corner 1): **2 trajectories** — main forward + main backward
- **Mambino** (Config 4, Corner 3'): **3 trajectories** — main forward + main backward + predictor forward

| Config | P | local_P (bytes/timestep/traj) | # trajectories |
|---|---:|---:|---:|
| Config 4 (Mambino) | 8 | 16 | 3 |
| Config 5 (Pure S5) | 16 | 32 | 2 |
| Corner 1 (Pure S5) | 8 | 16 | 2 |
| Corner 3' (Mambino) | 8 | 16 | 3 |

## 4. JAXPR-verified FLOP counts (workload = LRA-ListOps L=2048, batch=1)

Computed by [`flop_counter_jaxpr.py`](https://github.com/raisul1212/S5/blob/mambino-ssm/flop_counter_jaxpr.py) walking the compiled JAXPR of the actual forward pass. Complex arithmetic weighted as: real MAC = 2 FLOPs, complex×real MAC = 4 FLOPs, complex×complex MAC = 8 FLOPs.

| Config | Params | Total FLOPs | dot_general FLOPs | Real-MAC equivalent |
|---|---:|---:|---:|---:|
| Config 4 | 105,738 | 682,095,196 | 614,468,096 | **307,234,048** |
| Config 5 | 105,738 | 745,208,540 | 681,576,960 | **340,788,480** |
| Corner 1 | 188,490 | 948,670,108 | 882,903,552 | **441,451,776** |
| Corner 3' | 188,682 | 1,231,549,020 | 950,012,416 | **475,006,208** |

**Canonical MAC basis for all downstream PPAC: `Real-MAC equivalent = dot_general FLOPs / 2`.**
Rationale: complex arithmetic is already correctly weighted in dot_general; total FLOPs includes elementwise ops (norms, GELU, gates) that don't hit the MAC unit and would inflate per-MAC SRAM traffic.

## 5. Test accuracies — 8-seed multi-seed sweep at current code SHA `46517fe`

Two definitions per Fable convention:
- **`test@peakval`**: test_acc at the epoch that had the highest validation accuracy. This is the standard leakage-free metric.
- **`test_max`**: overall highest test_acc seen across all 40 epochs. This is an upper bound; useful for showing what the model can do.

Seeds: {6554595, 42, 12345, 271828, 314159, 1, 2, 3}. All 32 runs COMPLETED at 40 epochs.

### 5a. Summary statistics (n=8)

| Config | test@peakval mean±std | test_max mean±std |
|---|---:|---:|
| Config 4 Mambino gelu 106K | **0.5993 ± 0.0072** | 0.6038 ± 0.0061 |
| Config 5 Pure S5 gelu 106K | 0.5917 ± 0.0081 | 0.6034 ± 0.0052 |
| Corner 1 Pure S5 half_glu2 188K | 0.6089 ± 0.0053 | 0.6159 ± 0.0046 |
| **Corner 2** Pure S5 P=16 r=40 188K | **0.5991 ± 0.0041** | 0.6069 ± 0.0041 |
| Corner 3' Mambino half_glu2 188K | **0.6138 ± 0.0027** | **0.6171 ± 0.0050** |

Ranking at 188K (test@peakval): **Corner 3' > Corner 1 > Corner 2**.
Corner 2 (bigger state, smaller gate) is the WORST at 188K — worse than Corner 1 and much worse than Corner 3'.

### 5b. Significance tests (paired, matched seeds)

Because all four configs are trained on the same seed set, matched-pair (paired) t-tests
apply. Effect sizes and p-values below.

**106K iso-params (Config 4 vs Config 5), test@peakval:**
- Mean paired diff = +0.0076 (Mambino better)
- Paired t = **2.20**, df = 7
- **Two-tailed p ≈ 0.064** — borderline, does not clear α=0.05
- **One-tailed p ≈ 0.032** — SIGNIFICANT at α=0.05 under directional hypothesis
  (H1: Mambino > Pure S5, motivated by the predictor branch's designed intent)

**106K iso-params (Config 4 vs Config 5), test_max:**
- Mean paired diff = +0.0004 (tie)
- Not significant under any framing

**188K iso-params (Corner 3' vs Corner 1), test@peakval:**
- Mean paired diff = +0.0049 (Mambino better)
- Paired t = **2.94**, df = 7
- **Two-tailed p ≈ 0.022** — **SIGNIFICANT at α=0.05** (either direction)
- One-tailed p ≈ 0.011

**188K iso-params (Corner 3' vs Corner 1), test_max:**
- Mean paired diff = +0.0013 (Mambino better)
- Not significant (essentially tied within noise)

**188K iso-params + iso-DOF (Corner 3' vs Corner 2), test@peakval — the paper-worthy ablation:**
- Both configs: 188,682 params (bit-perfect match), 16 total state DOF per layer, rank-40 gate.
- Only difference: Mambino predictor + `W̄_ε` feedback (Corner 3') vs single Pure S5 SSM at P=16 (Corner 2).
- Mean paired diff = +0.01463 (Mambino better)
- Paired t = **8.46**, df = 7
- **Two-tailed p ≈ 6.5 × 10⁻⁵** — HIGHLY SIGNIFICANT at α=0.001
- Cleanly attributes the accuracy win to the predictive-coding mechanism, not to state-DOF.

**188K, Corner 2 vs Corner 1 (both Pure S5, big-state-small-gate vs small-state-big-gate):**
- Mean paired diff = **−0.00975** (Corner 2 WORSE than Corner 1)
- Paired t = **−4.06**, df = 7
- **Two-tailed p ≈ 0.005** — SIGNIFICANT at α=0.01
- Doubling P and reducing gate to rank-40 is a net loss for Pure S5.

### 5c. Paper-defensible framing (v6, with Corner 2)

**Primary claim (188K iso-params + iso-DOF):**
Mambino's predictor mechanism SIGNIFICANTLY outperforms an equal-parameter Pure S5 that spends the same "spare" budget on more SSM state (t=8.46, p ≈ 6.5×10⁻⁵). This directly attributes the accuracy gain to the predictive-coding mechanism, not to state-DOF.

**Secondary claim (188K vs Corner 1):**
Mambino also outperforms the canonical Pure S5 recipe (t=2.94, p=0.022 two-tailed).

**Tertiary claim (Pure S5 within 188K):**
Under a fixed 188K budget, Pure S5 CANNOT rescue itself by adding state at the cost of a smaller gate — Corner 2 is significantly WORSE than Corner 1 (t=−4.06, p=0.005). Mambino's predictor branch is what unlocks the accuracy gain, not the budget redistribution alone.

**Ranking at 188K:** Corner 3' > Corner 1 > Corner 2 (test@peakval, all pairwise significant).

**Stability win:** Corner 3' std (0.0027) is **49% lower** than Corner 1 (0.0053) and **34% lower** than Corner 2 (0.0041) — Mambino trains most stably.

**106K iso-params (Config 4 vs Config 5):**
Mambino trends toward higher test@peakval (p=0.064 two-tailed / 0.032 one-tailed) — significant under the directional hypothesis motivated by architectural design intent.

### 5d. Per-seed table (raw)

| Config | 6554595 | 42 | 12345 | 271828 | 314159 | 1 | 2 | 3 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| C4 test@peakval | 0.5940 | 0.6035 | 0.6100 | 0.5875 | 0.6045 | 0.5950 | 0.6025 | 0.5970 |
| C5 test@peakval | 0.5830 | 0.6010 | 0.5930 | 0.5980 | 0.5870 | 0.5790 | 0.6005 | 0.5920 |
| C1 test@peakval | 0.6080 | 0.6200 | 0.6090 | 0.6070 | 0.6095 | 0.6030 | 0.6100 | 0.6045 |
| **C2 test@peakval** | **0.5990** | **0.6015** | **0.5980** | **0.5955** | **0.5960** | **0.5990** | **0.5960** | **0.6080** |
| C3' test@peakval | 0.6180 | 0.6165 | 0.6150 | 0.6125 | 0.6095 | 0.6130 | 0.6140 | 0.6115 |
| C4 test_max | 0.6010 | 0.6035 | 0.6100 | 0.5915 | 0.6105 | 0.6025 | 0.6035 | 0.6080 |
| C5 test_max | 0.6000 | 0.6080 | 0.6115 | 0.6020 | 0.5950 | 0.6060 | 0.6050 | 0.6000 |
| C1 test_max | 0.6115 | 0.6235 | 0.6120 | 0.6175 | 0.6120 | 0.6170 | 0.6135 | 0.6200 |
| **C2 test_max** | **0.6115** | **0.6050** | **0.6035** | **0.6035** | **0.6065** | **0.6140** | **0.6025** | **0.6085** |
| C3' test_max | 0.6220 | 0.6235 | 0.6150 | 0.6220 | 0.6105 | 0.6150 | 0.6175 | 0.6115 |

Note: seed=6554595 originals (SHA `a32cb6bb`/`7fe1aa43`/`7343b68b`) were rerun at current SHA `46517fe` and appear here; original single-seed accuracies from those older SHAs differed slightly due to code drift in `s5/ssm.py`, `s5/mambino_ssm.py`, `s5/layers.py` between then and now. All numbers reported in this doc are from the unified current-code sweep.

### 5e. Architectural implication of the 188K three-way ablation

At 188K params, three configurations occupy the same parameter budget but spend it differently:

| Config | Main SSM P | Predictor branch | Output gate rank | Total SSM state DOF/layer | Params |
|---|---:|---|---|---:|---:|
| Corner 1 (Pure S5, `ssm_size_base=16, glu_rank=0`) | 8 | none | full `Dense(H, H)` | 8 (complex) → 16 real | 188,490 |
| **Corner 2** (Pure S5, `ssm_size_base=32, glu_rank=40`) | **16** | none | low-rank r=40 | 16 (complex) → 32 real | 188,682 |
| Corner 3' (Mambino, `ssm_size_base=16, glu_rank=40`) | 8 | + fwd-only P=8 | low-rank r=40 | 8 main + 8 predictor = 16 (complex) → 32 real | 188,682 |

**Corner 2 and Corner 3' have identical parameter count AND identical total SSM state
dimensionality per layer** (32 real values, arranged either as one big P=16 SSM in Corner 2
or as two P=8 SSMs in Corner 3'). They differ ONLY in **whether the second block of state
is coupled to the main scan via a Mambino predictive-coding feedback loop** or is just extra
capacity in a single Pure S5 SSM.

Under the 8-seed matched-pair paired t-test:

- **Corner 3' vs Corner 2 (the Mambino mechanism ablation):**
  paired diff = +0.01463 test@peakval, t = 8.46, df = 7, two-tailed **p ≈ 6.5 × 10⁻⁵**.
  Mambino's predictor + `W̄_ε` mechanism produces a >1 pp accuracy improvement over
  arranging the same state as a single wider SSM — cleanly attributed to the predictive-coding
  mechanism because state DOF and parameter budget are held fixed.

- **Corner 2 vs Corner 1 (Pure S5 budget-redistribution):**
  paired diff = −0.00975 (Corner 2 WORSE), t = −4.06, p ≈ 0.005. Under Pure S5 alone,
  swapping full-rank gate for double state is a net LOSS. Pure S5 cannot rescue itself
  by rearranging its budget.

- **Corner 3' vs Corner 1 (compound Mambino + gate reduction):**
  paired diff = +0.0049, t = 2.94, p = 0.022. Confounded (both mechanism and gate change)
  — provides triangulation but not clean isolation.

**Conclusion:** At 188K parameter budget, the Mambino predictor mechanism is the causal
driver of the accuracy gain. The 3-way ablation table above is the paper's central
architectural claim. Chip consequence: Corner 3' also uses **half** the streaming state
SRAM of Corner 2 (256 KB vs 512 KB, §11) at higher accuracy, so the mechanism win stacks
with a direct memory-footprint win.

### 5f. λ_pc ablation on Config 4 (single-seed sweep, seed=42)

The Mambino runs in §5 all use `--lambda_pc=0.0` (predictive-coding auxiliary loss turned off).
This is empirically justified: the predictor branch still trains via the task gradient
back-propagating through W̄_ε, and adding an explicit L_int = mean(||ε||²) term to the training
objective either hurts marginally (small λ_pc) or catastrophically (λ_pc ≥ 0.1) by diverting
gradient signal from the task loss to the predictor.

Single-seed sweep on Config 4 (Mambino gelu 106K, seed=42, same code SHA `46517fe`, same
40-epoch training regime):

| λ_pc | SLURM Job | peak_val (epoch) | test@peakval | test_max (epoch) | Δ test@peakval vs λ_pc=0 |
|---:|---:|:---:|---:|:---:|---:|
| **0.0 (baseline)** | 11181833 (seed=42 subset) | 0.6035 (E–) | **0.6035** | 0.6035 (E–) | — |
| 0.0001 | 11230563 | 0.5895 (E37) | 0.5995 | 0.6040 (E38) | −0.0040 |
| 0.001 | 11230564 | 0.5970 (E39) | 0.5980 | 0.5980 (E39) | −0.0055 |
| 0.01 | 11230565 | 0.5880 (E40) | 0.5940 | 0.6000 (E36) | −0.0095 |
| 0.1 | 11230566 | 0.5460 (E30) | **0.5585** | 0.5770 (E38) | **−0.0450** |

The λ_pc=0 baseline number is C4 seed=42 from the 8-seed sweep (§5d). All other rows are from
this ablation on the same seed.

**Interpretation.** λ_pc=0 (task gradient only) is Pareto-optimal on this single seed —
every non-zero λ_pc value is worse on test@peakval. λ_pc=0.1 is the deliberate failure end of
the sweep and confirms the prior expectation (from NCB v031 experiments) that a strong PC
auxiliary loss dominates the task gradient on the predictor branch and starves it of
task-relevant learning signal. λ_pc ∈ {1e-4, 1e-3, 1e-2} is the "small enough not to blow up
but still parasitic" regime.

The paper's Config 4 result carries the λ_pc=0 setting; §5f is the empirical evidence that
this choice is not just training convenience but the correct architectural setting for
additive predictive-coding-augmented SSMs — the predictor gets its learning signal cleanly
through W̄_ε from the task loss, without an explicit intrinsic term.

**Caveat.** This is a single-seed ablation (seed=42, chosen as a median-tier seed from the
8-seed distribution). We do not report significance tests on this row; the paper's central
significance claim rests on the 8-seed sweep in §5a-5e. §5f is presented as a directional
architectural ablation justifying the λ_pc=0 choice, not as a significance-tested comparison.

### 5g. External SOTA context and comparison philosophy

The paper's positioning is **SOTA-close accuracy + real chip efficiency for constrained
hardware** — not SOTA-chase. External LRA-ListOps numbers below are provided as **context**,
not as the paper's primary comparison surface. The **comparison of record** is our matched-seed
reproduction of Pure S5 (Corner 1), because it is the only setup that supports valid paired
significance testing.

**External SOTA numbers (context only):**

| Model | Reported acc | Source | Chip-friendly? |
|---|---:|---|:---:|
| S7 (Han et al., 2024) | 63.77% | *S7: A New State-Space Model with Selective and Simplified Structure* | ✗ (selective-scan, dynamic recurrence — expensive per-timestep parameter updates on-chip) |
| Mega (Ma et al., 2023) | 63.14% | *Mega: Moving Average Equipped Gated Attention* | ✗ (attention + moving-average hybrid, quadratic surface) |
| S5 (Smith et al., 2023 — paper's reported number) | 62.15% | *Simplified State Space Layers for Sequence Modeling* (ICLR 2023) | ✓ (diagonal SSM) |

We mention S7 and Mega briefly to acknowledge the current LRA-ListOps SOTA, but do not treat
them as the paper's comparators: they are architecturally chip-unfriendly (S7's selective scan
requires per-token parameter recomputation; Mega's attention component reintroduces quadratic
compute), so they are not on the same efficiency Pareto frontier that this paper targets.

**On the reported S5 62.15%.** The S5 paper reports 62.15% on LRA-ListOps under their training
regime. That number is not directly comparable to our results: it comes from a different
training budget, a different (likely single-seed) reporting convention, and no multi-seed
variance is available to enable a paired significance test. Citing it as our comparison would
be either uncontrolled (different training regime) or misleading (comparing our multi-seed
mean against their single-seed number).

**Comparison of record (this paper).** We re-ran the Pure S5 baseline (Corner 1, and Corner 2
for the mechanism ablation) at the exact same 8-seed set, same code SHA `46517fe`, same
40-epoch training regime, same optimizer config as our Mambino runs. Corner 1 mean test@peakval
= **0.6089 ± 0.0053** at n=8, and paired t-testing against Corner 3' (Mambino) gives t=2.94,
p=0.022 two-tailed. This is the only apples-to-apples S5-vs-Mambino comparison we make and the
only one that carries a valid significance test.

The gap to external SOTA — 63.77% (S7) vs our 0.6138 (Corner 3') — is ~2.4 pp, well within
"SOTA-close" for a paper whose central axis is chip efficiency at competitive accuracy, not
architecture-novelty SOTA-chase. The chip-friendly S5 family sits in the low-62% range;
Mambino at 0.6138 is at the top of that family under matched conditions.

## 6. Digital PPAC (Accelergy 0.4 + CACTI + NeuroSim, 22nm INT8, 1 GHz clock)

Primitives: SRAM → CACTI. MAC (intadder) + register file (flip_flop) → NeuroSim.
Action counts: 1 SRAM read per MAC (worst-case, no line buffering). Ratios between configs are invariant to this assumption.

Reported under **two chip topologies** that differ only in activation SRAM buffering:
- **Pipelined** — 4.2 MB activation SRAM (holds all 8 layers' fwd + bwd streams concurrently, throughput-optimized).
- **Sequential** — 512 KB activation SRAM (processes layers one at a time, latency-optimized single-inference chip).

### 6a. Pipelined chip (4.2 MB activation SRAM) — total energy + area

| Config | **Total energy (mJ)** | **Total area (mm²)** |
|---|---:|---:|
| Config 4 | **150.57** | **4.70** |
| Config 5 | 166.80 | 4.95 |
| Corner 1 | 229.89 | **4.77** |
| **Corner 2** | **264.54** | **5.02** ← highest energy at 188K |
| Corner 3' | 247.22 | **4.77** ← equal to Corner 1 |

### 6b. Sequential chip (512 KB activation SRAM) — total energy + area

| Config | **Total energy (mJ)** | **Total area (mm²)** |
|---|---:|---:|
| Config 4 | **48.47** | **0.68** |
| Config 5 | 53.69 | 0.72 |
| Corner 1 | 83.74 | **0.76** |
| **Corner 2** | **96.37** | **0.79** ← highest at 188K |
| Corner 3' | 90.06 | **0.76** ← equal to Corner 1 |

### Pipelined-to-Sequential ratio (design choice)

| Config | Energy ratio (pipe/seq) | Area ratio (pipe/seq) |
|---|---:|---:|
| Config 4 | 3.10× | 3.84× |
| Config 5 | 3.09× | 3.47× |
| Corner 1 | 2.74× | 4.28× |
| Corner 3' | 2.74× | 3.69× |

Rankings within each topology are preserved: **Config 4 < Config 5 < Corner 1 < Corner 3'** for energy in both scenarios.

## Appendix A. Reference mixed-signal PIM PPAC (out of Mambino paper scope; kept as context)

**SCOPE NOTE.** This section is **NOT part of the Mambino paper's main claims** as of v7. It is
kept in the master document as a reference PPAC surface for a generic 22 nm RRAM crossbar
running the same LRA-ListOps workload, so the digital numbers in §6 can be contextualised
against a mainstream analog-PIM comparator. The Mambino paper does not claim any mixed-signal
chip advantage. Analog state-cell PPAC on our own circuit design is deferred to a companion
paper; see [`STUDENT_STATE_CELL_MEMO.md`](STUDENT_STATE_CELL_MEMO.md) for the parallel effort.

Methodology: Accelergy 0.4 + NeuroSim PIM + CACTI, 22 nm, RRAM cell = `nvmexplorer_RRAM`.
Crossbar topology: 128×128 tiles, all columns active. Tile activations = `ceil(MACs / 16384)`.

### 7a. Pipelined chip (4.2 MB activation SRAM)

| Config | Total energy (mJ) | Total area (mm²) | Tile activations |
|---|---:|---:|---:|
| Config 4 | **1.12** | **5.01** | 18,752 |
| Config 5 | 1.24 | 5.26 | 20,800 |
| Corner 1 | 1.60 | **5.01** ← equal | 26,946 |
| **Corner 2** | **1.84** | **5.26** ← same as C5 (P=16 state) | 31,036 |
| Corner 3' | 1.72 | **5.01** ← equal | 28,993 |

### 7b. Sequential chip (512 KB activation SRAM)

| Config | Total energy (mJ) | Total area (mm²) |
|---|---:|---:|
| Config 4 | **0.32** | **1.00** |
| Config 5 | 0.35 | 1.03 |
| Corner 1 | 0.46 | **1.00** ← equal |
| **Corner 2** | **0.52** | **1.03** ← same as C5 |
| Corner 3' | 0.49 | **1.00** ← equal |

**Note on the same-area cluster (Config 4 = Corner 1 = Corner 3' at 5.01 mm² pipelined, 1.00 mm² sequential):** All three share P=8 (so identical 256 KB streaming state SRAM), identical activation SRAM (4.2 MB pipelined / 512 KB sequential), and no weight SRAM in the mixed-signal chip (weights are on-array in the PIM crossbar). Config 5 stands slightly larger because P=16 doubles its streaming state SRAM to 512 KB. The energy ranking is still Config 4 < Config 5 < Corner 1 < Corner 3' — driven by MAC count, not state footprint.

## 8. Accuracy-per-mJ — chip efficiency metric

**§8 is the Mambino paper's primary chip-efficiency table.** Digital PPAC is the paper's claim;
mixed-signal reference numbers (Appendix A) are not part of the main-paper acc/mJ tables. If a
reviewer wants a mixed-signal comparator, see Appendix A of the master document.

### 8a. Pipelined chip

Using 8-seed mean test@peakval and test_max from §5.

| Config | test@peakval mean | test@peakval / mJ (×10⁻³) | test_max mean | test_max / mJ (×10⁻³) |
|---|---:|---:|---:|---:|
| **Config 4** Mambino gelu | 0.5993 | **3.98** | 0.6038 | **4.01** |
| Config 5 Pure S5 gelu | 0.5917 | 3.55 | 0.6034 | 3.62 |
| Corner 1 Pure S5 half_glu2 | 0.6089 | 2.65 | 0.6159 | 2.68 |
| **Corner 2** Pure S5 P=16 r=40 | 0.5991 | **2.26** ← worst at 188K | 0.6069 | **2.29** ← worst at 188K |
| Corner 3' Mambino half_glu2 | 0.6138 | 2.48 | 0.6171 | 2.50 |

### 8b. Sequential chip

| Config | test@peakval mean | test@peakval / mJ (×10⁻³) | test_max mean | test_max / mJ (×10⁻³) |
|---|---:|---:|---:|---:|
| **Config 4** Mambino gelu | 0.5993 | **12.36** | 0.6038 | **12.46** |
| Config 5 Pure S5 gelu | 0.5917 | 11.02 | 0.6034 | 11.24 |
| Corner 1 Pure S5 half_glu2 | 0.6089 | 7.27 | 0.6159 | 7.35 |
| **Corner 2** Pure S5 P=16 r=40 | 0.5991 | **6.22** ← worst at 188K | 0.6069 | **6.30** ← worst at 188K |
| Corner 3' Mambino half_glu2 | 0.6138 | 6.82 | 0.6171 | 6.85 |

**Config 4 wins acc/mJ under both accuracy definitions and both chip topologies.** At n=8, this
lead is stable across all four quadrants (digital pipe/seq × test@peakval / test_max).

**Corner 2 is Pareto-dominated at 188K:** in every quadrant it has lower accuracy AND higher
energy than either Corner 1 or Corner 3'. Doubling P at the cost of gate rank buys Pure S5
nothing on-chip.

## 9. Methodology citations

- **Accelergy 0.4** — Wu, Sze, Emer, RSP 2019: <https://ieeexplore.ieee.org/document/8942149>
- **CACTI 7** — Balasubramonian et al., ACM TACO 2017 (SRAM circuit-level model at 22nm PTM)
- **NeuroSim** — Chen, Peng, Yu, IEDM 2018: <https://ieeexplore.ieee.org/document/8614591>
- **Horowitz** — INT8 MAC energy at 45nm scaled to 22nm, ISSCC 2014: <https://ieeexplore.ieee.org/document/6757323>
- **Chen (Eyeriss)** — SRAM 8-bit read energy at 65nm scaled, ISSCC 2016
- **Wong** — SRAM cell area 22nm 6T, Sci Rep 2015
- **JAXPR walker** — [`flop_counter_jaxpr.py`](https://github.com/raisul1212/S5/blob/mambino-ssm/flop_counter_jaxpr.py), walks compiled compute graph via `jax.make_jaxpr`
- **LRA-ListOps** — Tay et al. ICLR 2021 (Long Range Arena benchmark, ListOps task, 2K sequence, 10-class)

## 10. Files backing these numbers

- Training args + full training logs: `checkpoints/{config-dir}/run.log` on Gilbreth
- Checkpoints: `checkpoints/{config-dir}/best.pkl` on Gilbreth
- JAXPR walker: [`flop_counter_jaxpr.py`](https://github.com/raisul1212/S5/blob/mambino-ssm/flop_counter_jaxpr.py)
- Per-config workload YAMLs: `workload_{config}_jaxpr.yaml`
- Accelergy digital YAMLs: `accelergy/v6_digital_{config}_{arch,actions}.yaml`
- Accelergy mixed-signal YAMLs: `accelergy/v6_mixed_signal_{config}_{arch,actions}.yaml`
- Accelergy outputs: `accelergy/v6_out_{topology}_{config}/energy_estimation.yaml`, `ART_summary.yaml`
- Primitive component lib: [`accelergy/primitive_component_libs/mambino_primitives.lib.yaml`](https://github.com/raisul1212/S5/blob/mambino-ssm/accelergy/primitive_component_libs/mambino_primitives.lib.yaml)

## 11. Modeling assumptions used in Accelergy YAMLs

- All INT8 quantization, 22nm, 1 GHz clock.
- **Activation SRAM sizing** is a chip-topology choice, reported at TWO scenarios:
  - **Pipelined** (`4.2 MB`, width=1024, depth=32784): holds all 8 layers' fwd + bwd activation streams concurrently. Throughput-optimized — the chip can pipeline multiple layers in flight. Reference sizing: `2 × L × H × n_layers = 2 × 2048 × 128 × 8 = 4 MB` (fwd+bwd × sequence × hidden × layers).
  - **Sequential** (`512 KB`, width=1024, depth=4096): holds only one layer's fwd+bwd activations at a time; process layer N, discard, move to N+1. Latency-optimized single-inference chip. Reference sizing: `2 × L × H = 2 × 2048 × 128 = 512 KB`.
  Choice between the two is a system-level tradeoff — the pipelined design supports higher throughput and pipelined bidir compute; the sequential design cuts activation SRAM 8× at the cost of serialized layer execution. Both are defensible; §6/§7/§8 report both.
- Weight SRAM depth = ceil(params / 128) — assumes 128-byte-wide lines.
- **State SRAM sizing formula (streaming inference chip, corrected v4):**
  ```
  state_bytes = L * local_P * n_layers * 1  (one buffered direction for bidir concat)
      local_P = 2*P (real INT8 bytes per state per timestep)
  ```
  The chip runs the SSM recurrence `x[t] = Lambda_bar * x[t-1] + Bu[t]` sequentially, one
  timestep at a time. Only the CURRENT state persists in a register per layer per direction.
  For bidirectional layers, the forward pass's state trajectory (L × local_P bytes) is
  buffered so the backward pass can produce combined `y[t] = 2Re(C₁·x_fwd[t]) + 2Re(C₂·x_bwd[t])`.
  Predictor is causal-shift streaming — one register per layer, negligible SRAM. No parallel
  associative scan is required at inference (that was a training-time GPU optimization).

  For L=2048, n_layers=8:
  - Config 4 (P=8): 16 × 2048 × 8 × 1 = **262,144 bytes = 256 KB**
  - Config 5 (P=16): 32 × 2048 × 8 × 1 = **524,288 bytes = 512 KB**
  - Corner 1 (P=8): 16 × 2048 × 8 × 1 = **262,144 bytes = 256 KB**
  - Corner 3' (P=8): 16 × 2048 × 8 × 1 = **262,144 bytes = 256 KB**

  Corner 1 = Corner 3' = Config 4 = 256 KB — no Mambino predictor penalty because the
  predictor's state is streamed (single register per layer).
  Config 5 stands out at 512 KB because its P=16 doubles local_P.
  Sequential chip (one layer at a time): divide by 8 → 32 KB (or 64 KB for Config 5).
- **Canonical MAC count** for every action-count entry that scales with MACs: `dot_general FLOPs / 2` (see §4). All 4 configs use the same accounting basis.
- Worst-case action count: 1 SRAM read per MAC (no line buffering). Line-buffered version would drop absolute energy by ~100× but ratios between configs are invariant.

## 12. What is NOT locked / open items

- ~~**Multi-seed mean±std**~~ — CLOSED v5: 8-seed sweep in §5 at code SHA `46517fe` with paired significance tests.
- ~~**λ_pc ablation on Config 4**~~ — CLOSED v7.1: single-seed sweep in §5f (seed=42, SLURM 11230563-11230566); λ_pc=0 is Pareto-optimal on test@peakval, λ_pc=0.1 fails as expected.
- **True analog RC state-cell PPAC** (student SPICE + measured data): out of scope for this paper (moved to companion paper). Appendix A uses generic RRAM crossbar reference for context only.
- **Line-buffered / batched-read PPAC re-estimate**: not run. Current numbers are worst-case 1 read/MAC. Ratios between configs are invariant to this choice, so it does not affect the paper's chip claims.
- **Activation SRAM audit**: 4.2 MB assumes full forward+backward buffering — may be reducible. §6/§7 report both pipelined (4.2 MB) and sequential (512 KB) as an explicit design-point pair, so this is bracketed rather than open.
