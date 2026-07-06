# Master Document: Training Runs + JAXPR FLOPs + Accelergy PPAC

Locked 2026-07-06 (v4 — streaming inference model + 5-seed multi-seed sweep).
All numbers from executed code + published methodologies.

**v4 changes from v3:** (1) State SRAM sizing corrected from parallel-scan trajectory to
streaming inference (§11); (2) All PPAC tables re-run (§6/§7); (3) §5 replaced with 5-seed
multi-seed statistics from unified current-code SHA `46517fe`; (4) §8 acc/mJ uses the
5-seed means.

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

## 5. Test accuracies — 5-seed multi-seed sweep at current code SHA `46517fe`

Two definitions per Fable convention:
- **`test@peakval`**: test_acc at the epoch that had the highest validation accuracy. This is the standard leakage-free metric.
- **`test_max`**: overall highest test_acc seen across all 40 epochs. This is an upper bound; useful for showing what the model can do.

Seeds: {6554595, 42, 12345, 271828, 314159}. All 20 runs COMPLETED at 40 epochs.

| Config | test@peakval mean±std | test_max mean±std |
|---|---:|---:|
| Config 4 Mambino gelu | **0.5999 ± 0.0081** | 0.6033 ± 0.0069 |
| Config 5 Pure S5 gelu | 0.5924 ± 0.0067 | 0.6033 ± 0.0058 |
| Corner 1 Pure S5 half_glu2 | 0.6107 ± 0.0047 | 0.6153 ± 0.0047 |
| Corner 3' Mambino half_glu2 | **0.6143 ± 0.0030** | **0.6186 ± 0.0050** |

### Iso-params deltas

**106K (Config 4 vs Config 5):** Mambino wins test@peakval by +0.75 pp; ties on test_max.

**188K (Corner 3' vs Corner 1):** Mambino wins test@peakval by +0.36 pp AND test_max by +0.33 pp AND has **36% lower seed variance** (0.30% vs 0.47%).

### Ranking is stable across both metrics
Corner 3' > Corner 1 > Config 4 > Config 5 (test@peakval).

### Per-seed table (raw)

| Config | seed 6554595 | seed 42 | seed 12345 | seed 271828 | seed 314159 |
|---|:---:|:---:|:---:|:---:|:---:|
| Config 4 test@peakval | 0.5940 | 0.6035 | 0.6100 | 0.5875 | 0.6045 |
| Config 5 test@peakval | 0.5830 | 0.6010 | 0.5930 | 0.5980 | 0.5870 |
| Corner 1 test@peakval | 0.6080 | 0.6200 | 0.6090 | 0.6070 | 0.6095 |
| Corner 3' test@peakval | 0.6180 | 0.6165 | 0.6150 | 0.6125 | 0.6095 |

Note: seed=6554595 originals (SHA `a32cb6bb`/`7fe1aa43`/`7343b68b`) were rerun at current SHA `46517fe` and appear here; original single-seed accuracies from those older SHAs differed slightly due to code drift in `s5/ssm.py`, `s5/mambino_ssm.py`, `s5/layers.py` between then and now. All numbers reported in this doc are from the unified current-code sweep.

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
| Corner 3' | 247.22 | **4.77** ← equal to Corner 1 |

### 6b. Sequential chip (512 KB activation SRAM) — total energy + area

| Config | **Total energy (mJ)** | **Total area (mm²)** |
|---|---:|---:|
| Config 4 | **48.47** | **0.68** |
| Config 5 | 53.69 | 0.72 |
| Corner 1 | 83.74 | **0.76** |
| Corner 3' | 90.06 | **0.76** ← equal to Corner 1 |

### Pipelined-to-Sequential ratio (design choice)

| Config | Energy ratio (pipe/seq) | Area ratio (pipe/seq) |
|---|---:|---:|
| Config 4 | 3.10× | 3.84× |
| Config 5 | 3.09× | 3.47× |
| Corner 1 | 2.74× | 4.28× |
| Corner 3' | 2.74× | 3.69× |

Rankings within each topology are preserved: **Config 4 < Config 5 < Corner 1 < Corner 3'** for energy in both scenarios.

## 7. Mixed-signal PIM PPAC (Accelergy 0.4 + NeuroSim PIM + CACTI, 22nm)

**Caveat:** NeuroSim PIM with `nvmexplorer_RRAM` reference cell — represents a **generic 22nm RRAM analog crossbar**, NOT this paper's true analog RC state cell. See [`STUDENT_STATE_CELL_MEMO.md`](https://github.com/raisul1212/S5/blob/mambino-ssm/accelergy/STUDENT_STATE_CELL_MEMO.md) for characterization data needed to replace the RRAM reference with the actual state-cell PPAC.

Crossbar topology: 128×128 tiles, all columns active. Tile activations = `ceil(MACs / 16384)`.

### 7a. Pipelined chip (4.2 MB activation SRAM)

| Config | Total energy (mJ) | Total area (mm²) | Tile activations |
|---|---:|---:|---:|
| Config 4 | **1.12** | **5.01** | 18,752 |
| Config 5 | 1.24 | 5.26 | 20,800 |
| Corner 1 | 1.60 | **5.01** ← equal | 26,946 |
| Corner 3' | 1.72 | **5.01** ← equal | 28,993 |

### 7b. Sequential chip (512 KB activation SRAM)

| Config | Total energy (mJ) | Total area (mm²) |
|---|---:|---:|
| Config 4 | **0.32** | **1.00** |
| Config 5 | 0.35 | 1.03 |
| Corner 1 | 0.46 | **1.00** ← equal |
| Corner 3' | 0.49 | **1.00** ← equal |

**Note on the same-area cluster (Config 4 = Corner 1 = Corner 3' at 5.01 mm² pipelined, 1.00 mm² sequential):** All three share P=8 (so identical 256 KB streaming state SRAM), identical activation SRAM (4.2 MB pipelined / 512 KB sequential), and no weight SRAM in the mixed-signal chip (weights are on-array in the PIM crossbar). Config 5 stands slightly larger because P=16 doubles its streaming state SRAM to 512 KB. The energy ranking is still Config 4 < Config 5 < Corner 1 < Corner 3' — driven by MAC count, not state footprint.

## 8. Accuracy-per-mJ (digital) — chip efficiency metric

### 8a. Pipelined chip

Using 5-seed mean test@peakval and test_max from §5.

| Config | test@peakval mean | test@peakval / mJ (×10⁻³) | test_max mean | test_max / mJ (×10⁻³) |
|---|---:|---:|---:|---:|
| **Config 4** Mambino gelu | 0.5999 | **3.98** | 0.6033 | **4.01** |
| Config 5 Pure S5 gelu | 0.5924 | 3.55 | 0.6033 | 3.62 |
| Corner 1 Pure S5 half_glu2 | 0.6107 | 2.66 | 0.6153 | 2.68 |
| Corner 3' Mambino half_glu2 | 0.6143 | 2.48 | 0.6186 | 2.50 |

### 8b. Sequential chip

| Config | test@peakval mean | test@peakval / mJ (×10⁻³) | test_max mean | test_max / mJ (×10⁻³) |
|---|---:|---:|---:|---:|
| **Config 4** Mambino gelu | 0.5999 | **12.38** | 0.6033 | **12.44** |
| Config 5 Pure S5 gelu | 0.5924 | 11.03 | 0.6033 | 11.24 |
| Corner 1 Pure S5 half_glu2 | 0.6107 | 7.29 | 0.6153 | 7.35 |
| Corner 3' Mambino half_glu2 | 0.6143 | 6.82 | 0.6186 | 6.87 |

**Config 4 wins acc/mJ under both accuracy definitions and both chip topologies.**

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

- **Multi-seed mean±std** (jobs 11187522–11187541): currently held. Single-seed accuracies at ±1.5 pp separation (Corner 1 vs Corner 3') are within seed noise. Do not treat this doc as final until multi-seed lands.
- **True analog RC state-cell PPAC** (student SPICE + measured data): pending. Mixed-signal PPAC currently uses generic RRAM crossbar reference.
- **Line-buffered / batched-read PPAC re-estimate**: not run. Current numbers are worst-case 1 read/MAC.
- **Activation SRAM audit**: 4.2 MB assumes full forward+backward buffering — may be reducible.
