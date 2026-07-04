# Master Document: Training Runs + JAXPR FLOPs + Accelergy PPAC

Locked 2026-07-03 (v2 — corrections applied per independent review).
All numbers from executed code + published methodologies.

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

## 5. Test accuracies (single-seed, from best.pkl and full run.log)

Two definitions per Fable convention:
- **`test@peakval`**: test_acc at the epoch that had the highest validation accuracy. This is the standard leakage-free metric.
- **`test_max`**: overall highest test_acc seen across all 40 epochs. This is an upper bound; useful for showing what the model can do.

| Config | peak_val | **test@peakval** | **test_max** | Source |
|---|---:|---:|---:|---|
| Config 4 Mambino gelu | 0.6025 | **0.6055** | **0.6095** | `chip_mamb_gelu_11181833/run.log` |
| Config 5 Pure S5 gelu | 0.5845 | **0.5830** | **0.6000** | `train_pure_s5_gelu_p16_11187128/run.log` |
| Corner 1 Pure S5 half_glu2 | 0.6085 | **0.6155** | **0.6250** | `chip_pure_s5_11181830/run.log` |
| Corner 3' Mambino half_glu2 | 0.6005 | **0.6140** | **0.6205** | `chip_mamb_iso_11181831/run.log` |

All single-seed. Multi-seed reruns (jobs 11187522-11187541) currently held.

## 6. Digital PPAC (Accelergy 0.4 + CACTI + NeuroSim, 22nm INT8, 1 GHz clock)

Primitives: SRAM → CACTI. MAC (intadder) + register file (flip_flop) → NeuroSim.
Action counts: 1 SRAM read per MAC (worst-case, no line buffering). Ratios between configs are invariant to this assumption.

Reported under **two chip topologies** that differ only in activation SRAM buffering:
- **Pipelined** — 4.2 MB activation SRAM (holds all 8 layers' fwd + bwd streams concurrently, throughput-optimized).
- **Sequential** — 512 KB activation SRAM (processes layers one at a time, latency-optimized single-inference chip).

### 6a. Pipelined chip (4.2 MB activation SRAM) — total energy + area

| Config | weight_sram | activation_sram | state_sram | mac+reg | **Total energy (mJ)** | **Total area (mm²)** |
|---|---:|---:|---:|---:|---:|---:|
| Config 4 | 11.09 mJ | 139.45 mJ | 0.14 mJ | 27.7 μJ | **150.70** | **5.12** |
| Config 5 | 12.30 mJ | 154.47 mJ | 0.39 mJ | 30.8 μJ | **167.18** | **5.32** |
| Corner 1 | 30.33 mJ | 199.52 mJ | 0.11 mJ | 39.8 μJ | **230.00** | **4.94** |
| Corner 3' | 32.64 mJ | 214.54 mJ | 0.14 mJ | 42.9 μJ | **247.36** | **5.19** |

### 6b. Sequential chip (512 KB activation SRAM) — total energy + area

| Config | **Total energy (mJ)** | **Total area (mm²)** |
|---|---:|---:|
| Config 4 | **48.60** | **1.34** |
| Config 5 | **54.07** | **1.53** |
| Corner 1 | **83.85** | **1.16** |
| Corner 3' | **90.19** | **1.41** |

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
| Config 4 | **1.25** | **5.44** | 18,752 |
| Config 5 | **1.62** | **5.63** | 20,800 |
| Corner 1 | **1.71** | **5.18** | 26,946 |
| Corner 3' | **1.85** | **5.44** | 28,993 |

### 7b. Sequential chip (512 KB activation SRAM)

| Config | Total energy (mJ) | Total area (mm²) |
|---|---:|---:|
| Config 4 | **0.46** | **1.65** |
| Config 5 | **0.74** | **1.85** |
| Corner 1 | **0.57** | **1.40** |
| Corner 3' | **0.63** | **1.65** |

## 8. Accuracy-per-mJ (digital) — chip efficiency metric

### 8a. Pipelined chip (4.2 MB activation SRAM)

| Config | test@peakval | test@peakval / mJ (×10⁻³) | test_max | test_max / mJ (×10⁻³) |
|---|---:|---:|---:|---:|
| **Config 4** Mambino gelu | 0.6055 | **4.02** | 0.6095 | **4.04** |
| Config 5 Pure S5 gelu | 0.5830 | 3.49 | 0.6000 | 3.59 |
| Corner 1 Pure S5 half_glu2 | 0.6155 | 2.68 | 0.6250 | 2.72 |
| Corner 3' Mambino half_glu2 | 0.6140 | 2.48 | 0.6205 | 2.51 |

### 8b. Sequential chip (512 KB activation SRAM)

| Config | test@peakval | test@peakval / mJ (×10⁻³) | test_max | test_max / mJ (×10⁻³) |
|---|---:|---:|---:|---:|
| **Config 4** Mambino gelu | 0.6055 | **12.46** | 0.6095 | **12.54** |
| Config 5 Pure S5 gelu | 0.5830 | 10.78 | 0.6000 | 11.10 |
| Corner 1 Pure S5 half_glu2 | 0.6155 | 7.34 | 0.6250 | 7.45 |
| Corner 3' Mambino half_glu2 | 0.6140 | 6.81 | 0.6205 | 6.88 |

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
- **State SRAM sizing formula:**
  ```
  state_bytes = local_P * L * n_layers * n_traj
      local_P    = 2*P (real INT8 bytes per state per timestep)
      n_traj     = 2 for Pure S5 (main fwd + main bwd)
                   3 for Mambino (main fwd + main bwd + predictor fwd)
  ```
  For L=2048, n_layers=8:
  - Config 4 (P=8, Mambino): 16 × 2048 × 8 × 3 = **786,432 bytes = 768 KB**
  - Config 5 (P=16, Pure S5): 32 × 2048 × 8 × 2 = **1,048,576 bytes = 1024 KB**
  - Corner 1 (P=8, Pure S5): 16 × 2048 × 8 × 2 = **524,288 bytes = 512 KB**
  - Corner 3' (P=8, Mambino): 16 × 2048 × 8 × 3 = **786,432 bytes = 768 KB**
- **Canonical MAC count** for every action-count entry that scales with MACs: `dot_general FLOPs / 2` (see §4). All 4 configs use the same accounting basis.
- Worst-case action count: 1 SRAM read per MAC (no line buffering). Line-buffered version would drop absolute energy by ~100× but ratios between configs are invariant.

## 12. What is NOT locked / open items

- **Multi-seed mean±std** (jobs 11187522–11187541): currently held. Single-seed accuracies at ±1.5 pp separation (Corner 1 vs Corner 3') are within seed noise. Do not treat this doc as final until multi-seed lands.
- **True analog RC state-cell PPAC** (student SPICE + measured data): pending. Mixed-signal PPAC currently uses generic RRAM crossbar reference.
- **Line-buffered / batched-read PPAC re-estimate**: not run. Current numbers are worst-case 1 read/MAC.
- **Activation SRAM audit**: 4.2 MB assumes full forward+backward buffering — may be reducible.
