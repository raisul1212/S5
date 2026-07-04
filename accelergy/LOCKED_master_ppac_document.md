# Master Document: Training Runs + JAXPR FLOPs + Accelergy PPAC

Locked 2026-07-03. All numbers from executed code + published methodologies.

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

**Raw data on Purdue Gilbreth cluster** (not publicly linkable, but paths listed for reproducibility):

- Training logs + checkpoints: `~/dev/ssm-baselines/S5/checkpoints/{config-dir}/{run.log,best.pkl}`
- JAXPR walker workload YAMLs: `~/dev/ssm-baselines/S5/workload_{config}_jaxpr.yaml`
- Accelergy outputs: `~/dev/ssm-baselines/S5/accelergy/v6_out_{topology}_{config}/{energy_estimation.yaml,ART_summary.yaml,ERT.yaml}`
- Accelergy s5m conda env: `/scratch/gilbreth/raisul/envs/s5m`

**Companion locked docs:**
- [`LOCKED_digital_ppac_results.md`](https://github.com/raisul1212/S5/blob/mambino-ssm/accelergy/LOCKED_digital_ppac_results.md)
- [`LOCKED_mixed_signal_reference.md`](https://github.com/raisul1212/S5/blob/mambino-ssm/accelergy/LOCKED_mixed_signal_reference.md)
- [`STUDENT_STATE_CELL_MEMO.md`](https://github.com/raisul1212/S5/blob/mambino-ssm/accelergy/STUDENT_STATE_CELL_MEMO.md)

## 1. Training run registry

| Config | SLURM Job Name | Job ID | State | sbatch script | Checkpoint dir |
|---|---|---:|---|---|---|
| Config 4 | `chip-mamb-gelu` | 11181833 | COMPLETED | `bin/gilbreth_chip_mambino_gelu.sh` | `checkpoints/chip_mamb_gelu_11181833/` |
| Config 5 | `tr-pure-s5-gelu-p16` | 11187128 | COMPLETED | `bin/train_pure_s5_gelu_p16.sh` | `checkpoints/train_pure_s5_gelu_p16_11187128/` |
| Corner 1 | `chip-mamb-iso` (submitted with Corner 3') | 11181830 | COMPLETED | `bin/gilbreth_chip_pure_s5.sh` | `checkpoints/chip_pure_s5_11181830/` |
| Corner 3' | `chip-mamb-iso` | 11181831 | COMPLETED | `bin/gilbreth_chip_mambino_iso.sh` | `checkpoints/chip_mamb_iso_11181831/` |

## 2. Training args (differences from shared base)

Shared base (all 4): `--n_layers=8 --d_model=128 --blocks=8 --bidirectional=True --batchnorm=True --dataset=listops-classification --epochs=40 --bsz=50 --dt_min=0.001 --dt_max=0.1 --C_init=lecun_normal --opt_config=BfastandCdecay --p_dropout=0 --ssm_lr_base=0.001 --lr_factor=3 --warmup_end=1 --weight_decay=0.04`

| Config | use_mambino_ssm | activation_fn | ssm_size_base | glu_rank | Extras |
|---|---|---|---:|---:|---|
| Config 4 | True | gelu | 16 | 0 | — |
| Config 5 | False | gelu | 32 | 0 | — |
| Corner 1 | False | half_glu2 | 16 | 0 | — |
| Corner 3' | True | half_glu2 | 16 | 40 | `--lambda_pc=0.0` |

## 3. Model shapes

For each config with `conj_sym=True`: `P = ssm_size_base // 2`, `local_P (real) = 2 * P`.

| Config | P | local_P (real bytes) | # SSM traces per layer | Effective state size per timestep per layer |
|---|---:|---:|---:|---:|
| Config 4 (Mambino) | 8 | 16 | 2 (main + predictor) | 32 bytes |
| Config 5 (Pure S5) | 16 | 32 | 1 (main) | 32 bytes |
| Corner 1 (Pure S5) | 8 | 16 | 1 (main) | 16 bytes |
| Corner 3' (Mambino) | 8 | 16 | 2 (main + predictor) | 32 bytes |

## 4. JAXPR-verified FLOP counts (workload = LRA-ListOps L=2048, batch=1)

Computed by `flop_counter_jaxpr.py` walking the compiled JAXPR of the actual forward pass. Complex arithmetic weighted as: real MAC = 2 FLOPs, complex×real MAC = 4 FLOPs, complex×complex MAC = 8 FLOPs.

| Config | Params | Total FLOPs | dot_general FLOPs | Real-MAC equivalent (dot_general / 2) |
|---|---:|---:|---:|---:|
| Config 4 | 105,738 | 682,095,196 | 614,468,096 | 307,234,048 |
| Config 5 | 105,738 | 745,208,540 | 681,576,960 | 340,788,480 |
| Corner 1 | 188,490 | 948,670,108 | 882,903,552 | 441,451,776 |
| Corner 3' | 188,682 | 1,231,549,020 | 950,012,416 | 475,006,208 |

## 5. Test accuracies

| Config | Best test acc | Epoch | Source |
|---|---:|---:|---|
| Config 4 | 0.6055 | (from best.pkl) | `chip_mamb_gelu_11181833/best.pkl` |
| Config 5 | 0.5830 | (from best.pkl) | `train_pure_s5_gelu_p16_11187128/best.pkl` |
| Corner 1 | **0.6155** | 37 (peak 0.6160 at ep 32) | `chip_pure_s5_11181830/best.pkl` |
| Corner 3' | 0.6140 | 35 | `chip_mamb_iso_11181831/best.pkl` |

All single-seed. Multi-seed reruns (jobs 11187522-11187541) currently held.

## 6. Digital PPAC (Accelergy 0.4 + CACTI + NeuroSim, 22nm INT8, 1 GHz clock)

Primitives: SRAM → CACTI. MAC (intadder) + register file (flip_flop) → NeuroSim.
State SRAM sized to match §3: Mambino carries 2 state trajectories.

| Config | Weight SRAM (mJ) | Activation SRAM (mJ) | State SRAM (mJ) | MAC (μJ) | Reg (μJ) | **Total energy** | **Total area** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Config 4 | 12.31 | 154.58 | 0.68 | 2.93 | 27.85 | **167.64 mJ** | **6.38 mm²** |
| Config 5 | 13.45 | 168.71 | 0.96 | 3.20 | 30.43 | **183.14 mJ** | **6.38 mm²** |
| Corner 1 | 30.33 | 199.52 | 0.29 | 3.79 | 36.05 | **230.18 mJ** | **5.39 mm²** |
| Corner 3' | 32.64 | 214.54 | 0.68 | 4.07 | 38.79 | **247.94 mJ** | **6.45 mm²** |

Digital area breakdown (μm²):

| Config | weight_sram | activation_sram | state_sram | Total (mm²) |
|---|---:|---:|---:|---:|
| Config 4 | 89,020 | 4,353,200 | 1,938,200 | 6.38 |
| Config 5 | 89,020 | 4,353,200 | 1,938,200 | 6.38 |
| Corner 1 | 160,889 | 4,353,200 | 878,057 | 5.39 |
| Corner 3' | 160,889 | 4,353,200 | 1,938,200 | 6.45 |

## 7. Mixed-signal PIM PPAC (Accelergy 0.4 + NeuroSim PIM + CACTI, 22nm)

**Caveat:** NeuroSim PIM with `nvmexplorer_RRAM` reference cell — represents a **generic 22nm RRAM analog crossbar**, NOT this paper's true analog RC state cell. See `STUDENT_STATE_CELL_MEMO.md` for what's needed to replace the RRAM reference with actual state-cell characterization.

Primitives: PIM row/col drivers + memory cell + ADC → NeuroSim. Activation + state SRAM → CACTI. Accumulator adder → NeuroSim. Register file → NeuroSim.

| Config | **Total energy** | **Total area** |
|---|---:|---:|
| Config 4 | **1.95 mJ** | **6.69 mm²** |
| Config 5 | **2.31 mJ** | **6.69 mm²** |
| Corner 1 | **1.89 mJ** | **5.63 mm²** |
| Corner 3' | **2.43 mJ** | **6.69 mm²** |

## 8. Accuracy-per-mJ (digital) — chip efficiency metric

| Config | Test Acc | Digital mJ | Acc / mJ (× 10⁻³) |
|---|---:|---:|---:|
| Config 4 | 0.6055 | 167.64 | 3.61 |
| Config 5 | 0.5830 | 183.14 | 3.18 |
| Corner 1 | 0.6155 | 230.18 | 2.67 |
| Corner 3' | 0.6140 | 247.94 | 2.48 |

## 9. Methodology citations

- **Accelergy 0.4** — Wu, Sze, Emer, RSP 2019: <https://ieeexplore.ieee.org/document/8942149>
- **CACTI 7** — Balasubramonian et al., ACM TACO 2017 (SRAM circuit-level model at 22nm PTM)
- **NeuroSim** — Chen, Peng, Yu, IEDM 2018: <https://ieeexplore.ieee.org/document/8614591>
- **Horowitz** — INT8 MAC energy at 45nm scaled to 22nm, ISSCC 2014: <https://ieeexplore.ieee.org/document/6757323>
- **Chen (Eyeriss)** — SRAM 8-bit read energy at 65nm scaled, ISSCC 2016
- **Wong** — SRAM cell area 22nm 6T, Sci Rep 2015
- **JAXPR walker** — `flop_counter_jaxpr.py`, walks compiled compute graph via `jax.make_jaxpr`
- **LRA-ListOps** — Tay et al. ICLR 2021 (Long Range Arena benchmark, ListOps task, 2K sequence, 10-class)

## 10. Files backing these numbers

- Training args + full training logs: `checkpoints/{config-dir}/run.log` on Gilbreth
- Checkpoints: `checkpoints/{config-dir}/best.pkl` on Gilbreth
- JAXPR walker: `flop_counter_jaxpr.py`
- Per-config workload YAMLs: `workload_{config}_jaxpr.yaml`
- Accelergy digital YAMLs: `accelergy/v6_digital_{config}_{arch,actions}.yaml`
- Accelergy mixed-signal YAMLs: `accelergy/v6_mixed_signal_{config}_{arch,actions}.yaml`
- Accelergy outputs: `accelergy/v6_out_{topology}_{config}/energy_estimation.yaml`, `ART_summary.yaml`
- Primitive component lib: `accelergy/primitive_component_libs/mambino_primitives.lib.yaml`

## 11. Modeling assumptions used in Accelergy YAMLs

- All INT8 quantization, 22nm, 1 GHz clock.
- Activation SRAM sized to 4.2 MB (holds concat of forward + backward pass activations at H=128 for L=2048 × n_layers=8 minus overlap).
- Weight SRAM depth = ceil(params / 128) — assumes 128-byte-wide lines.
- State SRAM sized to `local_P × 2 × L × n_layers × 2 (bidir)` per config, doubled for Mambino to account for main + predictor state trajectories.
- Worst-case action count: 1 SRAM read per MAC (no line buffering). Line-buffered version would drop absolute energy by ~100× but ratios between configs are invariant.

## 12. What is NOT locked / open items

- Multi-seed mean±std (multi-seed jobs currently held): not run yet.
- True analog RC state-cell PPAC (student SPICE data): pending.
- Line-buffered / batched-read PPAC re-estimate: not run yet.
