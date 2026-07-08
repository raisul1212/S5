# Mambino paper — training + PPAC environment freeze

Frozen 2026-07-08 for archival integrity. All numbers in
[`LOCKED_master_ppac_document.md`](LOCKED_master_ppac_document.md) were produced under this
environment on Purdue's Gilbreth cluster.

## Training environment (Gilbreth `s5m` conda env at `/scratch/gilbreth/raisul/envs/s5m`)

**Interpreter:**
- Python 3.11.15 (GCC 14.3.0)

**Core numerical stack:**
- JAX 0.4.30
- jaxlib 0.4.30
- NumPy 1.26.4
- Flax 0.8.5
- chex 0.1.86
- einops 0.8.2

**Full pip freeze:** [`mambino_pip_freeze.txt`](mambino_pip_freeze.txt) — 111 packages.

**JAX runtime info:** [`mambino_jax_env.txt`](mambino_jax_env.txt).

## Hardware (SLURM compute nodes)

Every training job in §1 of the master doc used the `a30` partition on Gilbreth:

```
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
```

Node fleet (as reported by `nvidia-smi` on the front-end at time of freeze):

| Component | Value |
|---|---|
| GPU | NVIDIA A30 (single-GPU per job) |
| GPU memory | 24 576 MiB HBM |
| Driver | 590.48.01 |
| CUDA runtime | 13.1 |
| Host CPU | AMD EPYC-class (Gilbreth partition) |
| Host RAM per job | 16 GB |
| Host OS | Rocky/RHEL 9.7 (kernel 5.14.0-611.16.1.el9_7.x86_64) |
| Wall-clock per training run | ~1.5–2.5 h for 40 epochs LRA-ListOps L=2048 |

## PPAC / Accelergy toolchain

Installed from local source under `/home/raisul/accelergy_setup/`:
- accelergy (upstream 0.4)
- accelergy-cacti-plug-in (CACTI 7 for SRAM)
- accelergy-neurosim-plug-in (INT8 MAC via intadder + flip_flop)
- accelergy-aladdin-plug-in
- accelergy-adc-plug-in (mixed-signal reference only, Appendix A)

Cell library: [`accelergy/primitive_component_libs/mambino_primitives.lib.yaml`](primitive_component_libs/mambino_primitives.lib.yaml).

## Reproducibility notes

- **Determinism.** All runs use `--jax_seed=<int>` which fixes JAX's PRNG key. Reproducing bit-identical numbers on a different GPU / different cuDNN / different JAX release is not expected due to floating-point non-associativity in reductions and cuDNN kernel selection. Expected drift: ≤ 0.3 pp on test@peakval when the code is unchanged but the accelerator stack differs.
- **Random seeds used** for the 8-seed sweep in §5: `{6554595, 42, 12345, 271828, 314159, 1, 2, 3}`.
- **Seed used** for the λ_pc single-seed ablation in §5f: `42` (median-tier rank in the 8-seed distribution; see §5d).
- **JAX version pinning.** JAX 0.4.30 was current at freeze time. Future JAX releases may change compiled JAXPR shapes; the JAXPR walker output ([`workload_*.yaml`](../workload_config4_jaxpr.yaml)) should be treated as frozen.

## Companion artifacts

- Frozen code: git tag `mambino-paper-v1` at SHA `46517fe` (multi-seed sweep) plus subsequent analysis-only commits. See `git log mambino-paper-v1` for the exact bundle.
- Checkpoints, run.log files, and Accelergy input+output YAMLs: PURR deposit at DOI **`[TODO: populate after upload]`**.
