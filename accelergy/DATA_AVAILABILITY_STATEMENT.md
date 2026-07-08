# Data Availability + Reproducibility Statement

**Paper title:** *Mambino: Predictive-Coding-Augmented State-Space Models for Efficient Long-Range Reasoning on Constrained Hardware*

Paste-ready text for the paper's Data Availability / Reproducibility section.

**PURR DOI (populated 2026-07-08):** [10.4231/9ADT-WP13](https://doi.org/10.4231/9ADT-WP13)

---

## Full statement (long form — for appendix or full-length venues)

**Code.** All training, evaluation, PPAC-analysis, and figure-generation code is
publicly available at <https://github.com/raisul1212/S5>, tagged
`mambino-paper-v1` (commit `0a9f0c9`, built on training-code SHA `46517fe`
which produced the 8-seed sweep results reported in §5). The tag additionally
includes the paper's LOCKED master PPAC document, the Accelergy input and
output YAMLs for all 5 configurations under both pipelined and sequential
chip topologies, and the frozen conda environment specification.

**Trained checkpoints, run logs, and per-run accuracy manifest.** Available at
Purdue University Research Repository (PURR),
DOI [10.4231/9ADT-WP13](https://doi.org/10.4231/9ADT-WP13).
The deposit contains 54 training-run directories (~345 MB uncompressed / 242 MB
compressed): 40 runs from the 5-config × 8-seed sweep backing §5, 6 runs from
the λ_pc single-seed ablation backing §5f, and 8 runs from single-seed
exploration. Each run directory holds the best-validation and final Flax
checkpoints (`best.pkl`, `final.pkl`), the msgpack-only variants, per-checkpoint
metadata (epoch + args), and the full stdout `run.log` with per-epoch train,
validation, and test accuracies. The deposit is accompanied by
`manifest.jsonl` cross-referencing each SLURM job ID to (config, seed,
test@peakval, test_max, `matches_5d` flag) and a `hashes.txt` file with
SHA-256 tamper-evident hashes of every `best.pkl`, `final.pkl`, and `run.log`.

**Environment.** All training runs were executed on Purdue's Gilbreth cluster,
`a30` SLURM partition (NVIDIA A30 24 GB HBM, one GPU per SLURM job, 4 CPU cores,
16 GB host RAM), running Rocky Linux 9.7 with the CUDA 13.1 driver stack.
The Python environment used JAX 0.4.30 + jaxlib 0.4.30 on Python 3.11.15;
a full 111-package pip freeze is included under the git tag as
`accelergy/mambino_pip_freeze.txt`. Wall-clock is approximately 1.5–2.5 h
per training run at 40 epochs for LRA-ListOps L=2048.

**Randomness and determinism.** All runs use JAX's PRNG seeded via
`--jax_seed=<int>`. The 8-seed sweep uses seeds
`{6554595, 42, 12345, 271828, 314159, 1, 2, 3}`; the λ_pc single-seed ablation
in §5f uses `seed=42` (median-tier rank in the 8-seed distribution).
Bit-identical rerun on a different accelerator stack is not expected because of
floating-point non-associativity in reductions and cuDNN kernel selection.
Expected drift when reproducing on a different GPU / cuDNN version is
≤ 0.3 pp on test@peakval; larger drifts should be reported as a discrepancy.

**PPAC methodology.** Digital PPAC (paper's primary chip claim) uses Accelergy
0.4 with CACTI 7 for SRAM (weight, activation, state buffers) and NeuroSim for
INT8 MAC and register-file primitives, at 22 nm 1 GHz. Mixed-signal PIM PPAC
(Appendix A, reference-only context) uses Accelergy's NeuroSim-PIM plugin with
`nvmexplorer_RRAM` cell config on 128 × 128 tiles. The MAC-count basis for both
methodologies is JAXPR-walked `dot_general` divided by 2 (real-MAC equivalent
after accounting for the complex-arithmetic weighting used in state-space
kernels); see §4 of the master PPAC document under the git tag.

**How to reproduce a specific number.** Every number in §5–§8 of the paper is
traceable to (a) the code at the git tag, (b) the SLURM job ID listed in the
run manifest, and (c) the environment freeze under the git tag. Verification
protocol for a claimed number:
1. Download the deposit and unpack.
2. Verify the tarball SHA-256 matches the value published under the git tag
   (`SHA256SUMS.txt` in the deposit).
3. Locate the run directory in `checkpoints/` by SLURM job ID.
4. Cross-check the run's `run.log` peak-validation and test accuracy against
   the corresponding row of paper §5d and the deposit's `manifest.jsonl`.

---

## Short statement (for space-constrained venues; e.g. two-column conference)

The Mambino source code, LOCKED PPAC master document, Accelergy input/output
YAMLs, and Python environment freeze are available at
<https://github.com/raisul1212/S5> at git tag `mambino-paper-v1`. Trained
checkpoints, full run logs, per-run manifest, and SHA-256 hashes for all 54
training runs (5 configurations × 8 seeds + λ_pc ablation + exploration runs)
are archived at Purdue University Research Repository, DOI
[10.4231/9ADT-WP13](https://doi.org/10.4231/9ADT-WP13). Runs used
NVIDIA A30 GPUs on Purdue Gilbreth, Python 3.11 + JAX 0.4.30. Random seeds:
`{6554595, 42, 12345, 271828, 314159, 1, 2, 3}`.

---

## Upload checklist for PURR

Files to upload (from `C:\Users\raisul\dev\nc-block\Neurocognitive Block\mambino_paper_purr_deposit\`):

1. **`mambino-paper-v1-purr-deposit.tar.gz`** — 242 MB, primary artifact
2. **`README.md`** — deposit-level documentation (also inside the tarball, but
   PURR viewers benefit from having it un-nested)
3. **`SHA256SUMS.txt`** — one-line hash of the tarball for tamper-evident verification
4. **`manifest.jsonl`** — machine-readable per-run manifest (also inside the tarball;
   un-nested here for landing-page previews)
5. **`hashes.txt`** — per-file SHA-256 hashes of every `best.pkl`, `final.pkl`,
   and `run.log` in the deposit (also inside the tarball; un-nested here)

Suggested PURR metadata:
- **Title:** *Mambino: Predictive-Coding-Augmented State-Space Models for Efficient
  Long-Range Reasoning on Constrained Hardware — trained checkpoints, run logs, and
  per-run manifest (v1)*
- **Author:** Raisul Islam
- **Affiliation:** Purdue University
- **Keywords:** state-space model, S5, predictive coding, LRA-ListOps, chip
  efficiency, PPAC, Accelergy, 22 nm, deep learning reproducibility
- **License for data:** CC-BY-4.0
- **License for code (linked):** as declared at
  <https://github.com/raisul1212/S5/blob/mambino-paper-v1/LICENSE>
- **Related identifiers:** point to the GitHub tag `mambino-paper-v1` as the
  companion code artifact

## Post-upload action — COMPLETED 2026-07-08

- ✅ PURR DOI populated in both long form and short form above:
  [10.4231/9ADT-WP13](https://doi.org/10.4231/9ADT-WP13)
- ⏳ Copy the appropriate form into the paper's Data Availability section
  when assembling the manuscript.
