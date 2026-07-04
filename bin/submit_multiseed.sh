#!/bin/bash
# Canonical multi-seed sweep for Mambino paper (LRA-ListOps).
# Pins seed set + config-to-script mapping so paper's "seeds = {...}" is
# fully reproducible from repo state alone.
#
# Usage:
#   bin/submit_multiseed.sh              # dry-run: print sbatch commands
#   bin/submit_multiseed.sh --submit     # actually submit all 16 jobs, HELD
#   bin/submit_multiseed.sh --submit --release   # submit + immediately release
#
# The "5th seed" (6554595 - default) is the SEED baked into the ORIGINAL
# single-seed runs listed below. This script sweeps 4 ADDITIONAL seeds so
# multi-seed statistics have 5 total data points per config.

set -euo pipefail

# ============================================================================
# SEED SET (locked)
# ============================================================================
SEEDS=(42 12345 271828 314159)

# ============================================================================
# CONFIG-TO-SCRIPT MAPPING (locked)
# ============================================================================
# Format: "config_name:sbatch_script:original_single_seed_jobid"
# The original job produced the single-seed accuracy that appears in
# LOCKED_master_ppac_document.md.
CONFIGS=(
  "corner1:bin/gilbreth_chip_pure_s5.sh:11181830"
  "corner3p:bin/gilbreth_chip_mambino_iso.sh:11181831"
  "config4:bin/gilbreth_chip_mambino_gelu.sh:11181833"
  "config5:bin/train_pure_s5_gelu_p16.sh:11187128"
)

# ============================================================================
# EXECUTE
# ============================================================================
DO_SUBMIT=0
DO_RELEASE=0
for arg in "$@"; do
  case "$arg" in
    --submit)  DO_SUBMIT=1 ;;
    --release) DO_RELEASE=1 ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

echo "# Canonical multi-seed sweep"
echo "# Seeds: ${SEEDS[*]}"
echo "# Configs: ${#CONFIGS[@]}"
echo ""

SUBMITTED=()
for entry in "${CONFIGS[@]}"; do
  name=$(echo "$entry" | cut -d: -f1)
  script=$(echo "$entry" | cut -d: -f2)
  orig_jid=$(echo "$entry" | cut -d: -f3)
  echo "# ---- $name (original single-seed = job $orig_jid) ----"
  for seed in "${SEEDS[@]}"; do
    cmd="sbatch --hold $script $seed"
    if [ "$DO_SUBMIT" -eq 1 ]; then
      jid=$(sbatch --hold "$script" "$seed" | awk '{print $NF}')
      echo "sbatch --hold $script $seed  # -> job $jid"
      SUBMITTED+=("$jid")
    else
      echo "$cmd  # (dry-run; use --submit to actually submit)"
    fi
  done
done

if [ "$DO_SUBMIT" -eq 1 ] && [ "$DO_RELEASE" -eq 1 ]; then
  echo ""
  echo "# Releasing all ${#SUBMITTED[@]} submitted jobs..."
  scontrol release "${SUBMITTED[@]}"
  echo "# Released."
fi

if [ "$DO_SUBMIT" -eq 1 ] && [ "$DO_RELEASE" -eq 0 ]; then
  echo ""
  echo "# ${#SUBMITTED[@]} jobs held. Release with:"
  echo "#   scontrol release ${SUBMITTED[*]}"
fi
