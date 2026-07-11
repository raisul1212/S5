#!/bin/bash
# Queue the v2 gate PILOT on the Corner 3' slot: 4 arms x 3 seeds = 12 jobs.
#   dense_r40  = same-SHA baseline (methodologically cleaner than reusing v1's SHA 46517fe;
#                the v1 8-seed Corner 3' already has these seeds if you prefer to reuse).
#   dense_r32  = clean-rank control (the lean check: does a hardware-friendly rank alone win?)
#   monarch    = structured, cross-block, full-rank-capable (R=3 heads -> 9,216 params)
#   blockdiag  = structured, block-local (B=2 -> 8,192 params)
#
# Decision rule after the pilot: expand to the full 8-seed sweep ONLY if monarch/blockdiag
# >= dense on accuracy AND cheaper/cleaner on chip; if dense_r32 already ties, pivot to the
# lean "pick a hardware-friendly rank" result. Report the null honestly.
#
# Run PHASE 3 FIRST:  python bin/test_v2_gate.py --v1_ckpt <a v1 Corner 3' best>
# (must print ALL PASS before spending GPU-hours here.)
set -e
cd "$(dirname "$0")/.."

SEEDS="42 12345 271828"          # pilot seeds (median-tier subset of the 8-seed set)
ARMS="dense_r40 dense_r32 monarch blockdiag"
# To skip the same-SHA baseline and reuse v1 Corner 3' numbers instead, set:
#   ARMS="dense_r32 monarch blockdiag"   # -> 9 jobs

for ARM in $ARMS; do
  for SEED in $SEEDS; do
    echo "sbatch bin/gilbreth_v2_pilot.sh $ARM $SEED"
    sbatch bin/gilbreth_v2_pilot.sh "$ARM" "$SEED"
  done
done
echo "queued $(echo $ARMS | wc -w) arms x $(echo $SEEDS | wc -w) seeds"
