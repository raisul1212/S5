#!/usr/bin/env bash
# Generate 4 chip specs + 4 Accelergy ERTs for the array-size sweep.
# Buffer sizes held fixed at 256/128/64 KB (weight/act/state).
# Only the PE array size (and hence MAC/spad instance counts) change.
set -euo pipefail

CHIP=$(cd "$(dirname "$0")" && pwd)
ACCELERGY=/scratch/gilbreth/raisul/envs/s5m/bin/accelergy

cd "$CHIP"

# Dummy action counts — Accelergy needs some counts to walk the tree, but
# the ERT (per-access energy) it emits is independent of the counts. We
# use tiny placeholder counts so the run is fast.
cat > dummy_actions.yaml <<'EOF'
action_counts:
  version: 0.4
  local:
    - name: mambino_chip.weight_sram
      action_counts:
        - {name: read, counts: 1}
        - {name: write, counts: 1}
        - {name: leak, counts: 1}
    - name: mambino_chip.activation_sram
      action_counts:
        - {name: read, counts: 1}
        - {name: write, counts: 1}
        - {name: leak, counts: 1}
    - name: mambino_chip.state_sram
      action_counts:
        - {name: read, counts: 1}
        - {name: write, counts: 1}
        - {name: leak, counts: 1}
EOF

for SIZE in 8 16 32 64; do
  echo "===== $SIZE x $SIZE ====="
  python build_chip_specs.py --array_x $SIZE --array_y $SIZE --out_dir "$CHIP"
  OUT=out_accelergy_${SIZE}x${SIZE}
  rm -rf "$OUT"
  mkdir -p "$OUT"
  "$ACCELERGY" \
      arch_accelergy_${SIZE}x${SIZE}.yaml \
      components/*.yaml \
      dummy_actions.yaml \
      -o "$OUT/" \
      2>&1 | tail -20
  echo "[gen_ert] wrote $OUT/ERT.yaml"
done

echo "===== done ====="
ls -1 out_accelergy_*/ERT.yaml
