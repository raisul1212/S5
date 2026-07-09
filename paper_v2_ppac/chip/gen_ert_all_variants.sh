#!/usr/bin/env bash
# Generate Accelergy ERTs for BOTH SRAM variants across 4 array sizes.
#
# Per-config chip sizing (paper design rule: fit workload without DRAM):
#   384 KB activation SRAM chip: Config 4, Config 5, Corner 2, Corner 3'
#   512 KB activation SRAM chip: Corner 1 (full-rank gate needs bigger)
#
# Emits:
#   out_accelergy_384KB_{8x8,16x16,32x32,64x64}/ERT.yaml + ART.yaml
#   out_accelergy_512KB_{8x8,16x16,32x32,64x64}/ERT.yaml + ART.yaml
set -euo pipefail

CHIP=$(cd "$(dirname "$0")" && pwd)
ACCELERGY=/scratch/gilbreth/raisul/envs/s5m/bin/accelergy
cd "$CHIP"

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

for SRAM_KB in 288 384 512; do
  ACT_BYTES=$((SRAM_KB * 1024))
  for SIZE in 8 16 32 64; do
    TAG=${SRAM_KB}KB_${SIZE}x${SIZE}
    echo "===== ${TAG} ====="
    # Override CHIP_PARAMS activation_sram_bytes via a small Python monkeypatch
    python - <<PYEOF
import sys, os
sys.path.insert(0, os.getcwd())
import build_chip_specs as b
b.CHIP_PARAMS['activation_sram_bytes'] = ${ACT_BYTES}
# Emit arch YAMLs with SRAM tag in filename
p = dict(b.CHIP_PARAMS); p['array_x'] = ${SIZE}; p['array_y'] = ${SIZE}
open(f"arch_accelergy_${TAG}.yaml","w").write(
    __import__('yaml').safe_dump(b.build_accelergy_yaml(p), sort_keys=False, default_flow_style=False))
open(f"arch_timeloop_${TAG}.yaml","w").write(b.build_timeloop_yaml_text(p))
print(f"[build] {p['array_x']}x{p['array_y']} @ {p['activation_sram_bytes']//1024} KB activation SRAM")
PYEOF
    OUT=out_accelergy_${TAG}
    rm -rf "$OUT"
    mkdir -p "$OUT"
    "$ACCELERGY" arch_accelergy_${TAG}.yaml components/*.yaml dummy_actions.yaml \
                 -o "$OUT/" 2>&1 | tail -3
    echo "[gen_ert] wrote $OUT/ERT.yaml"
  done
done
echo "===== done ====="
ls -1d out_accelergy_[35]??KB_*/ERT.yaml 2>/dev/null | head
