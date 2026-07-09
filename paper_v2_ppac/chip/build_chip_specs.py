"""Single source of truth for the Mambino v2 chip architecture.

Emits both schemas from one Python dict:
  arch_accelergy.yaml  --  subtree/local format that Accelergy 0.4 wants
  arch_timeloop.yaml   --  nodes/!Container tag format that Timeloop wants

Both files describe the SAME chip. If we change chip parameters (array size,
buffer sizes, etc.), we edit only CHIP_PARAMS below, and both YAMLs regenerate.

Chip target (locked 2026-07-09):
  Edge inference class, ~1 mm^2 die, 22 nm INT8, 1 GHz
  16 x 16 systolic MAC array (256 PEs), weight-stationary
  256 KB weight SRAM, 128 KB activation SRAM, 64 KB state SRAM
  No off-chip DRAM (all configs' 106-188K weights fit on-chip)

Rationale for 16x16 (not 64x64):
  JAXPR extraction shows Shape 2 (B_bar @ x per timestep) has M = local_P
  which is 8 for P=8 configs and 16 for P=16 configs. A 64x64 array would
  waste 87.5% of PEs on this shape. 16x16 fits the natural minor dim of
  the paper's SSMs while still handling gate Denses (M=128, N=128) with
  standard tiling.
"""
from __future__ import annotations

import argparse
import os
import yaml


# =====================================================================
# CHIP PARAMETERS (single source of truth)
# =====================================================================
CHIP_PARAMS = dict(
    technology="22nm",
    clock_ns=1.0,

    # === Systolic MAC array ===
    array_x=16,           # 16 columns
    array_y=16,           # 16 rows -> 256 PEs total
    datawidth_int=8,      # INT8 activations/weights
    accum_width=32,       # INT32 accumulator

    # === On-chip buffer sizes (bytes) ===
    # width in bits = 64 (line = 8 bytes at INT8)
    weight_sram_bytes=256 * 1024,      # 256 KB
    activation_sram_bytes=128 * 1024,  # 128 KB
    state_sram_bytes=64 * 1024,        # 64 KB

    sram_line_bits=64,
    sram_line_bytes=8,
)


# =====================================================================
# Emitter helpers
# =====================================================================
def sram_depth(bytes_: int, params=CHIP_PARAMS) -> int:
    return bytes_ // params["sram_line_bytes"]


# ---------- Accelergy subtree/local schema ----------
def build_accelergy_yaml(p=CHIP_PARAMS) -> dict:
    """Emit Accelergy 0.4 subtree/local architecture spec.
    Component names use [0..N] notation for spatial replication.
    """
    n_pe = p["array_x"] * p["array_y"]
    common = dict(technology=int(p["technology"].rstrip("nm")),
                  global_cycle_seconds=p["clock_ns"] * 1e-9)

    return {
        "architecture": {
            "version": 0.4,
            "subtree": [
                {
                    "name": "mambino_chip",
                    "attributes": dict(common),
                    "local": [
                        {
                            "name": "weight_sram",
                            "class": "smartbuffer_SRAM",
                            "attributes": {
                                **common,
                                "depth": sram_depth(p["weight_sram_bytes"]),
                                "width": p["sram_line_bits"],
                                "n_banks": 8,
                                "datawidth": p["datawidth_int"],
                                "read_bandwidth": 16,
                                "write_bandwidth": 16,
                            },
                        },
                        {
                            "name": "activation_sram",
                            "class": "smartbuffer_SRAM",
                            "attributes": {
                                **common,
                                "depth": sram_depth(p["activation_sram_bytes"]),
                                "width": p["sram_line_bits"],
                                "n_banks": 8,
                                "datawidth": p["datawidth_int"],
                                "read_bandwidth": 16,
                                "write_bandwidth": 16,
                            },
                        },
                        {
                            "name": "state_sram",
                            "class": "smartbuffer_SRAM",
                            "attributes": {
                                **common,
                                "depth": sram_depth(p["state_sram_bytes"]),
                                "width": p["sram_line_bits"],
                                "n_banks": 4,
                                "datawidth": p["datawidth_int"],
                                "read_bandwidth": 16,
                                "write_bandwidth": 16,
                            },
                        },
                    ],
                    "subtree": [
                        {
                            "name": f"PE[0..{n_pe-1}]",
                            "attributes": {**common, "meshX": p["array_x"]},
                            "local": [
                                {
                                    "name": "weights_spad",
                                    "class": "smartbuffer_RF",
                                    "attributes": {
                                        **common,
                                        "depth": 1,
                                        "width": p["datawidth_int"],
                                        "datawidth": p["datawidth_int"],
                                        "read_bandwidth": 2,
                                        "write_bandwidth": 2,
                                    },
                                },
                                {
                                    "name": "psum_spad",
                                    "class": "smartbuffer_RF",
                                    "attributes": {
                                        **common,
                                        "depth": 1,
                                        "width": p["accum_width"],
                                        "datawidth": p["accum_width"],
                                        "read_bandwidth": 2,
                                        "write_bandwidth": 2,
                                        "update_fifo_depth": 2,
                                    },
                                },
                                {
                                    "name": "mac",
                                    "class": "intmac",
                                    "attributes": {
                                        **common,
                                        "multiplier_width": p["datawidth_int"],
                                        "adder_width": p["accum_width"],
                                    },
                                },
                            ],
                        }
                    ],
                }
            ],
        }
    }


# ---------- Timeloop v0.4 nodes/!Container schema ----------
# Using YAML tags requires custom dumper. Easiest path: emit YAML text
# with the !Container / !Component tags as literal strings.
def build_timeloop_yaml_text(p=CHIP_PARAMS) -> str:
    """Emit Timeloop v0.4 architecture spec with !Container/!Component/!Parallel tags.

    Structure (mirrors Eyeriss but sized for our chip):
      system (Container)
      weight_sram (Component, keep A)
      activation_sram (Component, keep B + Z)
      state_sram (Component)
      PE_column (Container, spatial meshX = 16)
      PE (Container, spatial meshY = 16)
      Parallel:
        weights_spad (keep A)
        psum_spad (keep Z)
      mac (Component)
    """
    return f"""# =====================================================================
# Mambino paper v2 chip architecture - Timeloop v0.4 nodes/Container schema
# Generated by build_chip_specs.py - do NOT hand-edit; edit CHIP_PARAMS.
# =====================================================================
architecture:
  version: 0.4
  nodes:
    - !Container
      name: system
      attributes:
        technology: "{p['technology']}"

    - !Component
      name: weight_sram
      class: smartbuffer_SRAM
      attributes:
        depth: {sram_depth(p['weight_sram_bytes'])}
        width: {p['sram_line_bits']}
        n_banks: 8
        datawidth: {p['datawidth_int']}
        read_bandwidth: 16
        write_bandwidth: 16
      constraints:
        dataspace: {{keep: [A], bypass: [B, Z]}}

    - !Component
      name: activation_sram
      class: smartbuffer_SRAM
      attributes:
        depth: {sram_depth(p['activation_sram_bytes'])}
        width: {p['sram_line_bits']}
        n_banks: 8
        datawidth: {p['datawidth_int']}
        read_bandwidth: 16
        write_bandwidth: 16
      constraints:
        dataspace: {{keep: [B, Z], bypass: [A]}}

    - !Container
      name: PE_column
      spatial: {{meshX: {p['array_x']}}}
      constraints:
        spatial:
          permutation: [K, M, N]
          factors: [K=1, M=1]
          split: 999

    - !Container
      name: PE
      spatial: {{meshY: {p['array_y']}}}
      constraints:
        spatial:
          split: 0
          permutation: [K, N, M]
          factors: [K=1, N=1]

    - !Parallel
      nodes:
      - !Component
        name: weights_spad
        class: smartbuffer_RF
        attributes:
          depth: 1
          width: {p['datawidth_int']}
          datawidth: {p['datawidth_int']}
          read_bandwidth: 2
          write_bandwidth: 2
        constraints:
          dataspace: {{keep: [A]}}
          temporal:
            permutation: [M, N, K]
            factors: [M=1, N=1]

      - !Component
        name: psum_spad
        class: smartbuffer_RF
        attributes:
          depth: 1
          width: {p['accum_width']}
          datawidth: {p['accum_width']}
          update_fifo_depth: 2
          read_bandwidth: 2
          write_bandwidth: 2
        constraints:
          dataspace: {{keep: [Z]}}
          temporal:
            permutation: [K, M, N]
            factors: [M=1, N=1, K=1]

    - !Component
      name: mac
      class: intmac
      attributes:
        multiplier_width: {p['datawidth_int']}
        adder_width: {p['accum_width']}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=".",
                    help="Directory to write arch_accelergy.yaml + arch_timeloop.yaml")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    p = CHIP_PARAMS
    print(f"[build] chip: {p['array_x']}x{p['array_y']} WS array = "
          f"{p['array_x']*p['array_y']} PEs at {p['technology']} INT8")
    print(f"[build] buffers: {p['weight_sram_bytes']//1024} KB weight, "
          f"{p['activation_sram_bytes']//1024} KB activation, "
          f"{p['state_sram_bytes']//1024} KB state")

    with open(os.path.join(args.out_dir, "arch_accelergy.yaml"), "w") as f:
        yaml.safe_dump(build_accelergy_yaml(p), f, sort_keys=False,
                       default_flow_style=False)
    print(f"[build] wrote arch_accelergy.yaml (subtree/local)")

    with open(os.path.join(args.out_dir, "arch_timeloop.yaml"), "w") as f:
        f.write(build_timeloop_yaml_text(p))
    print(f"[build] wrote arch_timeloop.yaml (nodes/!Container)")


if __name__ == "__main__":
    main()
