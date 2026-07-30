"""SCALE-Sim v2 cross-check for the v2 chip PPAC pipeline.

For each of the 10 unique GEMM shapes (across all 5 configs), runs
SCALE-Sim v2 on a 16×16 weight-stationary systolic chip and reports
cycles + memory accesses. Compares to Timeloop-model per-shape stats
to establish cross-tool agreement.

Timeloop uses an analytical model; SCALE-Sim v2 is cycle-accurate
including systolic pipeline fill/drain lag. Agreement within ~2-3x on
cycles is expected (Timeloop = steady-state throughput lower bound,
SCALE-Sim = wall-clock upper bound including latency).
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

import yaml

CHIP = Path(__file__).resolve().parent
WORKLOADS = CHIP.parent / "workloads"
SCALESIM_OUT = CHIP / "scalesim_crosscheck"
SCALESIM_OUT.mkdir(exist_ok=True)

ARRAY_X, ARRAY_Y = 16, 16

# ---- SCALE-Sim v2 config (16×16 WS array, 1 MB unified buffer, no DRAM) ----
config_content = f"""[general]
run_name = crosscheck

[architecture_presets]
ArrayHeight    : {ARRAY_Y}
ArrayWidth     : {ARRAY_X}
IfmapSramSzkB  : 512
FilterSramSzkB : 256
OfmapSramSzkB  : 256
IfmapOffset    : 0
FilterOffset   : 10000000
OfmapOffset    : 20000000
Bandwidth      : 10
Dataflow       : ws
MemoryBanks    : 1

[run_presets]
InterfaceBandwidth : CALC
"""
config_path = CHIP / "scalesim_crosscheck.cfg"
config_path.write_text(config_content)

# ---- Collect unique GEMM shapes across all 5 configs ----
configs = ["config4", "config5", "corner1", "corner2", "corner3p"]
unique_shapes = {}  # (M, N, K, dtype) -> Gemm
for cfg in configs:
    d = yaml.safe_load(open(WORKLOADS / f"workload_{cfg}_gemms.yaml"))
    for g in d['gemms']:
        key = (g['M'], g['N'], g['K'], g['dtype'])
        unique_shapes[key] = g
print(f"[scalesim] {len(unique_shapes)} unique GEMM shapes to cross-check")

# ---- Emit a combined SCALE-Sim topology CSV with all 10 GEMMs ----
topo_path = CHIP / "scalesim_gemms.csv"
with open(topo_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Layer", "M", "N", "K", "dtype_marker"])
    for i, (key, g) in enumerate(sorted(unique_shapes.items()), 1):
        M, N, K, dtype = key
        w.writerow([f"gemm_{i:02d}_{dtype}", M, N, K, dtype])

print(f"[scalesim] running SCALE-Sim on {topo_path.name}...")
proc = subprocess.run(
    [
        "python", "-m", "scalesim.scale",
        "-c", str(config_path),
        "-t", str(topo_path),
        "-p", str(SCALESIM_OUT),
        "-i", "gemm",
    ],
    capture_output=True, text=True, cwd=str(CHIP),
)
if proc.returncode != 0:
    print(f"[scalesim] FAILED: {proc.stderr[-800:]}")
    exit(1)
print("[scalesim] SCALE-Sim runs complete")

# ---- Load SCALE-Sim results ----
compute_report = SCALESIM_OUT / "crosscheck" / "COMPUTE_REPORT.csv"
bandwidth_report = SCALESIM_OUT / "crosscheck" / "BANDWIDTH_REPORT.csv"

if not compute_report.exists():
    # Try alternative path
    for p in SCALESIM_OUT.rglob("COMPUTE_REPORT.csv"):
        compute_report = p
        break
    for p in SCALESIM_OUT.rglob("BANDWIDTH_REPORT.csv"):
        bandwidth_report = p
        break

ss_rows = list(csv.DictReader(open(compute_report)))
bw_rows = list(csv.DictReader(open(bandwidth_report)))
print(f"[scalesim] parsed {len(ss_rows)} SCALE-Sim compute rows")

# ---- Load Timeloop per-shape results ----
per_gemm = json.load(open(CHIP / "per_gemm_ppac.json"))

# ---- Cross-check table ----
print()
print("=" * 130)
print(f"SCALE-Sim v2 vs Timeloop-model — 16×16 WS array, 10 unique GEMM shapes")
print("=" * 130)
print(f"{'GEMM shape':<32}{'SCALE-Sim cyc':>14}{'SCALE-Sim util%':>18}{'Timeloop cyc':>14}"
      f"{'Timeloop util%':>16}{'Ratio SS/TL':>13}")
print("-" * 130)

crosscheck_rows = []
for i, (key, g) in enumerate(sorted(unique_shapes.items()), 1):
    M, N, K, dtype = key
    shape_id = f"M{M}_N{N}_K{K}_{dtype}"
    layer = f"gemm_{i:02d}_{dtype}"
    # SCALE-Sim uses LayerID = index (i-1)
    ss_row = next((r for r in ss_rows if r.get('LayerID', '').strip() == str(i-1)), None)
    tl_row = per_gemm.get(f"{shape_id}__16x16")
    if ss_row is None or tl_row is None:
        print(f"{shape_id:<32}  MISSING (ss={ss_row is not None}, tl={tl_row is not None})")
        continue
    # SCALE-Sim column names have leading spaces
    def _g(row, key):
        for k in row:
            if k.strip() == key:
                return row[k]
        return None
    ss_cyc = int(float(_g(ss_row, "Total Cycles")))
    ss_util = float(_g(ss_row, "Overall Util %"))
    tl_cyc = tl_row['cycles']
    tl_util = tl_row['util_pct']
    ratio = ss_cyc / tl_cyc if tl_cyc > 0 else 0
    print(f"{shape_id:<32}{ss_cyc:>14,}{ss_util:>16.1f}%{tl_cyc:>14,}{tl_util:>14.1f}%{ratio:>13.2f}")
    crosscheck_rows.append(dict(
        shape=shape_id, dtype=dtype,
        M=M, N=N, K=K,
        scalesim_cycles=ss_cyc, scalesim_util_pct=ss_util,
        timeloop_cycles=tl_cyc, timeloop_util_pct=tl_util,
        ratio_ss_over_tl=round(ratio, 2),
    ))

# ---- Save the cross-check table ----
cc_csv = CHIP / "scalesim_vs_timeloop_16x16.csv"
with open(cc_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["shape","dtype","M","N","K",
                "scalesim_cycles","scalesim_util_pct",
                "timeloop_cycles","timeloop_util_pct","ratio_ss_over_tl"])
    for r in crosscheck_rows:
        w.writerow([r["shape"], r["dtype"], r["M"], r["N"], r["K"],
                    r["scalesim_cycles"], r["scalesim_util_pct"],
                    r["timeloop_cycles"], r["timeloop_util_pct"],
                    r["ratio_ss_over_tl"]])
print(f"\n[scalesim] wrote {cc_csv.name}")

# ---- Summary ----
if crosscheck_rows:
    ratios = [r["ratio_ss_over_tl"] for r in crosscheck_rows]
    mean_ratio = sum(ratios) / len(ratios)
    print(f"\n[summary] Mean SS/TL cycle ratio: {mean_ratio:.2f}")
    print(f"[summary] Min ratio: {min(ratios):.2f}  Max ratio: {max(ratios):.2f}")
    print(f"[summary] Expected range: 1.5–3.0 (SCALE-Sim includes pipeline fill/drain; Timeloop is analytical)")
    within = sum(1 for r in ratios if 1.0 <= r <= 3.5)
    print(f"[summary] Shapes within expected 1.0–3.5× band: {within}/{len(ratios)}")
