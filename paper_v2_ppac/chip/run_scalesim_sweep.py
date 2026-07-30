"""SCALE-Sim sweep across all 4 array sizes {8x8, 16x16, 32x32, 64x64}.
For each unique GEMM shape × array size, get SCALE-Sim cycles + util.
Then aggregate per-config totals (using each config's instance list ×
repeat count × dtype_scale) for direct comparison to Timeloop-model.

Emits:
  scalesim_sweep_per_gemm.csv     -- 10 shapes × 4 array sizes
  scalesim_sweep_per_config.csv   -- 5 configs × 4 array sizes
  ppac_all_methods.csv            -- Timeloop + SCALE-Sim per-config combined
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
SCALESIM_OUT = CHIP / "scalesim_sweep"
SCALESIM_OUT.mkdir(exist_ok=True)

ARRAY_SIZES = [(8, 8), (16, 16), (32, 32), (64, 64)]
FREQ_HZ = 1e9

configs = ["config4", "config5", "corner1", "corner2", "corner3p"]

# --- Load workloads ---
unique_shapes = {}       # (M, N, K, dtype) -> gemm dict
per_config_instances = {c: [] for c in configs}
for cfg in configs:
    d = yaml.safe_load(open(WORKLOADS / f"workload_{cfg}_gemms.yaml"))
    for g in d['gemms']:
        key = (g['M'], g['N'], g['K'], g['dtype'])
        unique_shapes[key] = g
        per_config_instances[cfg].append(g)
print(f"[scalesim-sweep] {len(unique_shapes)} unique GEMMs × {len(ARRAY_SIZES)} array sizes")

# --- Topology CSV (shared across all array sizes) ---
topo_path = CHIP / "scalesim_sweep_topology.csv"
shape_id_by_index = {}
with open(topo_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Layer", "M", "N", "K", "dtype_marker"])
    for i, (key, g) in enumerate(sorted(unique_shapes.items())):
        M, N, K, dtype = key
        shape_id = f"M{M}_N{N}_K{K}_{dtype}"
        w.writerow([f"gemm_{i:02d}_{dtype}", M, N, K, dtype])
        shape_id_by_index[i] = shape_id

# --- Run SCALE-Sim per array size ---
per_shape_results = {}   # (shape_id, ax, ay) -> {cycles, util}
for ax, ay in ARRAY_SIZES:
    cfg_path = CHIP / f"scalesim_config_{ax}x{ay}.cfg"
    cfg_path.write_text(f"""[general]
run_name = sweep_{ax}x{ay}

[architecture_presets]
ArrayHeight    : {ay}
ArrayWidth     : {ax}
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
""")
    print(f"[scalesim-sweep] {ax}x{ay} ...", end=" ", flush=True)
    proc = subprocess.run(
        ["python", "-m", "scalesim.scale",
         "-c", str(cfg_path), "-t", str(topo_path),
         "-p", str(SCALESIM_OUT), "-i", "gemm"],
        capture_output=True, text=True, cwd=str(CHIP),
    )
    if proc.returncode != 0:
        print(f"FAILED  {proc.stderr[-200:]}")
        continue
    report = SCALESIM_OUT / f"sweep_{ax}x{ay}" / "COMPUTE_REPORT.csv"
    if not report.exists():
        for p in SCALESIM_OUT.rglob("COMPUTE_REPORT.csv"):
            # Get most recent based on cfg
            if f"sweep_{ax}x{ay}" in str(p):
                report = p; break
    if not report.exists():
        print("NO REPORT")
        continue
    rows = list(csv.DictReader(open(report)))
    print(f"parsed {len(rows)} shapes")
    for r in rows:
        idx = int(r.get('LayerID', 0))
        shape_id = shape_id_by_index.get(idx)
        if not shape_id: continue
        def _g(row, key):
            for k in row:
                if k.strip() == key: return row[k]
            return None
        cycles = int(float(_g(r, "Total Cycles")))
        util = float(_g(r, "Overall Util %"))
        per_shape_results[(shape_id, ax, ay)] = dict(cycles=cycles, util_pct=util)

# --- Emit per-GEMM sweep table ---
per_gemm_csv = CHIP / "scalesim_sweep_per_gemm.csv"
with open(per_gemm_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["shape_id","dtype","M","N","K","array_x","array_y","n_pe",
                "cycles","util_pct"])
    for (shape_id, ax, ay), v in sorted(per_shape_results.items()):
        parts = shape_id.split("_")
        M = int(parts[0][1:]); N = int(parts[1][1:]); K = int(parts[2][1:])
        dtype = "_".join(parts[3:])
        w.writerow([shape_id, dtype, M, N, K, ax, ay, ax*ay,
                    v["cycles"], round(v["util_pct"], 2)])
print(f"[scalesim-sweep] wrote {per_gemm_csv.name} ({len(per_shape_results)} rows)")

# --- Aggregate per-config per-array totals ---
per_config_arr = {}   # (cfg, ax, ay) -> {cycles, latency, ...}
for cfg in configs:
    for ax, ay in ARRAY_SIZES:
        total_cycles = 0
        total_real_macs = 0
        ok = 0
        for g in per_config_instances[cfg]:
            shape_id = f"M{g['M']}_N{g['N']}_K{g['K']}_{g['dtype']}"
            r = per_shape_results.get((shape_id, ax, ay))
            if r is None: continue
            dtype_scale = g['per_mac_flops'] // 2
            total_cycles += r['cycles'] * dtype_scale * g['repeat']
            total_real_macs += g['total_real_macs']
            ok += 1
        latency_ms = total_cycles / FREQ_HZ * 1e3
        per_config_arr[(cfg, ax, ay)] = dict(
            cycles=total_cycles, latency_ms=latency_ms,
            throughput_ips=1e9/total_cycles if total_cycles else 0,
            n_shapes_ok=ok, total_real_macs=total_real_macs,
        )

per_cfg_csv = CHIP / "scalesim_sweep_per_config.csv"
with open(per_cfg_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["config","array_x","array_y","n_pe","total_cycles","latency_ms","throughput_ips"])
    for (cfg, ax, ay), v in sorted(per_config_arr.items()):
        w.writerow([cfg, ax, ay, ax*ay, v['cycles'],
                    round(v['latency_ms'], 4), round(v['throughput_ips'], 2)])
print(f"[scalesim-sweep] wrote {per_cfg_csv.name}")

# --- Combined table: Timeloop + SCALE-Sim per config × array size ---
tl_pcfg = json.load(open(CHIP / "per_config_ppac.json"))
combined_csv = CHIP / "ppac_all_methods.csv"
label = {'config4':'Config 4','config5':'Config 5','corner1':'Corner 1',
         'corner2':'Corner 2','corner3p':"Corner 3'"}
family = {'config4':'Mambino','config5':'Pure S5','corner1':'Pure S5',
          'corner2':'Pure S5','corner3p':'Mambino'}
band = {'config4':'106K','config5':'106K','corner1':'188K','corner2':'188K','corner3p':'188K'}
ACC = {"config4": 0.5993, "config5": 0.5917,
       "corner1": 0.6089, "corner2": 0.5991, "corner3p": 0.6138}

with open(combined_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "config","label","family","params_band","accuracy",
        "array_x","array_y","n_pe",
        # Timeloop numbers
        "tl_energy_mJ","tl_latency_ms","tl_power_mW","tl_throughput_ips",
        "tl_cycles","tl_pj_per_real_mac","tl_edp_mJ_ms",
        # SCALE-Sim numbers
        "ss_cycles","ss_latency_ms","ss_throughput_ips",
        # Cross-check
        "cycle_ratio_ss_over_tl",
    ])
    for cfg in configs:
        for ax, ay in ARRAY_SIZES:
            tl = tl_pcfg[cfg][f"{ax}x{ay}"]
            ss = per_config_arr.get((cfg, ax, ay), {})
            ss_cyc = ss.get('cycles', 0)
            tl_cyc = tl['total_cycles']
            ratio = ss_cyc / tl_cyc if tl_cyc else 0
            w.writerow([
                cfg, label[cfg], family[cfg], band[cfg], ACC[cfg],
                ax, ay, ax*ay,
                round(tl['energy_mJ'], 4),
                round(tl['latency_ms'], 4),
                round(tl['power_mW'], 2),
                round(tl['throughput_ips'], 2),
                tl_cyc,
                round(tl['pj_per_real_mac'], 3),
                round(tl['energy_mJ'] * tl['latency_ms'], 4),
                ss_cyc,
                round(ss.get('latency_ms', 0), 4),
                round(ss.get('throughput_ips', 0), 2),
                round(ratio, 2),
            ])
print(f"[scalesim-sweep] wrote {combined_csv.name}")

# --- Summary ---
print()
print("=" * 130)
print("Timeloop + SCALE-Sim cycles per (config × array size)")
print("=" * 130)
print(f"{'Config':<10}{'Array':<8}{'TL cyc':>14}{'TL lat':>10}{'SS cyc':>14}{'SS lat':>10}{'Ratio SS/TL':>14}")
print("-" * 130)
for cfg in configs:
    for ax, ay in ARRAY_SIZES:
        tl = tl_pcfg[cfg][f"{ax}x{ay}"]
        ss = per_config_arr.get((cfg, ax, ay), {})
        ratio = ss.get('cycles', 0) / tl['total_cycles'] if tl['total_cycles'] else 0
        print(f"{cfg:<10}{ax}x{ay:<6}"
              f"{tl['total_cycles']:>14,}{tl['latency_ms']:>7.3f}ms"
              f"{ss.get('cycles', 0):>14,}{ss.get('latency_ms', 0):>7.3f}ms"
              f"{ratio:>13.2f}")
    print()
