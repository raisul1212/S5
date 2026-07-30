"""SCALE-Sim v2 multi-array validation for the v4 chip PPAC simulator.

For each of ~8 representative shape × array-size combinations from the
multi_array_ppac_v4 ATP-optimal per-config assignments, run SCALE-Sim v2
in the specified rectangular WS array shape and compare the cycle count
to the H1 fill/drain analytical model in multi_array_ppac_v4.py.

SCALE-Sim v2 supports rectangular arrays (ArrayHeight != ArrayWidth) natively.
"""
from __future__ import annotations

import csv
import math
import os
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

CHIP = Path(__file__).resolve().parent
SCALESIM_OUT = CHIP / "scalesim_multi_array_validation"
SCALESIM_OUT.mkdir(exist_ok=True)

# Shape × array pairs from the v4 ATP-optimal assignments (across the 5 configs).
# Fields: (label, M, N, K, dtype, ay, ax, wc_rep, role, config)
# For A-stationary shapes, SCALE-Sim's B-stationary WS still computes the same
# total MACs; we just interpret ay|M and ax|K by feeding it as (M, K, N)-ordered
# GEMM. dtype_scale applied outside SCALE-Sim.

CASES = [
    # Corner 1 blocks at its ATP-optimal (target 32,768)
    ("Corner1 Encoder",        2048, 128, 20, "real_real",    8, 64, 1, "B", "corner1"),
    ("Corner1 SSM main",          8, 2048, 128, "complex_real", 8, 64, 8, "A", "corner1"),
    ("Corner1 SSM C-proj",      128, 2048, 16, "complex_complex", 64, 16, 8, "A", "corner1"),
    ("Corner1 Dense gate",     2048, 128, 128, "real_real",   64, 64, 8, "B", "corner1"),

    # Corner 3' blocks at its ATP-optimal (target 262,144)
    ("Corner3p Encoder",       2048, 128, 20, "real_real",     8,  8, 1, "B", "corner3p"),
    ("Corner3p SSM main",         8, 2048, 128, "complex_real", 8, 64, 16, "A", "corner3p"),  # wc_rep=16
    ("Corner3p Predictor Cproj", 128, 2048, 8, "complex_complex", 64, 8, 8, "A", "corner3p"),
    ("Corner3p SSM Cproj",      128, 2048, 16, "complex_complex", 64, 16, 8, "A", "corner3p"),
    ("Corner3p Gate DOWN",     2048,  40, 128, "real_real",   64,  8, 8, "B", "corner3p"),
    ("Corner3p Gate UP",       2048, 128,  40, "real_real",    8, 64, 8, "B", "corner3p"),
]

FREQ_HZ = 1e9

def analytical_cycles(M, N, K, ay, ax, dtype_scale, role, wc_rep):
    """multi_array_ppac_v4.py cycle model: per_tile = stream + ay + ax - 2."""
    def ldiv(x, m):
        if x <= m: return x
        b = 1
        for f in range(1, m+1):
            if x % f == 0 and f > b: b = f
        return b
    if role == "A":
        # A[M×K] stationary. Spatial: M on ay, K on ax. Stream: N.
        real_ay = ldiv(M, ay); real_ax = ldiv(K, ax)
        n_tiles_outer = math.ceil(M / real_ay)
        n_tiles_inner = math.ceil(K / real_ax)
        stream = N
    else:
        # B[K×N] stationary. Spatial: K on ay, N on ax. Stream: M.
        real_ay = ldiv(K, ay); real_ax = ldiv(N, ax)
        n_tiles_outer = math.ceil(K / real_ay)
        n_tiles_inner = math.ceil(N / real_ax)
        stream = M
    per_tile = stream + (ay + ax - 2)
    cyc_per_inst = per_tile * n_tiles_outer * n_tiles_inner * dtype_scale
    return cyc_per_inst * wc_rep

def scalesim_cycles(M, N, K, ay, ax, dtype_scale, wc_rep, label, role="B"):
    """Run SCALE-Sim v2 in WS mode on the given rectangular array.

    SCALE-Sim assumes B-stationary WS: weights are the K×N matrix, activations
    are M×K, output is M×N. For A-stationary shapes in our simulator (where
    A[M×K] is the small parameter matrix and N is the streaming dimension),
    we swap M↔N in the SCALE-Sim feed so its B-stationary interpretation
    lines up with our A-stationary mapping."""
    if role == "A":
        # Feed SCALE-Sim as (M'=N, N'=M, K'=K). SCALE-Sim then holds K×N'=K×M
        # stationary (which is our A operand) and streams M'=N activations.
        Mf, Nf, Kf = N, M, K
    else:
        Mf, Nf, Kf = M, N, K
    topo = SCALESIM_OUT / f"topo_{label.replace(' ','_')}.csv"
    with open(topo, "w") as f:
        w = csv.writer(f)
        w.writerow(["Layer", "M", "N", "K", "dtype"])
        w.writerow([f"gemm_{Mf}x{Nf}x{Kf}", Mf, Nf, Kf, "dtype"])
    # Emit config
    cfg = SCALESIM_OUT / f"cfg_{label.replace(' ','_')}.cfg"
    cfg.write_text(f"""[general]
run_name = validate_{label.replace(' ','_')}

[architecture_presets]
ArrayHeight    : {ay}
ArrayWidth     : {ax}
IfmapSramSzkB  : 256
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
    # Run SCALE-Sim (use full s5m env Python, not system /usr/bin/python)
    py = os.path.expanduser("~/../scratch/gilbreth/raisul/envs/s5m/bin/python")
    if not os.path.exists(py):
        py = "/scratch/gilbreth/raisul/envs/s5m/bin/python"
    proc = subprocess.run(
        [py, "-m", "scalesim.scale",
         "-c", str(cfg), "-t", str(topo),
         "-p", str(SCALESIM_OUT), "-i", "gemm"],
        capture_output=True, text=True, cwd=str(CHIP), timeout=120)
    if proc.returncode != 0:
        return None, f"SCALE-Sim failed: {proc.stderr[-200:]}"
    # Find COMPUTE_REPORT.csv
    report = None
    for p in SCALESIM_OUT.rglob("COMPUTE_REPORT.csv"):
        if f"validate_{label.replace(' ','_')}" in str(p):
            report = p; break
    if not report:
        return None, "no COMPUTE_REPORT"
    row = next(csv.DictReader(open(report)))
    ss_cyc = int(float(next(v for k, v in row.items() if k.strip() == "Total Cycles")))
    ss_util = float(next(v for k, v in row.items() if k.strip() == "Overall Util %"))
    # Multiply by dtype_scale * wc_rep to get total cycles for our workload
    return ss_cyc * dtype_scale * wc_rep, f"util={ss_util:.1f}%"

def main():
    print(f"{'Case':<32}{'Array':>10}{'Analytical cyc':>18}{'SCALE-Sim cyc':>16}"
          f"{'Ratio SS/An':>14}{'Notes':<25}")
    print("-" * 118)
    rows = []
    for (label, M, N, K, dtype, ay, ax, wc_rep, role, cfg) in CASES:
        ds = {"real_real": 1, "complex_real": 2, "complex_complex": 4}[dtype]
        an_cyc = analytical_cycles(M, N, K, ay, ax, ds, role, wc_rep)
        ss_cyc, notes = scalesim_cycles(M, N, K, ay, ax, ds, wc_rep, label, role=role)
        if ss_cyc is None:
            ratio_str = "---"
            ss_str = notes
        else:
            ratio = ss_cyc / an_cyc if an_cyc else 0
            ss_str = f"{ss_cyc:,}"
            ratio_str = f"{ratio:.2f}x"
        print(f"{label:<32}{f'{ay}x{ax}':>10}{an_cyc:>18,}{ss_str:>16}{ratio_str:>14}  {notes:<20}")
        rows.append((label, ay, ax, ds, wc_rep, an_cyc, ss_cyc, role))

    # Save CSV
    csv_path = SCALESIM_OUT / "validation_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "ay", "ax", "dtype_scale", "wc_rep",
                    "analytical_cyc", "scalesim_cyc", "role", "ratio"])
        for r in rows:
            r_ratio = r[6] / r[5] if r[5] and r[6] else 0
            w.writerow([*r, r_ratio])
    print(f"\nSaved {csv_path}")

    # Summary stats
    valid = [(a, s) for _, _, _, _, _, a, s, _ in rows if s]
    if valid:
        ratios = [s/a for a, s in valid if a > 0]
        print(f"\nMean SS/analytical cycle ratio: {sum(ratios)/len(ratios):.2f}")
        print(f"Min: {min(ratios):.2f}, Max: {max(ratios):.2f}")
        within = sum(1 for r in ratios if 0.8 <= r <= 1.5)
        print(f"Within 0.8-1.5x band: {within}/{len(ratios)}")

if __name__ == "__main__":
    main()
