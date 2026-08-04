"""Emit Origin-importable CSVs for the 4-config v1 paper.

Four configs (canonical names per memos/NAMING_AND_STATUS.md):
    S5-gateless  105,738   Pure S5, gelu            iso-param control
    Mambino-0    105,738   Mambino, gelu, no gate   isolates predictive coding
    Mambino-G    105,754   Mambino-0 + SWG          the hero
    Pure S5      188,490   Pure S5, half_glu2 r0    baseline of record

Every per-seed value is extracted from run.log and keyed on the seed recorded
INSIDE the checkpoint (best.meta.pkl['args'].jax_seed) -- never on directory
names, which have proven unreliable. Configs are identified by a full arg
signature so the lambda_pc sweep, the unsigned-gate arm, the v2pilot structured
gates and the superseded old-SHA chip_* singles cannot leak in.

Guiding metric is test@peakval (test acc at the highest-val epoch). test_max
(best test at any epoch) is emitted alongside as literature context only.

Usage on Gilbreth from the worktree root:
    /scratch/gilbreth/raisul/envs/s5m/bin/python paper_v2_ppac/make_figure_data_v1.py
Writes paper_v2_ppac/figure_data_v1/*.csv
"""
import csv
import glob
import os
import pickle
import re
import sys

import numpy as np

HOME = os.path.expanduser("~")
WT = f"{HOME}/dev/ssm-baselines"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure_data_v1")
SEEDS = [6554595, 42, 12345, 271828, 314159, 1, 2, 3]
# Canonical names per mambino-paper/CLAUDE.md section 1 (settled 2026-08-03).
# Ordering is fixed: the three 106 K configs first, the 188 K reference last.
ORDER = ["S5-0", "Mambino-0", "Mambino-G", "S5-Dense"]
PARAMS = {"S5-0": 105738, "Mambino-0": 105738, "Mambino-G": 105754, "S5-Dense": 188490}
# PPAC at each config's ATP-optimal design point, honest overlap (v5).
# Regenerate with paper_v2_ppac/chip/multi_array_ppac_v5.py::atp_optimal.
PPAC = {  # config: (PEs, area_mm2, energy_uJ, latency_ms, peak_mW)
    "S5-0":        (3648, 20.62, 421.2, 0.236, 949),
    "Mambino-0":   (2368, 16.01, 451.7, 0.528, 502),
    "Mambino-G":   (2368, 16.01, 471.4, 0.528, 502),
    "S5-Dense":    (6208, 32.09, 563.0, 0.300, 2302),
}


def arg(a, k, d=None):
    return a.get(k, d) if isinstance(a, dict) else getattr(a, k, d)


def classify(a):
    """Full signature -> canonical name, or None if it is a different model."""
    if float(arg(a, "lambda_pc", 0.0)) != 0.0:            # lambda_pc sweep
        return None
    if arg(a, "glu_structure", "dense") not in ("dense", None):   # v2pilot gates
        return None
    if int(arg(a, "epochs", 0)) != 40 or float(arg(a, "p_dropout", 0)) != 0:
        return None
    if arg(a, "dataset") != "listops-classification":
        return None
    if arg(a, "fast_weight", False):                       # Cluster B
        return None
    mam, act, sb = arg(a, "use_mambino_ssm", False), arg(a, "activation_fn"), arg(a, "ssm_size_base")
    sg, gr = arg(a, "surprise_gate", False), arg(a, "glu_rank", 0)
    if not mam:
        return "S5-0" if act == "gelu" else ("S5-Dense" if (sb == 16 and gr == 0) else None)
    if act != "gelu" or sb != 16:
        return None
    if sg and arg(a, "gate_range", "signed") != "signed":  # unsigned arm
        return None
    return "Mambino-G" if sg else "Mambino-0"


def curve(log):
    v, t = [], []
    for ln in open(log, errors="ignore"):
        if "Train Loss:" in ln:
            v.append(float(re.search(r"Val Accuracy:\s*([0-9.]+)", ln).group(1)))
            t.append(float(re.search(r"Test Accuracy:\s*([0-9.]+)", ln).group(1)))
    return np.array(v), np.array(t)


def collect():
    out = {}
    for w in ("S5", "S5-sgate", "S5-sgate2", "S5-clusterb"):
        for mp in glob.glob(f"{WT}/{w}/checkpoints/*/best.meta.pkl"):
            d = os.path.dirname(mp)
            if os.path.basename(d).startswith("chip_"):      # superseded old-SHA
                continue
            log = f"{d}/run.log"
            if not os.path.exists(log):
                continue
            try:
                a = pickle.load(open(mp, "rb"))["args"]
            except Exception:
                continue
            c = classify(a)
            if c is None:
                continue
            v, t = curve(log)
            if len(v) < 40:
                continue
            out.setdefault((c, int(arg(a, "jax_seed"))), []).append(
                (v[:40], t[:40], os.path.basename(d)))
    return out


def w(name, header, rows):
    p = os.path.join(OUT, name)
    with open(p, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(header)
        cw.writerows(rows)
    print(f"  wrote {name:<34} {len(rows)} rows")


def main():
    os.makedirs(OUT, exist_ok=True)
    R = collect()
    pv, mx, curves = {}, {}, {}
    bad = []
    for c in ORDER:
        for s in SEEDS:
            e = R.get((c, s))
            if not e:
                bad.append(f"{c} seed {s} MISSING")
                continue
            if len({round(float(x[1][int(np.argmax(x[0]))]), 6) for x in e}) > 1:
                bad.append(f"{c} seed {s} CONFLICTING duplicates: {[x[2] for x in e]}")
            v, t, _ = e[0]
            pv[(c, s)] = float(t[int(np.argmax(v))])
            mx[(c, s)] = float(t.max())
            curves[(c, s)] = (v, t)
    if bad:
        print("!! COVERAGE / CONFLICT ISSUES:")
        for b in bad:
            print("   " + b)

    # 1 per-seed accuracy (Origin: bar of mean + scatter overlay of the 8 points)
    w("fig1_accuracy_per_seed.csv", ["seed"] + ORDER + [c + "_testmax" for c in ORDER],
      [[s] + [f"{pv[(c,s)]:.4f}" for c in ORDER] + [f"{mx[(c,s)]:.4f}" for c in ORDER]
       for s in SEEDS])

    # 2 summary: accuracy vs params, with error bars
    rows = []
    for c in ORDER:
        a = np.array([pv[(c, s)] for s in SEEDS])
        m = np.array([mx[(c, s)] for s in SEEDS])
        rows.append([c, PARAMS[c], f"{a.mean():.4f}", f"{a.std(ddof=1):.4f}",
                     f"{a.std(ddof=1)/np.sqrt(len(a)):.4f}", f"{a.max():.4f}", len(a),
                     f"{m.mean():.4f}", f"{m.max():.4f}"])
    w("fig2_accuracy_vs_params.csv",
      ["config", "params", "mean", "std", "sem", "best_seed", "n",
       "testmax_mean", "testmax_best_seed"], rows)

    # 3 chip PPAC
    rows = []
    for c in ORDER:
        pe, ar, en, la, pk = PPAC[c]
        acc = np.mean([pv[(c, s)] for s in SEEDS])
        rows.append([c, PARAMS[c], pe, f"{ar:.2f}", f"{en:.1f}", f"{la:.3f}", pk, f"{acc:.4f}"])
    w("fig3_ppac.csv",
      ["config", "params", "PEs", "area_mm2", "energy_uJ", "latency_ms", "peak_mW", "acc"], rows)

    # 4 accuracy per unit chip cost
    rows = []
    for c in ORDER:
        pe, ar, en, la, pk = PPAC[c]
        acc = np.mean([pv[(c, s)] for s in SEEDS])
        rows.append([c, PARAMS[c], f"{acc/ar:.5f}", f"{1000*acc/en:.4f}",
                     f"{acc/(pk/1000):.4f}", f"{1e6*acc/PARAMS[c]:.4f}"])
    w("fig4_efficiency.csv",
      ["config", "params", "acc_per_mm2", "acc_per_mJ", "acc_per_W_peak",
       "acc_per_Mparam"], rows)

    # 5 training curves: mean +- std across the 8 seeds, per epoch
    hdr = ["epoch"]
    for c in ORDER:
        hdr += [f"{c}_val_mean", f"{c}_val_std", f"{c}_test_mean", f"{c}_test_std"]
    rows = []
    for e in range(40):
        r = [e + 1]
        for c in ORDER:
            V = np.array([curves[(c, s)][0][e] for s in SEEDS])
            T = np.array([curves[(c, s)][1][e] for s in SEEDS])
            r += [f"{V.mean():.4f}", f"{V.std(ddof=1):.4f}",
                  f"{T.mean():.4f}", f"{T.std(ddof=1):.4f}"]
        rows.append(r)
    w("fig5_training_curves.csv", hdr, rows)

    # 6 paired per-seed deltas -- the three claims, one column each
    P = [("Mambino-0", "S5-gateless"), ("Mambino-G", "Mambino-0"), ("Mambino-G", "Pure S5")]
    hdr = ["seed"] + [f"{a}_minus_{b}_pp" for a, b in P]
    rows = [[s] + [f"{100*(pv[(a,s)]-pv[(b,s)]):.2f}" for a, b in P] for s in SEEDS]
    for a, b in P:
        d = np.array([pv[(a, s)] - pv[(b, s)] for s in SEEDS])
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  paired {a} - {b}: {100*d.mean():+.2f} pp, t={d.mean()/se:.2f}, "
              f"ahead {int((d>0).sum())}/{len(d)}")
    w("fig6_paired_deltas.csv", hdr, rows)
    print(f"\nOutput dir: {OUT}")


if __name__ == "__main__":
    sys.exit(main())
