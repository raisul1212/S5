"""Emit the detailed per-corner PPAC CSV for the 4-config v1 paper.

SELF-REGENERATING BY DESIGN. Every chip number is obtained by EXECUTING the
PPAC model, not transcribed. The older `make_figure_data_v1.py` carries a
hardcoded PPAC dict; that dict was audited and found to be an accurate
transcription, but a hardcoded table cannot stay correct across model edits,
so this file calls the model instead.

    multi_array_ppac_v5.atp_optimal(cfg, honest=True)   rows 1,2,3,5
    ppac_v5_scheduler.barrier_point(cfg, 1, concurrent_predictor=True)  row 4
    peak_power_scenarios.scenarios(cfg)                 complete-chip peak band

`honest=True` IS LOAD-BEARING. It disables the v4 `wall_clock_rep` 24->16
predictor-overlap freebie, which grants free throughput on an array already at
100% utilization. Never regenerate a row with honest=False.

TWO POWER COLUMNS, AND THEY ARE NOT ON THE SAME BASIS
-----------------------------------------------------
`avg_mW` = energy_full / latency. This is the paper's convention (paper
CLAUDE.md section 3: "Never write 'peak power'"), and it is what tab:ppac,
tab:threeaxis and every IPW number use.

`peak_datapath_mW` covers the GEMM ARRAYS ONLY -- elementwise, state-SRAM and
leakage power are not on the v5 timeline (multi_array_ppac_v5.py:19-25), and
those account for 49-57% of total energy. That is why peak_datapath_mW is
BELOW avg_mW on four of five rows, which is otherwise physically impossible.
The two columns must never be divided by each other, and a figure must not mix
them. `peak_chip_lo/hi_mW` bracket the complete-chip peak and are the only
peak numbers safe to compare against avg_mW.

Usage (Windows or cluster; pure-Python analytical model, no GPU needed):
    python paper_v2_ppac/make_ppac_detailed_csv.py
Writes ppac_detailed_4corners.csv into every figure-data directory below.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "chip"))

import multi_array_ppac_v5 as m5          # noqa: E402
import peak_power_scenarios as pps        # noqa: E402
import ppac_v5_scheduler as sch           # noqa: E402

# Output lands next to the figure CSVs so Origin finds everything in one place.
OUT_DIRS = [
    os.path.join(HERE, "figure_data_v1"),
    os.path.expanduser("~/dev/nc-block/paper/mambino-paper/figure_data_v1"),
]

# Canonical paper name -> model config key.
CONFIGS = [
    ("S5-0",      "config5",     105738),
    ("Mambino-0", "config4",     105738),
    ("Mambino-G", "mambino2p0",  105754),
    ("S5-Dense",  "corner1",     188490),
]

# test@peakval means re-derived from the Gilbreth run.logs (n=8 matched seeds).
# The model bakes accuracy in; assert the two agree so a drift in either fails
# loudly instead of silently publishing a stale number.
ACC_FROM_LOGS = {
    "config5": 0.5933, "config4": 0.6008, "mambino2p0": 0.6018, "corner1": 0.6102,
}
# n=16 (2026-08-06): original eight seeds plus the pre-registered extension 4..11.
# Supersedes the n=8 values 0.5917 / 0.5993 / 0.6050 / 0.6089.

HEADER = [
    "config", "schedule", "params", "PEs", "area_mm2", "latency_ms",
    "throughput_ips", "energy_uJ", "avg_mW", "peak_datapath_mW",
    "peak_chip_lo_mW", "peak_chip_hi_mW", "acc", "acc_per_mJ", "acc_per_mm2",
    "IPW_acc_per_W_avg", "provenance",
]

# Per-inference energy by chip component. The eight fields sum to energy_full_uJ
# exactly; the generator asserts it, so a stacked bar built from these columns
# cannot silently disagree with the energy column of the main table.
E_FIELDS = ["e_mac_uJ", "e_wsram_uJ", "e_asram_uJ", "e_spad_uJ",
            "e_elem_uJ", "e_state_uJ", "e_noc_ctrl_uJ", "e_leak_uJ"]
E_HEADER = (["config", "params"]
            + [f[2:-3] + "_uJ" for f in E_FIELDS]      # strip "e_" prefix only
            + ["total_uJ", "elementwise_pct", "mac_pct"])


def derived(acc, area, energy_uJ, avg_mW):
    return [
        f"{1000.0 * acc / energy_uJ:.6f}",     # acc per mJ
        f"{acc / area:.8f}",                   # acc per mm2
        f"{acc / (avg_mW / 1000.0):.6f}",      # IPW, average-power basis
    ]


def concurrent_throughput(atp):
    """Throughput once the predictor shape gets its own array (rep 24 -> 16).

    Pipelined steady state = 1 GHz / max(per-block cycles), NOT 1000/latency.
    Derived rather than read off: barrier_point does not return throughput.
    """
    # NB: _predictor_collapse returns the rep RATIO (16/24), not a rep count.
    shape, rep_ratio, _area, _pow = sch._predictor_collapse("mambino2p0", True, None)
    worst = 0.0
    for b in atp["blocks"]:
        cyc = b["cycles"]
        if b["shape"] == shape:
            cyc = cyc * rep_ratio              # collapse, paid for in PEs+area
        worst = max(worst, cyc)
    return 1e9 / worst, shape, rep_ratio


def main():
    rows, erows = [], []
    for name, cfg, params in CONFIGS:
        r = m5.atp_optimal(cfg, honest=True)
        acc = r["acc"]

        # Energy breakdown. Assert the components reconstruct the total.
        parts = [r[f] for f in E_FIELDS]
        tot = r["energy_full_uJ"]
        assert abs(sum(parts) - tot) < 1e-6, f"{name}: components {sum(parts)} != {tot}"
        erows.append([name, params] + [f"{v:.6f}" for v in parts]
                     + [f"{tot:.6f}", f"{100 * r['e_elem_uJ'] / tot:.2f}",
                        f"{100 * r['e_mac_uJ'] / tot:.2f}"])
        assert abs(acc - ACC_FROM_LOGS[cfg]) < 1e-9, (
            f"{name}: model acc {acc} != log-verified {ACC_FROM_LOGS[cfg]}")

        # Complete-chip peak band: [array_peak, array_peak + background_power].
        _atp, bg_uJ, scen = pps.scenarios(cfg)
        serial = next(s for s in scen if s["n"] == 1)

        area, lat = r["total_area_mm2"], r["lat_serial_ms"]
        energy, avg = r["energy_full_uJ"], r["avg_power_serial_mW"]
        rows.append([
            name, "serial", params, r["total_pe"], f"{area:.6f}", f"{lat:.6f}",
            f"{r['throughput_ips']:.4f}", f"{energy:.6f}", f"{avg:.6f}",
            f"{r['peak_serial_mW']:.6f}",
            f"{serial['total_lo']:.6f}", f"{serial['total_hi']:.6f}",
            f"{acc:.4f}", *derived(acc, area, energy, avg), "executed",
        ])

        # Mambino-G additionally reports a concurrent-predictor schedule.
        if cfg == "mambino2p0":
            b = sch.barrier_point(cfg, 1, concurrent_predictor=True)
            tput, shape, rep_ratio = concurrent_throughput(r)
            # Pin the audited value; a model edit that moves it must be noticed.
            assert abs(tput - 6847.064) < 0.01, f"concurrent throughput drifted: {tput}"
            area_c, lat_c = b["area_mm2"], b["latency_ms"]
            energy_c, avg_c = b["energy_uJ"], b["avg_power_mW"]
            peak_c = b["peak_mW"]
            bg_pow = bg_uJ / lat_c                 # uJ/ms == mW
            rows.append([
                name, "concurrent", params, r["total_pe"] + 512,
                f"{area_c:.6f}", f"{lat_c:.6f}", f"{tput:.4f}",
                f"{energy_c:.6f}", f"{avg_c:.6f}", f"{peak_c:.6f}",
                f"{peak_c:.6f}", f"{peak_c + bg_pow:.6f}",
                f"{acc:.4f}", *derived(acc, area_c, energy_c, avg_c),
                f"executed; PEs +512 and throughput derived "
                f"({shape} rep x{rep_ratio:.4f})",
            ])

    # Keep S5-Dense last: the three 106 K configs first, then the 188 K reference.
    order = {"S5-0": 0, "Mambino-0": 1, "Mambino-G": 2, "S5-Dense": 3}
    rows.sort(key=lambda x: (order[x[0]], x[1] != "serial"))

    for d in OUT_DIRS:
        if not os.path.isdir(d):
            print(f"  SKIP (no such dir) {d}")
            continue
        p = os.path.join(d, "ppac_detailed_4corners.csv")
        with open(p, "w", newline="") as f:
            cw = csv.writer(f)
            cw.writerow(HEADER)
            cw.writerows(rows)
        print(f"  wrote {p}  ({len(rows)} rows)")

        pe = os.path.join(d, "energy_breakdown_4corners.csv")
        with open(pe, "w", newline="") as f:
            cw = csv.writer(f)
            cw.writerow(E_HEADER)
            cw.writerows(erows)
        print(f"  wrote {pe}  ({len(erows)} rows)")

    print()
    for r in rows:
        print(f"  {r[0]:<10} {r[1]:<11} PE={r[3]:<5} area={r[4]:>9} "
              f"lat={r[5]:>8} E={r[7]:>10} avg={r[8]:>11} peakDP={r[9]:>11}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
