"""peak_power_scenarios.py — what v5 lets us say about PEAK power, per scenario.

v5's scheduler gives the SYSTOLIC-ARRAY peak exactly: peak(n) = Σ top-min(n,n_stages) stage
powers (the co-active GEMM arrays at concurrency n). Three scenarios per config:
  - SERIAL (n=1)            : one array active  -> lowest peak, longest latency.
  - LATENCY-OPTIMAL (n*)    : the realistic min-latency operating point (fill/drain U-curve).
  - MAX-THROUGHPUT (n>=n_st): all arrays co-active -> highest peak, near-min latency.

HONESTY GAP (Fable-flagged, not yet closed): the array peak EXCLUDES the elementwise/state/
leak background (which runs on the vector unit + SRAM, not the arrays). We don't yet have the
elem TIMELINE (that needs the op-DAG's per-segment elem class_ops — the next refinement), so
here we BRACKET the complete-chip peak:
    total_peak ∈ [ array_peak ,  array_peak + background_avg_power ]
  lower = background not coincident with the array peak; upper = fully coincident.
  background_avg_power = (e_elem + e_state + e_leak) / latency  (rises as pipelining shortens
  latency). The true peak is inside this band; the elem-timeline refinement will tighten it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multi_array_ppac_v4 as m4
import multi_array_ppac_v5 as m5
import ppac_v5_scheduler as sch


def _background_uJ(config):
    """Non-array energy (elementwise + state SRAM + leak), honest overlap, at the ATP design."""
    b = sch._atp(config, honest=True)
    with m5.honest_overlap():
        v4 = m4.config_ppac(config, b["target_cyc"])
    return v4["e_elem_uJ"] + v4["e_state_uJ"] + v4["e_leak_uJ"]


def scenarios(config):
    b = sch._atp(config, honest=True)
    ns = len(b["blocks"])
    bg = _background_uJ(config)
    rows = []
    for name, n in [("serial (n=1)", 1),
                    ("latency-opt (n*)", sch.optimal_n(config)["n_chunks"]),
                    ("max-thruput (n>=n_st)", max(ns, 32))]:
        p = sch.pipeline_point(config, n, fill_drain=True)
        bg_pow = bg / p["latency_ms"]                      # uJ/ms = mW
        rows.append(dict(name=name, n=p["n_chunks"], lat=p["latency_ms"],
                         array_peak=p["peak_mW"], bg_pow=bg_pow,
                         total_lo=p["peak_mW"], total_hi=p["peak_mW"] + bg_pow))
    return b, bg, rows


def main():
    print("=" * 92)
    print("PEAK-POWER ESTIMATES BY SCENARIO (v5) — array peak exact; complete-chip = band w/ background")
    print("=" * 92)
    for cfg in ["corner1", "corner2", "corner3p", "config4", "config5", "mambino2p0"]:
        b, bg, rows = scenarios(cfg)
        tag = "Pure S5" if cfg in ("corner1", "corner2") else "Mambino"
        print(f"\n{cfg} ({tag}, acc {m4.ACC[cfg]:.4f}, area {b['total_area_mm2']:.1f} mm2, "
              f"background energy {bg:.0f} uJ):")
        print(f"    {'scenario':<22}{'n':>4}{'lat ms':>9}{'array pk':>10}{'bg pwr':>9}"
              f"{'  TOTAL peak band (mW)':>24}")
        for r in rows:
            print(f"    {r['name']:<22}{r['n']:>4}{r['lat']:>9.3f}{r['array_peak']:>10.0f}"
                  f"{r['bg_pow']:>9.0f}    {r['total_lo']:>6.0f} - {r['total_hi']:>6.0f}")

    # Fair iso-latency peak comparison (Pure S5 vs Mambino) WITH background — does the
    # background change the iso-latency winner? (Fable Q5)
    print("\n" + "=" * 92)
    print("FAIR iso-latency peak: min COMPLETE-chip peak (array + background) each config can hit")
    print("=" * 92)
    print(f"    {'lat<=ms':>8}{'S5 array':>10}{'S5 total(hi)':>14}{'Mamb array':>12}"
          f"{'Mamb total(hi)':>16}{'winner(hi)':>12}")
    for L in [0.30, 0.40, 0.50, 0.60, 0.80]:
        s5 = _min_total_peak("corner1", L)
        mb = _min_total_peak("corner3p", L)
        win = "--"
        if s5 and mb:
            win = "Pure S5" if s5["hi"] < mb["hi"] else ("Mambino" if mb["hi"] < s5["hi"] else "tie")
        sa = f"{s5['array']:.0f}" if s5 else "--"; sh = f"{s5['hi']:.0f}" if s5 else "--"
        ma = f"{mb['array']:.0f}" if mb else "--"; mh = f"{mb['hi']:.0f}" if mb else "--"
        print(f"    {L:>8.2f}{sa:>10}{sh:>14}{ma:>12}{mh:>16}{win:>12}")
    print("\n(band = [array-only, array+background]; 'total(hi)' is the conservative upper bracket.)")


def _min_total_peak(config, L):
    """Over all (target_cyc, n) with latency<=L, the point with min complete-chip upper peak."""
    best = None
    for t in m5.TARGETS:
        r = m5.analyze(config, t, honest=True)
        bg = _background_uJ_at(config, t, r)
        for n in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32]:
            lat = sch._makespan_cyc(r["blocks"], n, fill_drain=True) / m4.FREQ_HZ * 1e3
            if lat > L + 1e-9:
                continue
            P = [(bb["energy_pJ"] + m4.NOC_ENERGY_FRACTION_OF_MAC * bb["e_mac"]) / bb["cycles"]
                 for bb in r["blocks"]]
            array_pk = sum(sorted(P, reverse=True)[:min(n, len(P))])
            hi = array_pk + bg / lat
            if best is None or hi < best["hi"]:
                best = dict(array=array_pk, hi=hi, lat=lat, t=t, n=n)
    return best


def _background_uJ_at(config, t, r):
    with m5.honest_overlap():
        v4 = m4.config_ppac(config, t)
    return v4["e_elem_uJ"] + v4["e_state_uJ"] + v4["e_leak_uJ"]


if __name__ == "__main__":
    main()
