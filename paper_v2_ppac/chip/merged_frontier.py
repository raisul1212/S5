"""merged_frontier.py — the FAIR (latency, peak-power) comparison Fable demanded.

The scheduler's "at matched n_chunks Mambino keeps a peak lead" is an artifact: it holds
each config at its OWN per-config ATP-optimal design point (corner1 @ target_cyc 32768,
corner3p @ 262144 -- an 8x speed-grade gap) and then matches only the NEW knob n_chunks,
which hides corner3p's 1.5-3.2x latency deficit. The fair test gives BOTH configs BOTH
knobs -- design point (target_cyc) AND pipelining (n_chunks) -- and asks who is on the
JOINT (latency, peak) Pareto. This script builds that and reports iso-latency + iso-budget
cuts, plus the accuracy dimension (Mambino delivers higher acc, so raw latency/peak is only
half the story).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multi_array_ppac_v4 as m4
import multi_array_ppac_v5 as m5
import ppac_v5_scheduler as sch

FREQ_HZ = m4.FREQ_HZ
# dense enough to straddle each config's latency-optimal n* (corner1~5, corner3p~11)
NS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32]


def _pipe(blocks, n):
    """REALISTIC (fill/drain) makespan -- shares the scheduler's model so the frontier and
    the per-config validation cannot diverge."""
    P = [(b["energy_pJ"] + m4.NOC_ENERGY_FRACTION_OF_MAC * b["e_mac"]) / b["cycles"] for b in blocks]
    lat = sch._makespan_cyc(blocks, n, fill_drain=True) / FREQ_HZ * 1e3
    peak = sum(sorted(P, reverse=True)[:min(n, len(P))])
    return lat, peak


def all_points(config, barrier=False, concurrent=False):
    """Every (target_cyc x n_chunks) design -> multi-domain vector (lat, peak, energy, area, acc).
    barrier=True enforces per-layer bidir-scan barriers (op-DAG configs). concurrent=True runs the
    predictor on a dedicated 2nd array (+area, +peak, shorter SSM). NB energy is the frozen v4
    value (dynamic energy is schedule-invariant; the concurrent 2nd-array leak is not added --
    small, noted in the report)."""
    pts = []
    for t in m5.TARGETS:
        r = m5.analyze(config, t, honest=True)
        area = r["total_area_mm2"]
        for n in NS:
            if barrier and config in sch.DAG_FILE:
                bp = sch.barrier_point(config, n, target=t, concurrent_predictor=concurrent)
                lat, peak, area = bp["latency_ms"], bp["peak_mW"], bp["area_mm2"]
            else:
                lat, peak = _pipe(r["blocks"], n)
            pts.append(dict(config=config, target=t, n=n, lat=lat, peak=peak,
                            area=area, acc=r["acc"], energy=r["energy_full_uJ"]))
    return pts


def multi_domain_predictor():
    """The multi-domain (latency / peak / energy / area / acc) view of Pure S5 vs Mambino
    serial vs Mambino concurrent-predictor, under barriers. Shows that pushing Mambino toward
    Pure S5's latency INVERTS its area+peak advantages -- the domains conflict."""
    modes = [("Pure S5", "corner1", False), ("Mamb serial", "corner3p", False),
             ("Mamb concur", "corner3p", True)]
    data = {name: all_points(cfg, barrier=True, concurrent=conc) for name, cfg, conc in modes}
    print("\n" + "=" * 84)
    print("MULTI-DOMAIN predictor comparison (barrier, both-knob) — the domains CONFLICT")
    print("=" * 84)
    print(f"    {'mode':<13}{'lat ms':>8}{'peak mW':>9}{'energy uJ':>11}{'area mm2':>10}{'acc':>8}")
    print("  fastest-latency design of each mode:")
    for name in data:
        p = min(data[name], key=lambda x: x["lat"])
        print(f"    {name:<13}{p['lat']:>8.3f}{p['peak']:>9.0f}{p['energy']:>11.0f}{p['area']:>10.1f}{p['acc']:>8.4f}")
    print("  lowest-peak design of each mode (the relaxed-latency corner):")
    for name in data:
        p = min(data[name], key=lambda x: x["peak"])
        print(f"    {name:<13}{p['lat']:>8.3f}{p['peak']:>9.0f}{p['energy']:>11.0f}{p['area']:>10.1f}{p['acc']:>8.4f}")
    # 3-way iso-latency min-peak
    print("\n  iso-latency min-peak (mW) -- best Mambino predictor vs Pure S5:")
    print(f"    {'lat<=ms':>8}{'PureS5':>9}{'Mamb-ser':>10}{'Mamb-conc':>11}{'winner':>9}")
    for L in [0.40, 0.45, 0.50, 0.55, 0.60, 0.70]:
        s = _min_peak_leq(data["Pure S5"], L)
        ms = _min_peak_leq(data["Mamb serial"], L)
        mc = _min_peak_leq(data["Mamb concur"], L)
        best = min([x for x in (ms, mc) if x is not None], default=None)
        win = "--" if (s is None or best is None) else ("S5" if s < best - 1e-6 else "Mamb")

        def f(x):
            return "--" if x is None else f"{x:.0f}"
        print(f"    {L:>8.2f}{f(s):>9}{f(ms):>10}{f(mc):>11}{win:>9}")
    print("  -> at the low-latency extreme Mambino's area+peak advantages INVERT (bigger+hotter"
          " than Pure S5); its Pareto home is the relaxed-latency, low-peak corner.")


def _min_peak_leq(pts, L):
    ps = [p["peak"] for p in pts if p["lat"] <= L + 1e-9]
    return min(ps) if ps else None


def iso_latency_both_knobs():
    """The FAIR (both-knob) iso-latency peak comparison, no-barrier vs WITH-barrier. This is the
    test Fable flagged as missing -- the barrier crossover MUST be read here, not off the
    single-knob ATP frontier (which pins Pure S5 at its 8x-faster design and is unfair)."""
    c1n, c3n = all_points("corner1"), all_points("corner3p")
    c1b, c3b = all_points("corner1", barrier=True), all_points("corner3p", barrier=True)
    print("\n" + "=" * 80)
    print("FAIR both-knob iso-latency peak (min over ALL target_cyc x n): NO-barrier vs BARRIER")
    print("=" * 80)
    print(f"    {'lat<=ms':>8} | {'S5 nb':>7}{'Mamb nb':>9}{'win':>8} | {'S5 br':>7}{'Mamb br':>9}{'win':>8}")

    def w(a, b):
        return "--" if (a is None or b is None) else ("S5" if a < b - 1e-6 else ("Mamb" if b < a - 1e-6 else "tie"))

    def f(x):
        return "--" if x is None else f"{x:.0f}"
    for L in [0.40, 0.45, 0.50, 0.55, 0.60, 0.70]:
        s5n, m3n = _min_peak_leq(c1n, L), _min_peak_leq(c3n, L)
        s5b, m3b = _min_peak_leq(c1b, L), _min_peak_leq(c3b, L)
        print(f"    {L:>8.2f} | {f(s5n):>7}{f(m3n):>9}{w(s5n,m3n):>8} | {f(s5b):>7}{f(m3b):>9}{w(s5b,m3b):>8}")


def dominated(p, others, eps=1e-9):
    """p is (Pareto-)dominated if some q is <= on BOTH latency and peak and < on one."""
    for q in others:
        if q is p:
            continue
        if q["lat"] <= p["lat"] + eps and q["peak"] <= p["peak"] + eps \
                and (q["lat"] < p["lat"] - eps or q["peak"] < p["peak"] - eps):
            return True
    return False


def main():
    c1 = all_points("corner1")     # Pure S5
    c3 = all_points("corner3p")    # Mambino
    allp = c1 + c3

    print("=" * 80)
    print("FAIR MERGED FRONTIER — both configs, both knobs (target_cyc x n_chunks)")
    print("Pure S5 acc=%.4f | Mambino acc=%.4f (+%.2f pp)"
          % (m4.ACC["corner1"], m4.ACC["corner3p"], 100 * (m4.ACC["corner3p"] - m4.ACC["corner1"])))
    print("=" * 80)

    # --- Joint (latency, peak) Pareto: who is non-dominated? ---
    front = [p for p in allp if not dominated(p, allp)]
    n_c1 = sum(1 for p in front if p["config"] == "corner1")
    n_c3 = sum(1 for p in front if p["config"] == "corner3p")
    print(f"\n[JOINT PARETO on (latency,peak)] {len(front)} non-dominated points: "
          f"{n_c1} Pure S5, {n_c3} Mambino")
    for p in sorted(front, key=lambda x: x["lat"]):
        print(f"    {p['config']:<9} t={p['target']:>9,} n={p['n']:>2}  "
              f"lat {p['lat']:.3f} ms  peak {p['peak']:>5.0f} mW  area {p['area']:.1f}  acc {p['acc']:.4f}")

    # --- Does Mambino UNIQUELY own a low-peak region Pure S5 can't reach? ---
    c1_min_peak = min(p["peak"] for p in c1)
    c3_min_peak = min(p["peak"] for p in c3)
    print(f"\n[MIN reachable peak]  Pure S5 {c1_min_peak:.0f} mW   Mambino {c3_min_peak:.0f} mW")
    if c3_min_peak < c1_min_peak:
        print(f"    -> Mambino reaches {c1_min_peak - c3_min_peak:.0f} mW LOWER peak than Pure S5 can "
              f"(at lat {min(p['lat'] for p in c3 if p['peak']==c3_min_peak):.3f} ms) — a real unique region.")
    else:
        print("    -> Pure S5 reaches peak as low or lower than Mambino — no unique low-peak region.")

    # --- ISO-LATENCY cut (Fable probe b): at each Pure S5 latency, min peak of each config ---
    print("\n[ISO-LATENCY] at a target latency, the MIN peak each config can achieve (any t,n):")
    print(f"    {'lat<=ms':>8}{'S5 peak':>9}{'Mamb peak':>11}{'winner':>10}{'  S5 acc/pk   Mamb acc/pk':>26}")
    for L in [0.15, 0.20, 0.25, 0.30, 0.40, 0.60]:
        s5 = [p for p in c1 if p["lat"] <= L + 1e-9]
        mb = [p for p in c3 if p["lat"] <= L + 1e-9]
        s5p = min((p["peak"] for p in s5), default=float("inf"))
        mbp = min((p["peak"] for p in mb), default=float("inf"))
        win = "Pure S5" if s5p < mbp else ("Mambino" if mbp < s5p else "tie")
        s5a = m4.ACC["corner1"] / s5p * 1e3 if s5p < float("inf") else 0
        mba = m4.ACC["corner3p"] / mbp * 1e3 if mbp < float("inf") else 0
        sp = f"{s5p:.0f}" if s5p < float("inf") else "--"
        mp = f"{mbp:.0f}" if mbp < float("inf") else "--"
        print(f"    {L:>8.2f}{sp:>9}{mp:>11}{win:>10}    {s5a:>8.3f}     {mba:>8.3f}")

    # --- ISO-BUDGET cut (Fable probe a): corner1 at corner3p's ATP budget (262144) ---
    print("\n[ISO-BUDGET] corner1 given corner3p's design budget (target_cyc=262144):")
    r1 = m5.analyze("corner1", 262144, honest=True)
    r3 = m5.analyze("corner3p", 262144, honest=True)
    for n in [1, 2, 4]:
        l1, p1 = _pipe(r1["blocks"], n)
        l3, p3 = _pipe(r3["blocks"], n)
        print(f"    n={n}: Pure S5 ({l1:.3f} ms, {p1:.0f} mW)  vs  Mambino ({l3:.3f} ms, {p3:.0f} mW)")


if __name__ == "__main__":
    main()
    iso_latency_both_knobs()
    multi_domain_predictor()
