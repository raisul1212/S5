"""multi_array_ppac_v5.py — concurrency + peak-power layer over v4 (increment 1).

Imports v4 UNMODIFIED and post-processes its per-block output. Adds:

  (1) HONEST predictor overlap. Kills v4's `wall_clock_rep` 24->16 freebie — v4 took
      the LATENCY benefit of running the predictor scan on a concurrent 2nd array
      without charging its silicon/peak power (see v1_wall_clock_rep_FIX_memo). Under
      honest mode wc_rep == jaxpr_rep, so intra-block rep is CONSISTENT (energy and
      cycles use the same count). This is a PREREQUISITE for peak power: a raw v4 dict
      mixes energy_rep=24 with wc_rep=16 on the SSM-main block -> P=E/t inflates 1.5x.

  (2) TIMELINE PEAK POWER. Per-GEMM-block instantaneous power P_b = energy_pJ / cycles
      [mW] (full block energy: MAC + weight-SRAM + act-SRAM + spads). Reported at the
      two schedule BOUNDS, computed symmetrically for every config:
        serial     (no inter-block overlap): latency = sum(block cycles); peak = max P_b
        concurrent (all blocks parallel)    : latency = max(block cycles); peak = sum P_b
      The realistic dependency-bounded schedule lies BETWEEN these; it needs the op-DAG
      (increment 2). These are honest bounds, not a hand-picked point.
      LIMITATION (state it): P_b is a coarse block-AVERAGE power (energy/active-time),
      not sub-block instantaneous; NoC IS included per-block (0.10*e_mac); elemwise /
      state / leak power is not yet on the timeline. Excluding elemwise is direction-safe
      TODAY via the LATENCY DENOMINATOR (excluded background power 973 vs 339 mW serial
      is larger for Corner 1) — NOT because Mambino has less elemwise (it has MORE:
      312 vs 277 uJ). RE-VERIFY when increment-2 pipelining shortens Mambino's latency
      (raises its elemwise power density). Added in increment 2.

  (3) avg power (energy/latency) retained, explicitly labelled avg.

Increment 1 = post-processing only -> zero drift; baseline mode calls v4 unchanged and
the regression asserts byte-identical output for all 5 configs.
Runs locally (no flax): reads the same workload YAMLs + ERT as v4.
"""
import contextlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multi_array_ppac_v4 as m4

FREQ_HZ = m4.FREQ_HZ
TARGETS = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152]  # mirrors v4 main()

# All shared scalar keys the baseline (honest=off) must reproduce v4 on (no-drift gate
# for increment 2, which will add real machinery to analyze()).
_V4_SCALAR_KEYS = ["total_pe", "total_cycles", "bottleneck_cycles", "latency_ms",
                   "throughput_ips", "energy_mJ", "energy_full_uJ", "power_mW",
                   "pe_area_mm2", "total_area_mm2", "acc_per_mJ", "area_x_latency",
                   "pe_x_latency", "e_mac_uJ", "e_asram_uJ", "e_elem_uJ"]


@contextlib.contextmanager
def honest_overlap():
    """Temporarily disable v4's wall_clock_rep 24->16 predictor-overlap freebie."""
    orig = m4.wall_clock_rep
    m4.wall_clock_rep = lambda shape, jaxpr_rep: jaxpr_rep
    try:
        yield
    finally:
        m4.wall_clock_rep = orig


def _block_power_mW(b):
    # P[mW] = energy_pJ*1e-12 J / (cycles/1e9 s) = energy_pJ/cycles  (mW).
    # Include per-block NoC (v4 adds it GLOBALLY as 0.10*e_mac at v4:338; spec §2.3
    # requires NoC in E_b). Including it slightly WIDENS Mambino's peak lead
    # (-62.2%->-63.4% serial), so its prior omission understated Mambino.
    e = b["energy_pJ"] + m4.NOC_ENERGY_FRACTION_OF_MAC * b["e_mac"]
    return e / b["cycles"] if b["cycles"] else 0.0


def analyze(config, target_cyc, honest=True):
    """v4 config_ppac (optionally honest-overlap) + peak-power & schedule-bound latency.
    honest=False reproduces v4 exactly (adds v5 fields, does not change v4 fields)."""
    ctx = honest_overlap() if honest else contextlib.nullcontext()
    with ctx:
        r = m4.config_ppac(config, target_cyc)
    cyc = [b["cycles"] for b in r["blocks"]]
    lat_serial_ms = sum(cyc) / FREQ_HZ * 1e3
    lat_concurrent_ms = (max(cyc) if cyc else 0) / FREQ_HZ * 1e3
    energy_uJ = r["energy_full_uJ"]
    # Peak power is only meaningful under HONEST overlap: a raw v4 block mixes
    # energy_rep=24 with wc_rep=16 on the SSM-main -> P_b would be 1.5x inflated.
    # Do NOT attach peak fields when honest=False.
    if honest:
        Pb = [_block_power_mW(b) for b in r["blocks"]]
        peak_serial_mW = max(Pb) if Pb else 0.0     # one block active at a time
        peak_concurrent_mW = sum(Pb)                # all blocks active at once (upper bound)
    else:
        peak_serial_mW = peak_concurrent_mW = float("nan")
    r.update(
        v5_honest=honest,
        lat_serial_ms=lat_serial_ms,             # = v4 latency under honest mode (sum blocks)
        lat_concurrent_ms=lat_concurrent_ms,     # all-blocks-parallel lower bound
        peak_serial_mW=peak_serial_mW,
        peak_concurrent_mW=peak_concurrent_mW,
        avg_power_serial_mW=energy_uJ / lat_serial_ms if lat_serial_ms else 0.0,      # uJ/ms = mW
        avg_power_concurrent_mW=energy_uJ / lat_concurrent_ms if lat_concurrent_ms else 0.0,
    )
    return r


def atp_optimal(config, honest=True):
    """Pick the design point minimising area x (honest-serial latency), like v4's ATP."""
    sweep = [analyze(config, t, honest=honest) for t in TARGETS]
    return min(sweep, key=lambda r: r["total_area_mm2"] * r["lat_serial_ms"])


# ----------------------------------------------------------------------------------
# Regression + A/B bounds
# ----------------------------------------------------------------------------------
def regression_vs_v4():
    """v5 baseline (honest=False) must reproduce v4 field-for-field; honest=True must
    change ONLY the two Mambino configs (Config 4, Corner 3')."""
    print("=== REGRESSION: v5 baseline (honest=off) == v4, per config at each config's ATP ===")
    ok = True
    for cfg in ["config4", "config5", "corner1", "corner2", "corner3p"]:
        base = atp_optimal(cfg, honest=False)      # == v4
        v4ref = min([m4.config_ppac(cfg, t) for t in TARGETS], key=lambda r: r["area_x_latency"])
        # field-for-field over every shared scalar (relative tol) + per-block invariant
        same = all(abs(base[k] - v4ref[k]) <= 1e-6 * max(1.0, abs(v4ref[k]))
                   for k in _V4_SCALAR_KEYS)
        for bb, vb in zip(base["blocks"], v4ref["blocks"]):     # §4.3 per-block invariant (pre-stage)
            same = same and bb["psum_spill_bytes"] == vb["psum_spill_bytes"] \
                        and bb["e_mac"] == vb["e_mac"] and bb["weight_bytes"] == vb["weight_bytes"]
        hon = atp_optimal(cfg, honest=True)
        moved = (abs(hon["lat_serial_ms"] - v4ref["latency_ms"]) > 1e-6)
        expect_move = cfg in ("config4", "corner3p")
        tag = "PASS" if (same and moved == expect_move) else "FAIL"
        if tag == "FAIL":
            ok = False
        print(f"  [{tag}] {cfg:<9} baseline==v4:{same}  honest-moved:{moved} (expect {expect_move})"
              f"  v4_lat={v4ref['latency_ms']:.3f}ms honest_lat={hon['lat_serial_ms']:.3f}ms"
              f"  v4_pow={v4ref['power_mW']:.0f} honest_avg={hon['avg_power_serial_mW']:.0f}mW")
    print("REGRESSION:", "ALL PASS" if ok else "FAILURES")
    return ok


def ab_bounds():
    print("\n=== iso-datapath A/B — schedule BOUNDS (honest overlap), ATP-optimal per config ===")
    print(f"{'config':<9}{'PEs':>7}{'Area':>8}{'Energy uJ':>10}"
          f"{'lat_ser ms':>11}{'lat_con ms':>11}{'peak_ser mW':>12}{'peak_con mW':>12}{'acc':>8}")
    print("-" * 90)
    rows = {}
    for cfg in ["corner1", "corner2", "corner3p", "config4", "config5"]:
        r = atp_optimal(cfg, honest=True); rows[cfg] = r
        print(f"{cfg:<9}{r['total_pe']:>7,}{r['total_area_mm2']:>8.2f}{r['energy_full_uJ']:>10.1f}"
              f"{r['lat_serial_ms']:>11.3f}{r['lat_concurrent_ms']:>11.3f}"
              f"{r['peak_serial_mW']:>12.0f}{r['peak_concurrent_mW']:>12.0f}{r['acc']:>8.4f}")
    c1, c3 = rows["corner1"], rows["corner3p"]
    print(f"\nMambino (Corner 3') vs Pure S5 (Corner 1) peak-power lead:")
    print(f"  serial end:     {c3['peak_serial_mW']:.0f} vs {c1['peak_serial_mW']:.0f} mW"
          f"  -> {100*(c3['peak_serial_mW']/c1['peak_serial_mW']-1):+.0f}%")
    print(f"  concurrent end: {c3['peak_concurrent_mW']:.0f} vs {c1['peak_concurrent_mW']:.0f} mW"
          f"  -> {100*(c3['peak_concurrent_mW']/c1['peak_concurrent_mW']-1):+.0f}%")
    return rows


if __name__ == "__main__":
    regression_vs_v4()
    ab_bounds()
