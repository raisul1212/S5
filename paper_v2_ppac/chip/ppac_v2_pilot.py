"""Phase-6 chip PPAC for the v2 gate pilot (4 arms), reusing multi_array_ppac_v4's
config_ppac + ATP sweep. Does NOT touch main() and does NOT overwrite the archived
multi_array_sweep_v4.json — writes multi_array_sweep_v2pilot.json instead.

Includes corner3p (v1 Corner 3') as a cross-check: dense_r40 is byte-identical to
Corner 3', so its PPAC MUST reproduce corner3p's (~19.76 mm2, 816 mW) — a pipeline sanity gate.

Run on Gilbreth (needs the workload_*.yaml from extract_workload_from_jaxpr.py + the
out_accelergy_384KB_16x16 ERT), from the chip dir:
    python paper_v2_ppac/chip/ppac_v2_pilot.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multi_array_ppac_v4 as m

TARGETS = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152]

# Pilot arms: all Corner 3' slot (Mambino P=8, 3 trajectories, 384KB act-SRAM tier).
# acc = pilot test@peakval means (n=3).
ARMS = {
    "v2pilot_dense_r40": dict(tier="384KB", ntraj=3, acc=0.6147),
    "v2pilot_dense_r32": dict(tier="384KB", ntraj=3, acc=0.6035),
    "v2pilot_monarch":   dict(tier="384KB", ntraj=3, acc=0.6092),
    "v2pilot_blockdiag": dict(tier="384KB", ntraj=3, acc=0.6072),
}
for name, d in ARMS.items():
    m.CONFIG_TIER[name] = d["tier"]
    m.N_TRAJ[name] = d["ntraj"]
    m.ACC[name] = d["acc"]

# corner3p (v1 Corner 3') runs from its existing v1 workload as the cross-check.
ORDER = ["corner3p", "v2pilot_dense_r40", "v2pilot_dense_r32",
         "v2pilot_monarch", "v2pilot_blockdiag"]

best_by = {}
for name in ORDER:
    sweep = [m.config_ppac(name, t) for t in TARGETS]
    best_by[name] = min(sweep, key=lambda r: r["area_x_latency"])

print("=" * 118)
print("v2 GATE PILOT — chip PPAC (ATP-optimal per arm, v4 methodology). "
      "corner3p = v1 cross-check for dense_r40.")
print("=" * 118)
hdr = (f"{'arm':<20}{'acc':>8}{'PEs':>8}{'Area mm2':>10}{'Power mW':>10}"
       f"{'Energy uJ':>11}{'Latency ms':>12}{'acc/mm2 e-2':>13}{'acc/mW e-3':>12}")
print(hdr); print("-" * 118)
for name in ORDER:
    b = best_by[name]
    acc = m.ACC[name]
    area = b["total_area_mm2"]; pw = b["power_mW"]
    print(f"{name:<20}{acc:>8.4f}{b['total_pe']:>8,}{area:>10.2f}{pw:>10.0f}"
          f"{b['energy_full_uJ']:>11.1f}{b['latency_ms']:>12.3f}"
          f"{100*acc/area:>13.3f}{1000*acc/pw:>12.3f}")

# Deltas vs the v1 dense gate (dense_r40 == corner3p).
print("\n" + "=" * 70)
print("Δ vs dense_r40 (the v1 gate) — area / power / acc:")
print("=" * 70)
d40 = best_by["v2pilot_dense_r40"]
for name in ["v2pilot_dense_r32", "v2pilot_monarch", "v2pilot_blockdiag"]:
    b = best_by[name]
    da = 100 * (b["total_area_mm2"] / d40["total_area_mm2"] - 1)
    dp = 100 * (b["power_mW"] / d40["power_mW"] - 1)
    dacc = 100 * (m.ACC[name] - m.ACC["v2pilot_dense_r40"])
    print(f"  {name:<20} area {da:+6.1f}%   power {dp:+6.1f}%   acc {dacc:+.2f} pp")

xcheck = best_by["corner3p"]; d40b = best_by["v2pilot_dense_r40"]
print(f"\nCROSS-CHECK dense_r40 vs corner3p: area {d40b['total_area_mm2']:.2f} vs "
      f"{xcheck['total_area_mm2']:.2f} mm2, power {d40b['power_mW']:.0f} vs "
      f"{xcheck['power_mW']:.0f} mW  -> {'MATCH' if abs(d40b['total_area_mm2']-xcheck['total_area_mm2'])<0.01 and abs(d40b['power_mW']-xcheck['power_mW'])<1 else 'DIFFER (investigate)'}")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "multi_array_sweep_v2pilot.json")
with open(out, "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "blocks"}
               for k, v in best_by.items()}, f, indent=2, default=str)
print(f"\nwrote {out}")
