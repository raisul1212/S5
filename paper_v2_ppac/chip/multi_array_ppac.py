"""Multi-array PPAC with max-util + no-overprovision + area + memory-access energy.

For each block in a config's workload:
  1. Rule: pick SMALLEST 100%-util standard rectangle whose total_cyc <= target.
     If no std size divides the dim (K=20, N=10), fall back to smallest std giving
     the best achievable util.
  2. Sweep target bottleneck cycles to find the silicon-latency-product optimum.
  3. Sum per-block metrics: PE silicon, memory access energy (per-block ERT),
     activation SRAM area (per config's max footprint), weight/state SRAM area.

Assumes:
  - Each block has its own dedicated array (multi-array chip topology)
  - Blocks execute serially per sample (latency = sum of block cycles)
  - Throughput = 1 / max(block cycles)  [pipeline steady state]
  - Weight-stationary mapping per block
  - Per-block ERT selected by config's SRAM tier (320KB / 384KB / 512KB)
  - SRAM area: linear model calibrated to 22 nm high-density (~0.7 Mb/mm2)
"""
from __future__ import annotations
import csv, json, math, os, sys
from pathlib import Path
import yaml
sys.stdout.reconfigure(encoding='utf-8')

CHIP = Path(__file__).resolve().parent
WORKLOADS = CHIP.parent / "workloads"

STD = [8, 16, 32, 64]
FREQ_HZ = 1e9

# --- Silicon area model (22 nm INT8) ------------------------------------------
PE_AREA_UM2 = 3000                  # one INT8 MAC PE with weight+psum spad
SRAM_MBITS_PER_MM2 = 0.7            # 22 nm high-density SRAM
def sram_area_mm2(bytes_):
    return (bytes_ * 8 / 1e6) / SRAM_MBITS_PER_MM2

# --- Per-config chip tiers (from earlier per-config sizing analysis) ----------
CONFIG_TIER = {
    "config4":  "320KB",   # 106K Mambino gelu, max footprint 258 KB
    "config5":  "320KB",   # 106K Pure S5 gelu, max footprint 260 KB
    "corner1":  "512KB",   # 188K Pure S5 dense, max footprint 512 KB
    "corner2":  "384KB",   # 188K Pure S5 low-rank, max footprint 336 KB
    "corner3p": "384KB",   # 188K Mambino low-rank, max footprint 336 KB
}
TIER_KB = {"320KB": 320, "384KB": 384, "512KB": 512}

ACC = {"config4": 0.5993, "config5": 0.5917,
       "corner1": 0.6089, "corner2": 0.5991, "corner3p": 0.6138}

# Predictor s(t-1) overlap: applies ONLY to Corner 3' (jaxpr_rep=24), collapsing
# to wall-clock=16 for LATENCY purposes. Corner 1's SSM main has same shape but
# jaxpr_rep=8 (no predictor) — must not double-count. ENERGY is always charged at
# jaxpr_rep (overlap hides time, not joules — predictor's 8 SSM instances still
# consume MAC + memory energy).
def wall_clock_rep(shape, jaxpr_rep):
    if shape == (8, 2048, 128, 'complex_real') and jaxpr_rep == 24:
        return 16
    return jaxpr_rep

# --- Helpers ------------------------------------------------------------------
def ldiv(x, m):
    """Largest divisor of x that is <= m."""
    if x <= m: return x
    b = 1
    for f in range(1, m + 1):
        if x % f == 0 and f > b: b = f
    return b

def util(ay, ax, K, N):
    return 100 * (ldiv(K, ay) * ldiv(N, ax)) / (ay * ax)

def cyc_per_inst(M, N, K, ay, ax, dtype_scale):
    ray = ldiv(K, ay); rax = ldiv(N, ax)
    return M * math.ceil(K / ray) * math.ceil(N / rax) * dtype_scale

def parse_ert(ert_path):
    ert = yaml.safe_load(open(ert_path))["ERT_summary"]["table_summary"]
    out = {}
    for c in ert:
        short = c['name'].split('.')[-1]
        if short not in out:
            out[short] = {a['name']: float(a['energy']) for a in c['actions']}
    return out

# --- Block-level PPAC ---------------------------------------------------------
def block_ppac(g, ay, ax, ert, wc_rep, energy_rep):
    """Return dict of per-block metrics.
    wc_rep: wall-clock repeat (for latency/throughput; predictor overlap collapses).
    energy_rep: always g['repeat'] (JAXPR-faithful; overlap hides time, not joules).
    """
    M, N, K, dtype = g['M'], g['N'], g['K'], g['dtype']
    ds = g['per_mac_flops'] // 2
    output_scale = 2 if dtype == 'complex_complex' else 1
    cyc = cyc_per_inst(M, N, K, ay, ax, ds)
    total_cyc = cyc * wc_rep

    n_pe = ay * ax
    real_macs_per_inst = M * N * K * ds
    total_real_macs = real_macs_per_inst * energy_rep     # JAXPR-faithful

    # Memory bytes per instance (INT8)
    weight_bytes = K * N * ds
    K_tp = math.ceil(N / ldiv(N, ax))
    act_read_bytes = M * K * K_tp * ds
    act_write_bytes = M * N * output_scale

    # Energy (pJ) per block, summed over ENERGY_rep instances (not wc_rep)
    e_mac = total_real_macs * ert['mac']['compute']
    e_w_sram = math.ceil(weight_bytes / 8) * ert['weight_sram']['read'] * energy_rep
    e_a_sram = (math.ceil(act_read_bytes / 8) * ert['activation_sram']['read']
              + math.ceil(act_write_bytes / 8) * ert['activation_sram']['write']) * energy_rep
    e_w_spad = total_real_macs * ert['weights_spad']['read']
    e_p_spad = total_real_macs * (ert['psum_spad']['read'] + ert['psum_spad']['update'])
    energy = e_mac + e_w_sram + e_a_sram + e_w_spad + e_p_spad

    return dict(cycles=total_cyc, silicon_pe=n_pe, energy_pJ=energy,
                util_pct=util(ay, ax, K, N), array=f"{ay}x{ax}",
                e_mac_pJ=e_mac, e_wsram_pJ=e_w_sram, e_asram_pJ=e_a_sram,
                real_macs=total_real_macs)

# --- Rule: smallest 100%-util rectangle whose cycles fit under target ----------
def choose_array(g, target_cyc, wc_rep):
    M, N, K, dtype = g['M'], g['N'], g['K'], g['dtype']
    ds = g['per_mac_flops'] // 2

    # Enumerate all std x std rectangles
    candidates = []
    for ay in STD:
        for ax in STD:
            cyc = cyc_per_inst(M, N, K, ay, ax, ds) * wc_rep
            u = util(ay, ax, K, N)
            candidates.append((ay * ax, u, cyc, ay, ax))

    # Prefer 100% util AND cycles fit target
    max_u = max(c[1] for c in candidates)
    # If some rectangle hits 100% util, restrict to those
    if max_u == 100.0:
        pool = [c for c in candidates if c[1] == 100.0]
    else:
        # No 100% util possible; use best util
        pool = [c for c in candidates if c[1] == max_u]

    # Among pool, prefer smallest silicon whose cycles <= target
    fits = [c for c in pool if c[2] <= target_cyc]
    if fits:
        return min(fits, key=lambda c: c[0])
    # Nothing fits; return fastest available (smallest cycles)
    return min(pool, key=lambda c: c[2])

# --- Config-level PPAC --------------------------------------------------------
def config_ppac(config, target_cyc):
    gemms = yaml.safe_load(open(WORKLOADS / f"workload_{config}_gemms.yaml"))["gemms"]
    tier = CONFIG_TIER[config]
    ert = parse_ert(CHIP / f"out_accelergy_{tier}_16x16" / "ERT_summary.yaml")

    blocks = []
    total_pe = 0; total_cyc = 0; total_energy_pJ = 0.0
    total_real_macs = 0
    for g in gemms:
        wc_rep = wall_clock_rep((g['M'], g['N'], g['K'], g['dtype']), g['repeat'])
        energy_rep = g['repeat']       # JAXPR-faithful for energy
        _, _, _, ay, ax = choose_array(g, target_cyc, wc_rep)
        b = block_ppac(g, ay, ax, ert, wc_rep, energy_rep)
        blocks.append({'shape': f"{g['M']}x{g['N']}x{g['K']}_{g['dtype']}",
                       'rep_wc': wc_rep, 'rep_jaxpr': energy_rep, **b})
        total_real_macs += b['real_macs']
        total_pe += b['silicon_pe']
        total_cyc += b['cycles']
        total_energy_pJ += b['energy_pJ']

    bottleneck_cyc = max(b['cycles'] for b in blocks)
    latency_s = total_cyc / FREQ_HZ
    throughput = FREQ_HZ / bottleneck_cyc if bottleneck_cyc else 0

    # Area breakdown
    pe_area_mm2 = total_pe * PE_AREA_UM2 / 1e6
    wsram_area = sram_area_mm2(256 * 1024)                # 256 KB fixed
    asram_area = sram_area_mm2(TIER_KB[CONFIG_TIER[config]] * 1024)
    ssram_area = sram_area_mm2(64 * 1024)
    total_area_mm2 = pe_area_mm2 + wsram_area + asram_area + ssram_area

    return dict(
        config=config, target_cyc=target_cyc,
        total_pe=total_pe, total_cycles=total_cyc,
        bottleneck_cycles=bottleneck_cyc,
        latency_ms=latency_s * 1e3, throughput_ips=throughput,
        energy_mJ=total_energy_pJ / 1e9,
        pe_area_mm2=pe_area_mm2, sram_area_mm2=wsram_area + asram_area + ssram_area,
        total_area_mm2=total_area_mm2,
        acc=ACC[config],
        acc_per_mJ=ACC[config] / (total_energy_pJ / 1e9) if total_energy_pJ > 0 else 0,
        # ATP proxies
        pe_x_latency=total_pe * latency_s * 1e3,    # PE * ms
        area_x_latency=total_area_mm2 * latency_s * 1e3,
        blocks=blocks,
    )

# --- Sweep to find ATP optimum ------------------------------------------------
def sweep_atp(config, targets):
    """Sweep bottleneck target, return list of PPAC dicts."""
    results = []
    for t in targets:
        r = config_ppac(config, t)
        results.append(r)
    return results

# --- Main ---------------------------------------------------------------------
def main():
    # Target sweep: aim for bottleneck cycles from tight (8k, big chip) to loose (2M, small chip)
    targets = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152]

    all_results = {}
    for cfg in ["corner1", "corner3p"]:
        results = sweep_atp(cfg, targets)
        all_results[cfg] = results

    # Report the sweep
    print("=" * 118)
    print(f"{'Config':<10}{'Target bt':>10}{'Total PE':>10}{'Bot cyc':>12}{'Latency':>11}"
          f"{'Throughput':>12}{'Energy':>10}{'PE*ms':>12}{'Area(mm2)':>12}{'Area*ms':>12}")
    print("=" * 118)
    for cfg in ["corner1", "corner3p"]:
        for r in all_results[cfg]:
            print(f"{cfg:<10}{r['target_cyc']:>10,}{r['total_pe']:>10,}"
                  f"{r['bottleneck_cycles']:>12,}{r['latency_ms']:>9.3f}ms"
                  f"{r['throughput_ips']:>10.0f}/s{r['energy_mJ']*1000:>7.2f}uJ"
                  f"{r['pe_x_latency']:>12.1f}{r['total_area_mm2']:>11.2f}"
                  f"{r['area_x_latency']:>10.3f}")
        print()

    # Find ATP-optimum (area × latency minimum) per config
    print("=" * 100)
    print("Area·Latency-Product (ATP) optima per config:")
    print("=" * 100)
    for cfg in ["corner1", "corner3p"]:
        best = min(all_results[cfg], key=lambda r: r['area_x_latency'])
        print(f"\n{cfg} ATP-optimal at target_cyc = {best['target_cyc']:,}:")
        print(f"  PEs: {best['total_pe']:,}  Latency: {best['latency_ms']:.3f} ms  "
              f"Throughput: {best['throughput_ips']:.0f}/s")
        print(f"  Total area: {best['total_area_mm2']:.2f} mm2  "
              f"(PE: {best['pe_area_mm2']:.2f}, SRAM: {best['sram_area_mm2']:.2f})")
        print(f"  Energy: {best['energy_mJ']*1000:.2f} uJ  acc/mJ: {best['acc_per_mJ']:.2f}")
        print(f"  Per-block silicon assignments:")
        for b in best['blocks']:
            print(f"    {b['shape']:<35} {b['array']:>7} ({b['silicon_pe']:>4} PE, "
                  f"util {b['util_pct']:>5.1f}%, cyc {b['cycles']:>10,})")

    # Save results
    with open(CHIP / "multi_array_sweep.json", "w") as f:
        json.dump({cfg: [{k: v for k, v in r.items() if k != 'blocks'}
                        for r in results] for cfg, results in all_results.items()},
                  f, indent=2, default=str)
    print(f"\nSaved sweep to multi_array_sweep.json")

if __name__ == "__main__":
    main()
