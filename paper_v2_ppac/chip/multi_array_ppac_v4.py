"""Multi-array PPAC v4 — all four Fable H1-H4 fixes applied.

H1: fill/drain per weight-tile boundary (depth-1 spad forces reload).
H2: operand-role detection — for each GEMM, the SMALLER matrix is the parameter
    (stationary weight); the larger is the streaming activation. Reverses the
    weight-vs-activation SRAM attribution for SSM matmuls.
H3: elemwise ops charge memory traffic (2 reads + 1 write per scalar op through
    activation SRAM) in addition to compute cost.
H4: partial-sum spill through activation SRAM when C_tp > 1
    (depth-1 psum spad can't hold cross-tile running accumulator).

Everything else matches multi_array_ppac.py:
  - Per-block dedicated arrays under smallest-100%-util-under-target rule
  - Predictor overlap: SSM main (shape-matched with jaxpr_rep=24) collapses to
    wall-clock rep=16 for LATENCY. Energy always uses jaxpr_rep.
  - Per-config chip tier (320/384/512 KB) selects ERT.
  - Area model: 3000 μm² per PE + SRAM at 0.7 Mb/mm² (22 nm HD-SRAM).
"""
from __future__ import annotations
import csv, json, math, os, sys
from pathlib import Path
import yaml
sys.stdout.reconfigure(encoding='utf-8')

CHIP = Path(os.path.expanduser("~/dev/ssm-baselines/S5/paper_v2_ppac/chip"))
WORKLOADS = Path(os.path.expanduser("~/dev/ssm-baselines/S5/paper_v2_ppac/workloads"))

STD = [8, 16, 32, 64]
FREQ_HZ = 1e9
PE_AREA_UM2 = 3000
SRAM_MBITS_PER_MM2 = 0.7
LINE_BYTES = 8

CONFIG_TIER = {"config4": "320KB", "config5": "320KB",
               "corner1": "512KB", "corner2": "384KB", "corner3p": "384KB",
               "default": "384KB"}
TIER_KB = {"320KB": 320, "384KB": 384, "512KB": 512, "default": 384}
_cfg_ctx = "default"    # set inside config_ppac to route TIER lookup for F1 fix
ACC = {"config4": 0.5993, "config5": 0.5917,
       "corner1": 0.6089, "corner2": 0.5991, "corner3p": 0.6138}

# Elemwise cost coefficients (units of mac_energy per scalar op) — H3 keeps compute
# side of the earlier model but adds memory traffic below.
ELEMWISE_COMPUTE_COEFF = {"trivial": 1.0, "moderate": 3.0, "transcendental": 10.0,
                         "reduction": 2.0, "structural": 0.0}
# F3 fix (Fable v4 audit): structural ops split between "materializing" (pad,
# concatenate, transpose — real data movement through SRAM) and "addressing-only"
# (reshape, squeeze, broadcast_in_dim, convert_element_type, slice — compiler-level,
# no SRAM round-trip). Only the materializing group charges memory traffic.
# Applied per-op-class since JAXPR walker aggregates at class level.
# Approximation: use 30% of structural ops as materializing (empirical).
STRUCTURAL_MATERIALIZING_FRAC = 0.30

def sram_area_mm2(bytes_): return (bytes_ * 8 / 1e6) / SRAM_MBITS_PER_MM2

def wall_clock_rep(shape, jaxpr_rep):
    """Predictor overlap collapses SSM main only for Corner 3' (jaxpr_rep=24)."""
    if shape == (8, 2048, 128, 'complex_real') and jaxpr_rep == 24:
        return 16
    return jaxpr_rep

def ldiv(x, m):
    if x <= m: return x
    b = 1
    for f in range(1, m + 1):
        if x % f == 0 and f > b: b = f
    return b

def util(ay, ax, spatial_dim_y, spatial_dim_x):
    return 100 * (ldiv(spatial_dim_y, ay) * ldiv(spatial_dim_x, ax)) / (ay * ax)

def parse_ert(path):
    ert = yaml.safe_load(open(path))["ERT_summary"]["table_summary"]
    out = {}
    for c in ert:
        short = c['name'].split('.')[-1]
        if short not in out:
            out[short] = {a['name']: float(a['energy']) for a in c['actions']}
    return out

# --- H2: operand role detection ------------------------------------------------
def role(M, N, K):
    """Return 'A' if A[M,K] is smaller (i.e., the parameter/weight matrix).
    Return 'B' if B[K,N] is smaller. In WS the smaller matrix stays stationary."""
    return 'A' if M * K < K * N else 'B'

# --- Per-block PPAC with H1-H4 fixes ------------------------------------------
def block_ppac(g, ay, ax, ert, wc_rep, energy_rep, elem_energy_pJ_share=0):
    M, N, K, dtype = g['M'], g['N'], g['K'], g['dtype']
    ds = g['per_mac_flops'] // 2                       # dtype_scale
    output_scale = 2 if dtype == 'complex_complex' else 1
    r = role(M, N, K)

    if r == 'A':
        # A[M,K] is stationary; ay|M, ax|K. Sequence B[K,N] streams over N cycles.
        real_ay = ldiv(M, ay); real_ax = ldiv(K, ax)
        n_tiles_outer = math.ceil(M / real_ay)           # outer M-tile count
        n_tiles_inner = math.ceil(K / real_ax)           # outer K-tile count (drives psum spill)
        stream_cyc = N                                    # activation stream length
        # H1: fill/drain per weight-tile boundary
        per_tile = stream_cyc + (ay + ax - 2)            # fill/drain simplified
        cyc_per_inst = per_tile * n_tiles_outer * n_tiles_inner * ds
        # Memory bytes per instance
        weight_bytes = M * K * ds                        # A is small param matrix
        act_read_bytes = N * K * n_tiles_outer * ds      # B re-streamed per M-tile
        C_tp = n_tiles_inner                             # outer K-tile count
        util_pct = 100 * (real_ay * real_ax) / (ay * ax)
    else:
        # B[K,N] is stationary; ay|K, ax|N. Sequence A[M,K] streams over M cycles.
        real_ay = ldiv(K, ay); real_ax = ldiv(N, ax)
        n_tiles_K = math.ceil(K / real_ay)                # outer K-tile count (drives psum spill)
        n_tiles_N = math.ceil(N / real_ax)                # outer N-tile count
        stream_cyc = M
        per_tile = stream_cyc + (ay + ax - 2)             # H1 fill/drain
        cyc_per_inst = per_tile * n_tiles_K * n_tiles_N * ds
        weight_bytes = K * N * ds
        act_read_bytes = M * K * n_tiles_N * ds
        C_tp = n_tiles_K
        util_pct = 100 * (real_ay * real_ax) / (ay * ax)

    act_write_bytes = M * N * output_scale
    # H4: psum spill through activation SRAM (INT32 psum, 4 bytes/elem)
    # F1 fix (Fable v4 audit): running M×N INT32 psum can exceed activation SRAM.
    # M-chunk the matmul so each chunk's psum residency fits: M' = act_sram_budget / (N*4).
    # act_sram_budget approximated from CONFIG_TIER; use 75% for psum residency budget
    # (25% headroom for weight/act traffic staging).
    act_sram_bytes = TIER_KB.get(CONFIG_TIER.get(_cfg_ctx, 'default'), 384) * 1024
    psum_budget = int(act_sram_bytes * 0.75)
    psum_full = M * N * 4
    if psum_full > psum_budget and psum_full > 0:
        n_m_chunks = math.ceil(psum_full / psum_budget)
    else:
        n_m_chunks = 1
    M_chunk = math.ceil(M / n_m_chunks)
    psum_per_chunk = M_chunk * N * 4
    # Traffic per chunk × chunk count. Weights reload per chunk. Fill/drain also
    # scales with chunk count.
    psum_spill_bytes = 0
    if C_tp > 1:
        psum_spill_bytes = (C_tp - 1) * psum_per_chunk * 2 * n_m_chunks
    # Extra weight reloads and fill/drain for M-chunking
    if n_m_chunks > 1:
        # Weight cost increases: reload each chunk (weight already counted once per instance)
        weight_bytes = weight_bytes * n_m_chunks
        # Cycles increase: per_tile fill/drain paid n_m_chunks times more
        cyc_per_inst += (ay + ax - 2) * (n_m_chunks - 1) * (n_tiles_inner if r == 'A' else n_tiles_K) * ds
        total_cyc = cyc_per_inst * wc_rep

    n_pe = ay * ax
    total_real_macs = M * N * K * ds * energy_rep
    total_cyc = cyc_per_inst * wc_rep

    e_mac = total_real_macs * ert['mac']['compute']
    # Weight SRAM: params loaded once per instance, weight tiles reloaded per outer tile
    # For simplicity charge full weight matrix reload per instance × energy_rep
    e_w_sram = math.ceil(weight_bytes / LINE_BYTES) * ert['weight_sram']['read'] * energy_rep
    # Activation SRAM: reads + writes + H4 psum spill
    ar_lines = math.ceil(act_read_bytes / LINE_BYTES)
    aw_lines = math.ceil(act_write_bytes / LINE_BYTES)
    # Split psum spill roughly half-read half-write
    spill_r_lines = math.ceil(psum_spill_bytes / 2 / LINE_BYTES)
    spill_w_lines = math.ceil(psum_spill_bytes / 2 / LINE_BYTES)
    e_a_sram = ((ar_lines + spill_r_lines) * ert['activation_sram']['read']
              + (aw_lines + spill_w_lines) * ert['activation_sram']['write']) * energy_rep
    e_w_spad = total_real_macs * ert['weights_spad']['read']
    e_p_spad = total_real_macs * (ert['psum_spad']['read'] + ert['psum_spad']['update'])
    energy_pJ = e_mac + e_w_sram + e_a_sram + e_w_spad + e_p_spad

    return dict(cycles=total_cyc, silicon_pe=n_pe, energy_pJ=energy_pJ,
                util_pct=util_pct, array=f"{ay}x{ax}", role=r,
                weight_bytes=weight_bytes, act_read_bytes=act_read_bytes,
                psum_spill_bytes=psum_spill_bytes, C_tp=C_tp,
                e_mac=e_mac, e_w_sram=e_w_sram, e_a_sram=e_a_sram,
                real_macs=total_real_macs)

# --- Array picker: max util + no over-provision -------------------------------
def choose_array(g, target_cyc, wc_rep):
    """Smallest 100%-util standard rectangle whose total cycles fit under target.
    Search over the correct spatial-dim pair based on operand role."""
    M, N, K, dtype = g['M'], g['N'], g['K'], g['dtype']
    ds = g['per_mac_flops'] // 2
    r = role(M, N, K)
    # Spatial dims depend on role
    spatial_y = M if r == 'A' else K
    spatial_x = K if r == 'A' else N
    candidates = []
    for ay in STD:
        for ax in STD:
            u = util(ay, ax, spatial_y, spatial_x)
            b = block_ppac(g, ay, ax, ERT_DUMMY, wc_rep, energy_rep=1)  # cycles only
            candidates.append((ay * ax, u, b['cycles'] / 1, ay, ax))
    max_u = max(c[1] for c in candidates)
    pool = [c for c in candidates if c[1] == 100.0] if max_u == 100.0 else \
           [c for c in candidates if c[1] == max_u]
    # F2 fix (Fable v4 audit): b['cycles'] already includes wc_rep (block_ppac
    # multiplies by wc_rep internally). Compare directly to target — no second wc_rep.
    fits = [c for c in pool if c[2] <= target_cyc]
    if fits:
        return min(fits, key=lambda c: c[0])
    return min(pool, key=lambda c: c[2])

# --- Elemwise energy WITH H3 memory traffic -----------------------------------
def elemwise_energy(elem_yaml, ert):
    mac_pJ = ert['mac']['compute']
    act_r = ert['activation_sram']['read']
    act_w = ert['activation_sram']['write']
    total_pJ = 0
    per_class = {}
    for cls, tot in elem_yaml['energy_class_totals'].items():
        ops = tot['total_scalar_ops']
        e_compute = ops * mac_pJ * ELEMWISE_COMPUTE_COEFF.get(cls, 1.0)
        # F3 fix (Fable v4 audit): structural class (reshape/squeeze/broadcast
        # /convert/slice — 0-FLOP addressing ops) mostly doesn't touch SRAM.
        # Only the "materializing" fraction (pad, concatenate, transpose) does.
        # Approximate by fraction; anchor per-op counts to the JAXPR walker rules
        # in workload_*_elemwise.yaml's `note` field.
        if cls == "structural":
            effective_ops = int(ops * STRUCTURAL_MATERIALIZING_FRAC)
        else:
            effective_ops = ops
        read_lines = math.ceil(effective_ops * 2 / LINE_BYTES)
        write_lines = math.ceil(effective_ops * 1 / LINE_BYTES)
        e_mem = read_lines * act_r + write_lines * act_w
        e = e_compute + e_mem
        per_class[cls] = e
        total_pJ += e
    return total_pJ, per_class

# --- State-SRAM (unchanged from direct_ppac) ----------------------------------
N_TRAJ = {"config4": 3, "config5": 2, "corner1": 2, "corner2": 2, "corner3p": 3}
def state_energy(manifest, config, ert):
    args = manifest["args"]
    L = 2048; n_layers = args["n_layers"]
    n_dir = 2 if args["bidirectional"] else 1
    P = args["ssm_size_base"] // 2
    n_traj = N_TRAJ[config]
    lines_per_step = math.ceil(2 * P / LINE_BYTES)
    reads = L * n_layers * n_dir * n_traj * lines_per_step
    return (reads * ert['state_sram']['read'] + reads * ert['state_sram']['write']), reads

# --- Config-level PPAC --------------------------------------------------------
ERT_DUMMY = None  # set at runtime

def config_ppac(config, target_cyc):
    global ERT_DUMMY, _cfg_ctx
    _cfg_ctx = config
    gemms = yaml.safe_load(open(WORKLOADS / f"workload_{config}_gemms.yaml"))["gemms"]
    elem = yaml.safe_load(open(WORKLOADS / f"workload_{config}_elemwise.yaml"))
    manif = yaml.safe_load(open(WORKLOADS / f"workload_{config}_manifest.yaml"))
    tier = CONFIG_TIER[config]
    ert = parse_ert(CHIP / f"out_accelergy_{tier}_16x16" / "ERT_summary.yaml")
    ERT_DUMMY = ert

    blocks = []
    total_pe = 0; total_cyc = 0; total_energy_pJ = 0.0
    total_real_macs = 0
    for g in gemms:
        wc_rep = wall_clock_rep((g['M'], g['N'], g['K'], g['dtype']), g['repeat'])
        energy_rep = g['repeat']
        _, _, _, ay, ax = choose_array(g, target_cyc, wc_rep)
        b = block_ppac(g, ay, ax, ert, wc_rep, energy_rep)
        blocks.append({'shape': f"{g['M']}x{g['N']}x{g['K']}_{g['dtype']}",
                       'rep_wc': wc_rep, 'rep_jaxpr': energy_rep, **b})
        total_pe += b['silicon_pe']
        total_cyc += b['cycles']
        total_energy_pJ += b['energy_pJ']
        total_real_macs += b['real_macs']

    # H3 elemwise
    e_elem, per_class = elemwise_energy(elem, ert)
    total_energy_pJ += e_elem
    e_state, _ = state_energy(manif, config, ert)
    total_energy_pJ += e_state

    bottleneck_cyc = max(b['cycles'] for b in blocks)
    latency_s = total_cyc / FREQ_HZ
    throughput = FREQ_HZ / bottleneck_cyc if bottleneck_cyc else 0

    pe_area_mm2 = total_pe * PE_AREA_UM2 / 1e6
    wsram_area = sram_area_mm2(256 * 1024)
    asram_area = sram_area_mm2(TIER_KB[CONFIG_TIER[config]] * 1024)
    ssram_area = sram_area_mm2(64 * 1024)
    total_area_mm2 = pe_area_mm2 + wsram_area + asram_area + ssram_area

    return dict(config=config, target_cyc=target_cyc,
                total_pe=total_pe, total_cycles=total_cyc,
                bottleneck_cycles=bottleneck_cyc,
                latency_ms=latency_s * 1e3, throughput_ips=throughput,
                energy_mJ=total_energy_pJ / 1e9,
                pe_area_mm2=pe_area_mm2, sram_area_mm2=wsram_area + asram_area + ssram_area,
                total_area_mm2=total_area_mm2,
                acc=ACC[config],
                acc_per_mJ=ACC[config] / (total_energy_pJ / 1e9) if total_energy_pJ > 0 else 0,
                pe_x_latency=total_pe * latency_s * 1e3,
                area_x_latency=total_area_mm2 * latency_s * 1e3,
                elem_energy_pJ=e_elem, state_energy_pJ=e_state,
                blocks=blocks)

# --- Main ---------------------------------------------------------------------
def main():
    targets = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152]
    all_results = {}
    for cfg in ["corner1", "corner3p"]:
        all_results[cfg] = [config_ppac(cfg, t) for t in targets]

    print("=" * 128)
    print(f"MULTI-ARRAY PPAC v4 — H1 (fill/drain) + H2 (operand role) + H3 (elemwise mem) + H4 (psum spill)")
    print("=" * 128)
    print(f"{'Config':<10}{'Target':>10}{'PEs':>8}{'Bot cyc':>12}{'Latency':>11}"
          f"{'Tput':>10}{'Energy':>10}{'Area(mm2)':>12}{'Area*ms':>10}{'acc/mJ':>9}")
    print("-" * 128)
    for cfg in ["corner1", "corner3p"]:
        for r in all_results[cfg]:
            print(f"{cfg:<10}{r['target_cyc']:>10,}{r['total_pe']:>8,}"
                  f"{r['bottleneck_cycles']:>12,}{r['latency_ms']:>9.3f}ms"
                  f"{r['throughput_ips']:>8.0f}/s{r['energy_mJ']*1000:>7.2f}uJ"
                  f"{r['total_area_mm2']:>11.2f}{r['area_x_latency']:>10.3f}"
                  f"{r['acc_per_mJ']:>9.2f}")
        print()

    print("=" * 100)
    print("ATP (Area × Latency) OPTIMA at v4 accounting:")
    print("=" * 100)
    for cfg in ["corner1", "corner3p"]:
        best = min(all_results[cfg], key=lambda r: r['area_x_latency'])
        print(f"\n{cfg}: target_cyc = {best['target_cyc']:,}")
        print(f"  PEs: {best['total_pe']:,}  Area: {best['total_area_mm2']:.2f} mm2  "
              f"Latency: {best['latency_ms']:.3f} ms  Throughput: {best['throughput_ips']:.0f}/s")
        print(f"  Energy: {best['energy_mJ']*1000:.2f} uJ  acc/mJ: {best['acc_per_mJ']:.2f}")
        print(f"  (elem: {best['elem_energy_pJ']/1e6:.2f} uJ, state: {best['state_energy_pJ']/1e6:.2f} uJ)")
        print(f"  Per-block:")
        for b in best['blocks']:
            print(f"    {b['shape']:<35} {b['array']:>7} role={b['role']} "
                  f"({b['silicon_pe']:>4} PE, C_tp={b['C_tp']:>2}, "
                  f"spill={b['psum_spill_bytes']/1024:>7.0f} KB, cyc {b['cycles']:>10,})")

    with open(CHIP / "multi_array_sweep_v4.json", "w") as f:
        json.dump({cfg: [{k: v for k, v in r.items() if k != 'blocks'}
                        for r in results] for cfg, results in all_results.items()},
                  f, indent=2, default=str)

if __name__ == "__main__":
    main()
