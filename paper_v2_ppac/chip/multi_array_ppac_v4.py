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

# Repo-relative so the pipeline is reproducible in any clone/worktree (this file
# lives at <repo>/paper_v2_ppac/chip/; workloads sit alongside at ../workloads).
CHIP = Path(__file__).resolve().parent
WORKLOADS = CHIP.parent / "workloads"

STD = [8, 16, 32, 64]
FREQ_HZ = 1e9
PE_AREA_UM2 = 3000
SRAM_MBITS_PER_MM2 = 0.7
LINE_BYTES = 8
# NoC + control logic overhead, anchored on Chen et al. (Eyeriss ISSCC 2016) and
# Sze 2020 review. Mesh NoC ~15% of PE-array area, control + instruction fetch ~5%.
# Scales linearly with PE count; because Corner 3' has fewer PEs than Corner 1,
# including NoC widens Mambino's area advantage rather than narrowing it.
NOC_AREA_FRACTION_OF_PE = 0.15
CONTROL_AREA_FRACTION_OF_PE = 0.05
# NoC dynamic energy per MAC (routers, wires, arbitration). Chen et al. report
# NoC at ~7% of Eyeriss inference energy; Sze 2020 gives 5-10% range.
NOC_ENERGY_FRACTION_OF_MAC = 0.10

CONFIG_TIER = {"config4": "320KB", "config5": "320KB",
               "corner1": "512KB", "corner2": "384KB", "corner3p": "384KB",
               "mambino2p0": "320KB",   # Mambino 2.0 = config4 arch + surprise gate (same tier)
               # Cluster-B family: config4 arch + fast weight. No new large-footprint
               # GEMM (biggest is the CxC=64x64 chunk attention and W_o at 8x128), so
               # the 320KB activation tier still holds. Names per NAMING_AND_STATUS.md.
               "mambinoF": "320KB", "mambinoFo": "320KB", "mambinoGF": "320KB",
               "default": "384KB"}
TIER_KB = {"320KB": 320, "384KB": 384, "512KB": 512,
           "1024KB": 1024, "1600KB": 1600, "2048KB": 2048, "default": 384}
# Ladder for auto_tier(); extended past 512 KB for the L=4096, d_model=256 tasks.
TIER_LADDER = [320, 384, 512, 1024, 1600, 2048, 2560, 3072]
_cfg_ctx = "default"    # set inside config_ppac to route TIER lookup for F1 fix
# ---------------------------------------------------------------------------
# n=16 UPDATE (2026-08-06). The four v1-paper configurations are now at SIXTEEN
# matched seeds: the original eight (6554595/42/12345/271828/314159/1/2/3) plus
# the pre-registered extension (4..11). These supersede the 8-seed values.
#     config5    S5-0        0.5917 -> 0.5933  (sd 0.0071)
#     config4    Mambino-0   0.5993 -> 0.6008  (sd 0.0065)
#     mambino2p0 Mambino-G   0.6050 -> 0.6018  (sd 0.0052)
#     corner1    S5-Dense    0.6089 -> 0.6102  (sd 0.0054)
# What the extension changed: the predictor claim hardened (+0.75 pp, p=0.0013,
# 13/16), the surprise-gate claim collapsed to null (+0.10 pp, p=0.70, 8/16),
# and the gap to the baseline doubled and became significant (-0.83 pp,
# p=0.0029). corner2/corner3p are RETIRED configs still at n=8; do not mix them
# with the four above in any comparison without stating the differing n.
# ---------------------------------------------------------------------------
# Weight SRAM is sized PER CONFIGURATION from its actual parameter count, not
# fixed. At INT8 one parameter is one byte; capacity is rounded up to the next
# 64 KB bank granularity. The previous hardcoded 256 KB was defensible on
# ListOps, where every configuration fits under it, but it gave a configuration
# using 56% of the baseline's parameters no area credit for doing so, and it is
# not physically valid at IMDB scale, where ~1.31 M parameters need ~1.28 MB.
#
# CAVEAT, stated rather than hidden: the Accelergy ERTs are generated per
# ACTIVATION tier, so per-access weight-SRAM ENERGY still uses the 256 KB-tier
# figures. A smaller SRAM has lower per-access energy, so the energy numbers are
# conservative for the smaller configurations; only area responds to this change.
PARAM_COUNT = {
    # ListOps (d_model=128, 8 layers)
    "config5": 105_738, "config4": 105_738, "mambino2p0": 105_754,
    "corner1": 188_490, "corner2": 139_402, "corner3p": 188_682,
    # LRA-Text / IMDB (d_model=256, 6 layers). Read from the training logs;
    # newer extractions carry n_params in the workload manifest and
    # param_count() prefers that.
    "imdb-S5-0": 1_314_230, "imdb-Mambino-0": 1_314_230,
    "imdb-Mambino-G": 1_314_242, "imdb-S5-Dense": 1_321_154,
    "imdb-S5-0-R": 926_402, "imdb-Mambino-0-R": 926_402,
    "imdb-Mambino-G-R": 926_414,
}
WSRAM_GRANULARITY = 64 * 1024


def weight_sram_bytes(config):
    """INT8 weights, rounded up to the next 64 KB bank."""
    need = param_count(config)          # 1 byte per parameter at INT8
    banks = -(-need // WSRAM_GRANULARITY)
    return banks * WSRAM_GRANULARITY


ACC = {"config4": 0.6008, "config5": 0.5933,
       "corner1": 0.6102, "corner2": 0.5991, "corner3p": 0.6138,
       "mambino2p0": 0.6018,   # 16-seed signed-gate mean (sd 0.0052)
       # Cluster-B accuracies are 4-SEED means (6554595/42/12345/271828) -- 8-seed
       # sweeps still running. Do NOT compare acc/mJ against the 8-seed entries above
       # without matching n: on these 4 seeds corner1=0.6110, config4=0.5988,
       # mambino2p0=0.6025, i.e. the subset runs high for corner1 and low for the gate.
       "mambinoF": 0.6051, "mambinoFo": 0.5966, "mambinoGF": 0.6024}

# Gate ranks that do not tile onto a standard {8,16,32,64} array width without
# extra contraction tiles, and therefore get the util-vs-latency relaxation.
# Rank 40 reaches 100% util only on an 8-wide axis (40 = 5x8), which costs 5
# contraction tiles and can exceed the per-config bottleneck target; the relaxed
# rule lets the picker take a 64-wide array at 62.5% util with one tile instead.
# Powers of two (8, 16, 32, 64) tile exactly and need no relaxation.
#
# This is SHAPE-driven, not name-driven. It used to be gated on a config
# allowlist (RELAX_SET = corner2, corner3p, v2pilot_dense_r40), which meant an
# identical architecture registered under a different config name would be held
# to 100% util and priced differently. The rank sweep registers rank-40 points
# under new names, so the allowlist would have made rs_Mambino_r40 and corner3p
# disagree about the same design. Behaviour is unchanged for every existing
# config: the retired ones have K=40 blocks and were already in the set, and the
# four v1 configs contain no K=40 block at all.
RELAX_RANKS = {40}
RELAX_SET = {"corner2", "corner3p", "v2pilot_dense_r40"}   # retained for reference

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


def workload_footprint_kb(config):
    """Max activation footprint over the config's GEMMs, in KB.

    The rule the model actually implements (direct_ppac.py:155-159):
        footprint(GEMM) = M*K + M*N bytes, ONE byte per element for every dtype,
        no partial-sum term. M-chunking is applied against the tier afterwards
        and is not part of the sizing quantity.
    Verified to reproduce every hand-assigned tier in CONFIG_TIER.
    """
    g = yaml.safe_load(open(WORKLOADS / f"workload_{config}_gemms.yaml"))["gemms"]
    return max(k["M"] * k["K"] + k["M"] * k["N"] for k in g) / 1024.0


def auto_tier(config):
    """Smallest ladder tier that holds the config's max GEMM footprint."""
    need = workload_footprint_kb(config)
    for t in TIER_LADDER:
        if t >= need:
            return f"{t}KB"
    raise ValueError(f"{config}: footprint {need:.0f} KB exceeds the tier ladder")


def tier_of(config):
    """Hand-assigned tier if one exists, otherwise derive it from the workload."""
    if config in CONFIG_TIER:
        return CONFIG_TIER[config]
    t = auto_tier(config)
    TIER_KB.setdefault(t, int(t[:-2]))
    return t


def param_count(config):
    """Parameter count from the workload manifest, falling back to the table."""
    if config in PARAM_COUNT:
        return PARAM_COUNT[config]
    man = yaml.safe_load(open(WORKLOADS / f"workload_{config}_manifest.yaml"))
    for k in ("n_params", "params", "trainable_parameters", "param_count"):
        if man.get(k):
            return int(man[k])
    raise KeyError(f"{config}: no parameter count in PARAM_COUNT or manifest")

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
    batch = int(g.get('batch', 1))                     # batched GEMM (structured gates)
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
    act_sram_bytes = TIER_KB.get(tier_of(_cfg_ctx) if _cfg_ctx != 'default' else 'default', 384) * 1024
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
    # Batched GEMM: `batch` independent matmuls with DISTINCT weights run
    # sequentially on the array. Array size (n_pe) and per-matmul psum residency
    # (M_chunk x N) are unchanged; all WORK scales x batch. batch==1 for every v1
    # workload, so this is a no-op there (corner3p cross-check is the regression gate).
    weight_bytes *= batch
    act_read_bytes *= batch
    act_write_bytes *= batch
    psum_spill_bytes *= batch
    cyc_per_inst *= batch
    total_real_macs = M * N * K * ds * energy_rep * batch
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
def choose_array(g, target_cyc, wc_rep, min_util=100.0):
    """Smallest rectangle with util >= min_util whose total cycles fit under target.
    Search over the correct spatial-dim pair based on operand role.
    min_util=100: strict rule — requires 100% utilization on the chosen axis.
    min_util<100: relaxed rule for the util-vs-latency tradeoff. For rank-40
        low-rank gates, an 8-wide axis reaches 100% util (40=5x8) but requires
        5 contraction tiles and can exceed the per-config bottleneck target T;
        the relaxed rule lets the picker choose a wider array (e.g. 64-wide, 1
        contraction tile, 40/64=62.5% util) that meets T at the ATP-optimal
        design point. The 62.5% figure is a util-vs-latency policy, not a
        divisibility limitation.
    """
    M, N, K, dtype = g['M'], g['N'], g['K'], g['dtype']
    ds = g['per_mac_flops'] // 2
    r = role(M, N, K)
    spatial_y = M if r == 'A' else K
    spatial_x = K if r == 'A' else N
    candidates = []
    for ay in STD:
        for ax in STD:
            u = util(ay, ax, spatial_y, spatial_x)
            b = block_ppac(g, ay, ax, ERT_DUMMY, wc_rep, energy_rep=1)
            candidates.append((ay * ax, u, b['cycles'], ay, ax))
    max_u = max(c[1] for c in candidates)
    # If min_util==100 and shape supports 100%, restrict to those. Otherwise allow >= min_util.
    if min_util == 100.0 and max_u == 100.0:
        pool = [c for c in candidates if c[1] == 100.0]
    elif max_u < min_util:
        pool = [c for c in candidates if c[1] == max_u]        # best achievable
    else:
        pool = [c for c in candidates if c[1] >= min_util]
    # F2 fix: cycles already includes wc_rep.
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
# Fast-weight M is carried PER CHUNK (d x d, 64 entries at d=8), not per timestep,
# so it adds no L-scaled state trajectory: the Cluster-B configs keep Mambino's 3
# (main fwd + main bwd + predictor fwd).
N_TRAJ = {"config4": 3, "config5": 2, "corner1": 2, "corner2": 2, "corner3p": 3, "mambino2p0": 3,
          "mambinoF": 3, "mambinoFo": 3, "mambinoGF": 3}

# Sequence length per task. state_energy() previously hardcoded L = 2048, which
# is the ListOps length; any other task silently got the wrong state traffic
# (IMDB at L = 4096 would have been charged half). ListOps values are unchanged
# by this fix because 2048 is what the table returns for it.
DATASET_L = {"listops-classification": 2048, "imdb-classification": 4096,
             "aan-classification": 4000, "lra-cifar-classification": 1024}


def seq_len_of(manifest, gemms=None):
    """Sequence length from the task, falling back to the widest GEMM dimension."""
    ds = manifest["args"].get("dataset")
    if ds in DATASET_L:
        return DATASET_L[ds]
    if manifest["args"].get("seq_len"):
        return int(manifest["args"]["seq_len"])
    if gemms:
        return max(max(g["M"], g["N"]) for g in gemms)
    raise KeyError(f"cannot determine sequence length for dataset {ds!r}")


def n_traj_of(manifest, config):
    """State trajectories held per timestep.

    Two for a plain bidirectional S5 layer (forward + backward main scan); three
    when a predictor branch is present, which adds one forward trajectory.
    """
    if config in N_TRAJ:
        return N_TRAJ[config]
    return 3 if manifest["args"].get("use_mambino_ssm") else 2


def state_energy(manifest, config, ert):
    args = manifest["args"]
    L = seq_len_of(manifest); n_layers = args["n_layers"]
    n_dir = 2 if args["bidirectional"] else 1
    P = args["ssm_size_base"] // 2
    n_traj = n_traj_of(manifest, config)
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
    tier = tier_of(config)
    ert = parse_ert(CHIP / f"out_accelergy_{tier}_16x16" / "ERT_summary.yaml")
    ERT_DUMMY = ert

    blocks = []
    total_pe = 0; total_cyc = 0; total_energy_pJ = 0.0
    total_real_macs = 0
    # Per-config util-vs-latency policy: Corner 2 and Corner 3' have K=40 low-rank
    # gate blocks whose 100%-util mapping IS available (40 = 5x8, an 8-wide axis
    # divides 40 evenly with 5 contraction tiles) but that mapping does not meet
    # the bottleneck target T at the ATP-optimal design point. Relaxing the util
    # floor to 62.5% lets the picker choose a wider array (e.g. 64-wide, 1 tile,
    # 40/64=62.5% util) that meets T. Corner 1 has no K=40 shape and is unaffected.
    for g in gemms:
        wc_rep = wall_clock_rep((g['M'], g['N'], g['K'], g['dtype']), g['repeat'])
        energy_rep = g['repeat']
        # Relax util threshold only for low-rank gate shapes where K=40 or N=40 in
        # Corner 2 (Pure S5 low-rank) and Corner 3' (Mambino low-rank). This is a
        # util-vs-latency policy, not a divisibility limitation.
        is_low_rank_gate = (g['K'] in RELAX_RANKS or g['N'] in RELAX_RANKS)
        min_util = 62.5 if is_low_rank_gate else 100.0
        _, _, _, ay, ax = choose_array(g, target_cyc, wc_rep, min_util=min_util)
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
    noc_area_mm2 = pe_area_mm2 * NOC_AREA_FRACTION_OF_PE
    control_area_mm2 = pe_area_mm2 * CONTROL_AREA_FRACTION_OF_PE
    wsram_area = sram_area_mm2(weight_sram_bytes(config))
    asram_area = sram_area_mm2(TIER_KB[tier_of(config)] * 1024)
    ssram_area = sram_area_mm2(64 * 1024)
    total_area_mm2 = pe_area_mm2 + noc_area_mm2 + control_area_mm2 + wsram_area + asram_area + ssram_area

    # Energy breakdown per component (sum over blocks)
    e_mac_total = sum(b['e_mac'] for b in blocks)
    e_wsram_total = sum(b['e_w_sram'] for b in blocks)
    e_asram_total = sum(b['e_a_sram'] for b in blocks)
    # NoC + control dynamic energy: scaled from MAC energy (Chen ISSCC 2016).
    e_noc_ctrl_total = e_mac_total * NOC_ENERGY_FRACTION_OF_MAC
    e_spad_pe_total = sum(b.get('e_p_spad_pJ', b.get('e_p_spad', 0)) +
                          b.get('e_w_spad_pJ', b.get('e_w_spad', 0)) for b in blocks)
    # spads not in block dict currently; compute from real_macs
    e_spad_pe_total = sum(b['real_macs'] * (ert['weights_spad']['read']
                          + ert['psum_spad']['read'] + ert['psum_spad']['update'])
                          for b in blocks)

    # SRAM leakage across all cycles
    leak_pJ_per_cyc = (ert['weight_sram'].get('leak', 0.0)
                    + ert['activation_sram'].get('leak', 0.0)
                    + ert['state_sram'].get('leak', 0.0))
    e_leak = leak_pJ_per_cyc * total_cyc

    # Add NoC+control to total energy
    total_energy_pJ += e_noc_ctrl_total

    total_check = e_mac_total + e_wsram_total + e_asram_total + e_spad_pe_total + e_elem + e_state + e_leak + e_noc_ctrl_total

    latency_ms = latency_s * 1e3
    power_mW = (total_energy_pJ / 1e9) / latency_ms  # mJ / ms = W = 1000 mW

    return dict(config=config, target_cyc=target_cyc,
                total_pe=total_pe, total_cycles=total_cyc,
                bottleneck_cycles=bottleneck_cyc,
                latency_ms=latency_ms, throughput_ips=throughput,
                energy_mJ=total_energy_pJ / 1e9,
                power_mW=power_mW * 1000,   # W to mW
                # Area breakdown (mm2)
                pe_area_mm2=pe_area_mm2,
                noc_area_mm2=noc_area_mm2,
                control_area_mm2=control_area_mm2,
                weight_sram_area_mm2=wsram_area,
                act_sram_area_mm2=asram_area,
                state_sram_area_mm2=ssram_area,
                sram_area_mm2=wsram_area + asram_area + ssram_area,
                total_area_mm2=total_area_mm2,
                # Energy breakdown (uJ)
                e_mac_uJ=e_mac_total / 1e6,
                e_wsram_uJ=e_wsram_total / 1e6,
                e_asram_uJ=e_asram_total / 1e6,
                e_spad_uJ=e_spad_pe_total / 1e6,
                e_elem_uJ=e_elem / 1e6,
                e_state_uJ=e_state / 1e6,
                e_leak_uJ=e_leak / 1e6,
                e_noc_ctrl_uJ=e_noc_ctrl_total / 1e6,
                # Add leakage to total (was omitted before)
                energy_full_uJ=(total_energy_pJ + e_leak) / 1e6,   # already includes NoC in total_energy_pJ
                acc=ACC[config],
                acc_per_mJ=ACC[config] / (total_energy_pJ / 1e9) if total_energy_pJ > 0 else 0,
                pe_x_latency=total_pe * latency_ms,
                area_x_latency=total_area_mm2 * latency_ms,
                elem_energy_pJ=e_elem, state_energy_pJ=e_state,
                blocks=blocks)

# --- Main ---------------------------------------------------------------------
def main():
    targets = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152]
    all_results = {}
    for cfg in ["config4", "config5", "corner1", "corner2", "corner3p"]:
        all_results[cfg] = [config_ppac(cfg, t) for t in targets]

    print("=" * 128)
    print(f"MULTI-ARRAY PPAC v4 — H1 (fill/drain) + H2 (operand role) + H3 (elemwise mem) + H4 (psum spill)")
    print("=" * 128)
    print(f"{'Config':<10}{'Target':>10}{'PEs':>8}{'Bot cyc':>12}{'Latency':>11}"
          f"{'Tput':>10}{'Energy':>10}{'Area(mm2)':>12}{'Area*ms':>10}{'acc/mJ':>9}")
    print("-" * 128)
    for cfg in ["config4", "config5", "corner1", "corner2", "corner3p"]:
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
    for cfg in ["config4", "config5", "corner1", "corner2", "corner3p"]:
        best = min(all_results[cfg], key=lambda r: r['area_x_latency'])
        print(f"\n{cfg}: target_cyc = {best['target_cyc']:,}")
        print(f"  PEs: {best['total_pe']:,}  Latency: {best['latency_ms']:.3f} ms  "
              f"Throughput: {best['throughput_ips']:.0f}/s")
        print(f"  AREA BREAKDOWN (mm2):")
        print(f"     PE array:        {best['pe_area_mm2']:>7.2f}  ({100*best['pe_area_mm2']/best['total_area_mm2']:>5.1f}%)")
        print(f"     NoC (mesh):      {best['noc_area_mm2']:>7.2f}  ({100*best['noc_area_mm2']/best['total_area_mm2']:>5.1f}%)")
        print(f"     Control logic:   {best['control_area_mm2']:>7.2f}  ({100*best['control_area_mm2']/best['total_area_mm2']:>5.1f}%)")
        print(f"     Weight SRAM:     {best['weight_sram_area_mm2']:>7.2f}  ({100*best['weight_sram_area_mm2']/best['total_area_mm2']:>5.1f}%)")
        print(f"     Act SRAM:        {best['act_sram_area_mm2']:>7.2f}  ({100*best['act_sram_area_mm2']/best['total_area_mm2']:>5.1f}%)")
        print(f"     State SRAM:      {best['state_sram_area_mm2']:>7.2f}  ({100*best['state_sram_area_mm2']/best['total_area_mm2']:>5.1f}%)")
        print(f"     TOTAL:           {best['total_area_mm2']:>7.2f}")
        print(f"  ENERGY BREAKDOWN (uJ):")
        e_tot = best['energy_full_uJ']
        print(f"     PE compute (MAC): {best['e_mac_uJ']:>7.2f}  ({100*best['e_mac_uJ']/e_tot:>5.1f}%)")
        print(f"     NoC + control:    {best['e_noc_ctrl_uJ']:>7.2f}  ({100*best['e_noc_ctrl_uJ']/e_tot:>5.1f}%)")
        print(f"     Weight SRAM acc:  {best['e_wsram_uJ']:>7.2f}  ({100*best['e_wsram_uJ']/e_tot:>5.1f}%)")
        print(f"     Act SRAM acc:     {best['e_asram_uJ']:>7.2f}  ({100*best['e_asram_uJ']/e_tot:>5.1f}%)")
        print(f"     Per-PE spads:     {best['e_spad_uJ']:>7.2f}  ({100*best['e_spad_uJ']/e_tot:>5.1f}%)")
        print(f"     Elementwise:      {best['e_elem_uJ']:>7.2f}  ({100*best['e_elem_uJ']/e_tot:>5.1f}%)")
        print(f"     State SRAM:       {best['e_state_uJ']:>7.2f}  ({100*best['e_state_uJ']/e_tot:>5.1f}%)")
        print(f"     SRAM leakage:     {best['e_leak_uJ']:>7.2f}  ({100*best['e_leak_uJ']/e_tot:>5.1f}%)")
        print(f"     TOTAL:            {e_tot:>7.2f}")
        print(f"  POWER: {best['power_mW']:>8.1f} mW  (energy / latency)")
        print(f"  acc/mJ: {best['acc_per_mJ']:.2f}")

    with open(CHIP / "multi_array_sweep_v4.json", "w") as f:
        json.dump({cfg: [{k: v for k, v in r.items() if k != 'blocks'}
                        for r in results] for cfg, results in all_results.items()},
                  f, indent=2, default=str)

if __name__ == "__main__":
    main()
