"""ppac_v5_scheduler.py — increment 2b (first cut): chunk-pipeline scheduler.

Computes an INTERPOLATION BOUND for pipelined latency + timeline peak power BETWEEN
increment 1's serial and concurrent bounds, as a function of the concurrency knob n_chunks,
symmetrically for every config. Built ON TOP of multi_array_ppac_v5 (honest overlap) —
imports it; v4/v5 unchanged. It is a BOUND, not a realizable schedule (see model boundary).

FAIRNESS WARNING: pipeline_point()/pareto() vary ONLY n_chunks, holding each config at its
own per-config ATP design point. Comparing two configs at "matched n_chunks" is NOT fair —
they sit at different latencies (different speed grades), so it hides a latency deficit. The
honest cross-config comparison gives BOTH configs BOTH knobs (target_cyc x n_chunks) and
lives in merged_frontier.py. Do not quote matched-n_chunks A/B as an architecture result.

MODEL (first cut — block granularity; DAG-fine per-layer barriers + fusion are the
next refinement, guided by the Fable check):
  Pipeline stages = v4's per-shape blocks (honest, frozen ATP arrays). Each stage i has a
  whole-L duration d_i (cycles) and instantaneous power P_i = (energy_pJ + NoC)/cycles.
  Chunk L into n_chunks; fill+steady+drain:
      latency(n) = ( sum(d_i) + (n-1)*max(d_i) ) / n / FREQ          # -> sum at n=1, ->max_d as n->inf
      peak(n)    = sum of the top-min(n, n_stages) stage powers      # co-active pipeline stages
  ENDPOINTS: n=1 reproduces increment-1 SERIAL EXACTLY (latency=sum d, peak=max P).
  PEAK saturates at increment-1's CONCURRENT peak (sum P) once n>=n_stages (all stages
  co-active). LATENCY approaches increment-1's CONCURRENT bound (max_d, the throughput
  limit) only ASYMPTOTICALLY as n->inf -- with finite chunking there is always fill/drain
  overhead, so max_d is a lower bound, never exactly reached. The knob trades latency for
  peak power -> the Pareto. Silicon/area/energy are v4's honest values, frozen.

  MODEL BOUNDARY — refinement (a) is now APPLIED (default fill_drain=True); (b),(c) remain:
    (a) DONE per-chunk fill/drain (realistic model): per-chunk stage time = fd_i + stream_i/n
        (fd re-paid PER CHUNK), so latency is U-shaped in n (interior optimum; see optimal_n).
        pipeline_point(..., fill_drain=False) still exposes the free-chunk interpolation bound.
        Sensitivity to the M-chunk fold: ~10-12% on min-latency (see _block_filldrain_frac).
    (b) DONE per-layer bidir barriers (barrier_point/validate_barriers): both configs carry 8
        dag 'rev' segments; a reverse scan cannot stream chunks across the barrier. Segmented
        the pipeline at each -> 9 short pipelines in series. Conservation (n=1==serial) +
        dominance verified. On the FAIR both-knob frontier barriers HELP Mambino's iso-latency
        case (Pure S5's low-peak mid-latency points are slow-target deep-chunk designs that
        barriers penalize most); the crossover stays ~[0.50, 0.60] ms. (The per-ATP-point
        "Mambino inflates 1.34x" is a barrier-blind design-point artifact, NOT architecture.)
        Model boundary: forbids cross-SEGMENT streaming but allows post-L/pre-(L+1) overlap;
        a fwd/bwd-split scan mapping would relieve Pure S5 more than Mambino (disclose).
    (c) OPEN gate/GLU fusion ('logistic' segments) not modeled.
    (d) predictor scheduling: all of the above is the SERIAL predictor (honest, 24 SSM
        instances time-multiplexed on one array). The CONCURRENT predictor (2nd 512-PE array)
        is a separate design point (increment 2d).
  Background power (elem+state+leak) is off the peak timeline here -- see peak_power_scenarios.py
  for the complete-chip peak BAND (adding it flips no iso-latency winner; Fable-confirmed).
  Stages are per-SHAPE blocks (rep folded in), so cross-layer dependencies are not enforced and
  the op-DAG topology is not yet consumed. Treat latency outputs as a model estimate, not silicon.
"""
import functools
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multi_array_ppac_v4 as m4
import multi_array_ppac_v5 as m5

FREQ_HZ = m4.FREQ_HZ


@functools.lru_cache(maxsize=None)
def _atp(config, honest=True):
    """Cached ATP-optimal (each config's design point is invariant to the concurrency
    knob; config_ppac re-reads YAMLs+ERT so caching is essential for speed)."""
    return m5.atp_optimal(config, honest=honest)


def _stage_power_mW(b):
    return (b["energy_pJ"] + m4.NOC_ENERGY_FRACTION_OF_MAC * b["e_mac"]) / b["cycles"] if b["cycles"] else 0.0


def _shape_MN(shape):
    m, n, _k = shape.split("_")[0].split("x")
    return int(m), int(n)


def _block_filldrain_frac(b):
    """Fraction of a block's cycles that is per-tile fill/drain -- re-paid PER CHUNK under
    sequence chunking (v4 block_ppac: per_tile = stream_cyc + (ay+ax-2), so this fraction is
    (ay+ax-2)/(stream_cyc+(ay+ax-2)), independent of the common n_tiles*ds*rep*batch factor).
    stream_cyc = N (role A) or M (role B).

    FIRST-ORDER: this counts only the per-tile fill/drain; v4's extra M-chunk fill/drain
    (v4:160, present when psum spills) is NOT separated out -- it stays in the streaming part
    and so gets divided by n. Fable-corrected framing: that fold is defensible NOT because it
    is a 'lower bound' (a lower fd fraction would OPTIMISTICALLY under-count the re-paid
    penalty), but because sequence chunking physically SHRINKS per-chunk psum residency
    (M/n * N * 4 for role-B; earlier contraction-tile completion for role-A), so M-chunking
    itself decays with n and vanishes at n >= n_m_chunks -- i.e. the M-chunk fd is genuinely
    ~stream-like, not fixed per chunk. SENSITIVITY (frozen-mapping worst case, M-chunk fd
    fixed per chunk instead): min-latency ~10-12% lower and n* ~2-4 higher than reported
    (corner1 0.199->0.220 n*5->6; corner3p 0.361->0.408 n*11->7). Verdict robust either way
    (Fable: no iso-latency winner flips under the pessimistic bracket)."""
    ay, ax = (int(x) for x in b["array"].split("x"))
    M, N = _shape_MN(b["shape"])
    stream_cyc = N if b["role"] == "A" else M
    fd = ay + ax - 2
    return fd / (stream_cyc + fd) if (stream_cyc + fd) > 0 else 0.0


def _makespan_cyc(blocks, n, fill_drain=True):
    """Chunk-pipeline makespan (cycles). fill_drain=True (realistic): each stage's per-chunk
    time = fd_i + stream_i/n, so latency is U-SHAPED in n (chunking too finely re-pays
    fill/drain n times). fill_drain=False: the free-chunk interpolation bound d_i/n (monotone,
    clean serial<->max_d endpoints). Both give exactly serial (sum d) at n=1."""
    if fill_drain:
        t = [b["cycles"] * _block_filldrain_frac(b)
             + b["cycles"] * (1 - _block_filldrain_frac(b)) / n for b in blocks]
        return sum(t) + (n - 1) * max(t)
    d = [b["cycles"] for b in blocks]
    return (sum(d) + (n - 1) * max(d)) / n


def pipeline_point(config, n_chunks, honest=True, fill_drain=True):
    """One (latency, peak) point at concurrency n_chunks, from the honest ATP-optimal design.
    fill_drain=True is the realistic (U-shaped) model; False is the free-chunk bound."""
    r = _atp(config, honest=honest)
    blocks = r["blocks"]
    P = [_stage_power_mW(b) for b in blocks]
    n_stages = len(blocks)
    n = max(1, int(n_chunks))                          # sequence chunks (up to L)
    lat_cyc = _makespan_cyc(blocks, n, fill_drain=fill_drain)
    latency_ms = lat_cyc / FREQ_HZ * 1e3
    co_active = min(n, n_stages)                        # pipeline DEPTH: at most n_stages co-active
    peak_mW = sum(sorted(P, reverse=True)[:co_active])
    return dict(config=config, n_chunks=n, n_stages=n_stages,
                latency_ms=latency_ms, peak_mW=peak_mW,
                area_mm2=r["total_area_mm2"], energy_uJ=r["energy_full_uJ"],
                total_pe=r["total_pe"], acc=r["acc"],
                avg_power_mW=r["energy_full_uJ"] / latency_ms if latency_ms else 0.0)


def pareto(config, honest=True, max_fifo=64, fill_drain=True):
    """The (latency, peak-power) frontier over the concurrency knob. Default fill_drain=True
    is the REALISTIC model -> latency is U-shaped in n (see optimal_n). max_fifo caps the knob."""
    r = _atp(config, honest=honest)
    ns = len(r["blocks"])
    knobs = sorted(set([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, ns, max_fifo]))
    return [pipeline_point(config, n, honest, fill_drain=fill_drain) for n in knobs if n <= max_fifo]


def optimal_n(config, honest=True, nmax=256):
    """The n that MINIMIZES realistic (fill/drain) latency -- the useful operating point, since
    chunking too finely re-pays fill/drain. Returns that pipeline_point."""
    return min((pipeline_point(config, n, honest, fill_drain=True) for n in range(1, nmax + 1)),
               key=lambda p: p["latency_ms"])


# ------------------------------------------------------------------
# Barrier model (increment 2c) — consumes the op-DAG. The bidirectional-scan 'rev' segments
# (one per layer) are hard flushes: a reverse scan needs the WHOLE sequence, so chunks cannot
# stream across them. The pipeline is therefore SEGMENTED at each 'rev' -> many short pipelines
# in series (pay fill/drain per segment, less overlap) instead of one long backbone pipe. The
# barrier adds NO work (conservation: n=1 barrier makespan == serial), it only limits overlap.
# ------------------------------------------------------------------
DAG_FILE = {"corner1": "workload_corner1_dag_dag.yaml", "corner3p": "workload_corner3p_dag_dag.yaml",
            "mambino2p0": "workload_mambino2p0_dag_dag.yaml"}


@functools.lru_cache(maxsize=None)
def _analyze(config, target=None, honest=True):
    """Cached design-point result. target=None -> ATP-optimal; else that target_cyc. Caching is
    essential: barrier_point + _shape_timing both need it, and config_ppac re-reads YAMLs+ERT."""
    return _atp(config, honest=honest) if target is None else m5.analyze(config, target, honest=honest)


@functools.lru_cache(maxsize=None)
def _shape_timing(config, honest=True, target=None):
    """shape -> (per_instance_cycles, fill_drain_frac, power_mW). target=None -> the ATP-optimal
    design; an explicit target_cyc -> that design point (needed for the FAIR both-knob barrier
    frontier). per_instance = block_cycles / rep_wc (rep_wc == #DAG nodes of that shape)."""
    return {b["shape"]: (b["cycles"] / b["rep_wc"], _block_filldrain_frac(b), _stage_power_mW(b))
            for b in _analyze(config, target, honest)["blocks"]}


@functools.lru_cache(maxsize=None)
def _barrier_segments(config):
    """Walk the op-DAG in id order; group GEMM-node shapes into segments split at each 'rev'
    elem barrier. Returns a tuple of segments, each a tuple of (shape, count) -- same-shape
    instances in one segment SHARE the shape's dedicated array (they serialize)."""
    d = yaml.safe_load(open(m4.WORKLOADS / DAG_FILE[config]))
    segs, cur = [], {}
    for n in sorted(d["nodes"], key=lambda x: x["id"]):
        if n["kind"] == "gemm":
            s = f"{n['M']}x{n['N']}x{n['K']}_{n['dtype']}"
            cur[s] = cur.get(s, 0) + 1
        elif "rev" in (n.get("prims") or {}):
            segs.append(tuple(cur.items()))
            cur = {}
    if cur:
        segs.append(tuple(cur.items()))
    return tuple(segs)


@functools.lru_cache(maxsize=None)
def _predictor_collapse(config, honest=True, target=None):
    """The SSM shape whose raw-v4 wall_clock_rep collapses (the predictor-overlap shape), with
    the CONCURRENT design's (factor, 2nd-array area cost mm2, 2nd-array power mW) AT THIS DESIGN
    POINT. Under the concurrent predictor, `factor`*count of that shape's instances serialize on
    the main array; the rest overlap on a dedicated 2nd array (+area, +peak). None for Pure S5.
    target must be threaded (Fable 2d): the SSM array size shrinks at large target_cyc, so the
    2nd-array cost is design-point-dependent -- freezing it at the ATP point over-charges
    concurrent at t>=524288. Anchor: corner3p ATP -> 8x2048x128_complex_real, 16/24, +1.84 mm2
    (19.76->21.60, memo), 377 mW."""
    for b in _analyze(config, target, honest)["blocks"]:
        parts = b["shape"].split("_")
        mnk = parts[0].split("x")
        tup = (int(mnk[0]), int(mnk[1]), int(mnk[2]), "_".join(parts[1:]))
        raw = m4.wall_clock_rep(tup, b["rep_jaxpr"])
        if raw != b["rep_jaxpr"]:
            extra = b["silicon_pe"] * m4.PE_AREA_UM2 / 1e6 * (
                1 + m4.NOC_AREA_FRACTION_OF_PE + m4.CONTROL_AREA_FRACTION_OF_PE)
            return (b["shape"], raw / b["rep_jaxpr"], extra, _stage_power_mW(b))
    return None


def _seg_stage_makespan(stages, n, fill_drain):
    """One barrier-segment's chunk-pipeline makespan. stages = [(duration_cyc, fd_frac), ...]."""
    if not stages:
        return 0.0
    if fill_drain:
        t = [d * f + d * (1 - f) / n for d, f in stages]
    else:
        t = [d / n for d, _ in stages]
    return sum(t) + (n - 1) * max(t)


def barrier_point(config, n_chunks, honest=True, fill_drain=True, target=None, concurrent_predictor=False):
    """(latency, peak) WITH per-layer bidir barriers enforced. Latency = sum over barrier-
    segments of each segment's internal chunk-pipeline makespan (no cross-segment streaming).
    Peak = max over segments (barriers serialize the segments -> one active at a time) of the
    segment's co-active power. Silicon/area/energy identical to pipeline_point (frozen).
    target=None uses the ATP-optimal design; pass a target_cyc for the both-knob frontier.
    concurrent_predictor=True: the predictor SSM runs on a dedicated 2nd array (factor*count of
    the SSM shape serialize on the main; the rest overlap) -> shorter SSM stage, +area, +peak."""
    if config not in DAG_FILE:
        raise ValueError(f"no op-DAG for {config}; barrier model needs an extracted DAG")
    r = _analyze(config, target, honest)
    sh = _shape_timing(config, honest, target)
    pc = _predictor_collapse(config, honest, target) if concurrent_predictor else None
    n = max(1, int(n_chunks))
    total_cyc, peak = 0.0, 0.0
    for seg in _barrier_segments(config):
        stages, powers = [], []
        for s, c in seg:
            eff, is_ssm = c, bool(pc) and s == pc[0]
            if is_ssm:
                eff = c * pc[1]            # predictor overlap: factor*count serialize on main array
                                            # (eff is integer when count%3==0, as in these DAGs)
            stages.append((sh[s][0] * eff, sh[s][1]))
            powers.append((sh[s][2], is_ssm))
        total_cyc += _seg_stage_makespan(stages, n, fill_drain)
        co = sorted(powers, key=lambda x: -x[0])[:min(n, len(powers))]      # co-active stages
        extra = pc[3] if (pc and any(i for _p, i in co)) else 0.0          # 2nd array draws power
        peak = max(peak, sum(p for p, _i in co) + extra)                    # only when SSM co-active
    latency_ms = total_cyc / FREQ_HZ * 1e3
    area = r["total_area_mm2"] + (pc[2] if pc else 0.0)
    return dict(config=config, n_chunks=n, latency_ms=latency_ms, peak_mW=peak,
                area_mm2=area, energy_uJ=r["energy_full_uJ"], acc=r["acc"],
                predictor="concurrent" if concurrent_predictor else "serial",
                avg_power_mW=r["energy_full_uJ"] / latency_ms if latency_ms else 0.0)


def barrier_optimal_n(config, honest=True, nmax=256):
    return min((barrier_point(config, n, honest, fill_drain=True) for n in range(1, nmax + 1)),
               key=lambda p: p["latency_ms"])


@functools.lru_cache(maxsize=None)
def _dag_topological(config):
    """True iff every DAG edge has src_id < dst_id -- i.e. id-order is a valid topological order.
    This is what makes id-order barrier segmentation CONSERVATIVE: a gemm placed after a 'rev'
    cannot be a predecessor of it, so we never optimistically stream a barrier-dependent op
    across the barrier (Fable-flagged: guards a future DAG with an aux side-branch gemm)."""
    d = yaml.safe_load(open(m4.WORKLOADS / DAG_FILE[config]))
    return all(e["src"] < e["dst"] for e in d["edges"])


def validate_barriers():
    """Barrier-model invariants (op-DAG configs only): (0) TOPOLOGY -- id-order is a valid
    topological order (all edges src<dst), so segmentation is conservative; (1) CONSERVATION --
    barrier n=1 latency == serial (barriers add no work, only forbid cross-segment overlap);
    (2) DOMINANCE -- barrier latency >= no-barrier latency at every n.
    NOTE the reported per-config floor inflation is a PER-ATP-DESIGN-POINT number, NOT an
    architecture claim -- most of Mambino's larger inflation is its barrier-blind ATP encoder
    (8x8 array, ~30% of the floor). With BOTH knobs free the inflation is near-symmetric
    (~1.06x vs ~1.03x), and on the FAIR both-knob frontier barriers HELP Mambino's iso-latency
    case (see merged_frontier.iso_latency_both_knobs). Read the crossover THERE, never here."""
    print("=== VALIDATION: barrier model (op-DAG configs) ===")
    ok = True
    for cfg in DAG_FILE:
        b = _atp(cfg, honest=True)
        topo = _dag_topological(cfg)
        cons = abs(barrier_point(cfg, 1)["latency_ms"] - b["lat_serial_ms"]) < 1e-6
        dom = all(barrier_point(cfg, n)["latency_ms"] >= pipeline_point(cfg, n)["latency_ms"] - 1e-9
                  for n in [1, 2, 4, 8, 16, 32])
        nb, br = optimal_n(cfg), barrier_optimal_n(cfg)
        tag = "PASS" if (topo and cons and dom) else "FAIL"
        if not (topo and cons and dom):
            ok = False
        print(f"  [{tag}] {cfg:<9} topo={topo} conservation(n=1==serial)={cons} barrier>=no-barrier={dom} | "
              f"ATP-pt floor {nb['latency_ms']:.3f}->{br['latency_ms']:.3f} ms ({br['latency_ms']/nb['latency_ms']:.2f}x, "
              f"design-pt artifact -- read crossover on the fair both-knob frontier)")
    print("BARRIER VALIDATION:", "ALL PASS" if ok else "FAILURES")
    return ok


# ------------------------------------------------------------------
# Validation + A/B
# ------------------------------------------------------------------
def validate_bounds():
    """(A) INTERPOLATION BOUND (fill_drain=False): n=1 == SERIAL exact (sum d, max P); peak
        saturates at CONCURRENT peak (sum P) for n>=n_stages; latency -> CONCURRENT max_d
        asymptotically (n=1e7, from above); monotone decreasing.
    (B) REALISTIC (fill_drain=True): n=1 == SERIAL exact; latency U-SHAPED -- an interior
        optimum <= serial, and latency at very large n exceeds that optimum (fill/drain tax)."""
    print("=== VALIDATION: (A) interpolation bound + (B) realistic U-shape ===")
    ok = True
    for cfg in ["corner1", "corner2", "corner3p", "config4", "config5"]:
        b = _atp(cfg, honest=True)
        # (A) free-chunk interpolation bound
        p1 = pipeline_point(cfg, 1, fill_drain=False)
        pSat = pipeline_point(cfg, len(b["blocks"]), fill_drain=False)
        pInf = pipeline_point(cfg, 10 ** 7, fill_drain=False)
        ser = abs(p1["latency_ms"] - b["lat_serial_ms"]) < 1e-9 and abs(p1["peak_mW"] - b["peak_serial_mW"]) < 1e-6
        peak_sat = abs(pSat["peak_mW"] - b["peak_concurrent_mW"]) < 1e-6
        lat_asym = abs(pInf["latency_ms"] - b["lat_concurrent_ms"]) / b["lat_concurrent_ms"] < 1e-3 \
            and pInf["latency_ms"] >= b["lat_concurrent_ms"] - 1e-12
        sweepA = [pipeline_point(cfg, n, fill_drain=False) for n in [1, 2, 4, 8, 16, 32, 64]]
        monoA = all(sweepA[i]["latency_ms"] >= sweepA[i + 1]["latency_ms"] - 1e-12 for i in range(len(sweepA) - 1))
        # (B) realistic
        pr1 = pipeline_point(cfg, 1, fill_drain=True)
        rser = abs(pr1["latency_ms"] - b["lat_serial_ms"]) < 1e-9
        opt = optimal_n(cfg)
        big = pipeline_point(cfg, 4096, fill_drain=True)
        ushape = opt["latency_ms"] <= pr1["latency_ms"] + 1e-12 and big["latency_ms"] > opt["latency_ms"] + 1e-12
        tag = "PASS" if (ser and peak_sat and lat_asym and monoA and rser and ushape) else "FAIL"
        if tag == "FAIL":
            ok = False
        print(f"  [{tag}] {cfg:<9} (A) serial {p1['latency_ms']:.3f}=={b['lat_serial_ms']:.3f} "
              f"peak_sat {pSat['peak_mW']:.0f}=={b['peak_concurrent_mW']:.0f} asym->{b['lat_concurrent_ms']:.3f} mono {monoA} "
              f"| (B) opt n*={opt['n_chunks']} lat {opt['latency_ms']:.3f} (serial {pr1['latency_ms']:.3f}, "
              f"n=4096 {big['latency_ms']:.3f})")
    print("VALIDATION:", "ALL PASS" if ok else "FAILURES")
    return ok


def ab_pareto():
    print("\n=== per-config (latency vs peak-power) frontier — REALISTIC (fill/drain), honest ===")
    for cfg in ["corner1", "corner2", "corner3p"]:
        opt = optimal_n(cfg)
        print(f"\n{cfg} (area {_atp(cfg,honest=True)['total_area_mm2']:.2f} mm2, "
              f"acc {m4.ACC[cfg]:.4f}); latency-optimal n*={opt['n_chunks']} @ {opt['latency_ms']:.3f} ms:")
        print(f"    {'n_chunks':>8}{'latency ms':>12}{'peak mW':>10}{'avg mW':>9}")
        for p in pareto(cfg):
            star = " *" if p["n_chunks"] == opt["n_chunks"] else ""
            print(f"    {p['n_chunks']:>8}{p['latency_ms']:>12.3f}{p['peak_mW']:>10.0f}{p['avg_power_mW']:>9.0f}{star}")
    # Matched-n_chunks A/B — shown ONLY to expose why it is NOT fair (different latencies =
    # different delivered work). For the honest cross-config verdict run merged_frontier.py.
    print("\n=== matched n_chunks A/B — UNFAIR (different latencies); see merged_frontier.py ===")
    for n in [1, 2, 4]:
        c3 = pipeline_point("corner3p", n); c1 = pipeline_point("corner1", n)
        print(f"  n={n}: Mambino {c3['latency_ms']:.3f} ms / {c3['peak_mW']:.0f} mW vs "
              f"Pure S5 {c1['latency_ms']:.3f} ms / {c1['peak_mW']:.0f} mW "
              f"-- Mambino is {c3['latency_ms']/c1['latency_ms']:.2f}x SLOWER here, so the "
              f"peak gap is not iso-work")


def barrier_ab():
    print("\n=== bidir-barrier effect on the ATP-point latency floor (NOT the crossover) ===")
    print("  (per-config ATP design point; the fair crossover is in merged_frontier.iso_latency_both_knobs)")
    for cfg in DAG_FILE:
        nb, br = optimal_n(cfg), barrier_optimal_n(cfg)
        tag = "Pure S5" if cfg == "corner1" else "Mambino"
        print(f"  {cfg} ({tag}): floor {nb['latency_ms']:.3f} -> {br['latency_ms']:.3f} ms "
              f"({br['latency_ms']/nb['latency_ms']:.2f}x, design-pt artifact), peak {br['peak_mW']:.0f} mW @ n*={br['n_chunks']}")


if __name__ == "__main__":
    validate_bounds()
    validate_barriers()
    ab_pareto()
    barrier_ab()
