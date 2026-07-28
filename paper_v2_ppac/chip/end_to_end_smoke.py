"""end_to_end_smoke.py — full-chain smoke test of the v5 datapath PPAC.

Walks the WHOLE pipeline and asserts at every stage boundary, so a single run tells us
the JAXPR->workload->DAG->v4->v5->scheduler chain is connected and self-consistent:

  STAGE 1  extraction artifacts : workload GEMM/elem YAMLs load; op-DAG loads; DAG's
                                  gemm_count == sum of workload GEMM repeats; the per-layer
                                  bidir barrier ('rev') and gate/GLU chain ('logistic') are
                                  present in the DAG (the structure the scheduler keys on).
  STAGE 2  v4 config_ppac       : blocks non-empty; every block has cycles>0, energy>0.
  STAGE 3  frozen-silicon       : the scheduler passes v5's area/energy through UNCHANGED for
                                  every concurrency n (the knob must not leak into silicon).
  STAGE 4  scheduler invariants : validate_bounds() passes -- (A) free-chunk interpolation
                                  bound (serial exact, concurrent-peak exact, latency->max_d
                                  asymptote, monotone) and (B) realistic fill/drain model
                                  (serial exact, latency U-shaped with interior optimum).

Asserts ONLY model invariants + the FLOP invariant + frozen silicon -- NEVER which config
wins (the fair cross-config comparison is merged_frontier.py / peak_power_scenarios.py).
Exit 0 = all stages PASS. Any assertion failure prints [FAIL] and exits 1.
"""
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multi_array_ppac_v4 as m4
import multi_array_ppac_v5 as m5
import ppac_v5_scheduler as sch

WL = m4.WORKLOADS
CONFIGS = ["corner1", "corner2", "corner3p", "config4", "config5"]
DAG_CONFIGS = {"corner1": "workload_corner1_dag_dag.yaml",
               "corner3p": "workload_corner3p_dag_dag.yaml"}

_fail = [0]


def check(cond, msg):
    tag = "PASS" if cond else "FAIL"
    if not cond:
        _fail[0] += 1
    print(f"    [{tag}] {msg}")
    return cond


def stage1_artifacts():
    print("STAGE 1 — extraction artifacts (workload + op-DAG) load & self-consistent")
    for cfg in CONFIGS:
        g = yaml.safe_load(open(WL / f"workload_{cfg}_gemms.yaml"))["gemms"]
        e = yaml.safe_load(open(WL / f"workload_{cfg}_elemwise.yaml"))
        check(len(g) > 0, f"{cfg}: {len(g)} GEMM shapes, elem present={bool(e)}")
        # HARD FLOP INVARIANT (the locked PPAC rule): sum(emitted GEMM FLOPs) == JAXPR
        # walker dot_general total. Catches a corrupted M/N/K even with correct repeats.
        man = yaml.safe_load(open(WL / f"workload_{cfg}_manifest.yaml"))
        emitted = sum(int(x["total_real_macs"]) * 2 for x in g)
        walker = man["jaxpr_walker_totals"]["dot_general"]
        check(emitted == walker,
              f"{cfg}: FLOP invariant emitted {emitted:,} == walker {walker:,}")
    for cfg, fn in DAG_CONFIGS.items():
        d = yaml.safe_load(open(WL / fn))
        nodes = d["nodes"]
        gemm_nodes = [n for n in nodes if n["kind"] == "gemm"]
        elem_nodes = [n for n in nodes if n["kind"] == "elem"]
        check(len(nodes) == d["node_count"] and len(gemm_nodes) == d["gemm_count"],
              f"{cfg} DAG: {len(nodes)} nodes ({len(gemm_nodes)} gemm) == header")
        # DAG gemm_count (per-instance) == sum of workload GEMM repeats
        g = yaml.safe_load(open(WL / f"workload_{cfg}_gemms.yaml"))["gemms"]
        rep_sum = sum(int(x.get("repeat", 1)) for x in g)
        check(d["gemm_count"] == rep_sum,
              f"{cfg} DAG: gemm_count {d['gemm_count']} == sum(workload repeats) {rep_sum}")
        # scheduler-relevant structure: bidir barrier + gate chain findable
        rev = [n for n in elem_nodes if "rev" in (n.get("prims") or {})]
        gate = [n for n in elem_nodes if "logistic" in (n.get("prims") or {})]
        check(len(rev) > 0, f"{cfg} DAG: bidir barrier ('rev') findable — {len(rev)} nodes")
        if cfg == "corner3p":
            check(len(gate) > 0, f"{cfg} DAG: gate/GLU chain ('logistic') findable — {len(gate)} nodes")


def stage2_v4():
    print("STAGE 2 — v4 config_ppac blocks well-formed")
    for cfg in CONFIGS:
        r = sch._atp(cfg, honest=True)
        blocks = r["blocks"]
        ok = len(blocks) > 0 and all(b["cycles"] > 0 and b["energy_pJ"] > 0 for b in blocks)
        check(ok, f"{cfg}: {len(blocks)} blocks, all cycles>0 & energy>0; "
                  f"area {r['total_area_mm2']:.2f} mm2, energy {r['energy_full_uJ']:.0f} uJ")


def stage3_v5():
    # The scheduler is a post-processing layer that recomputes ONLY latency + peak. Its
    # silicon (area/energy/acc) must equal v5's honest atp_optimal for EVERY concurrency n
    # -- the knob must not leak into area/energy. (Note: v5-honest legitimately differs from
    # v4-raw for the 2 Mambino configs because honest mode disables wall_clock_rep; that is
    # the documented predictor-overlap correction, not a bug -- so we anchor to v5, not v4.)
    print("STAGE 3 — FROZEN-SILICON invariant (scheduler passes v5 area/energy through, all n)")
    for cfg in CONFIGS:
        b = sch._atp(cfg, honest=True)
        ok = all(abs(sch.pipeline_point(cfg, n)["area_mm2"] - b["total_area_mm2"]) < 1e-9
                 and abs(sch.pipeline_point(cfg, n)["energy_uJ"] - b["energy_full_uJ"]) < 1e-6
                 for n in [1, 2, 4, 16, 64])
        check(ok, f"{cfg}: pipeline_point area/energy == v5 atp "
                  f"{b['total_area_mm2']:.3f} mm2 / {b['energy_full_uJ']:.1f} uJ for all n")


def stage4_scheduler():
    print("STAGE 4 — scheduler model invariants (interp-bound + realistic U-shape)")
    ok = sch.validate_bounds()
    check(ok, "validate_bounds(): (A) interp-bound endpoints+monotone, (B) realistic U-shape, all 5 configs")
    for cfg in ["corner1", "corner3p"]:
        # (A) the free-chunk interpolation bound is monotone decreasing in n
        A = [sch.pipeline_point(cfg, n, fill_drain=False) for n in [1, 2, 4, 8, 16, 32, 64]]
        monoA = all(A[i]["latency_ms"] >= A[i + 1]["latency_ms"] - 1e-12
                    and A[i]["peak_mW"] <= A[i + 1]["peak_mW"] + 1e-9 for i in range(len(A) - 1))
        # (B) the realistic model has an interior latency optimum below serial (chunking helps)
        opt = sch.optimal_n(cfg)
        ser = sch.pipeline_point(cfg, 1)["latency_ms"]
        check(monoA and opt["latency_ms"] < ser - 1e-9,
              f"{cfg}: interp-bound monotone={monoA}; realistic optimum n*={opt['n_chunks']} @ "
              f"{opt['latency_ms']:.3f} ms (< serial {ser:.3f})")
    # (C) barrier model (op-DAG configs): conservation + barrier>=no-barrier
    bok = sch.validate_barriers()
    check(bok, "validate_barriers(): n=1==serial conservation + barrier>=no-barrier, DAG configs")
    # NEUTRAL REPORT (not an assertion): the matched-n_chunks A/B is NOT a fair comparison --
    # it holds each config at its own per-config ATP design point (different speed grades) and
    # hides Mambino's latency deficit. The fair (target_cyc x n_chunks) merged frontier lives
    # in merged_frontier.py. Printed here for visibility, deliberately asserting NOTHING about
    # which config wins (a smoke test must not encode a contested conclusion as a pass gate).
    print("    [note] matched-n_chunks A/B (NOT the fair comparison -- see merged_frontier.py):")
    for n in [1, 2, 4]:
        c3 = sch.pipeline_point("corner3p", n)
        c1 = sch.pipeline_point("corner1", n)
        print(f"           n={n}: Mambino ({c3['latency_ms']:.3f} ms, {c3['peak_mW']:.0f} mW)  "
              f"Pure S5 ({c1['latency_ms']:.3f} ms, {c1['peak_mW']:.0f} mW)  "
              f"[different latencies -> not iso-work]")


if __name__ == "__main__":
    print("=" * 78)
    print("END-TO-END SMOKE TEST — v5 datapath PPAC (JAXPR -> DAG -> v4 -> v5 -> scheduler)")
    print("=" * 78)
    stage1_artifacts()
    stage2_v4()
    stage3_v5()
    stage4_scheduler()
    print("=" * 78)
    if _fail[0] == 0:
        print("END-TO-END SMOKE: ALL STAGES PASS")
        sys.exit(0)
    print(f"END-TO-END SMOKE: {_fail[0]} FAILURE(S)")
    sys.exit(1)
