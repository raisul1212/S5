"""Extract chip-PPAC workload files from the compiled JAXPR of a trained
Mambino / S5 checkpoint's forward pass.

Emits, per config:
  workload_<cfg>_gemms.csv       -- one row per dot_general (SCALE-Sim compatible)
  workload_<cfg>_gemms.yaml      -- Timeloop-friendly problem instances by shape
  workload_<cfg>_elemwise.yaml   -- aggregated elementwise ops MARKED by class
  workload_<cfg>_manifest.yaml   -- provenance + hard invariant check result

HARD METHODOLOGY RULE (locked 2026-07-08):
  For the extracted workload to be valid, the aggregate FLOPs the emitter
  reconstructs from its per-op records MUST equal the JAXPR walker's totals
  exactly.  Any deviation is a bug; the script errors out.

  emitter.dot_general_flops   ==  walker.total_dot_general
  emitter.elementwise_flops   ==  walker.total_elementwise

This enforces "no hand-authored workload" — every op tracked here is
exactly one op the compiler actually emitted in the trained model's
forward pass.

Usage:
  python extract_workload_from_jaxpr.py \
      --ckpt_prefix checkpoints/corner1_seed42_11187522/best \
      --config_name corner1 \
      --out_dir     paper_v2_ppac/workloads/ \
      --use_mambino_ssm False --ssm_size_base 16 \
      --activation_fn half_glu2 --glu_rank 0 --bidirectional True \
      --blocks 8 --n_layers 8 --d_model 128 --dataset listops-classification
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from functools import partial

import numpy as np
import jax
from jax import random
import jax.numpy as jnp
from jax.scipy.linalg import block_diag

import yaml

# Reuse existing walker's helpers where possible
from flop_counter_jaxpr import (
    _is_complex_dtype,
    _prod,
    _dot_general_flops,
    _elementwise_flops,
    _get_inner_jaxpr,
    walk_jaxpr,
)


# ============================================================
# Elementwise op classification
#
# Each JAX primitive is placed in one of five classes.  Downstream
# PPAC tools should model each class differently:
#
#   trivial       : add/sub/neg — folds into MAC array pipeline
#                   (basically free; no separate energy line item)
#   moderate      : mul/div/pow — small ALU, ~1-3 real FLOPs per elem
#                   at real precision, more at complex
#   transcendental: exp/log/tanh/sig/rsqrt/etc. — needs LUT or Taylor
#                   expansion; expensive per-elem (~4-10x MAC energy)
#   reduction     : reduce_sum/max/prod — chain of adds, memory-bound
#   structural    : reshape/transpose/broadcast/slice — 0 FLOPs, data
#                   movement (still costs bytes moved through buffers)
# ============================================================
ELEMWISE_CLASS = {
    # trivial (~1 FLOP/elem real)
    "add": "trivial", "sub": "trivial", "neg": "trivial", "abs": "trivial",
    # moderate (~1-3 FLOPs/elem)
    "mul": "moderate", "div": "moderate",
    "integer_pow": "moderate", "pow": "moderate",
    # transcendental (~4-10 FLOPs/elem, LUT-approximated on chip)
    "exp": "transcendental", "log": "transcendental",
    "log1p": "transcendental", "expm1": "transcendental",
    "sqrt": "transcendental", "rsqrt": "transcendental",
    "sin": "transcendental", "cos": "transcendental",
    "tan": "transcendental", "tanh": "transcendental",
    "sinh": "transcendental", "cosh": "transcendental",
    "asin": "transcendental", "acos": "transcendental", "atan": "transcendental",
    "erf": "transcendental", "logistic": "transcendental",
    # reductions
    "reduce_sum": "reduction", "reduce_max": "reduction",
    "reduce_min": "reduction", "reduce_prod": "reduction",
    "reduce_and": "reduction", "reduce_or": "reduction",
    # complex real/imag/conj — essentially free at chip layout
    "real": "structural", "imag": "structural",
    "complex": "structural", "conj": "structural",
    # structural (0 FLOP data movement or metadata)
    "broadcast_in_dim": "structural", "reshape": "structural",
    "transpose": "structural", "squeeze": "structural",
    "concatenate": "structural", "slice": "structural",
    "dynamic_slice": "structural", "dynamic_update_slice": "structural",
    "gather": "structural", "scatter": "structural",
    "iota": "structural", "convert_element_type": "structural",
    "bitcast_convert_type": "structural",
    "select_n": "structural", "cond": "structural",
    "pad": "structural", "reverse": "structural", "rev": "structural",
    "sort": "structural", "argmax": "structural", "argmin": "structural",
    "stop_gradient": "structural", "device_put": "structural",
    "random_seed": "structural", "random_unwrap": "structural",
    "random_clone": "structural", "random_wrap": "structural",
    "random_split": "structural",
    # comparisons / min-max reductions (structural for chip-PPAC purposes)
    "lt": "structural", "le": "structural", "gt": "structural",
    "ge": "structural", "eq": "structural", "ne": "structural",
    "max": "structural", "min": "structural",
}


# ============================================================
# The emitter
# ============================================================
class WorkloadEmitter:
    """Collects per-op records as we walk the JAXPR. Two rules:
       (a) every dot_general becomes its own GEMM record;
       (b) every elementwise op is aggregated by (primitive, shape,
           dtype-signature) with a scalar-op count.
       The emitter recomputes total FLOPs from its records, and the
       caller cross-checks against the walker's independent totals.
    """

    def __init__(self):
        # Each GEMM record:
        #   dict(name, M, N, K, dtype_lhs, dtype_rhs, per_mac_flops,
        #        repeat, flops, source_op)
        self.gemms: list[dict] = []
        # Elementwise: aggregate by (primitive, shape_tuple, dtype_sig)
        self.elemwise: dict[tuple, dict] = {}
        # Provenance stack of source ops (from pjit/scan nesting)
        self.gemm_seq = 0

    # ---- Add records ----
    def add_dot_general(self, params, invars, outvars, multiplier: int,
                        source_stack: list[str]) -> int:
        """Add one GEMM record. Returns the FLOP count contributed."""
        dim_nums = params["dimension_numbers"]
        (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dim_nums
        lhs_shape = invars[0].aval.shape
        rhs_shape = invars[1].aval.shape
        lhs_complex = _is_complex_dtype(invars[0].aval.dtype)
        rhs_complex = _is_complex_dtype(invars[1].aval.dtype)

        batch_size = _prod([lhs_shape[i] for i in lhs_batch])
        lhs_out = [lhs_shape[i] for i in range(len(lhs_shape))
                   if i not in lhs_contract and i not in lhs_batch]
        rhs_out = [rhs_shape[i] for i in range(len(rhs_shape))
                   if i not in rhs_contract and i not in rhs_batch]
        M = _prod(lhs_out)
        N = _prod(rhs_out)
        K = _prod([lhs_shape[i] for i in lhs_contract])

        if lhs_complex and rhs_complex:
            per_mac = 8
            dtype_sig = "complex_complex"
        elif lhs_complex ^ rhs_complex:
            per_mac = 4
            dtype_sig = "complex_real"
        else:
            per_mac = 2
            dtype_sig = "real_real"

        # One dot_general instance contributes: batch × M × N × K MACs
        # (with per_mac FLOPs per MAC).  The outer `multiplier` captures
        # times the JAXPR walker would enter this eqn (scan iterations,
        # jit re-runs, etc.), so real MACs = multiplier × batch × M × N × K.
        macs_per_instance = int(batch_size * M * N * K)
        flops_per_instance = macs_per_instance * per_mac
        total_flops = int(multiplier * flops_per_instance)

        self.gemm_seq += 1
        rec = dict(
            name=f"gemm_{self.gemm_seq:04d}",
            M=int(M),
            N=int(N),
            K=int(K),
            batch=int(batch_size),
            dtype=dtype_sig,
            per_mac_flops=per_mac,
            macs_per_instance=macs_per_instance,
            repeat=int(multiplier),
            total_flops=total_flops,
            total_real_macs=total_flops // 2,  # dot_general/2 basis (paper §4)
            source_op="/".join(source_stack) or "root",
        )
        self.gemms.append(rec)
        return total_flops

    def add_elemwise(self, primitive: str, eqn, multiplier: int,
                     source_stack: list[str]) -> int:
        """Add / aggregate one elementwise op. Returns FLOP count."""
        # Cost table (same as flop_counter_jaxpr for parity)
        def _cost(cost_real, cost_complex, cost_complex_real=None):
            return _elementwise_flops(eqn,
                                      cost_real=cost_real,
                                      cost_complex=cost_complex,
                                      cost_complex_real=cost_complex_real)

        if primitive in ("add", "sub"):
            per_instance = _cost(cost_real=1, cost_complex=2, cost_complex_real=1)
        elif primitive == "mul":
            per_instance = _cost(cost_real=1, cost_complex=6, cost_complex_real=2)
        elif primitive == "div":
            per_instance = _cost(cost_real=1, cost_complex=10, cost_complex_real=2)
        elif primitive in ("exp", "log", "log1p", "expm1", "sqrt",
                           "rsqrt", "sin", "cos", "tan", "tanh",
                           "sinh", "cosh", "asin", "acos", "atan",
                           "erf", "logistic"):
            per_instance = _cost(cost_real=4, cost_complex=8)
        elif primitive in ("integer_pow", "pow"):
            per_instance = _cost(cost_real=3, cost_complex=8)
        elif primitive in ("neg", "abs"):
            per_instance = _cost(cost_real=1, cost_complex=2)
        elif primitive in ("real", "imag", "complex", "conj"):
            per_instance = _cost(cost_real=0, cost_complex=0)
        elif primitive in ("reduce_sum", "reduce_max", "reduce_min",
                           "reduce_prod", "reduce_and", "reduce_or"):
            if eqn.outvars and eqn.invars:
                in_sz = _prod(eqn.invars[0].aval.shape)
                out_sz = _prod(eqn.outvars[0].aval.shape) if eqn.outvars[0].aval.shape else 1
                per_instance = max(0, in_sz - out_sz)
            else:
                per_instance = 0
        else:
            # structural (reshape, transpose, etc.)
            per_instance = 0

        total_flops = int(multiplier * per_instance)

        # Aggregate key: (primitive, out_shape_tuple, dtype_sig)
        if eqn.outvars:
            out_aval = eqn.outvars[0].aval
            out_shape = tuple(out_aval.shape)
            out_complex = _is_complex_dtype(out_aval.dtype)
        else:
            out_shape = ()
            out_complex = False
        in_complex_flags = tuple(
            _is_complex_dtype(v.aval.dtype) if hasattr(v, "aval") else False
            for v in eqn.invars
        )
        dtype_sig = ("complex" if out_complex else "real") + \
                    ("_c-in" if any(in_complex_flags) and not all(in_complex_flags) else "")

        key = (primitive, out_shape, dtype_sig)
        if key not in self.elemwise:
            self.elemwise[key] = dict(
                primitive=primitive,
                energy_class=ELEMWISE_CLASS.get(primitive, "unknown"),
                out_shape=list(out_shape),
                dtype_sig=dtype_sig,
                per_instance_flops=int(per_instance),
                instance_count=0,
                total_flops=0,
                total_scalar_ops=0,
                source_ops=set(),
            )
        rec = self.elemwise[key]
        rec["instance_count"] += multiplier
        rec["total_flops"] += total_flops
        rec["total_scalar_ops"] += int(multiplier * _prod(out_shape) if out_shape else multiplier)
        rec["source_ops"].add("/".join(source_stack) or "root")
        return total_flops

    # ---- Reconstruct totals for invariant check ----
    def total_dot_general_flops(self) -> int:
        return sum(r["total_flops"] for r in self.gemms)

    def total_elementwise_flops(self) -> int:
        return sum(r["total_flops"] for r in self.elemwise.values())

    # ---- Emit files ----
    def emit_scalesim_csv(self, path: str):
        """SCALE-Sim CSV: one row per unique GEMM shape × dtype × repeat."""
        # SCALE-Sim reads GEMM as M,N,K per row (its `-i gemm` mode).
        # Column format (required minimum 4 comma-separated fields):
        #   Layer, M, N, K,
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Layer", "M", "N", "K", "dtype", "batch", "repeat",
                        "per_mac_flops", "total_real_macs", "source_op"])
            for r in self.gemms:
                w.writerow([
                    r["name"], r["M"], r["N"], r["K"],
                    r["dtype"], r["batch"], r["repeat"],
                    r["per_mac_flops"], r["total_real_macs"],
                    r["source_op"],
                ])

    def emit_timeloop_yaml(self, path: str):
        """Timeloop-friendly summary: unique GEMM shapes with repeat counts."""
        # Group by (M, K, N, dtype, batch) — Timeloop's `problem.instance` is
        # keyed on the tensor shape. Same-shape GEMMs share a mapping search.
        groups: dict[tuple, dict] = {}
        for r in self.gemms:
            key = (r["M"], r["N"], r["K"], r["batch"], r["dtype"])
            if key not in groups:
                groups[key] = dict(
                    M=r["M"], N=r["N"], K=r["K"], batch=r["batch"],
                    dtype=r["dtype"], per_mac_flops=r["per_mac_flops"],
                    repeat=0, total_real_macs=0,
                    source_ops=[],
                )
            g = groups[key]
            g["repeat"] += r["repeat"]
            g["total_real_macs"] += r["total_real_macs"]
            g["source_ops"].append(dict(name=r["name"], src=r["source_op"]))

        payload = dict(
            format="timeloop-problem-summary-v1",
            unique_gemm_count=len(groups),
            total_gemm_instances=sum(r["repeat"] for r in self.gemms),
            gemms=[
                dict(
                    M=int(g["M"]), N=int(g["N"]), K=int(g["K"]),
                    batch=int(g["batch"]),
                    dtype=g["dtype"], per_mac_flops=int(g["per_mac_flops"]),
                    repeat=int(g["repeat"]),
                    total_real_macs=int(g["total_real_macs"]),
                    source_ops=g["source_ops"],
                )
                for g in groups.values()
            ],
        )
        with open(path, "w") as f:
            yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)

    def emit_elemwise_yaml(self, path: str):
        """Aggregated elementwise ops, one entry per (primitive, shape, dtype).

        Each entry is tagged with `energy_class` so downstream PPAC tools
        know how to model it (see ELEMWISE_CLASS docstring at top of file).
        """
        by_class: dict[str, list] = defaultdict(list)
        for rec in self.elemwise.values():
            by_class[rec["energy_class"]].append(dict(
                primitive=rec["primitive"],
                out_shape=rec["out_shape"],
                dtype_sig=rec["dtype_sig"],
                per_instance_flops=rec["per_instance_flops"],
                instance_count=int(rec["instance_count"]),
                total_flops=int(rec["total_flops"]),
                total_scalar_ops=int(rec["total_scalar_ops"]),
                source_ops=sorted(rec["source_ops"]),
            ))

        class_totals = {
            cls: dict(
                total_flops=sum(e["total_flops"] for e in entries),
                total_scalar_ops=sum(e["total_scalar_ops"] for e in entries),
                primitive_count=len(entries),
            )
            for cls, entries in by_class.items()
        }

        payload = dict(
            format="elementwise-workload-v1",
            energy_class_totals=class_totals,
            note=(
                "energy_class semantics: "
                "trivial (add/sub) folds into MAC pipeline; "
                "moderate (mul/div) small ALU ~1-3 FLOPs/elem; "
                "transcendental (exp/log/tanh/sig/rsqrt) LUT ~4-10 FLOPs/elem; "
                "reduction (reduce_sum/max/prod) chain of adds, memory-bound; "
                "structural (reshape/transpose/broadcast/etc.) 0 FLOPs, data movement only."
            ),
            entries_by_class=dict(by_class),
        )
        with open(path, "w") as f:
            yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)

    def emit_manifest(self, path: str, walker_totals: dict,
                      checkpoint_path: str, args: dict,
                      git_sha: str, invariant_ok: bool,
                      invariant_msg: str):
        """Master manifest with provenance + hard invariant check result."""
        payload = dict(
            format="workload-manifest-v1",
            generated_by="extract_workload_from_jaxpr.py",
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            git_sha=git_sha,
            checkpoint_path=checkpoint_path,
            args=args,
            jaxpr_walker_totals={
                k: int(v) for k, v in walker_totals.items()
                if not k.startswith("_")
            },
            emitter_totals=dict(
                gemm_count=len(self.gemms),
                unique_gemm_shapes=len({(r["M"], r["N"], r["K"], r["batch"],
                                        r["dtype"]) for r in self.gemms}),
                total_dot_general_flops=self.total_dot_general_flops(),
                total_elementwise_flops=self.total_elementwise_flops(),
                elemwise_unique_signatures=len(self.elemwise),
            ),
            hard_invariant_check=dict(
                passed=invariant_ok,
                message=invariant_msg,
                rule=(
                    "sum(emitter.dot_general) == walker.dot_general AND "
                    "sum(emitter.elementwise) == sum(walker.elementwise)"
                ),
            ),
        )
        with open(path, "w") as f:
            yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)


# ============================================================
# JAXPR walker for emission (parallel structure to flop_counter_jaxpr)
# ============================================================
def emit_from_jaxpr(jaxpr, emitter: WorkloadEmitter,
                    multiplier: int = 1,
                    depth: int = 0,
                    source_stack: list[str] | None = None):
    """Walk a jaxpr and call emitter.add_* per primitive.

    Recurses into scan / while / pjit / call / custom_jvp.  The
    multiplier tracks how many times an inner op is executed (scan
    length, etc.) so the emitted totals equal wall-clock inference work.
    """
    source_stack = source_stack or []
    for eqn in jaxpr.eqns:
        prim = eqn.primitive.name

        # Higher-order: recurse with adjusted multiplier
        if prim == "scan":
            inner = _get_inner_jaxpr(eqn.params, "jaxpr")
            length = int(eqn.params.get("length", 1))
            if inner is not None:
                emit_from_jaxpr(inner, emitter, multiplier * length,
                                depth + 1, source_stack + [f"scan[L={length}]"])
            continue

        if prim == "while":
            body = _get_inner_jaxpr(eqn.params, "body_jaxpr")
            # Iterations unknown; log presence in source_stack for now
            if body is not None:
                emit_from_jaxpr(body, emitter, multiplier, depth + 1,
                                source_stack + ["while"])
            continue

        if prim in ("pjit", "jit", "call", "xla_call",
                    "custom_jvp_call", "custom_vjp_call", "checkpoint"):
            inner = (_get_inner_jaxpr(eqn.params, "jaxpr")
                     or _get_inner_jaxpr(eqn.params, "call_jaxpr"))
            if inner is not None:
                name = eqn.params.get("name") or prim
                emit_from_jaxpr(inner, emitter, multiplier, depth + 1,
                                source_stack + [str(name)])
            continue

        # Leaf primitives
        if prim == "dot_general":
            emitter.add_dot_general(eqn.params, eqn.invars, eqn.outvars,
                                    multiplier, source_stack)
        else:
            emitter.add_elemwise(prim, eqn, multiplier, source_stack)


# ============================================================
# Op-DAG extractor (increment 2) — producer->consumer graph at GEMM +
# elementwise-SEGMENT granularity for the v5 pipeline scheduler + fusion.
# Same JAXPR walk as the emitter (scan/pjit recursion), plus var-dependency
# tracking. Predictor vs main-scan GEMMs are distinguished by source_op path.
# ============================================================
def _vkey(v):
    return id(v)   # unique per live Var object; we hold refs through the walk


def _bytes_of(v):
    try:
        n = 1
        for s in v.aval.shape:
            n *= int(s)
        return n * (2 if _is_complex_dtype(v.aval.dtype) else 1)   # INT8: 1 byte/real
    except Exception:
        return 0


class DagBuilder:
    def __init__(self):
        self.nodes = []          # gemm: {id,kind,M,N,K,batch,dtype,repeat,source}
        self.edges = {}          # (src,dst) -> bytes (max seen)

    def new_node(self, rec):
        rec["id"] = len(self.nodes)
        self.nodes.append(rec)
        return rec["id"]

    def add_edge(self, src, dst, b):
        if src is None or src == dst:
            return
        self.edges[(src, dst)] = max(self.edges.get((src, dst), 0), int(b))


def build_dag(jaxpr, dag: DagBuilder, env: dict, multiplier: int, source_stack):
    """env: vkey -> producer node id for this jaxpr's free (in)vars.
    Returns vkey -> producer node id for this jaxpr's OUTVARS (for the caller)."""
    prod = dict(env)
    pending = []   # consecutive elementwise eqns -> one segment node

    def flush():
        if not pending:
            return
        inside = {_vkey(v) for e in pending for v in e.outvars}
        seg_in, seg_out, nscalar = [], [], 0
        for e in pending:
            for v in e.invars:
                if hasattr(v, "aval") and _vkey(v) not in inside:
                    seg_in.append(v)
            seg_out.extend(e.outvars)
            if e.outvars and hasattr(e.outvars[0], "aval"):
                n = 1
                for s in e.outvars[0].aval.shape:
                    n *= int(s)
                nscalar += n
        nid = dag.new_node(dict(kind="elem", n_scalar_ops=int(nscalar * multiplier),
                                source="/".join(source_stack) or "root"))
        for v in seg_in:
            dag.add_edge(prod.get(_vkey(v)), nid, _bytes_of(v))
        for v in seg_out:
            prod[_vkey(v)] = nid
        pending.clear()

    for eqn in jaxpr.eqns:
        prim = eqn.primitive.name
        if prim in ("scan", "while", "pjit", "jit", "call", "xla_call",
                    "custom_jvp_call", "custom_vjp_call", "checkpoint"):
            flush()
            if prim == "scan":
                inner = _get_inner_jaxpr(eqn.params, "jaxpr")
                L = int(eqn.params.get("length", 1)); mm = multiplier * L; tag = f"scan[L={L}]"
            elif prim == "while":
                inner = _get_inner_jaxpr(eqn.params, "body_jaxpr"); mm = multiplier; tag = "while"
            else:
                inner = (_get_inner_jaxpr(eqn.params, "jaxpr")
                         or _get_inner_jaxpr(eqn.params, "call_jaxpr"))
                mm = multiplier; tag = str(eqn.params.get("name") or prim)
            if inner is not None:
                inner_env = {_vkey(iv): prod.get(_vkey(eqn.invars[i]))
                             for i, iv in enumerate(inner.invars) if i < len(eqn.invars)}
                out_prod = build_dag(inner, dag, inner_env, mm, source_stack + [tag])
                for i, ov in enumerate(eqn.outvars):
                    if i < len(inner.outvars):
                        prod[_vkey(ov)] = out_prod.get(_vkey(inner.outvars[i]))
            continue

        if prim == "dot_general":
            flush()
            (lc, rc), (lb, rb) = eqn.params["dimension_numbers"]
            ls, rs = eqn.invars[0].aval.shape, eqn.invars[1].aval.shape
            batch = _prod([ls[i] for i in lb])
            M = _prod([ls[i] for i in range(len(ls)) if i not in lc and i not in lb])
            N = _prod([rs[i] for i in range(len(rs)) if i not in rc and i not in rb])
            K = _prod([ls[i] for i in lc])
            lcx = _is_complex_dtype(eqn.invars[0].aval.dtype)
            rcx = _is_complex_dtype(eqn.invars[1].aval.dtype)
            dtype = ("complex_complex" if (lcx and rcx)
                     else "complex_real" if (lcx ^ rcx) else "real_real")
            nid = dag.new_node(dict(kind="gemm", M=int(M), N=int(N), K=int(K),
                                    batch=int(batch), dtype=dtype, repeat=int(multiplier),
                                    source="/".join(source_stack) or "root"))
            for v in eqn.invars:
                if hasattr(v, "aval"):
                    dag.add_edge(prod.get(_vkey(v)), nid, _bytes_of(v))
            for v in eqn.outvars:
                prod[_vkey(v)] = nid
        else:
            pending.append(eqn)
    flush()
    return {_vkey(v): prod.get(_vkey(v)) for v in jaxpr.outvars}


def emit_dag_yaml(dag: DagBuilder, path, config):
    payload = dict(format="op-dag-v1", config=config,
                   node_count=len(dag.nodes),
                   gemm_count=sum(1 for n in dag.nodes if n["kind"] == "gemm"),
                   edge_count=len(dag.edges),
                   nodes=dag.nodes,
                   edges=[dict(src=s, dst=d, bytes=int(b))
                          for (s, d), b in sorted(dag.edges.items())])
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)


# ============================================================
# Model loading (parallel to flop_counter_jaxpr.py's main; inlined so
# both scripts stay stable independently of each other)
# ============================================================
def build_forward(args):
    """Load model + checkpoint the SAME way flop_counter_jaxpr.py does,
    so the JAXPR traced here reflects exactly what the training code executed.

    Returns (forward_fn, params, dummy_input, integ) suitable for
    jax.make_jaxpr(forward_fn)(params, dummy_input, integ).
    """
    from s5.utils.util import str2bool  # noqa: F401 -- matches flop_counter_jaxpr
    from s5.train_helpers import create_train_state, load_checkpoint_msgpack
    from s5.dataloading import Datasets
    from s5.seq_model import BatchClassificationModel
    from s5.ssm import init_S5SSM
    from s5.ssm_init import make_DPLR_HiPPO
    from s5.mambino_ssm import init_MambinoSSM

    padded = args.dataset in ("imdb-classification",
                              "listops-classification",
                              "aan-classification")
    # Dims override: skip the (heavy) dataset loader when seq_len/in_dim/n_classes are
    # given.  Only needed for model-shape construction, so hardcoding known dims yields
    # a byte-identical JAXPR while avoiding a broken `datasets`/`packaging` import.
    if args.seq_len > 0 and args.in_dim > 0 and args.n_classes > 0:
        seq_len, in_dim, n_classes = args.seq_len, args.in_dim, args.n_classes
        print(f"[extract] dims OVERRIDE (no dataset load): L={seq_len} in_dim={in_dim} "
              f"n_classes={n_classes}")
    else:
        create_dataset_fn = Datasets[args.dataset]
        trainloader, valloader, testloader, aux, n_classes, seq_len, in_dim, tsize = \
            create_dataset_fn(args.dir_name, seed=args.jax_seed, bsz=args.bsz)
        print(f"[extract] dataset={args.dataset} L={seq_len} in_dim={in_dim} "
              f"n_classes={n_classes}")

    ssm_size = args.ssm_size_base
    block_size = int(ssm_size / args.blocks)
    Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)
    if args.conj_sym:
        block_size = block_size // 2
        ssm_size = ssm_size // 2
    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T
    Lambda = (Lambda * jnp.ones((args.blocks, block_size))).ravel()
    V = block_diag(*([V] * args.blocks))
    Vinv = block_diag(*([Vc] * args.blocks))

    if args.use_mambino_ssm:
        ssm_init_fn = init_MambinoSSM(
            H=args.d_model, P=ssm_size,
            Lambda_re_init=Lambda.real, Lambda_im_init=Lambda.imag,
            V=V, Vinv=Vinv, C_init=args.C_init,
            discretization=args.discretization,
            dt_min=args.dt_min, dt_max=args.dt_max,
            conj_sym=args.conj_sym, clip_eigs=args.clip_eigs,
            bidirectional=args.bidirectional,
            bidir_predictor=args.bidir_predictor,
        )
    else:
        ssm_init_fn = init_S5SSM(
            H=args.d_model, P=ssm_size,
            Lambda_re_init=Lambda.real, Lambda_im_init=Lambda.imag,
            V=V, Vinv=Vinv, C_init=args.C_init,
            discretization=args.discretization,
            dt_min=args.dt_min, dt_max=args.dt_max,
            conj_sym=args.conj_sym, clip_eigs=args.clip_eigs,
            bidirectional=args.bidirectional,
        )

    model_cls = partial(
        BatchClassificationModel,
        ssm=ssm_init_fn, d_output=n_classes, d_model=args.d_model,
        n_layers=args.n_layers, padded=padded,
        activation=args.activation_fn, dropout=args.p_dropout,
        mode=args.mode, prenorm=args.prenorm, batchnorm=args.batchnorm,
        bn_momentum=args.bn_momentum, glu_rank=args.glu_rank,
        glu_structure=args.glu_structure,
        glu_monarch_heads=args.glu_monarch_heads,
        glu_monarch_b=args.glu_monarch_b,
        glu_monarch_residual_rank=args.glu_monarch_residual_rank,
        glu_blockdiag_blocks=args.glu_blockdiag_blocks,
    )

    key = random.PRNGKey(args.jax_seed)
    init_rng, _ = random.split(key)
    state = create_train_state(
        model_cls, init_rng, padded, retrieval=False,
        in_dim=in_dim, bsz=args.bsz, seq_len=seq_len,
        weight_decay=args.weight_decay, batchnorm=args.batchnorm,
        opt_config=args.opt_config,
        ssm_lr=args.ssm_lr_base, lr=args.ssm_lr_base * args.lr_factor,
        dt_global=args.dt_global,
    )
    try:
        state, meta = load_checkpoint_msgpack(args.ckpt_prefix, state)
        n_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
        print(f"[extract] loaded {args.ckpt_prefix}: {n_params} params")
    except (ValueError, FileNotFoundError, OSError) as e:
        n_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
        print(f"[extract] ckpt load failed ({type(e).__name__}: {e}); "
              f"using random init with {n_params} params — JAXPR shapes are "
              f"still correct for workload extraction")

    if padded:
        # Padded datasets (LRA-ListOps, IMDB, AAN) expect (data, seq_length) tuple
        dummy_input = (jnp.ones((args.bsz, seq_len, in_dim)),
                       jnp.ones(args.bsz))
    else:
        dummy_input = jnp.ones((args.bsz, seq_len, in_dim))
    integ = jnp.ones((args.bsz, seq_len,))

    model = model_cls(training=False)

    def forward(params, x, integ):
        return model.apply(
            {"params": params,
             "batch_stats": state.batch_stats},
            x, integ,
            rngs={"params": jax.random.PRNGKey(0),
                  "dropout": jax.random.PRNGKey(1),
                  "noise": jax.random.PRNGKey(2)})

    return forward, state.params, dummy_input, integ


def resolve_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ============================================================
# Main
# ============================================================
def main():
    from s5.utils.util import str2bool
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_prefix", required=True,
                   help="Path prefix to checkpoint (e.g. checkpoints/corner1_.../best)")
    p.add_argument("--config_name", required=True,
                   help="Short tag for output files (e.g. corner1, corner3p, config4)")
    p.add_argument("--out_dir", default="./paper_v2_ppac/workloads/",
                   help="Where to emit workload_<cfg>_*.{csv,yaml}")
    p.add_argument("--emit_dag", action="store_true",
                   help="Also emit workload_<cfg>_dag.yaml (op-DAG: producer->consumer "
                        "edges at GEMM + elementwise-segment granularity, for the v5 "
                        "pipeline scheduler). Predictor vs main-scan GEMMs distinguished "
                        "by source_op path.")
    p.add_argument("--dir_name", type=str, default="./cache_dir")
    # Model args (mirror flop_counter_jaxpr.py / run_train.py)
    p.add_argument("--use_mambino_ssm", type=str2bool, default=False)
    p.add_argument("--ssm_size_base", type=int, required=True)
    p.add_argument("--activation_fn", default="gelu")
    p.add_argument("--glu_rank", type=int, default=0)
    p.add_argument("--glu_structure", type=str, default="dense")
    p.add_argument("--glu_monarch_heads", type=int, default=3)
    p.add_argument("--glu_monarch_b", type=int, default=0)
    p.add_argument("--glu_monarch_residual_rank", type=int, default=0)
    p.add_argument("--glu_blockdiag_blocks", type=int, default=2)
    # >0 for all three skips the dataset loader (avoids datasets/packaging import).
    p.add_argument("--seq_len", type=int, default=0)
    p.add_argument("--in_dim", type=int, default=0)
    p.add_argument("--n_classes", type=int, default=0)
    p.add_argument("--bidirectional", type=str2bool, default=True)
    p.add_argument("--batchnorm", type=str2bool, default=True)
    p.add_argument("--bn_momentum", type=float, default=0.95)
    p.add_argument("--prenorm", type=str2bool, default=True)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--dataset", default="listops-classification")
    p.add_argument("--C_init", type=str, default="lecun_normal")
    p.add_argument("--discretization", type=str, default="zoh")
    p.add_argument("--mode", type=str, default="pool")
    p.add_argument("--conj_sym", type=str2bool, default=True)
    p.add_argument("--clip_eigs", type=str2bool, default=False)
    p.add_argument("--dt_min", type=float, default=0.001)
    p.add_argument("--dt_max", type=float, default=0.1)
    p.add_argument("--bidir_predictor", type=str2bool, default=False)
    p.add_argument("--bsz", type=int, default=1)
    p.add_argument("--p_dropout", type=float, default=0.0)
    p.add_argument("--jax_seed", type=int, default=0)
    p.add_argument("--ssm_lr_base", type=float, default=1e-3)
    p.add_argument("--lr_factor", type=float, default=3.0)
    p.add_argument("--weight_decay", type=float, default=0.04)
    p.add_argument("--opt_config", type=str, default="BfastandCdecay")
    p.add_argument("--dt_global", type=str2bool, default=False)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[extract] loading model + ckpt {args.ckpt_prefix}", flush=True)
    forward_fn, params, dummy_input, integ = build_forward(args)

    _input_desc = (f"({dummy_input[0].shape} + seqlen)" if isinstance(dummy_input, tuple)
                   else f"{dummy_input.shape} {dummy_input.dtype}")
    print(f"[extract] tracing JAXPR of forward pass (input {_input_desc}) ...", flush=True)
    closed_jaxpr = jax.make_jaxpr(forward_fn)(params, dummy_input, integ)
    jaxpr = closed_jaxpr.jaxpr
    print(f"[extract] jaxpr top-level eqns: {len(jaxpr.eqns)}", flush=True)

    # 1) Independent walker totals (parity with §4 of the master doc)
    print(f"[extract] running independent walker for totals...", flush=True)
    walker_totals = walk_jaxpr(jaxpr, multiplier=1, depth=0)

    # 2) Emit workload records
    print(f"[extract] emitting per-op workload records...", flush=True)
    emitter = WorkloadEmitter()
    emit_from_jaxpr(jaxpr, emitter, multiplier=1, depth=0)

    # 3) Hard invariant check
    walker_dg = walker_totals.get("dot_general", 0)
    emitter_dg = emitter.total_dot_general_flops()

    # Elementwise totals: sum every counted primitive except dot_general and
    # meta counters (_scan_iterations, _while_bodies, structural ops with 0).
    walker_ew_total = sum(
        v for k, v in walker_totals.items()
        if k != "dot_general" and not k.startswith("_") and v > 0
    )
    emitter_ew = emitter.total_elementwise_flops()

    invariant_ok = (walker_dg == emitter_dg) and (walker_ew_total == emitter_ew)
    invariant_msg = (
        f"dot_general: walker={walker_dg:,} emitter={emitter_dg:,} "
        f"({'OK' if walker_dg == emitter_dg else 'MISMATCH'}); "
        f"elementwise: walker={walker_ew_total:,} emitter={emitter_ew:,} "
        f"({'OK' if walker_ew_total == emitter_ew else 'MISMATCH'})"
    )

    print(f"[extract] {invariant_msg}", flush=True)

    # 4) Emit output files
    stem = os.path.join(args.out_dir, f"workload_{args.config_name}")
    emitter.emit_scalesim_csv(f"{stem}_gemms.csv")
    emitter.emit_timeloop_yaml(f"{stem}_gemms.yaml")
    emitter.emit_elemwise_yaml(f"{stem}_elemwise.yaml")
    if args.emit_dag:
        print(f"[extract] building op-DAG...", flush=True)
        dag = DagBuilder()
        build_dag(jaxpr, dag, env={}, multiplier=1, source_stack=[])
        emit_dag_yaml(dag, f"{stem}_dag.yaml", args.config_name)
        # Cross-check: one DAG gemm node per emitter GEMM record (same dot_general set).
        dag_gemms = sum(1 for n in dag.nodes if n["kind"] == "gemm")
        dag_ok = (dag_gemms == len(emitter.gemms))
        print(f"[extract] op-DAG: {len(dag.nodes)} nodes ({dag_gemms} gemm), "
              f"{len(dag.edges)} edges; gemm-count vs emitter "
              f"({dag_gemms} vs {len(emitter.gemms)}): {'OK' if dag_ok else 'MISMATCH'}",
              flush=True)
    emitter.emit_manifest(f"{stem}_manifest.yaml",
                          walker_totals=walker_totals,
                          checkpoint_path=args.ckpt_prefix,
                          args=vars(args),
                          git_sha=resolve_git_sha(),
                          invariant_ok=invariant_ok,
                          invariant_msg=invariant_msg)

    print(f"[extract] wrote to {stem}_*.{{csv,yaml}}", flush=True)

    if not invariant_ok:
        print(f"[extract] HARD INVARIANT FAILED — extractor drops ops "
              f"vs the walker.  DO NOT USE THIS WORKLOAD.",
              file=sys.stderr, flush=True)
        sys.exit(2)

    print(f"[extract] HARD INVARIANT PASSED — workload matches walker exactly.",
          flush=True)


if __name__ == "__main__":
    main()
