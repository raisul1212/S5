"""First-principles FLOP counter for our SSM configs.

Computes per-op FLOPs from tensor shapes using textbook formulas.
No XLA cost_analysis (which is unreliable inside scan loops).
No hand-calculation.  Just walk the ops in order.

FLOP conventions (Sze 2020, standard DL literature):
  - MAC (multiply-accumulate) = 2 FLOPs (one multiply + one add)
  - Real matmul (M,N) @ (N,K) = 2 * M * N * K FLOPs
  - Complex multiplication (a+bi)(c+di):
      - Full: 4 real mults + 2 real adds = 6 FLOPs
      - Fused mul-add (as in accumulate): 4 mults + 4 adds = 8 FLOPs
  - Complex matmul (M,N)_c @ (N,)_c per column: 8*M*N real FLOPs
  - Associative scan (L steps, P state dims, complex Λ, complex h, complex Bu):
      Each combine: λ_a·λ_b (6) + λ_b·x_a (6) + x_b (2) = 14 FLOPs per element
      Total combines: 2*L (Blelloch tree)
      Per direction: 2 * L * P * 14 = 28*L*P FLOPs

Note on the `.real` optimization: the code does `(C_tilde @ h).real` which
formally only needs the real part.  XLA may not optimize this away, so we
report the full complex matmul cost as the DEFENSIBLE upper bound.
If XLA does optimize, real cost is ~half.

Usage:
  python flop_counter.py                              # default configs
  python flop_counter.py --config config5             # single config
  python flop_counter.py --csv flop_results.csv       # dump CSV
"""
import argparse
import csv
from dataclasses import dataclass, field
from typing import List, Tuple


# ============================================================
# Config specs
# ============================================================
@dataclass
class Config:
    name: str
    arch: str                    # "pure_s5" or "mambino"
    ssm_size_base: int           # raw P before conj_sym
    d_model: int = 128           # H
    n_layers: int = 8            # L (depth)
    seq_len: int = 2048          # L_seq (sequence length)
    blocks: int = 8              # HiPPO diagonal blocks
    conj_sym: bool = True
    bidirectional: bool = True   # main SSM
    bidir_predictor: bool = False # Mambino predictor
    activation: str = "gelu"     # "gelu" or "half_glu2" etc.
    glu_rank: int = 0            # 0 = no rank compression / no gate
    vocab_size: int = 20         # ListOps
    n_classes: int = 10

    @property
    def self_P(self) -> int:
        """SSM state dim after conj_sym halving."""
        return self.ssm_size_base // 2 if self.conj_sym else self.ssm_size_base

    @property
    def local_P(self) -> int:
        """Sampling dim used for B,C matrices (2*self.P under conj_sym)."""
        return 2 * self.self_P if self.conj_sym else self.self_P

    @property
    def C_read_dim(self) -> int:
        """C_tilde's second dim -- 2*local_P if bidirectional (concat), else local_P."""
        return 2 * self.local_P if self.bidirectional else self.local_P

    @property
    def has_gate(self) -> bool:
        return self.activation in ("half_glu2", "half_glu1", "full_glu")


# ============================================================
# FLOP formulas
# ============================================================
def flop_real_matmul(M: int, N: int, batch: int = 1) -> int:
    """Real matmul (M,N) @ (N,) per batch element, `batch` batch elements.
    Returns FLOPs = 2 * batch * M * N."""
    return 2 * batch * M * N


def flop_complex_matmul(M: int, N: int, batch: int = 1) -> int:
    """Complex-complex matmul: (M,N)_c @ (N,)_c per batch, `batch` elements.
    Per output: N complex mul-adds * 8 real flops = 8*N
    Total: batch * M * 8 * N = 8 * batch * M * N FLOPs."""
    return 8 * batch * M * N


def flop_complex_real_matmul(M: int, N: int, batch: int = 1) -> int:
    """Complex-real matmul: complex weight, real input.
    Per output: 2*N real mults (real and imag parts of complex weight)
    plus 2*(N-1) adds ≈ 4*N flops.
    Total: 4 * batch * M * N FLOPs."""
    return 4 * batch * M * N


def flop_associative_scan(L: int, P: int) -> int:
    """Associative scan over L steps, P state dims (complex).
    Blelloch tree work: ~2*L combines per element.
    Per combine: 14 real FLOPs (2 complex muls + 1 complex add).
    Total: 2 * L * P * 14 = 28 * L * P FLOPs."""
    return 28 * L * P


def flop_pointwise(numel: int, ops_per_element: int = 1) -> int:
    """Pointwise op on numel elements, ops_per_element real FLOPs each."""
    return numel * ops_per_element


# ============================================================
# Per-layer walkers
# ============================================================
@dataclass
class Op:
    name: str
    op_type: str            # "matmul_real", "matmul_complex", "matmul_creal", "scan", "pointwise", "add", "memory"
    shape_str: str
    flops: int


def walk_pure_s5_ssm(cfg: Config) -> List[Op]:
    """Walk one Pure S5 SSM layer's operations."""
    H = cfg.d_model
    L = cfg.seq_len
    P = cfg.local_P
    ops = []

    # B_bar @ x_seq: (local_P, H) real @ (L, H) -> (L, local_P)
    # This is a complex-real matmul (B_bar is complex after discretization)
    ops.append(Op(
        name="B_bar @ x",
        op_type="matmul_creal",
        shape_str=f"({P}, {H}) complex-real @ ({L}, {H}) real",
        flops=flop_complex_real_matmul(P, H, batch=L),
    ))

    # Scan forward: L steps, P state dims complex
    ops.append(Op(
        name="associative_scan (fwd)",
        op_type="scan",
        shape_str=f"L={L} steps × P={P} complex state",
        flops=flop_associative_scan(L, P),
    ))

    if cfg.bidirectional:
        ops.append(Op(
            name="associative_scan (bwd)",
            op_type="scan",
            shape_str=f"L={L} steps × P={P} complex state (reverse)",
            flops=flop_associative_scan(L, P),
        ))
        ops.append(Op(
            name="concat [xs_fwd, xs_bwd]",
            op_type="memory",
            shape_str=f"(L, P) + (L, P) -> (L, 2P) = (L, {2*P})",
            flops=0,
        ))

    # C_tilde @ h: (H, C_read_dim) complex @ (L, C_read_dim) complex -> (L, H) complex, take .real
    ops.append(Op(
        name="C_tilde @ h (.real)",
        op_type="matmul_complex",
        shape_str=f"({H}, {cfg.C_read_dim}) complex @ ({L}, {cfg.C_read_dim}) complex",
        flops=flop_complex_matmul(H, cfg.C_read_dim, batch=L),
    ))

    # D * u (feedthrough): (H,) real * (L, H) real
    ops.append(Op(
        name="D * u (feedthrough)",
        op_type="pointwise",
        shape_str=f"({H},) * ({L}, {H})",
        flops=flop_pointwise(L * H, ops_per_element=1),  # 1 mult per element
    ))

    # ys + Du
    ops.append(Op(
        name="ys + Du (analog sum)",
        op_type="add",
        shape_str=f"({L}, {H}) + ({L}, {H})",
        flops=flop_pointwise(L * H, ops_per_element=1),
    ))

    return ops


def walk_mambino_ssm(cfg: Config) -> List[Op]:
    """Walk one MambinoSSM layer's operations: predictor + main."""
    H = cfg.d_model
    L = cfg.seq_len
    P = cfg.local_P
    ops = []

    # ---- PREDICTOR BRANCH ----
    ops.append(Op(
        name="B_s @ x (predictor)",
        op_type="matmul_creal",
        shape_str=f"({P}, {H}) complex-real @ ({L}, {H}) real",
        flops=flop_complex_real_matmul(P, H, batch=L),
    ))

    ops.append(Op(
        name="associative_scan (predictor fwd)",
        op_type="scan",
        shape_str=f"L={L} × P={P} complex",
        flops=flop_associative_scan(L, P),
    ))

    if cfg.bidir_predictor:
        ops.append(Op(
            name="associative_scan (predictor bwd)",
            op_type="scan",
            shape_str=f"L={L} × P={P} complex (reverse)",
            flops=flop_associative_scan(L, P),
        ))

    # Predictor readout: C_s @ s_read
    # If bidir predictor: readout uses (H, 2P); else (H, P)
    C_s_dim = 2 * P if cfg.bidir_predictor else P
    ops.append(Op(
        name="C_s @ s_read -> x_hat",
        op_type="matmul_complex",
        shape_str=f"({H}, {C_s_dim}) complex @ ({L}, {C_s_dim}) complex",
        flops=flop_complex_matmul(H, C_s_dim, batch=L),
    ))

    # eps = x - x_hat
    ops.append(Op(
        name="eps = x - x_hat (surprise)",
        op_type="pointwise",
        shape_str=f"({L}, {H}) - ({L}, {H})",
        flops=flop_pointwise(L * H, ops_per_element=1),
    ))

    # ---- MAIN SSM WITH ADDITIVE PC ----
    ops.append(Op(
        name="B_bar @ x (main)",
        op_type="matmul_creal",
        shape_str=f"({P}, {H}) complex-real @ ({L}, {H}) real",
        flops=flop_complex_real_matmul(P, H, batch=L),
    ))

    ops.append(Op(
        name="W_eps @ eps",
        op_type="matmul_creal",
        shape_str=f"({P}, {H}) complex-real @ ({L}, {H}) real",
        flops=flop_complex_real_matmul(P, H, batch=L),
    ))

    ops.append(Op(
        name="Bu_total = Bu_x + Bu_eps (additive PC)",
        op_type="add",
        shape_str=f"({L}, {P}) + ({L}, {P}) complex",
        flops=flop_pointwise(L * P, ops_per_element=2),  # complex add = 2 real
    ))

    # Main scan
    ops.append(Op(
        name="associative_scan (main fwd)",
        op_type="scan",
        shape_str=f"L={L} × P={P} complex",
        flops=flop_associative_scan(L, P),
    ))

    if cfg.bidirectional:
        ops.append(Op(
            name="associative_scan (main bwd)",
            op_type="scan",
            shape_str=f"L={L} × P={P} complex (reverse)",
            flops=flop_associative_scan(L, P),
        ))
        ops.append(Op(
            name="concat [xs_fwd, xs_bwd]",
            op_type="memory",
            shape_str=f"(L, P) + (L, P) -> (L, {2*P})",
            flops=0,
        ))

    # C_tilde @ h  (main readout)
    ops.append(Op(
        name="C_tilde @ h (.real)",
        op_type="matmul_complex",
        shape_str=f"({H}, {cfg.C_read_dim}) complex @ ({L}, {cfg.C_read_dim}) complex",
        flops=flop_complex_matmul(H, cfg.C_read_dim, batch=L),
    ))

    ops.append(Op(
        name="D * u (feedthrough)",
        op_type="pointwise",
        shape_str=f"({H},) * ({L}, {H})",
        flops=flop_pointwise(L * H, ops_per_element=1),
    ))

    ops.append(Op(
        name="ys + Du",
        op_type="add",
        shape_str=f"({L}, {H}) + ({L}, {H})",
        flops=flop_pointwise(L * H, ops_per_element=1),
    ))

    return ops


def walk_sequence_layer(cfg: Config) -> List[Op]:
    """Walk one full SequenceLayer (BN + SSM + activation + skip + BN)."""
    H = cfg.d_model
    L = cfg.seq_len

    ops = []

    # Prenorm BN
    ops.append(Op(
        name="BatchNorm (prenorm)",
        op_type="pointwise",
        shape_str=f"({L}, {H})",
        flops=flop_pointwise(L * H, ops_per_element=3),  # subtract mean, scale/var, shift/beta
    ))

    # SSM ops
    if cfg.arch == "mambino":
        ops.extend(walk_mambino_ssm(cfg))
    else:
        ops.extend(walk_pure_s5_ssm(cfg))

    # Activation & gate (if any)
    if cfg.has_gate:
        # half_glu2 default -- 2 GELU + out2 matmul + sigmoid + multiply
        # This paper's configs are gelu-only so we skip this branch
        if cfg.glu_rank > 0:
            r = cfg.glu_rank
            # out2_down: (H, r) real, out2_up: (r, H) real
            ops.append(Op(
                name="out2_down (low-rank gate)",
                op_type="matmul_real",
                shape_str=f"({r}, {H}) @ ({L}, {H})",
                flops=flop_real_matmul(r, H, batch=L),
            ))
            ops.append(Op(
                name="out2_up (low-rank gate)",
                op_type="matmul_real",
                shape_str=f"({H}, {r}) @ ({L}, {r})",
                flops=flop_real_matmul(H, r, batch=L),
            ))
        else:
            # Full-rank gate: (H, H) real
            ops.append(Op(
                name="out2 (full gate)",
                op_type="matmul_real",
                shape_str=f"({H}, {H}) @ ({L}, {H})",
                flops=flop_real_matmul(H, H, batch=L),
            ))
        ops.append(Op(
            name="sigmoid(gate_raw)",
            op_type="pointwise",
            shape_str=f"({L}, {H})",
            flops=flop_pointwise(L * H, ops_per_element=4),  # sigmoid ~4 flops
        ))
        ops.append(Op(
            name="signal * sigmoid",
            op_type="pointwise",
            shape_str=f"({L}, {H})",
            flops=flop_pointwise(L * H, ops_per_element=1),
        ))
    else:
        # gelu only
        ops.append(Op(
            name="gelu",
            op_type="pointwise",
            shape_str=f"({L}, {H})",
            flops=flop_pointwise(L * H, ops_per_element=8),  # gelu ~8 flops (tanh approx)
        ))

    # Skip connection
    ops.append(Op(
        name="+ skip",
        op_type="add",
        shape_str=f"({L}, {H}) + ({L}, {H})",
        flops=flop_pointwise(L * H, ops_per_element=1),
    ))

    # Postnorm BN
    ops.append(Op(
        name="BatchNorm (postnorm)",
        op_type="pointwise",
        shape_str=f"({L}, {H})",
        flops=flop_pointwise(L * H, ops_per_element=3),
    ))

    return ops


def walk_full_model(cfg: Config) -> Tuple[List[Tuple[str, List[Op]]], List[Op]]:
    """Walk the full model.  Returns (per_layer_ops, model_level_ops)."""
    H = cfg.d_model
    L = cfg.seq_len
    N = cfg.n_layers

    per_layer = []
    for i in range(N):
        layer_ops = walk_sequence_layer(cfg)
        per_layer.append((f"Layer {i}", layer_ops))

    # Encoder Dense: (H, vocab) @ (L, vocab_one_hot) -- but really embedding lookup
    # For a fair FLOP count: embedding = memory lookup, ~0 flops (just gather)
    # If treated as one-hot matmul: 2*L*H*vocab flops
    model_ops = []
    model_ops.append(Op(
        name="Encoder embedding lookup",
        op_type="memory",
        shape_str=f"({cfg.vocab_size}, {H}) table, {L} lookups",
        flops=0,  # gather, no arithmetic
    ))

    # Mean pool: sum across L, divide by L
    model_ops.append(Op(
        name="Mean pool across L",
        op_type="pointwise",
        shape_str=f"sum({L}, {H}) / {L}",
        flops=(L - 1) * H + H,  # L-1 adds per output + H divides
    ))

    # Decoder Dense: (n_classes, H) @ (H,)
    model_ops.append(Op(
        name="Decoder Dense",
        op_type="matmul_real",
        shape_str=f"({cfg.n_classes}, {H}) @ ({H},)",
        flops=flop_real_matmul(cfg.n_classes, H, batch=1),
    ))

    # Softmax: exp + sum + divide over n_classes
    model_ops.append(Op(
        name="log_softmax",
        op_type="pointwise",
        shape_str=f"({cfg.n_classes},)",
        flops=cfg.n_classes * 3,  # exp + sum + divide
    ))

    return per_layer, model_ops


# ============================================================
# Report
# ============================================================
def summarize(ops: List[Op]) -> dict:
    """Sum FLOPs by op_type."""
    totals = {}
    for op in ops:
        totals[op.op_type] = totals.get(op.op_type, 0) + op.flops
    totals["_total"] = sum(op.flops for op in ops)
    return totals


def fmt_flops(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:6.2f} GFLOPs"
    if n >= 1e6:
        return f"{n/1e6:6.2f} MFLOPs"
    if n >= 1e3:
        return f"{n/1e3:6.2f} KFLOPs"
    return f"{n:6d} FLOPs"


def report(cfg: Config, csv_writer=None):
    print(f"\n{'='*74}")
    print(f"Config: {cfg.name}  |  arch={cfg.arch}  |  activation={cfg.activation}")
    print(f"ssm_size_base={cfg.ssm_size_base}  self.P={cfg.self_P}  local_P={cfg.local_P}")
    print(f"L={cfg.seq_len}  n_layers={cfg.n_layers}  H={cfg.d_model}")
    print(f"{'='*74}\n")

    per_layer, model_ops = walk_full_model(cfg)

    # Print one layer's ops (all layers identical)
    layer_name, layer_ops = per_layer[0]
    layer_total = 0
    print(f"--- Per-layer ops ({layer_name}, same for all {cfg.n_layers} layers) ---\n")
    print(f"{'Op':<40s} {'Type':<15s} {'FLOPs':>13s}")
    print("-" * 74)
    for op in layer_ops:
        print(f"{op.name:<40s} {op.op_type:<15s} {fmt_flops(op.flops):>13s}")
        layer_total += op.flops
        if csv_writer:
            csv_writer.writerow([cfg.name, layer_name, op.name, op.op_type,
                                 op.shape_str, op.flops])
    print("-" * 74)
    print(f"{'Per-layer total':<40s} {'':<15s} {fmt_flops(layer_total):>13s}\n")

    # Model-level ops (once per inference)
    print(f"--- Model-level ops (once per inference) ---\n")
    print(f"{'Op':<40s} {'Type':<15s} {'FLOPs':>13s}")
    print("-" * 74)
    model_total = 0
    for op in model_ops:
        print(f"{op.name:<40s} {op.op_type:<15s} {fmt_flops(op.flops):>13s}")
        model_total += op.flops
        if csv_writer:
            csv_writer.writerow([cfg.name, "model", op.name, op.op_type,
                                 op.shape_str, op.flops])
    print("-" * 74)
    print(f"{'Model-level total':<40s} {'':<15s} {fmt_flops(model_total):>13s}\n")

    # Grand total: layer_total * n_layers + model_total
    grand_total = layer_total * cfg.n_layers + model_total
    print("=" * 74)
    print(f"{'GRAND TOTAL':<40s} {'':<15s} {fmt_flops(grand_total):>13s}")
    print(f"  Per layer:        {fmt_flops(layer_total):>13s}")
    print(f"  x n_layers ({cfg.n_layers}):    {fmt_flops(layer_total * cfg.n_layers):>13s}")
    print(f"  + model-level:    {fmt_flops(model_total):>13s}")
    print("=" * 74)

    return {
        "name": cfg.name,
        "per_layer": layer_total,
        "model_level": model_total,
        "grand_total": grand_total,
        "layer_breakdown": summarize(layer_ops),
    }


# ============================================================
# Config catalogue (matches the trained checkpoints)
# ============================================================
CONFIGS = {
    "corner1": Config(
        name="Corner_1_PureS5_hg2",
        arch="pure_s5", ssm_size_base=16,
        activation="half_glu2", glu_rank=0),
    "corner3p": Config(
        name="Corner_3p_Mambino_hg2",
        arch="mambino", ssm_size_base=16,
        activation="half_glu2", glu_rank=40),
    "config3": Config(
        name="Config_3_PureS5_gelu_P8",
        arch="pure_s5", ssm_size_base=16, activation="gelu"),
    "config4": Config(
        name="Config_4_Mambino_gelu_P8",
        arch="mambino", ssm_size_base=16, activation="gelu"),
    "config5": Config(
        name="Config_5_PureS5_gelu_P16",
        arch="pure_s5", ssm_size_base=32, activation="gelu"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="all",
                   choices=list(CONFIGS.keys()) + ["all"])
    p.add_argument("--csv", type=str, default="")
    p.add_argument("--summary_csv", type=str, default="")
    args = p.parse_args()

    csv_writer = None
    csv_file = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "config", "scope", "op_name", "op_type", "shape", "flops"
        ])

    summaries = []
    configs_to_run = (list(CONFIGS.values())
                     if args.config == "all"
                     else [CONFIGS[args.config]])
    for cfg in configs_to_run:
        s = report(cfg, csv_writer=csv_writer)
        summaries.append(s)

    if csv_file:
        csv_file.close()
        print(f"\nWrote per-op CSV to {args.csv}")

    # Cross-config comparison
    if len(summaries) > 1:
        print("\n" + "=" * 74)
        print("Cross-config comparison")
        print("=" * 74)
        print(f"{'Config':<32s} {'Per layer':>14s} {'Total':>14s}")
        print("-" * 74)
        for s in summaries:
            print(f"{s['name']:<32s} "
                  f"{fmt_flops(s['per_layer']):>14s} "
                  f"{fmt_flops(s['grand_total']):>14s}")
        print("-" * 74)

        # iso-params gelu comparison
        c4 = next((s for s in summaries if "Config_4" in s["name"]), None)
        c5 = next((s for s in summaries if "Config_5" in s["name"]), None)
        if c4 and c5:
            print("\niso-params gelu 106K:  Config 4 Mambino vs Config 5 Pure S5")
            print(f"  Config 4 (Mambino):  {fmt_flops(c4['grand_total'])}")
            print(f"  Config 5 (Pure S5):  {fmt_flops(c5['grand_total'])}")
            delta_pct = 100.0 * (c4['grand_total'] - c5['grand_total']) / c5['grand_total']
            winner = "Mambino" if delta_pct < 0 else "Pure S5"
            print(f"  Delta: {delta_pct:+.1f}% ({winner} uses fewer FLOPs)")

    if args.summary_csv:
        with open(args.summary_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["config", "per_layer_flops", "grand_total_flops"])
            for s in summaries:
                w.writerow([s["name"], s["per_layer"], s["grand_total"]])
        print(f"Wrote summary CSV to {args.summary_csv}")


if __name__ == "__main__":
    main()
