"""First-principles FLOP counter that walks the ACTUAL jaxpr
of a trained model checkpoint -- not my formulas.

Uses:
  jax.make_jaxpr(fn)(...)              -- gets JAX's intermediate representation
                                          with real shapes traced through the code
  flax.linen.tabulate                   -- per-Flax-module summary (params, output
                                          shapes) so we can attribute totals to
                                          Config's own module hierarchy

FLOP conventions per primitive (Sze 2020 / textbook):
  - dot_general (real):    2 * batch * out_size * contract_size
  - dot_general (complex): 8 * batch * out_size * contract_size
  - add/sub/mul/div:       1 (real) or 2/6 (complex) per output element
  - transcendentals:       ~4 per output element (exp/log/tanh/sin/cos)
  - scan:                  inner_jaxpr FLOPs * length (# iterations)
  - while:                 body_jaxpr FLOPs (iterations unknown, warning)
  - pjit/jit/call:         recurse into inner jaxpr
  - Everything else (reshape/broadcast/transpose/gather/slice): 0

Usage:
  python flop_counter_jaxpr.py --ckpt_prefix checkpoints/... --use_mambino_ssm True \\
      --ssm_size_base 16 --activation_fn gelu
"""
import argparse
from collections import defaultdict
from functools import partial

import numpy as np
import jax
from jax import random
import jax.numpy as jnp
from jax.scipy.linalg import block_diag

from s5.utils.util import str2bool
from s5.train_helpers import create_train_state, load_checkpoint_msgpack
from s5.dataloading import Datasets
from s5.seq_model import BatchClassificationModel
from s5.ssm import init_S5SSM
from s5.ssm_init import make_DPLR_HiPPO
from s5.mambino_ssm import init_MambinoSSM


# ============================================================
# JAXPR walker -- counts FLOPs per primitive by recursing into
# scan / while / pjit / call subgraphs.
# ============================================================
def _prod(xs):
    r = 1
    for x in xs:
        r *= int(x)
    return r


def _is_complex_dtype(d):
    try:
        return jnp.issubdtype(d, jnp.complexfloating)
    except Exception:
        return False


def _dot_general_flops(params, invars):
    """FLOPs for dot_general primitive.

    params['dimension_numbers'] = ((lhs_contract, rhs_contract),
                                    (lhs_batch, rhs_batch))

    Per-MAC FLOP cost by dtype combination:
      - both real       -> 2 flops per MAC (1 mul + 1 add)
      - complex + real  -> 4 flops per MAC (2 real muls + 2 real adds:
                            W = A + Bi times real x = A*x + (B*x)i;
                            each output is 2 real muls; accumulate = 2 real adds)
      - both complex    -> 8 flops per MAC (4 real muls + 4 real adds:
                            (a+bi)(c+di) fused mul-add = 4 mul + 4 add)
    """
    dim_nums = params["dimension_numbers"]
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dim_nums

    lhs_aval = invars[0].aval
    rhs_aval = invars[1].aval
    lhs_shape = lhs_aval.shape
    rhs_shape = rhs_aval.shape

    lhs_complex = _is_complex_dtype(lhs_aval.dtype)
    rhs_complex = _is_complex_dtype(rhs_aval.dtype)

    # Batch size (product of batch dims -- shared between LHS and RHS)
    batch_dims_lhs = [lhs_shape[i] for i in lhs_batch]
    batch_size = _prod(batch_dims_lhs)

    # Output size: non-batch, non-contracting dims from both LHS and RHS
    lhs_out = [lhs_shape[i] for i in range(len(lhs_shape))
               if i not in lhs_contract and i not in lhs_batch]
    rhs_out = [rhs_shape[i] for i in range(len(rhs_shape))
               if i not in rhs_contract and i not in rhs_batch]
    out_size = _prod(lhs_out) * _prod(rhs_out)

    # Contracting dim
    contract_size = _prod([lhs_shape[i] for i in lhs_contract])

    # 3-way dtype cost
    if lhs_complex and rhs_complex:
        per_mac = 8      # complex-complex
    elif lhs_complex ^ rhs_complex:
        per_mac = 4      # complex-real (exactly one operand complex)
    else:
        per_mac = 2      # real-real

    return int(batch_size * out_size * per_mac * contract_size)


def _elementwise_flops(eqn, cost_real: int, cost_complex: int,
                       cost_complex_real: int = None):
    """FLOPs for element-wise primitive.

    Three-way cost based on input dtypes:
      - both real       -> cost_real per output element
      - both complex    -> cost_complex per output element
      - complex + real  -> cost_complex_real per output element (defaults to
                            average if not specified)
    """
    if not eqn.outvars:
        return 0
    out_aval = eqn.outvars[0].aval
    n = _prod(out_aval.shape)
    out_complex = _is_complex_dtype(out_aval.dtype)

    # Check input dtypes
    in_complex = [_is_complex_dtype(v.aval.dtype)
                  if hasattr(v, "aval") else False for v in eqn.invars]
    all_complex = all(in_complex) if in_complex else False
    any_complex = any(in_complex) if in_complex else False
    mixed_complex_real = any_complex and not all_complex

    if mixed_complex_real and cost_complex_real is not None:
        return int(n * cost_complex_real)
    if all_complex or (any_complex and out_complex):
        return int(n * cost_complex)
    return int(n * cost_real)


def _get_inner_jaxpr(params, key: str):
    """Extract inner jaxpr from a higher-order primitive's params."""
    inner = params.get(key)
    if inner is None:
        return None
    # inner might be ClosedJaxpr or Jaxpr
    if hasattr(inner, "jaxpr"):
        return inner.jaxpr
    return inner


def walk_jaxpr(jaxpr, multiplier=1, depth=0):
    """Walk a jaxpr recursively, return dict {primitive: total_flops}."""
    counts = defaultdict(int)

    for eqn in jaxpr.eqns:
        prim_name = eqn.primitive.name

        # ---- Higher-order primitives: recurse ----
        if prim_name == "scan":
            inner = _get_inner_jaxpr(eqn.params, "jaxpr")
            length = int(eqn.params.get("length", 1))
            if inner is not None:
                inner_counts = walk_jaxpr(inner, multiplier * length, depth + 1)
                for k, v in inner_counts.items():
                    counts[k] += v
                counts["_scan_iterations"] += length
            continue

        if prim_name == "while":
            body = _get_inner_jaxpr(eqn.params, "body_jaxpr")
            cond = _get_inner_jaxpr(eqn.params, "cond_jaxpr")
            # Iterations unknown; conservatively skip.  Log to see what while loops exist.
            counts["_while_bodies"] += 1
            if body is not None:
                inner_counts = walk_jaxpr(body, multiplier, depth + 1)
                for k, v in inner_counts.items():
                    counts[f"while_body_{k}"] += v
            continue

        if prim_name in ("pjit", "jit", "call", "xla_call", "custom_jvp_call",
                         "custom_vjp_call", "checkpoint"):
            inner = _get_inner_jaxpr(eqn.params, "jaxpr") or \
                    _get_inner_jaxpr(eqn.params, "call_jaxpr")
            if inner is not None:
                inner_counts = walk_jaxpr(inner, multiplier, depth + 1)
                for k, v in inner_counts.items():
                    counts[k] += v
            continue

        # ---- Leaf primitives ----
        if prim_name == "dot_general":
            f = _dot_general_flops(eqn.params, eqn.invars)
        elif prim_name in ("add", "sub"):
            # complex + real = 1 real add (imag part unchanged, no work)
            # complex + complex = 2 real adds
            f = _elementwise_flops(eqn, cost_real=1, cost_complex=2,
                                   cost_complex_real=1)
        elif prim_name in ("mul",):
            # complex * complex = 4 muls + 2 adds = 6 flops
            # complex * real    = 2 real muls (a+bi)*x = ax + (bx)i = 2 muls
            # real * real       = 1 mul
            f = _elementwise_flops(eqn, cost_real=1, cost_complex=6,
                                   cost_complex_real=2)
        elif prim_name in ("div",):
            # complex / complex = ~10 flops (numerator*conj(denom) / |denom|^2)
            # complex / real    = 2 real divs
            f = _elementwise_flops(eqn, cost_real=1, cost_complex=10,
                                   cost_complex_real=2)
        elif prim_name in ("exp", "log", "log1p", "expm1", "sqrt",
                           "rsqrt", "sin", "cos", "tan", "tanh",
                           "sinh", "cosh", "asin", "acos", "atan",
                           "erf", "logistic"):
            f = _elementwise_flops(eqn, cost_real=4, cost_complex=8)
        elif prim_name in ("integer_pow", "pow"):
            f = _elementwise_flops(eqn, cost_real=3, cost_complex=8)
        elif prim_name in ("neg", "abs"):
            f = _elementwise_flops(eqn, cost_real=1, cost_complex=2)
        elif prim_name in ("real", "imag", "complex", "conj"):
            f = _elementwise_flops(eqn, cost_real=0, cost_complex=0)  # essentially free
        elif prim_name in ("reduce_sum", "reduce_max", "reduce_min",
                           "reduce_prod", "reduce_and", "reduce_or"):
            # For sum-reduce: about (input_size - output_size) adds
            if eqn.outvars and eqn.invars:
                in_sz = _prod(eqn.invars[0].aval.shape)
                out_sz = _prod(eqn.outvars[0].aval.shape) if eqn.outvars[0].aval.shape else 1
                f = max(0, in_sz - out_sz)
            else:
                f = 0
        elif prim_name in ("broadcast_in_dim", "reshape", "transpose",
                           "squeeze", "concatenate", "slice",
                           "dynamic_slice", "dynamic_update_slice", "gather",
                           "scatter", "iota", "convert_element_type",
                           "bitcast_convert_type", "select_n", "cond",
                           "pad", "reverse", "sort", "argmax", "argmin",
                           "stop_gradient", "device_put"):
            f = 0
        else:
            # Unknown -- log for later inspection
            counts[f"_unknown_{prim_name}"] += 1
            f = 0

        counts[prim_name] += f * multiplier

    return counts


# ============================================================
# Runner
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_prefix", type=str, required=True)
    p.add_argument("--dataset", type=str, default="listops-classification")
    p.add_argument("--dir_name", type=str, default="./cache_dir")
    p.add_argument("--use_mambino_ssm", type=str2bool, default=False)
    p.add_argument("--activation_fn", type=str, default="gelu")
    p.add_argument("--ssm_size_base", type=int, required=True)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--glu_rank", type=int, default=0)
    p.add_argument("--C_init", type=str, default="lecun_normal")
    p.add_argument("--discretization", type=str, default="zoh")
    p.add_argument("--mode", type=str, default="pool")
    p.add_argument("--conj_sym", type=str2bool, default=True)
    p.add_argument("--clip_eigs", type=str2bool, default=False)
    p.add_argument("--bidirectional", type=str2bool, default=True)
    p.add_argument("--dt_min", type=float, default=0.001)
    p.add_argument("--dt_max", type=float, default=0.1)
    p.add_argument("--bidir_predictor", type=str2bool, default=False)
    p.add_argument("--prenorm", type=str2bool, default=True)
    p.add_argument("--batchnorm", type=str2bool, default=True)
    p.add_argument("--bn_momentum", type=float, default=0.95)
    p.add_argument("--bsz", type=int, default=1)
    p.add_argument("--p_dropout", type=float, default=0.0)
    p.add_argument("--jax_seed", type=int, default=0)
    p.add_argument("--ssm_lr_base", type=float, default=1e-3)
    p.add_argument("--lr_factor", type=float, default=3.0)
    p.add_argument("--weight_decay", type=float, default=0.04)
    p.add_argument("--opt_config", type=str, default="BfastandCdecay")
    p.add_argument("--dt_global", type=str2bool, default=False)
    p.add_argument("--config_name", type=str, default="model")
    args = p.parse_args()

    padded = args.dataset in ("imdb-classification",
                              "listops-classification",
                              "aan-classification")
    create_dataset_fn = Datasets[args.dataset]
    trainloader, valloader, testloader, aux, n_classes, seq_len, in_dim, tsize \
        = create_dataset_fn(args.dir_name, seed=args.jax_seed, bsz=args.bsz)

    print(f"[flop-jaxpr] dataset={args.dataset} L={seq_len} in_dim={in_dim} "
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
        print(f"[flop-jaxpr] loaded {args.ckpt_prefix}: {n_params} params")
    except (ValueError, FileNotFoundError, OSError) as e:
        n_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
        print(f"[flop-jaxpr] ckpt load failed ({type(e).__name__}: {e}); using random init with {n_params} params")

    if padded:
        dummy_input = (jnp.ones((args.bsz, seq_len, in_dim)),
                       jnp.ones(args.bsz))
        integ = jnp.ones((args.bsz, seq_len,))
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

    # ---- Flax module-level summary via nn.tabulate ----
    print("\n" + "=" * 78)
    print("Flax module hierarchy (via nn.tabulate)")
    print("=" * 78)
    try:
        from flax.linen import tabulate as flax_tabulate
        table = flax_tabulate(model, jax.random.PRNGKey(0),
                              console_kwargs={"width": 130})(
            dummy_input, integ,
            rngs={"params": jax.random.PRNGKey(0),
                  "dropout": jax.random.PRNGKey(1),
                  "noise": jax.random.PRNGKey(2)},
        )
        print(table)
    except Exception as e:
        print(f"[flop-jaxpr] nn.tabulate not available or failed: {e}")

    # ---- Trace forward pass to jaxpr and walk primitives ----
    print("\n" + "=" * 78)
    print("Tracing forward pass with jax.make_jaxpr ...")
    print("=" * 78)
    closed_jaxpr = jax.make_jaxpr(forward)(state.params, dummy_input, integ)
    jaxpr = closed_jaxpr.jaxpr
    print(f"[flop-jaxpr] jaxpr has {len(jaxpr.eqns)} top-level eqns")

    print("\nWalking primitives ...")
    counts = walk_jaxpr(jaxpr, multiplier=1)

    # ---- Report ----
    print("\n" + "=" * 78)
    print(f"FLOP counts per primitive for {args.config_name}")
    print("=" * 78)
    total = 0
    diagnostics = 0
    diag_items = {}
    for k in sorted(counts.keys(), key=lambda k: -counts[k]):
        v = counts[k]
        if k.startswith("_"):
            diag_items[k] = v
            continue
        pct = 100.0 * v / max(1, sum(x for kk, x in counts.items() if not kk.startswith("_")))
        print(f"  {k:<40s} {v:>18,d}  ({pct:5.1f}%)")
        total += v

    print("-" * 78)
    print(f"  {'TOTAL FLOPS':<40s} {total:>18,d}")
    print(f"  {'  GFLOPs':<40s} {total/1e9:>18.3f}")

    if diag_items:
        print("\nDiagnostics (not counted in total):")
        for k, v in diag_items.items():
            print(f"  {k}: {v}")

    # Save YAML for Accelergy input
    with open(f"workload_{args.config_name}_jaxpr.yaml", "w") as f:
        f.write(f"# Auto-generated by flop_counter_jaxpr.py\n")
        f.write(f"# Model: {args.config_name}\n")
        f.write(f"# Checkpoint: {args.ckpt_prefix}\n")
        f.write(f"# Params: {n_params}\n")
        f.write(f"# Total FLOPs: {total} ({total/1e9:.3f} G)\n")
        f.write(f"# batch=1, L={seq_len}, H={args.d_model}, "
                f"n_layers={args.n_layers}\n\n")
        f.write("flops_by_primitive:\n")
        for k in sorted(counts.keys(), key=lambda k: -counts[k]):
            v = counts[k]
            if k.startswith("_"):
                continue
            f.write(f"  {k}: {v}\n")
        f.write(f"\ntotal_flops: {total}\n")
        f.write(f"total_params: {n_params}\n")
    print(f"\nWrote workload_{args.config_name}_jaxpr.yaml")


if __name__ == "__main__":
    main()
