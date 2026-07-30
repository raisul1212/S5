"""Correctness gate for Cluster B (role 2: surprise-gated fast weight).

Modelled on bin/test_v2_gate.py.  Run on a flax env (locally or on Gilbreth).

Tests
  1 BYTE-EQUIVALENCE  fast_weight=False -> no fw_* params AND logits bit-identical
                      to the pre-change code.  Cross-worktree: dump a reference
                      from the v2/surprise-gate tree, check it here.
  2 IMPL EQUIVALENCE  seq (ground truth) == scan == chunk, swept over gamma, C, d,
                      both read conventions, and L NOT divisible by C.
  3 PARAM COUNTS      measured fw_* totals == the plan's Sec 4 D3 table.
  4 GRADIENTS         every fw_* param receives a nonzero gradient after one step
                      (guards the zero-init W_o escape and stray stop_gradients).
  6 CAUSALITY         perturbing x_t moves o_s only for s > t -- and under the
                      EXCLUSIVE read, never for s == t.  This is the test that
                      catches an off-by-one silently reintroducing the
                      self-read shortcut the exclusive read exists to remove.
  7 BOUNDEDNESS       ||M|| stays bounded as gamma -> 1; finite at gamma -> 0.
  8 DETERMINISM       same seed => identical logits.
  5 CKPT COMPAT       optional, needs --v1_ckpt.

Usage
  # in the v2/surprise-gate worktree (pre-change code):
  python bin/test_cluster_b_fastweight.py --dump_ref /tmp/fw_ref.npy
  # here:
  python bin/test_cluster_b_fastweight.py --check_ref /tmp/fw_ref.npy
  python bin/test_cluster_b_fastweight.py --v1_ckpt checkpoints/<run>/best
"""
import argparse
from functools import partial

import numpy as onp
import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.linalg import block_diag

from s5.seq_model import BatchClassificationModel
from s5.ssm_init import make_DPLR_HiPPO
from s5.mambino_ssm import init_MambinoSSM, MambinoSSM

# ListOps-shaped dims, shrunk so the gate runs in seconds.
H, N_LAYERS, BLOCKS, SSM_BASE = 128, 8, 8, 16
SEQ_LEN, IN_DIM, N_CLASSES, BSZ = 64, 20, 10, 2


def build_model(**fw):
    ssm_size, block_size = SSM_BASE, SSM_BASE // BLOCKS
    Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)
    block_size //= 2
    ssm_size //= 2
    Lambda, V = Lambda[:block_size], V[:, :block_size]
    Vc = V.conj().T
    Lambda = (Lambda * jnp.ones((BLOCKS, block_size))).ravel()
    V, Vinv = block_diag(*([V] * BLOCKS)), block_diag(*([Vc] * BLOCKS))
    ssm = init_MambinoSSM(H=H, P=ssm_size, Lambda_re_init=Lambda.real,
                          Lambda_im_init=Lambda.imag, V=V, Vinv=Vinv,
                          C_init="lecun_normal", discretization="zoh",
                          dt_min=0.001, dt_max=0.1, conj_sym=True,
                          clip_eigs=False, bidirectional=True, **fw)
    return partial(BatchClassificationModel, ssm=ssm, d_output=N_CLASSES,
                   d_model=H, n_layers=N_LAYERS, padded=True, activation="gelu",
                   dropout=0.0, mode="pool", prenorm=False, batchnorm=True,
                   bn_momentum=0.9)


def bare_ssm(**fw):
    """An UNBOUND MambinoSSM, for calling the _fw_* recurrences directly.
    They take q,k,v,s,gamma explicitly and touch no self.param, so no Flax
    binding is needed -- only the dataclass config fields matter."""
    z = jnp.zeros(4)
    return MambinoSSM(H=H, P=8, Lambda_re_init=z, Lambda_im_init=z,
                      V=jnp.eye(4), Vinv=jnp.eye(4), C_init="lecun_normal",
                      discretization="zoh", dt_min=0.001, dt_max=0.1,
                      conj_sym=True, clip_eigs=False, bidirectional=True, **fw)


def inputs():
    x = (jnp.ones((BSZ, SEQ_LEN, IN_DIM)), jnp.ones(BSZ))
    return x, jnp.ones((BSZ, SEQ_LEN))


def init_vars(mc, seed=0):
    x, integ = inputs()
    return mc(training=True).init(
        {"params": random.PRNGKey(seed), "dropout": random.PRNGKey(seed + 1),
         "noise": random.PRNGKey(seed + 2)}, x, integ)


def forward(mc, vs):
    x, integ = inputs()
    return mc(training=False).apply(
        {"params": vs["params"], "batch_stats": vs["batch_stats"]}, x, integ,
        rngs={"params": random.PRNGKey(0), "dropout": random.PRNGKey(1),
              "noise": random.PRNGKey(2)})


def fw_params(params):
    return {"/".join(str(getattr(kk, "key", kk)) for kk in p): leaf
            for p, leaf in jax.tree_util.tree_leaves_with_path(params)
            if "fw_" in "/".join(str(getattr(kk, "key", kk)) for kk in p)}


def rand_qkvs(key, L, d):
    k1, k2, k3, k4 = random.split(key, 4)
    n = lambda u: u / jnp.sqrt(jnp.sum(u * u, -1, keepdims=True) + 1e-6)
    return (n(random.normal(k1, (L, d))), n(random.normal(k2, (L, d))),
            n(random.normal(k3, (L, d))), jax.nn.sigmoid(random.normal(k4, (L,))))


# ────────────────────────────────────────────────────────────────────────
def t1_byte_equivalence(args, fails):
    # No kwargs at all, so this same call also runs against the PRE-CHANGE
    # module in the v2/surprise-gate worktree (whose factory has no fw_*).
    mc = build_model()
    vs = init_vars(mc)
    n_fw = len(fw_params(vs["params"]))
    y = forward(mc, vs)
    ok_tree = (n_fw == 0)
    print(f"[{'PASS' if ok_tree else 'FAIL'}] T1a param tree: {n_fw} fw_* leaves when off (want 0)")
    if not ok_tree:
        fails.append("T1a")
    if args.dump_ref:
        onp.save(args.dump_ref, onp.asarray(y))
        print(f"       wrote reference logits -> {args.dump_ref}  (run --check_ref in the other worktree)")
    if args.check_ref:
        ref = onp.load(args.check_ref)
        exact = bool(onp.array_equal(ref, onp.asarray(y)))
        print(f"[{'PASS' if exact else 'FAIL'}] T1b logits BIT-IDENTICAL to pre-change code "
              f"(max|diff|={onp.max(onp.abs(ref - onp.asarray(y))):.3e})")
        if not exact:
            fails.append("T1b")
    else:
        print("       T1b skipped (no --check_ref); run with --dump_ref in the v2/surprise-gate tree first")


def t2_impl_equivalence(fails):
    bad = []
    for read in ["exclusive", "inclusive"]:
        for gamma in [0.0, 0.5, 0.9, 0.99, 1.0]:
            for L, C, d in [(64, 16, 4), (70, 16, 8), (33, 8, 4), (128, 64, 8), (17, 64, 4)]:
                m = bare_ssm(fw_read=read, fw_chunk=C)
                q, k, v, s = rand_qkvs(random.PRNGKey(L + C + d), L, d)
                g = jnp.asarray(gamma)
                o_seq = m._fw_seq(q, k, v, s, g)
                o_scan = m._fw_scan(q, k, v, s, g)
                o_chunk = m._fw_chunk(q, k, v, s, g)
                e1 = float(jnp.max(jnp.abs(o_seq - o_scan)))
                e2 = float(jnp.max(jnp.abs(o_seq - o_chunk)))
                if not (e1 < 2e-5 and e2 < 2e-5):
                    bad.append(f"{read} g={gamma} L={L} C={C} d={d} scan={e1:.2e} chunk={e2:.2e}")
    print(f"[{'PASS' if not bad else 'FAIL'}] T2 seq==scan==chunk over "
          f"2 reads x 5 gammas x 5 (L,C,d) incl. L%C!=0  ({len(bad)} mismatches)")
    for b in bad[:6]:
        print(f"        {b}")
    if bad:
        fails.append("T2")


def t3_param_counts(fails):
    bad = []
    for proj in ["shared", "separate"]:
        for d in [4, 8, 16]:
            want_layer = (2 * H * d + 3 * d * d + 3) if proj == "shared" else (4 * H * d + 3)
            vs = init_vars(build_model(fast_weight=True, fw_proj=proj, fw_dim=d))
            got = sum(int(x.size) for x in fw_params(vs["params"]).values())
            want = want_layer * N_LAYERS
            tag = "ok" if got == want else "MISMATCH"
            if got != want:
                bad.append(f"{proj} d={d}: got {got:,} want {want:,}")
            print(f"       T3 {proj:<9} d={d:<3} per-layer {want_layer:>6,}  x8 = {got:>7,}  [{tag}]")
    print(f"[{'PASS' if not bad else 'FAIL'}] T3 param counts match plan Sec 4 D3")
    if bad:
        fails.append("T3")


def t4_gradients(fails):
    mc = build_model(fast_weight=True, fw_impl="chunk", fw_chunk=16)
    vs = init_vars(mc)
    x, integ = inputs()

    def loss(p):
        y, _ = mc(training=True).apply(
            {"params": p, "batch_stats": vs["batch_stats"]}, x, integ,
            rngs={"dropout": random.PRNGKey(1), "noise": random.PRNGKey(2)},
            mutable=["batch_stats"])
        return jnp.sum(y ** 2)

    g = jax.grad(loss)(vs["params"])
    dead = [n for n, leaf in fw_params(g).items() if float(jnp.max(jnp.abs(leaf))) == 0.0]
    # At step 0, W_o=0 => only W_o has a gradient; it escapes, then the rest turn on.
    step0 = {n.split("/")[-1] for n, leaf in fw_params(g).items()
             if float(jnp.max(jnp.abs(leaf))) > 0.0}
    p2 = jax.tree_util.tree_map(lambda a, b: a + 0.1 * b, vs["params"], g)
    g2 = jax.grad(loss)(p2)
    dead2 = sorted({n.split("/")[-1] for n, leaf in fw_params(g2).items()
                    if float(jnp.max(jnp.abs(leaf))) == 0.0})
    ok = (dead2 == [])
    print(f"[{'PASS' if ok else 'FAIL'}] T4 gradients: step0 nonzero={sorted(step0)}; "
          f"after one step dead={dead2 or 'none'}")
    if not ok:
        fails.append("T4")
    if "fw_W_o" not in step0:
        print("        WARNING: W_o had no gradient at step 0 -- the branch cannot escape zero-init")
        fails.append("T4-escape")


def t6_causality(fails):
    L, d, C = 24, 4, 8
    bad = []
    for read in ["exclusive", "inclusive"]:
        m = bare_ssm(fw_read=read, fw_chunk=C)
        q, k, v, s = rand_qkvs(random.PRNGKey(7), L, d)
        t = 9
        o0 = m._fw_chunk(q, k, v, s, jnp.asarray(0.9))
        # perturb EVERYTHING written at step t (v and k) and the gate
        v2 = v.at[t].set(v[t] + 1.0)
        k2 = k.at[t].set(k[t] * -1.0)
        s2 = s.at[t].set(1.0 - s[t])
        o1 = m._fw_chunk(q, k2, v2, s2, jnp.asarray(0.9))
        diff = jnp.max(jnp.abs(o1 - o0), axis=-1)
        past = float(jnp.max(diff[:t]))
        self_ = float(diff[t])
        future = float(jnp.max(diff[t + 1:]))
        if past > 1e-8:
            bad.append(f"{read}: FUTURE LEAK -- steps < t moved by {past:.2e}")
        if future < 1e-6:
            bad.append(f"{read}: write at t had no effect on s>t ({future:.2e}) -- memory is dead")
        if read == "exclusive" and self_ > 1e-8:
            bad.append(f"exclusive: SELF-READ -- o_t moved by {self_:.2e}; the off-by-one is back")
        if read == "inclusive" and self_ < 1e-6:
            bad.append(f"inclusive: o_t did NOT move ({self_:.2e}); read convention not applied")
    print(f"[{'PASS' if not bad else 'FAIL'}] T6 causality: no future leak; exclusive read "
          f"does not see its own write")
    for b in bad:
        print(f"        {b}")
    if bad:
        fails.append("T6")


def t7_boundedness(fails):
    L, d = 256, 8
    m = bare_ssm(fw_read="exclusive", fw_chunk=32)
    q, k, v, s = rand_qkvs(random.PRNGKey(3), L, d)
    rows = []
    bad = []
    for gamma in [0.0, 0.5, 0.9, 0.99, 0.999, 1.0]:
        o = m._fw_chunk(q, k, v, s, jnp.asarray(gamma))
        mx = float(jnp.max(jnp.abs(o)))
        rows.append(f"gamma={gamma:<6} max|o|={mx:.4f}")
        if not onp.isfinite(mx):
            bad.append(f"gamma={gamma}: non-finite")
        if mx > 1.5:            # ||M||<=1 with normalized v,k => |o| = |M q| <= 1
            bad.append(f"gamma={gamma}: max|o|={mx:.3f} > 1.5, (1-gamma) coupling not bounding M")
    print(f"[{'PASS' if not bad else 'FAIL'}] T7 boundedness: " + "; ".join(rows))
    for b in bad:
        print(f"        {b}")
    if bad:
        fails.append("T7")


def t8_determinism(fails):
    mc = build_model(fast_weight=True)
    y1 = forward(mc, init_vars(mc, seed=0))
    y2 = forward(mc, init_vars(mc, seed=0))
    ok = bool(jnp.array_equal(y1, y2)) and bool(jnp.all(jnp.isfinite(y1)))
    print(f"[{'PASS' if ok else 'FAIL'}] T8 determinism + finiteness")
    if not ok:
        fails.append("T8")


def t5_ckpt(args, fails):
    from s5.train_helpers import load_checkpoint_msgpack, create_train_state
    mc = build_model(fast_weight=True)
    st = create_train_state(mc, random.PRNGKey(0), padded=True, retrieval=False,
                            in_dim=IN_DIM, bsz=BSZ, seq_len=SEQ_LEN,
                            weight_decay=0.04, batchnorm=True,
                            opt_config="BfastandCdecay", ssm_lr=1e-3, lr=3e-3,
                            dt_global=False)
    try:
        st2, _ = load_checkpoint_msgpack(args.v1_ckpt, st)
        ok = True
    except Exception as e:
        print(f"        {type(e).__name__}: {e}")
        ok = False
    print(f"[{'PASS' if ok else 'FAIL'}] T5 Cluster-A ckpt loads into fast_weight model")
    if not ok:
        fails.append("T5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_ref", default=None)
    ap.add_argument("--check_ref", default=None)
    ap.add_argument("--v1_ckpt", default=None)
    args = ap.parse_args()
    fails = []
    print("=" * 78)
    t1_byte_equivalence(args, fails)
    if args.dump_ref:       # reference-dump mode also runs against pre-change code
        print("dump-only mode: skipping the fast-weight tests")
        raise SystemExit(1 if fails else 0)
    t4_gradients(fails)
    print("-" * 78)
    t2_impl_equivalence(fails)
    t6_causality(fails)
    t7_boundedness(fails)
    print("-" * 78)
    t3_param_counts(fails)
    t8_determinism(fails)
    if args.v1_ckpt:
        t5_ckpt(args, fails)
    print("=" * 78)
    print("ALL PASS" if not fails else f"FAILURES: {fails}")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
