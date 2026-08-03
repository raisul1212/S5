"""Mambino-LM Stage-1 mechanism proof (self-contained, CPU).

QUESTION (the §9 load-bearing risk): does the 2-level surprise-gated ESCALATION gate
train to a SELECTIVE, NON-COLLAPSED policy -- escalate on hard regions, skip on easy --
and does the slow-ticking top genuinely HELP the escalated tokens? If the STE hard-alpha
gate + alpha-curriculum + Rao-Ballard per-level loss can't avoid collapse (alpha->0 top
starved, or alpha->1 no adaptive compute), the architecture is wrong, not under-tuned;
do not spend cluster time on enwik8.

Task = BLOCK-REGIME stream (matched to trailing-surprise's regime-level strength, per Stage-0):
  EASY block: pure deterministic-Markov filler -> bottom predicts it -> low surprise -> SKIP top.
  HARD block: a context token c at block start, then repeated (filler,filler,Q,ANS) where
    ANS depends on c (long-range) -> the fast/local bottom (fixed strong leak) CANNOT hold c ->
    repeated errors -> high trailing surprise -> ESCALATE -> the slow top (holds c) supplies
    the nudge that resolves ANS.

Architecture (2-level, NO inference-time learning yet -- that is Stage 2):
  bottom : diagonal linear SSM, FIXED small decay (fast/local, can't hold context)
  top    : diagonal linear SSM, ticks every s tokens on POOLED input (holds context), learned decay
  gate   : alpha(t) = STE( z0(t) > theta ),  z0 = trailing causal EMA of bottom's OWN (pre-nudge) CE
  logits : p0(t) + alpha(t)*beta*nudge(t)          # confident tokens skip the top entirely
Losses: task CE  +  Rao-Ballard (top predicts pooled bottom error, trains top even when NOT escalated)
        +  ponder (penalize mean alpha; annealed IN after an alpha=1 curriculum warmup).
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax, jax.numpy as jnp
import optax
from functools import partial

# ---------------- config ----------------
V = 16
CTX0, CTX1, Q, ANS0, ANS1 = 0, 1, 2, 3, 4
FILLER = list(range(5, V))                 # 11 filler symbols
D, D1 = 64, 48                             # bottom / top state
S = 4                                      # top tick stride (slow ticking = FIX #1)
LB = 24                                    # block length
NBLOCK = 10                                # blocks per sequence  -> T = 240
BSZ = 48
STEPS = 4000
WARMUP = 600                               # ponder OFF (alpha high -> the TOP learns) then ramp in
LAM_AUX = 0.1                              # Rao-Ballard weight
LAM_POND_MAX = 0.05                        # ponder (escalation-cost) weight, ramped in
DECAY0_FIXED = 0.0                         # bottom = FEEDFORWARD (only current token) -> cannot hold c
EMA_A = 0.9
SEED = 0
T = NBLOCK * LB
rng = np.random.RandomState(SEED)
filler_next = {f: FILLER[(i + 3) % len(FILLER)] for i, f in enumerate(FILLER)}   # order-1 map

# labels for metrics
EASY, HARD = 0, 1

def gen_sequence(seed):
    r = np.random.RandomState(seed)
    toks, blk, pos = [], [], []            # token, block-type, position-type(0 filler,1 answer,2 special)
    for _ in range(NBLOCK):
        hard = r.rand() < 0.5
        if not hard:                       # EASY: pure markov filler
            f = r.choice(FILLER)
            for _ in range(LB):
                toks.append(f); blk.append(EASY); pos.append(0)
                f = filler_next[f]
        else:                              # HARD: context c, then (filler,filler,Q,ANS)*
            c = r.choice([CTX0, CTX1]); ans = ANS0 if c == CTX0 else ANS1
            toks.append(c); blk.append(HARD); pos.append(2)
            f = r.choice(FILLER); k = 1
            while k < LB:
                phase = (k - 1) % 4
                if phase in (0, 1):
                    toks.append(f); blk.append(HARD); pos.append(0); f = filler_next[f]
                elif phase == 2:
                    toks.append(Q); blk.append(HARD); pos.append(2)
                else:
                    toks.append(ans); blk.append(HARD); pos.append(1)
                k += 1
    toks = np.array(toks[:T], np.int32)
    return toks, np.array(blk[:T], np.int32), np.array(pos[:T], np.int32)

def batch(seed0):
    xs, bs, ps = [], [], []
    for b in range(BSZ):
        t, bl, po = gen_sequence(seed0 + b)
        xs.append(t); bs.append(bl); ps.append(po)
    xs = np.stack(xs)
    return xs[:, :-1], xs[:, 1:], np.stack(bs)[:, :-1], np.stack(ps)[:, :-1]   # x, y, blk, pos

# ---------------- params ----------------
def init_params(key):
    k = jax.random.split(key, 8)
    sc = 1.0 / np.sqrt(D)
    return dict(
        emb=jax.random.normal(k[0], (V, D)) * 0.1,
        B0=jax.random.normal(k[1], (D, D)) * sc,
        C0=jax.random.normal(k[2], (D, V)) * sc,
        b0=jnp.zeros(V),
        a1=jnp.full((D1,), 2.0),                       # top decay logit (long memory)
        B1=jax.random.normal(k[3], (D, D1)) * sc,      # top reads pooled emb (D) -> D1
        C1=jax.random.normal(k[4], (D1, V)) * sc,      # top nudge head
        Caux=jax.random.normal(k[5], (D1, 1)) * sc,    # Rao-Ballard: predict pooled bottom error
        baux=jnp.zeros(1),
        kappa=jnp.array(4.0), theta=jnp.array(0.0), beta=jnp.array(1.0),
    )

def ste(hard, soft):                                   # forward=hard, grad=soft
    return soft + jax.lax.stop_gradient(hard - soft)

def ce(logits, y):
    return optax.softmax_cross_entropy_with_integer_labels(logits, y)

def run_seq(p, x, y, force):
    """force: -2 soft gate (train), -1 hard-threshold gate (eval inference),
    0 force alpha=0, 1 force alpha=1. Returns per-step dict."""
    emb = p["emb"][x]                                  # (T,D)
    decay1 = jax.nn.sigmoid(p["a1"])

    def step(carry, inp):
        h0, h1, m, v, cnt, ps, pc, pe, tstep = carry
        e_t, y_t = inp
        # trailing causal surprise z0 (bias-corrected EMA of bottom pre-nudge CE)
        cntc = cnt + 1.0
        mu = m / (1 - EMA_A ** cntc)
        z0 = mu
        soft_a = jax.nn.sigmoid(p["kappa"] * (z0 - p["theta"]))
        hard_a = (z0 > p["theta"]).astype(jnp.float32)
        # -2: soft (train, gradients stay alive); -1: hard threshold (eval inference); >=0: forced
        alpha = jnp.where(force == -2, soft_a,
                 jnp.where(force == -1, hard_a, force.astype(jnp.float32)))
        # bottom (fast, fixed leak) + nudge from CURRENT (possibly stale) top state
        h0 = DECAY0_FIXED * h0 + e_t @ p["B0"]
        p0 = h0 @ p["C0"] + p["b0"]                     # bottom pre-nudge logits
        nudge = h1 @ p["C1"]
        logits = p0 + alpha * p["beta"] * nudge
        eb = ce(p0, y_t)                               # bottom's OWN realized error -> drives z0
        # update EMA with eb (for next step's z0) ; accumulate pools
        m2 = EMA_A * m + (1 - EMA_A) * eb
        v2 = EMA_A * v + (1 - EMA_A) * eb * eb
        ps2 = ps + e_t; pc2 = pc + 1.0; pe2 = pe + eb
        # slow tick: update top on pooled emb, emit Rao-Ballard (aux_pred, e_pool)
        is_tick = ((tstep + 1) % S == 0)
        pooled = ps2 / pc2
        h1_new = decay1 * h1 + pooled @ p["B1"]
        e_pool = pe2 / pc2
        aux_pred = (h1_new @ p["Caux"] + p["baux"])[0]
        h1 = jnp.where(is_tick, h1_new, h1)
        ps2 = jnp.where(is_tick, 0.0, ps2); pc2 = jnp.where(is_tick, 0.0, pc2); pe2 = jnp.where(is_tick, 0.0, pe2)
        out = dict(logits=logits, alpha=hard_a, soft=soft_a, z0=z0,
                   aux_pred=aux_pred, e_pool=e_pool, tick=is_tick.astype(jnp.float32))
        carry = (h0, h1, m2, v2, cntc, ps2, pc2, pe2, tstep + 1.0)
        return carry, out

    c0 = (jnp.zeros(D), jnp.zeros(D1), 0.0, 0.0, 0.0, jnp.zeros(D), 0.0, 0.0, 0.0)
    _, outs = jax.lax.scan(step, c0, (emb, y))
    return outs

def loss_fn(p, x, y, lam_pond, force):
    outs = jax.vmap(lambda xi, yi: run_seq(p, xi, yi, force))(x, y)
    task = ce(outs["logits"].reshape(-1, V), y.reshape(-1)).mean()
    aux = (((outs["aux_pred"] - jax.lax.stop_gradient(outs["e_pool"])) ** 2) * outs["tick"]).sum() / (outs["tick"].sum() + 1e-6)
    pond = outs["soft"].mean()
    return task + LAM_AUX * aux + lam_pond * pond, (task, aux, pond, outs)

@partial(jax.jit, static_argnums=())
def train_step(p, os_, x, y, lam_pond, force):
    (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p, x, y, lam_pond, force)
    u, os_ = tx.update(g, os_, p); p = optax.apply_updates(p, u)
    return p, os_, l, aux

params = init_params(jax.random.PRNGKey(SEED))
tx = optax.adam(3e-3); opt_state = tx.init(params)

print("[*] training 2-level surprise-gated escalation (soft gate in fwd, hard at eval) ...")
force_soft = jnp.array(-2)                                                # train with the soft gate
for step in range(STEPS):
    x, y, _, _ = batch(10_000 + step * BSZ)
    ramp = 0.0 if step < WARMUP else min(1.0, (step - WARMUP) / 1000.0)   # ponder OFF during warmup, then ramp
    lam_pond = jnp.array(LAM_POND_MAX * ramp)
    params, opt_state, l, (task, aux, pond, _) = train_step(params, opt_state, jnp.array(x), jnp.array(y), lam_pond, force_soft)
    if step % 500 == 0 or step == STEPS - 1:
        print(f"    step {step:5d}  loss {float(l):.4f}  task {float(task):.4f}  aux {float(aux):.4f}  gate<soft> {float(pond):.3f}"
              f"  kappa {float(params['kappa']):.2f} theta {float(params['theta']):.3f} beta {float(params['beta']):.2f}")

# ---------------- evaluation ----------------
def evaluate(force):
    xa, ya, blk, pos = batch(777)
    outs = jax.vmap(lambda xi, yi: run_seq(params, xi, yi, jnp.array(force)))(jnp.array(xa), jnp.array(ya))
    logits = np.array(outs["logits"]); alpha = np.array(outs["alpha"]); soft = np.array(outs["soft"])
    pred = logits.argmax(-1); correct = (pred == ya)
    easy = blk == EASY; hard_ans = (blk == HARD) & (pos == 1); hard_fill = (blk == HARD) & (pos == 0)
    return dict(
        a_easy=alpha[easy].mean(), a_hardfill=alpha[hard_fill].mean(), a_hardans=alpha[hard_ans].mean(),
        a_mean=alpha.mean(),
        s_easy=soft[easy].mean(), s_hardans=soft[hard_ans].mean(),
        acc_ans=correct[hard_ans].mean(), acc_easy=correct[easy].mean(), acc_hardfill=correct[hard_fill].mean(),
    )

print("\n" + "=" * 72)
print("STAGE-1 MECHANISM PROOF  (2-level surprise-gated escalation, no TTT)")
print("=" * 72)
learned = evaluate(-1); noesc = evaluate(0); allesc = evaluate(1)
print(f"\n[SELECTIVITY]  hard-threshold escalation rate (what the chip does) by region:")
print(f"    EASY blocks        : {learned['a_easy']:.3f}  (soft {learned['s_easy']:.3f})  <- should be LOW (skip top)")
print(f"    HARD filler tokens : {learned['a_hardfill']:.3f}")
print(f"    HARD answer tokens : {learned['a_hardans']:.3f}  (soft {learned['s_hardans']:.3f})  <- should be HIGH (need top)")
print(f"    overall escalation : {learned['a_mean']:.3f}   (collapse check: not ~0 and not ~1)")
print(f"\n[USEFULNESS]  hard-block ANSWER accuracy (does the top actually resolve answers?):")
print(f"    alpha=0 (no top)   : {noesc['acc_ans']:.3f}   <- bottom alone CANNOT know context c")
print(f"    learned alpha      : {learned['acc_ans']:.3f}")
print(f"    alpha=1 (always)   : {allesc['acc_ans']:.3f}   <- top ceiling")
print(f"\n[NO-HARM]  easy acc {learned['acc_easy']:.3f}  hard-filler acc {learned['acc_hardfill']:.3f}  (bottom handles, ~1.0)")

sel = learned['a_hardans'] - learned['a_easy']
lift = learned['acc_ans'] - noesc['acc_ans']
ceil_gap = allesc['acc_ans'] - noesc['acc_ans']          # is the task even top-requiring?
collapsed = (learned['a_mean'] < 0.02) or (learned['a_mean'] > 0.98)
print("\nVERDICT:")
print(f"    task is top-requiring (acc alpha=1 - alpha=0) = {ceil_gap:+.3f}   {'OK' if ceil_gap > 0.2 else 'TASK-TOO-EASY'}")
print(f"    selectivity (esc_hardans - esc_easy)          = {sel:+.3f}   {'PASS' if sel > 0.3 else 'WEAK'}")
print(f"    top usefulness (acc lift from escalation)     = {lift:+.3f}   {'PASS' if lift > 0.1 else 'WEAK'}")
print(f"    gate not collapsed                            = {not collapsed}")
print("    => PASS if task is top-requiring, gate is SELECTIVE, escalation LIFTS answer accuracy, no collapse.")
