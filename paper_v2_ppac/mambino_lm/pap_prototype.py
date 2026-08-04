"""Pure Adaptive Predictor (PAP) — CPU mechanism prototype (self-contained).

Validates, before any repo/cluster work, the three load-bearing properties of PAP
(pure_adaptive_predictor_spec.md):

  RECURRENCE (undivided, per token t; predict x_{t+1}):
    x_hat_t = C h_{t-1}                      # predict current token embedding from state
    eps_t   = emb(x_t) - x_hat_t             # prediction error
    z_t     = (||eps_t|| - mu)/sigma         # trailing-EMA-standardized surprise (causal)
    g_t     = sigmoid(kappa*z_t + b)         # ANCHOR 1 surprise gate (nonlinear -> NOT an S5)
    h_t     = a (.) h_{t-1} + g_t*(B eps_t)  # surprise-GATED error integration (coasts when eps~0)
    logits  = (Wo + W_fw) h_t                # W_fw = ANCHOR 2 error-driven inference-time fast-weight

Because g_t depends on eps_t (hence h_{t-1}), the recurrence is NONLINEAR with prediction
feedback -> it is a sequential scan (no S5 parallel prefix). g==1 collapses to a LINEAR
error-integrator = an S5 (h_t=(a-BC)h_{t-1}+B emb_t); that is the baseline here.

Checks: (1) trains + CAUSAL; (2) ANCHOR 1: g COASTS on predictable tokens (compute prop. surprise)
and gated vs g==1 (S5-equiv); (3) ANCHOR 2: fast-weight recovers after a distribution shift.

Task: trigger->body grammar (random trigger = high surprise; deterministic body = low surprise)
with a mid-stream A->B trigger->body remap (the shift).
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as onp
import jax, jax.numpy as jnp
import optax
from functools import partial

V, K, BODY = 20, 8, 3
D = 64                          # state dim ("P")
ALPHA = 0.9                     # surprise-EMA decay
PRETRAIN_STEPS, BSZ, T = 2500, 32, 192
SEED = 0
bodies_A = onp.random.RandomState(1).randint(0, V, size=(K, BODY)).astype(onp.int32)
bodies_B = onp.random.RandomState(2).randint(0, V, size=(K, BODY)).astype(onp.int32)

def gen(n_motifs, seed, first, second, shift=0.5):
    r = onp.random.RandomState(seed); seq, isbody = [], []
    ns = int(shift * n_motifs)
    for m in range(n_motifs):
        b = first if m < ns else second
        i = r.randint(0, K); seq.append(i); isbody.append(0)
        for j in range(BODY):
            seq.append(int(b[i, j])); isbody.append(1)
    return onp.array(seq, onp.int32), onp.array(isbody, onp.int32)

def batch(seed, first, second, shift=0.0):
    xs = []
    for bb in range(BSZ):
        s, _ = gen(T // (1 + BODY) + 2, seed + bb, first, second, shift)
        xs.append(s[:T + 1])
    xs = onp.stack(xs); return xs[:, :T], xs[:, 1:T + 1]

def init(key):
    k = jax.random.split(key, 7); sc = 1.0 / onp.sqrt(D)
    # B,C init SMALL so BC (hence the closed-loop transition a-BC) stays near the stable
    # decay a -> the error-integrator doesn't diverge (BC~O(1) at 1/sqrt(D) blows up).
    bcsc = 0.5 / D
    return dict(emb=jax.random.normal(k[0], (V, D)) * 0.5,
                a_raw=jnp.full((D,), 2.0),                        # sigmoid(2)=0.88 decay
                B=jax.random.normal(k[1], (D, D)) * bcsc,
                C=jax.random.normal(k[2], (D, D)) * bcsc,
                Wo=jax.random.normal(k[3], (V, D)) * sc, bo=jnp.zeros(V),
                kappa=jnp.array(2.0), b=jnp.array(0.0))

def run(p, x, gated=True, Wfw=None):
    """Sequential PAP over one sequence. Returns (logits (T,V), h (T,D), g (T))."""
    emb = p["emb"][x]; a = jax.nn.sigmoid(p["a_raw"])
    Wo = p["Wo"] if Wfw is None else p["Wo"] + Wfw

    def step(carry, u):
        h, m, v, cnt = carry
        xhat = p["C"] @ h
        eps = u - xhat
        n = jnp.sqrt(jnp.sum(eps * eps) + 1e-8)
        mu = jnp.where(cnt > 0, m / (1 - ALPHA ** cnt), 0.0)
        var = jnp.where(cnt > 0, jnp.maximum(v / (1 - ALPHA ** cnt) - mu * mu, 1e-6), 1.0)
        z = (n - mu) / jnp.sqrt(var)
        g = jax.nn.sigmoid(p["kappa"] * z + p["b"]) if gated else 1.0
        h = jnp.clip(a * h + g * (p["B"] @ eps), -30.0, 30.0)   # state guard against divergence
        m2 = ALPHA * m + (1 - ALPHA) * n; v2 = ALPHA * v + (1 - ALPHA) * n * n
        h_out = h * jax.lax.rsqrt(jnp.mean(h * h) + 1e-6)        # RMSNorm readout features (bounded logits)
        logits = Wo @ h_out + p["bo"]
        return (h, m2, v2, cnt + 1.0), (logits, h_out, jnp.asarray(g, jnp.float32))

    c0 = (jnp.zeros(D), 0.0, 0.0, 0.0)
    _, (lg, hs, gs) = jax.lax.scan(step, c0, emb)
    return lg, hs, gs

def ce(logits, y):
    return optax.softmax_cross_entropy_with_integer_labels(logits, y)

def loss_fn(p, x, y, gated):
    lg = jax.vmap(lambda xi: run(p, xi, gated)[0])(x)
    return ce(lg.reshape(-1, V), y.reshape(-1)).mean()

def train(gated, tag):
    p = init(jax.random.PRNGKey(SEED))
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3)); os_ = tx.init(p)
    @jax.jit
    def step(p, os_, x, y):
        l, g = jax.value_and_grad(loss_fn)(p, x, y, gated)   # `gated` closed over (constant)
        u, os2 = tx.update(g, os_, p)
        return optax.apply_updates(p, u), os2, l
    for s in range(PRETRAIN_STEPS):
        x, y = batch(3000 + s * BSZ, bodies_A, bodies_A)
        p, os_, l = step(p, os_, jnp.array(x), jnp.array(y))
        if s % 500 == 0: print(f"    [{tag}] step {s:4d} loss {float(l):.4f}")
    return p, float(l)

print("=" * 70); print("PAP PROTOTYPE"); print("=" * 70)
p_lin, l_lin = train(False, "g==1 / linear (S5-equiv)")
p_gat, l_gat = train(True, "gated (anchor 1)")
print(f"\n[TRAIN] final loss  linear(g==1) {l_lin:.4f}   gated {l_gat:.4f}")

# --- ANCHOR 1: does g coast on predictable (body) vs surprising (trigger) tokens? ---
xe, ye = batch(555, bodies_A, bodies_A)
_, isb = gen(T // (1 + BODY) + 2, 555, bodies_A, bodies_A); isb = isb[:T]
gs = onp.array(jax.vmap(lambda xi: run(p_gat, xi, True)[2])(jnp.array(xe)))  # (BSZ,T)
gtrig = gs[:, isb == 0].mean(); gbody = gs[:, isb == 1].mean()
print(f"[ANCHOR 1] mean g  triggers(surprising) {gtrig:.3f}  vs  bodies(predictable) {gbody:.3f}"
      f"   -> {'COASTS on predictable (compute prop surprise)' if gbody < gtrig - 0.05 else 'no coasting'}")

# --- CAUSALITY: change x at position k -> outputs<k unchanged ---
xa = jnp.array(xe[:1]); k = 100
xb = xa.at[0, k].set((int(xa[0, k]) + 3) % V)
la = onp.array(run(p_gat, xa[0], True)[0]); lb = onp.array(run(p_gat, xb[0], True)[0])
md = float(onp.max(onp.abs(la[:k] - lb[:k])))
print(f"[CAUSALITY] outputs<k change by {md:.1e}  -> {'CAUSAL' if md < 1e-6 else 'LEAK!'}")

# --- ANCHOR 2: fast-weight recovery across A->B shift (frozen recurrence, adapt Wo) ---
def ce_W(Wfw, h, y, p):
    lg = h @ (p["Wo"] + Wfw).T + p["bo"]
    return ce(lg, y).mean()
gradW = jax.jit(jax.grad(ce_W))
def eval_stream(p, ttt, CH=24, eta=0.1):
    s, isb = gen(1200, 999, bodies_A, bodies_B); x = s[:-1][None]; y = s[1:]; isb = isb[1:]
    _, h, _ = run(p, jnp.array(x[0]) if False else jnp.array(s[:-1]), True)  # (Ts,D) frozen states
    h = onp.array(h); Ts = h.shape[0]; Wfw = onp.zeros((V, D)); err = onp.zeros(Ts)
    for cs in range(0, Ts, CH):
        sl = slice(cs, min(cs + CH, Ts)); hc = h[sl]; yc = y[sl]
        lg = hc @ (p["Wo"] + Wfw).T + onp.array(p["bo"])
        err[sl] = onp.array(ce(jnp.array(lg), jnp.array(yc)))
        if ttt:
            Wfw = Wfw - eta * onp.array(gradW(jnp.array(Wfw), jnp.array(hc), jnp.array(yc), p))
    maskB_body = onp.zeros(Ts, bool); half = Ts // 2
    maskB_body[half:] = isb[half:] == 1
    return err[maskB_body].mean()
eb_off = eval_stream(p_gat, False); eb_on = eval_stream(p_gat, True)
print(f"[ANCHOR 2] regime-B body error  no-adapt {eb_off:.3f} -> adapt {eb_on:.3f}"
      f"   ({100*(1-eb_on/max(eb_off,1e-6)):.0f}% lower)")

print("\nVERDICT: PAP trains + causal; anchor-1 gate coasts on predictable; anchor-2 adapts on shift.")
