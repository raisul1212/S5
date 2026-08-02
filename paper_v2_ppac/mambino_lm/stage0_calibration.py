"""Mambino-LM Stage-0 calibration test (self-contained, CPU).

QUESTION (go/no-go): does TRAILING realized-surprise z (EMA of ||prediction error||)
track per-token realized error better than SOFTMAX-CONFIDENCE, and stay honest under a
test-time update while confidence inflates? If surprise tracks and confidence decouples
under TTT -> gate escalation on surprise, Mambino-LM survives. If BOTH decouple -> the
escalation premise is dead, don't build the hierarchy.

Synthetic 'trigger->body' grammar with a mid-stream regime shift:
  a random trigger symbol (unpredictable) is followed by a DETERMINISTIC body (predictable
  once the trigger->body map is learned). Regime A (first half) and Regime B (second half)
  use DIFFERENT trigger->body maps. A model pretrained on A is CONFIDENTLY WRONG on B's
  bodies until it adapts -- the exact miscalibration Fable predicted.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

# ---- config ----
V, K, BODY = 16, 8, 3          # vocab, #triggers (symbols 0..K-1), body length
D, LAYERS = 96, 2
PRETRAIN_STEPS, BSZ, TRAIN_T = 3000, 32, 256
EVAL_MOTIFS = 1400
CHUNK = 32                     # TTT update granularity (tokens)
ETA_TTT = 0.5                  # test-time step size
EMA_A = 0.9
SEED = 0

def make_bodies(seed):
    return np.random.RandomState(seed).randint(0, V, size=(K, BODY)).astype(np.int32)

bodies_A, bodies_B = make_bodies(1), make_bodies(2)   # same triggers, different maps

def gen_stream(n_motifs, seed, first, second, shift=0.5):
    r = np.random.RandomState(seed)
    seq, reg, body = [], [], []
    n_shift = int(shift * n_motifs)
    for m in range(n_motifs):
        b = first if m < n_shift else second
        rg = 0 if m < n_shift else 1
        i = r.randint(0, K)
        seq.append(i); reg.append(rg); body.append(0)            # trigger (unpredictable)
        for j in range(BODY):
            seq.append(int(b[i, j])); reg.append(rg); body.append(1)   # body (deterministic)
    return np.array(seq, np.int32), np.array(reg, np.int32), np.array(body, np.int32)

class GRULM(nn.Module):
    @nn.compact
    def __call__(self, x):
        h = nn.Embed(V, D)(x)
        for _ in range(LAYERS):
            h = nn.RNN(nn.GRUCell(features=D))(h)
        return nn.Dense(V)(h), h            # (logits, feat)

model = GRULM()

def sample_batch(step):
    xs = []
    for b in range(BSZ):
        s, _, _ = gen_stream(TRAIN_T // (1 + BODY) + 2, 2000 + step * BSZ + b, bodies_A, bodies_A)
        xs.append(s[:TRAIN_T + 1])
    xs = np.stack(xs)
    return xs[:, :TRAIN_T], xs[:, 1:TRAIN_T + 1]

x0, y0 = sample_batch(0)
params = model.init(jax.random.PRNGKey(SEED), jnp.array(x0))
tx = optax.adam(1e-3); opt_state = tx.init(params)

def loss_fn(p, x, y):
    logits, _ = model.apply(p, x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

@jax.jit
def train_step(p, os_, x, y):
    l, g = jax.value_and_grad(loss_fn)(p, x, y)
    u, os_ = tx.update(g, os_); p = optax.apply_updates(p, u)
    return p, os_, l

print("[*] pretraining on regime A ...")
for step in range(PRETRAIN_STEPS):
    x, y = sample_batch(step)
    params, opt_state, l = train_step(params, opt_state, jnp.array(x), jnp.array(y))
    if step % 500 == 0:
        print(f"    step {step:5d}  loss {float(l):.4f}")

# ---- eval stream A -> B ----
seq, reg, body = gen_stream(EVAL_MOTIFS, 999, bodies_A, bodies_B)
x_eval, y_eval = seq[:-1], seq[1:]; reg, body = reg[1:], body[1:]
base_logits, feat = model.apply(params, jnp.array(x_eval)[None])
base_logits, feat = np.array(base_logits[0]), np.array(feat[0])
T = base_logits.shape[0]

def ce_of_W(W, fc, bc, tc):
    logits = bc + fc @ W.T
    return optax.softmax_cross_entropy_with_integer_labels(logits, tc).mean()
grad_ce = jax.jit(jax.grad(ce_of_W))

def run_eval(ttt_on):
    W = np.zeros((V, D), np.float32)
    e = np.zeros(T); cd = np.zeros(T); z = np.zeros(T)
    m = 0.0; v = 0.0; cnt = 0
    for cs in range(0, T, CHUNK):
        ce_ = slice(cs, min(cs + CHUNK, T))
        fc, bc, tc = feat[ce_], base_logits[ce_], y_eval[ce_]
        logits = bc + fc @ W.T
        logits = logits - logits.max(-1, keepdims=True)
        p = np.exp(logits); p /= p.sum(-1, keepdims=True)
        for k in range(fc.shape[0]):
            t = cs + k
            et = -np.log(p[k, tc[k]] + 1e-12)
            cd[t] = 1.0 - p[k].max()                    # confidence-difficulty (high = model unsure)
            cnt += 1                                    # trailing surprise BEFORE folding e(t) in
            mu = m / (1 - EMA_A ** cnt); var = max(v / (1 - EMA_A ** cnt) - mu * mu, 0.0)
            z[t] = (et_ema := mu)                        # trailing EMA level of realized error
            m = EMA_A * m + (1 - EMA_A) * et; v = EMA_A * v + (1 - EMA_A) * et * et
            e[t] = et
        if ttt_on:
            g = grad_ce(jnp.array(W), jnp.array(fc), jnp.array(bc), jnp.array(tc))
            W = W - ETA_TTT * np.array(g)
    return e, cd, z

def spearman(a, b):
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])

print("\n" + "=" * 68)
print("STAGE-0 CALIBRATION RESULT  (regime A first half -> regime B second half)")
print("=" * 68)
maskB = reg == 1; maskB_body = (reg == 1) & (body == 1)
for ttt_on in [False, True]:
    e, cd, z = run_eval(ttt_on)
    eB, eBb = e[maskB].mean(), e[maskB_body].mean()
    tag = "TTT ON " if ttt_on else "TTT OFF"
    mA = reg == 0
    print(f"\n[{tag}]  A: err={e[mA].mean():.3f} confDiff={cd[mA].mean():.3f} surp={z[mA].mean():.3f}   ||   "
          f"B: err={eB:.3f} confDiff={cd[maskB].mean():.3f} surp={z[maskB].mean():.3f}   (B-bodies err={eBb:.3f})")
    print(f"           A->B jump:  err x{eB/max(e[mA].mean(),1e-6):.1f}   confDiff x{cd[maskB].mean()/max(cd[mA].mean(),1e-6):.1f}   surp x{z[maskB].mean()/max(z[mA].mean(),1e-6):.1f}   (does the signal jump WITH the error?)")
    for name, m in [("all-B", maskB), ("B-bodies", maskB_body)]:
        sc = spearman(cd[m], e[m]); sz = spearman(z[m], e[m])
        print(f"    {name:9s}: Spearman(confidence-diff, err) = {sc:+.3f}   "
              f"Spearman(trailing-surprise, err) = {sz:+.3f}   -> surprise{' WINS' if (sz>sc) else ' <= conf'}")

# adaptation check: does TTT lower regime-B body error?
eoff, _, _ = run_eval(False); eon, _, _ = run_eval(True)
print(f"\nAdaptation (B-bodies): err {eoff[maskB_body].mean():.3f} (no TTT) -> "
      f"{eon[maskB_body].mean():.3f} (TTT)  = {100*(1-eon[maskB_body].mean()/max(eoff[maskB_body].mean(),1e-6)):.0f}% lower")
print("\nVERDICT: PASS if trailing-surprise tracks err on B-bodies (positive, > confidence)")
print("         AND confidence stays low/decoupled while err is high (confidently wrong).")
