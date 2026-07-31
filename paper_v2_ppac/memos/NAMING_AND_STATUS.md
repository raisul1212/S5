# Mambino variants — canonical naming, block diagrams, and status

**Written: 2026-07-31 13:52 EDT.** Supersedes all earlier arm names.
Repo: `github.com/raisul1212/S5`, branch `v2/cluster-b-fastweight`.

---

## 0. Why this document exists

The working names collided badly. The worst offender was **"surprise"**, used for
three different things at once:

1. the physical quantity `eps = x - x_hat` (the prediction error);
2. `ARM=surprise`, one Cluster-B arm;
3. the property of Cluster A's gate, which is *also* surprise-driven.

That produced sentences like *"surprise beats Cluster A"* — incoherent, because
Cluster A is surprise-gated too. **Every Mambino variant is surprise-driven,
including the base**, which already writes `W_eps_bar . eps` into the main scan.
The differences are (a) *what* surprise modulates and (b) whether that modulation
is *adaptive* or *fixed*.

---

## 1. Canonical names

**"Cluster A" and "Cluster B" are WORKSTREAM names — research programs, not models.**
Models get their own names. Never label a model "Cluster A" again.

### The two mechanisms

| short | full name | what it is | roles | params |
|---|---|---|---|---|
| **SWG** | **State-Write Gate** | scalar `g(t)` multiplying the `W_eps_bar . eps` write into the MAIN SCAN | 1 + 3 + 4 | +2/layer (**16**) |
| **FWM** | **Fast-Weight Memory** | `d x d` matrix state `M` written from eps, read by a query, added to the block OUTPUT | 2 | +2,243/layer (**17,944** at d=8) |

### The models

| canonical name | = base + | FWM write gate | params | old names (retired) |
|---|---|---|---|---|
| **Mambino-0** | — | — | 105,738 | Config 4, "gateless", "Mambino-gelu", baseline |
| **Mambino-G** | SWG | — | 105,754 | **Cluster A**, mambino2p0, "Mambino 2.0", "signed" |
| **Mambino-F** | FWM | **adaptive** (`kappa_B` learned) | 123,682 | **`ARM=surprise`**, "arm 2", "B fw-only" |
| **Mambino-F°** | FWM | **fixed** (`kappa_B` pinned 0) | 123,682 | **`ARM=const`**, "arm 3", "B const" |
| **Mambino-GF** | SWG + FWM | adaptive | 123,698 | **`ARM=stacked`**, "arm 8", "Cluster B stacked" |
| **Mambino-GF°** | SWG + FWM | fixed | 123,698 | not built yet (control for GF) |

Read the suffix as composition: **G** = State-Write Gate, **F** = Fast-Weight
Memory, **°** = that model's FWM write gate is fixed rather than adaptive.

### Terms to stop using

| retired | use instead |
|---|---|
| "surprise" as an arm name | **Mambino-F** (the model) or **adaptive write gate** (the setting) |
| "const" as an arm name | **Mambino-F°** or **fixed write gate** |
| "Cluster A" / "Cluster B" for a model | **Mambino-G** / **Mambino-F** or **-GF** |
| "the gate" (ambiguous) | **SWG** or **FWM write gate** — say which |
| "surprise gate" | ambiguous; both gates are surprise-driven. Name the target. |

CLI flags are unchanged (`--surprise_gate`, `--fw_gate_mode=surprise|const`,
`ARM=surprise|const|stacked`) — renaming them would break the launchers and the
committed checkpoints. **The mapping table above is the bridge.**

---

## 2. Block diagrams

### 2.1 Mambino-0 — the base every variant builds on

```
 input (L,H) ──► [DAC] ──► x ═══════════════════════════════════════╗
                            ║                               (x bus) ║
 ┌──────────────────────────▼──────────────────────────────────┐    ║
 │ PREDICTOR                                    consumes: x    │    ║
 │   p(t) = Ls_bar·p(t-1) + Bs_bar·x(t)          [fwd only]    │    ║
 │   x_hat(t) = Cs·p(t-1)                        [causal shift]│    ║
 └──────────────────────────┬──────────────────────────────────┘    ║
                            ▼                                       ║
              eps(t) = x(t) - x_hat(t)      ← THE SURPRISE (a VECTOR)║
                            │                                       ║
 ┌──────────────────────────▼──────────────────────────────────┐    ║
 │ MAIN SSM                                                    │    ║
 │   u(t) = B_bar·x(t) + W_eps_bar·eps(t)      ← surprise is   │    ║
 │   <SCAN fwd>  <SCAN bwd>                      ALREADY here  │    ║
 │   ys = C_tilde·[h_fwd | h_bwd]                              │    ║
 └──────────────────────────┬──────────────────────────────────┘    ║
                            │                                       ║
                    ys + D·x  ◄════════════════════════════════════╝
                            │
                        [ADC] ──► output (L,H)
```

Note the base is **already** surprise-driven: `W_eps_bar . eps` is the signed
first moment (role 1). Nothing added below "introduces surprise" — they change
what it modulates.

### 2.2 The shared normalizer (used by both additions)

```
              eps(t) (L,H)  ──►  m(t) = ||eps(t)||       (L,H) -> scalar
                                 mu,sigma = causal EMA   (2 tiny scans)
                                 z(t) = (m - mu)/sigma   normalized scalar
```
`z` is param-free and computed ONCE. Each mechanism puts its OWN `(kappa, bias)`
on top of it — that is why `_surprise_gate` was split to expose `_surprise_z`.

### 2.3 Mambino-G = Mambino-0 + SWG  (adds 16 params)

```
   z(t) ──► g(t) = tanh( kappa_A·z(t) + b_A )        SIGNED, in [-1,1]
                            │
   MAIN SSM input becomes:  ▼
       u(t) = B_bar·x(t) + g(t)·( W_eps_bar·eps(t) )
                            ▲
                   push (+) / hold (0) / erase (-)
```
Modulates the existing role-1 write. Nothing else changes. **Zero silicon cost**
(elementwise only).

### 2.4 Mambino-F = Mambino-0 + FWM  (adds 17,944 params at d=8)

```
   z(t) ──► w(t) = sigmoid( kappa_B·z(t) + b_B )     UNSIGNED, in (0,1)
                                                     (Mambino-F°: kappa_B ≡ 0)
   x(t) ──► k = norm(W_k·x),  q = norm(W_q·x)        H -> d   shared address
   eps(t) ─► v = norm(W_v·eps)                       H -> d   SIGNED content

        M(t) = gamma·M(t-1) + (1-gamma)·w(t)·( v(t) k(t)^T )    d x d STATE
        o(t) = M(t-1)·q(t)                            EXCLUSIVE read

   OUTPUT becomes:  ys + D·x + W_o·o(t)
```
Attached at the block **output**, not inside the scan. So FWM is a **SIBLING** of
the main scan — both consume `(x, eps)`, neither consumes the other:

```
   PREDICTOR ──► { MAIN fwd || MAIN bwd || FWM } ──► sum
```
Width, not depth. Costs **+22.5% MACs** and `d^2` of state per layer.

### 2.5 Mambino-GF = Mambino-0 + SWG + FWM  (123,698 params)

Both of the above, sharing one `z`, with **independent** `(kappa, bias)` pairs:

```
                         z(t)
                    ┌─────┴─────┐
        g = tanh(kA·z + bA)   w = sigmoid(kB·z + bB)
          SIGNED                UNSIGNED
            │                       │
   u = B·x + g·(W_eps·eps)   M = gamma·M + (1-gamma)·w·(v k^T)
        MAIN SCAN                FAST WEIGHT
            │                       │
            └───────► ys + D·x + W_o·o ───────►
```
**Open question this is built to answer:** both gates read the SAME `z`. If most
of the benefit is simply "attend harder to surprising tokens", they may be two
routes to one effect and stack sublinearly. Untested until the 8-seed run lands.

---

## 3. Metric policy (locked)

**`test@peakval` — test accuracy at the highest-validation epoch — is THE metric.**
Leakage-free, and it is what every internal comparison uses.

`test_max` (best test over all epochs) is reported **only** as a single context
row, computed identically for every config, solely to relate to published numbers
that used a single-seed best-epoch convention. **It is never used to argue an
effect up or down.** Mixing the two was a real error earlier in this work.

Corollary already observed: single-epoch val selection is high-variance. Seed
12345's Mambino-F° run selected epoch 31, which happened to dip to 0.5810 while
neighbours sat at 0.59-0.60 (its `test_max` was 0.6035, train loss normal). That
one artifact moved a 4-seed mean by ~0.6 pp.

---

## 4. Status at 2026-07-31 13:52 EDT

### 4.1 Results — LRA-ListOps, 40 epochs, `test@peakval`

| model | params | 8-seed | 4-seed subset |
|---|---:|---:|---:|
| Mambino-0 | 105,738 | **0.5993** | 0.5988 |
| Mambino-G | 105,754 | **0.6050** | 0.6025 |
| Mambino-F | 123,682 | — | 0.6051 (n=4) |
| Mambino-F° | 123,682 | — | 0.5966 (n=4) |
| **Mambino-GF** | 123,698 | **RUNNING** | — |
| Pure S5 (Corner 1) | 188,490 | **0.6089** | 0.6110 |
| Mambino-lowrank (Corner 3') | 188,682 | **0.6138** | 0.6155 |

4-seed subset = {6554595, 42, 12345, 271828}. It runs **high** for Pure S5 (+0.21)
and **low** for Mambino-G (−0.25) versus their 8-seed values — a 0.46 pp swing
from seed selection alone. Quote 8-seed numbers wherever they exist.

**Per-seed deltas vs Mambino-0** (pp), the honest view at n=4:

| seed | F° − 0 | F − 0 | F − F° |
|---|---:|---:|---:|
| 6554595 | +0.90 | +1.10 | +0.20 |
| 42 | +0.30 | +0.60 | +0.30 |
| 12345 | −2.90 (artifact) | −0.40 | +2.50 |
| 271828 | +0.85 | +1.25 | +0.40 |
| **mean** | −0.22 | +0.63 | +0.85 |
| **positive** | 3/4 | 3/4 | **4/4** |

Reading: FWM helps on 3/4 seeds even with a fixed write gate; making the write
gate adaptive adds on top on 4/4. All within noise at n=4; nothing here is
resolved. **Additivity of SWG and FWM is UNMEASURED** — no completed run has both.

### 4.2 Results — S7 protocol (200 ep, dropout 0.235, warmup 10), 2 seeds

| model | 40 ep | 200 ep | delta |
|---|---:|---:|---:|
| Pure S5 (Corner 1) | 0.6140 | 0.6153 | **+0.13** |
| Mambino-G | 0.6030 | 0.6093 | **+0.63** |
| Mambino-GF | — | RUNNING (49/200) | — |

**Key negative:** 5x the budget plus heavy dropout moves Pure S5 by ~0.1 pp, so
the ~2.9 pp gap to S7's published 63.77% is **NOT training budget** — it is
capacity (~1.08 M vs 188 K) and/or S7's input-dependent selectivity.
**Key positive:** Mambino gains ~5x more from budget than Pure S5; the gap
narrows from 1.10 pp to 0.60 pp.

### 4.3 In flight

| jobs | what | ETA |
|---|---|---|
| 11429476/7 | Mambino-GF, 200 ep, seeds 6554595/42 | ~9 h (49/200) |
| 11438611-3 | enwik8 char-LM (separate worktree) | — |
| 11438928-35 | **Mambino-GF, 40 ep, all 8 seeds** | ~1 day |

**The bar: 0.6089** (Pure S5, 8-seed) at **66% of its parameters**.

---

## 5. Naming rules going forward

1. **Cluster A/B are workstreams, never model names.**
2. **Never use "surprise" to distinguish variants** — all of them are surprise-driven.
   Say **adaptive** vs **fixed** write gate.
3. **Name the gate's target**: "SWG" (state write) or "FWM write gate". Never bare "the gate".
4. **One guiding metric** (`test@peakval`) applied to every config; `test_max` is
   context only, never an argument.
5. **Always state n** when quoting a mean, and prefer 8-seed over subsets.
