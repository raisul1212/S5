# Mambino-LM — full design spec (Fable-audited)

**Audience:** the implementing agent. Self-contained. **Supersedes** `predictor_primary_pclm_design.md`
(that was the pre-audit sketch). **Status:** design only, ZERO code. Build strictly in the staged
order in §7; Stage 0 is a go/no-go gate — do not build the hierarchy until it passes.

Mambino-LM = a **hierarchical, dual-process, predictive-coding LM** that spawns from **Mambino-3-G**
(the *gated* Mambino — the surprise gate is inherited, not re-derived) and adds inference-time
learning. It is the char-LM redesign; it is NOT Mambino-3-G with more layers.

---

## 1. Provenance & the finding that forces this

- **Parent = Mambino-3-G:** a predictive-coding SSM (main SSM + predictor + `W̄_ε` error-write) with
  the **surprise gate** `g=tanh(κ·z+b)`, `z=EMA-norm ‖ε‖`, ε = prediction error. Proven on reasoning
  (ListOps: gate helps, signed>unsigned 8/8).
- **The finding (enwik8, iso-param ~795K, 50k steps, seed 42, causal), Test BPC:** Pure S5 (state
  160) **1.798 WINS** > Mambino gateless (state 64) 1.864 > Mambino-2.0 (+gate) 1.871; with the
  half_glu2 head, Pure-S5 1.778 > Mambino-r40 1.824. **Diagnosis:** on a *prediction* task the
  auxiliary predictor is REDUNDANT with the main SSM (two predictors split one budget, lose to one
  big SSM). Predictive-coding augmentation helps only when main-task ≠ prediction.
- **The fix is structural** (re-wire so there is ONE prediction path + a genuinely different memory),
  not a mixer/gate patch.

---

## 2. Thesis & honest novelty (read before believing the pitch)

**Primary novelty = hardware/efficiency co-design, NOT an ML-benchmark first.** The ML pieces mostly
exist in isolation: multiscale hierarchy = HM/Clockwork-RNN; confidence/surprise-gated skipping of
upper compute = **CALM early-exit (2022)**; gradient test-time updates = **TTT-layers (2024)**;
surprise-gated test-time memory = **Titans (2025)**. On a pure-ML axis Mambino-LM is a recombination,
and the frozen-vs-adaptive contrast is fair vs Mamba but NOT vs TTT/Titans — reviewers will say so.
**We do not lead with the ML claim.**

**What no one has built is the substrate-level version of this loop.** Mambino-LM is designed so every
mechanism is a *physical* lever on a compute-in-physics inference substrate
([[chip_pitch_substrate_not_architecture]], [[inference_is_the_chip_bottleneck]]):
- the **hard surprise gate α literally skips silicon** — an un-escalated token does not clock the
  upper level's array (energy *not spent*, not merely FLOPs not counted);
- the **slow-ticking upper level** (FIX #1) is a duty-cycled, low-toggle-rate block — the exact regime
  a mixed-signal/analog SSM tile is most efficient in;
- **inference-time learning migrates deliberation DOWN the hierarchy**, so energy-per-token *falls over
  a stream* — the chip does measurably less work the longer it runs on one source.

**The ML mechanism that DRIVES the energy story — the automation curve — is the supporting claim, not
the headline:**
> Inference-time learning at a *lower* level measurably *reduces* adaptive-compute escalation over a
> stream — skill migrates *down* the hierarchy at inference, and on the substrate that shows up as a
> **declining energy-per-token curve**.

Nobody has published that curve, and — the moat — nobody has published it **as a measured energy
result on a real inference substrate**. That pairing (a self-supervised migration mechanism + its PPAC
consequence) is what separates Mambino-LM from "TTT + Titans + ACT stacked." **Distinct from the
neighbors:** Titans = flat gradient-surprise memory, no hierarchy/escalation, no hardware; CALM =
confidence early-exit, no learning, no hardware; TTT = flat gradient state, no surprise gating, no
hardware. Mambino-LM = prediction-error surprise gating *hierarchical escalation* + test-time learning
that migrates deliberation downward, **quantified as energy-per-token on the Mambino substrate.** If
Stage 2 can't show the migration curve *with the controls in §8* AND its PPAC consequence, there is no
paper — treat that as the project's crux.

---

## 3. Architecture

### 3.1 Hierarchy = dual process (depth is the System-1→System-2 axis)
Levels ℓ = 0..L−1. Bottom (ℓ=0) = fast/automatic/local; higher = slow/deliberate/abstract. The
"System-2 memory that nudges level ℓ" IS level ℓ+1. Surprise (error) propagates UP; predictions
(nudges) propagate DOWN.

### 3.2 FIX #1 — slow ticking is a HARD invariant (else it's a deep SSM + overhead)
A deep SSM already self-organizes a fast→slow timescale hierarchy (our own v0.4-pro τ result);
residuals already carry "what lower layers couldn't explain." So levels are non-redundant **only if
temporally separated.** Enforce: **level ℓ+1's *state update* fires at most every `τ_ℓ = s^ℓ`
tokens** (geometric stride, s≈4 to start), and its **input is the pooled prediction error of level ℓ
over that window** (`e_pool = pool(ε_ℓ, window=τ)`). Upper levels thus see a *subsampled, compressed*
stream a bigger state cannot represent — that is what the budget split must buy. (Stage-3 variant:
surprise-triggered ticking instead of fixed stride.)

### 3.3 FIX #2 — escalation gates on REALIZED TRAILING SURPRISE, never self-reported confidence
CE-confidence rewards sharpness not correctness, is controlled by the level that benefits from
inflating it, and test-time learning makes it *worse* (fits recent window → confidently wrong on
shift). Use the inherited Mambino gate on **realized** error: `z_ℓ(t) = EMA-norm ‖ε_ℓ(t−1..)‖`
(trailing, causal — post-hoc measurement, not a forecast). Escalate when recent tokens were
surprising. This is the Mambino-3-G surprise gate **promoted from gating the error-write to gating
deliberation depth** — the spine of the design, not a casualty.

**Stage-0-validated refinement (2026-08-02, see §7):** trailing surprise is the *primary* gate — on a
distribution shift it tracks the error jump (×13.5) while CE-confidence stays flat (×1.1, *confidently
wrong*), and within-regime confidence is *negatively* correlated with error (−0.43). CE-confidence is
**not useless**, but only as a **secondary, fine-grained signal once escalation is already on** (during
active adaptation it tracks the per-token error gradient better than trailing surprise, +0.81 vs +0.40,
because surprise lags). It must **never** be the primary escalation trigger. If Stage 1 wants a finer
in-escalation modulation, confidence may feed the *magnitude* β of an already-fired nudge — not α.

### 3.4 FIX #3 — the inference-time learner is a TTT/Titans-style *small* fast-weight module
Naive SGD on CE over a full SSM level at inference diverges, has no chunk-parallel form, and blows up
per-sequence memory. Restrict the learner to a **small linear fast-weight module** (state-sized, not
layer-sized), **meta-learned step size**, normalization, chunk-wise updates (the TTT machinery).
Crucially, its **inner loss is on the level's PRE-NUDGE logits** (so the bottom learns to predict on
its own and *internalizes* what the nudge gave it — distillation-in-time; on fused logits it never
internalizes and the loop is unstable).

### 3.5 Forward pass — 2-level instance (Stage 1/2)

```
# ---- Level 0 (bottom), every token t ----
z0(t)  = EMA_norm( ‖ε0(t−1)‖ )                       # trailing realized surprise (Mambino gate)
g0(t)  = sigmoid(κ0·z0(t) + b0)                       # surprise-gated write (inherited)
h0(t)  = Λ0·h0(t−1) + g0(t)·( B0·emb(x_t) )           # bottom memory
p0(t)  = C0·h0(t−1) [ + FastWeight_0(x_<t) ]          # PRE-NUDGE prediction; [ ] added in Stage 2 (TTT)

# ---- Level 1 (top), state updates only when t % s == 0 ----
if t % s == 0:
    e_pool   = pool( ε0 over last s tokens )          # surprise propagated up
    h1       = Λ1·h1 + B1·e_pool                      # coarse, slow memory
nudge(t) = C1·h1                                      # cheap read of current (possibly stale) top state

# ---- Escalation (adaptive compute) ----
α(t) = 1[ z0(t) > θ ]                                 # HARD gate on realized surprise (STE in training)
logits(t) = p0(t) + α(t)·β·nudge(t)                   # confident ⇒ skip the nudge entirely
```
Kill-switch: `α≡1, s=1, FastWeight off` ⇒ a plain 2-level SSM (byte-equivalence test).
`use_mambino_lm=off` ⇒ the existing `LMModel` path, bit-identical.

### 3.6 Stage-2 addition — the learner (the novelty)
```
# after observing x_t, update the bottom's fast weights on its PRE-NUDGE loss:
FastWeight_0 ← FastWeight_0 − η(meta) · ∇ CE( p0_prenudge(t), x_t )
```
As FastWeight_0 lowers p0's error → z0 falls → α falls → escalation falls → **automation**.

---

## 4. Training (FIX #5 — it is NOT plain next-token CE)

Honest loss (≥4 terms; the λ balance is real work, budget for it):
1. **Task CE:** `CE(logits(t), x_t)` — main objective.
2. **Per-level predictive loss (Rao-Ballard):** top predicts the pooled bottom error →
   `MSE/CE(nudge-target, e_pool)`. **This trains the top even when it is NOT escalated** — the fix
   for gradient starvation / MoE-collapse (α→0 starves the top). Do not omit it.
3. **Ponder / escalation cost:** penalize escalation frequency (compute). **Anneal in** (curriculum:
   α forced high early so the top trains; penalty ramps later). Fights term 2 by construction — the
   curriculum is what keeps them from deadlocking.
4. **TTT inner loss (Stage 2):** `CE(p0_prenudge, x_t)` drives the fast-weight update; the outer
   objective backprops *through* the inner steps (double loop — keep inner steps few, module small).

Known conflict, from our own logs: Cluster-A's detach arm died because "detach needs λ_pc>0 but
λ_pc>0 starves task." Terms 2/3 here are the same tension × #levels. Mitigation = the α-curriculum +
keeping per-level losses weak. Pre-register: a 4-term loss with ≥3 interacting λs.

---

## 5. Baseline to beat & metrics

**The arm to beat (Fable): a monolith S5 (or Mamba) + CALM-style confidence early-exit**, iso-param
AND iso-FLOP, on the **accuracy–compute Pareto**. If 2-level Mambino-LM can't beat that Pareto, stop.

**Headline plots (not BPC alone):**
- **Automation curve:** escalation-rate (mean α) vs. position/step over a structured stream — must
  *fall* as the bottom learns.
- **Compute–vs–BPC Pareto** vs. the CALM monolith.

---

## 6. FIX (compute accounting) — the net-positive inequality, computed BEFORE building at scale
TTT adds ~2–3× FLOPs at the **always-on bottom**. Adaptive compute is net-positive only if
`cost(bottom+TTT) + P(escalate)·cost(top) < cost(monolith+CALM)`. **Compute this in Stage 2 with
measured `P(escalate)` and module sizes.** If it's negative, the honest claim pivots from "cheaper"
to "adapts better at equal/again-measured compute" — decide the pitch by the number, don't assume the
FLOP win.

---

## 7. Staging (strict order; each stage gates the next)

- **Stage 0 — the calibration test (go/no-go, NO hierarchy/TTT code). ✅ PASS (2026-08-02).** On a synthetic
  repeated-motif stream with a **mid-stream distribution shift**, using existing Mambino/S5 code +
  one minimal linear fast-weight module: measure whether **(i) softmax-confidence** and **(ii)
  trailing-surprise-z** each track **realized** per-token error, *with and without* the test-time
  update. Predicted: confidence decouples from correctness under TTT; trailing-surprise does not.
  - **PASS** (surprise tracks error, confidence doesn't) → proceed, gate on surprise.
  - **BOTH decouple** → the escalation premise is dead; **do not build the hierarchy.**
  - **RESULT** (`paper_v2_ppac/mambino_lm/stage0_calibration.py`, CPU, GRU-LM stand-in, trigger→body grammar,
    mid-stream A→B regime shift, seed 0):
    - *Shift detection (frozen model — the load-bearing case):* at A→B, realized **error ×13.6**,
      **trailing-surprise ×13.5** (tracks), **confidence-difficulty ×1.1** (flat — confidently wrong).
      A confidence-gated escalation would NOT fire → stays wrong; a surprise-gated one fires → gets
      help. Exactly the Fable-predicted failure mode.
    - *Token-level miscalibration:* within regime B (frozen), confidence-difficulty is **negatively**
      correlated with error (Spearman −0.43) — the most-wrong tokens are the most confident.
    - *TTT works:* the linear fast-weight adapted at inference, cutting B-body error **8.03 → 0.83 (90%)**.
    - *Nuance → FIX #2 refinement:* once actively adapting, confidence tracks the per-token gradient
      better than trailing surprise (+0.81 vs +0.40, surprise lags) ⇒ **surprise = primary gate,
      confidence = secondary-only**.
    - *Limits:* GRU stand-in (not the SSM), synthetic grammar, one seed — the signal property is
      architecture-agnostic; **re-confirm on the real SSM at Stage 1.**
- **Stage 1 — 2 levels, NO learning.** Bottom + one slow-ticking top; surprise-gated escalation; show
  escalation works, confident tokens skip the top (adaptive compute), and it beats the CALM-monolith
  Pareto (§5). Kill-switch green first.
- **Stage 2 — add TTT to the bottom (the novelty demo).** Measure the **automation curve** with ALL
  controls in §8. Compute the §6 inequality.
- **Stage 3 — deepen** (>2 levels), cross-level learning, surprise-triggered ticking.

Each new task needs fresh baselines; paired-seed analysis; pre-register effect thresholds.

---

## 8. The controls the novelty claim REQUIRES (Fable) — without these, Stage 2 proves nothing
- **Frozen-bottom ablation:** escalation-rate must **NOT** fall when TTT is off (else "escalation
  falls" is just "loss falls on a structured stream" repackaged).
- **Calibration audit:** show escalation falls because **competence rose** (realized error dropped),
  not because a signal inflated — hence gating on realized surprise, not confidence.
- **Compute accounting:** the §6 inequality, measured — is the migrated skill retained *cheaper* than
  re-escalating each time?
- **Distribution-shift probe:** on a mid-stream shift, escalation must *rise* (deliberate on novelty)
  then *fall* (re-automate) — the signature of a continual learner.

---

## 9. Biggest risk & kill criteria
- **Load-bearing risk = the escalation gate.** Gate collapse (α→0, top starved, never consulted) or
  gate inflation (α→1, no adaptive compute) each kills the pitch. The Rao-Ballard per-level loss +
  α-curriculum + gating-on-realized-surprise are the three defenses; if the gate still collapses in
  Stage 1, the architecture is wrong, not under-tuned.
- **System-level recurrence of enwik8:** at iso-param/FLOP the whole apparatus fails to beat
  monolith+CALM on the Pareto → stop at Stage 1.
- **Automation curve is trivial** (falls with TTF off) → the novelty is unmeasurable → stop at Stage 2.

---

## 10. Kill-switch / reuse / repo (house rules)
- Flag `--mambino_lm=off|1level|2level` (+ `--mlm_ttt=on/off`, `--mlm_stride=s`, `--mlm_alpha_thresh`).
  `off` ⇒ existing `LMModel` bit-identical; α≡1,s=1,ttt=off ⇒ plain 2-level SSM (byte-equivalence test).
- Reuse: `_surprise_gate` (the escalation signal), the SSM scan primitives, the char-LM harness
  (`--task=lm`, `lm_train_epoch`/`lm_validate`, step schedule, `bin/gilbreth_enwik8.sh`), and the
  Cluster-B fast-weight kernel (`[[cluster_b_fastweight_memo]]`) as the Stage-1 cheap learner /
  reference for the TTT module.
- Worktree `v2/mambino-lm` off `v2/surprise-gate`; push from the local box (Gilbreth has no GitHub
  auth); env `/scratch/gilbreth/raisul/envs/s5m`. Fable-verify the double-loop gradient + the STE gate
  BEFORE training.
- **Causal-normalization prerequisite (Fable, char-LM audit):** any chip-noise / ADC / quantization
  eval on the LM path MUST normalize activations **causally** (running/trailing stats), NOT over the
  whole sequence — sequence-wide normalization at `bits>0` or `σ>0` leaks future statistics and is
  acausal. Fix this before ANY noise/PPAC-robustness number on Mambino-LM (the energy story of §2 dies
  if the noise eval is silently acausal).

---

## 11. Open decisions (raise with the author)
- **D1** stride `s` value + fixed-stride vs surprise-triggered ticking (start fixed s=4).
- **D2** the TTT module form (linear fast-weight à la Cluster-B `M_t` vs a tiny MLP) + the meta step-size.
- **D3** the STE for the hard α gate (straight-through vs Gumbel vs ACT ponder) — affects trainability.
- **D4** how many levels at Stage 3, and whether cross-level TTT (learning at every level) or bottom-only.
- **D5** the synthetic Stage-0 stream design (motif alphabet, shift magnitude) — must have automatable structure.

## 12. First actions
1. Re-read §2 (novelty) and §7 Stage 0. Hand Fable the Stage-0 protocol + the §5 baseline definition.
2. Build Stage 0 (calibration test) — it needs *no* new architecture, only existing code + one fast-weight module. Its outcome decides the whole project.
3. Only on PASS: Stage 1 (kill-switch + byte-equivalence green first), with the CALM-monolith baseline as arm (c).

Related: [[predictor_primary_pclm_design]] (superseded), [[cluster_a_surprise_gate]] (the inherited gate),
[[cluster_b_fastweight_memo]] (the fast-weight learner), [[mambino2p0_ppac_result]], [[charlm_plan]],
[[sony_award_proposal]] (adaptive-compute-on-sensor sibling), [[user_long_term_vision]],
[[feedback_always_have_fallback]], [[feedback_lean_by_default]].
