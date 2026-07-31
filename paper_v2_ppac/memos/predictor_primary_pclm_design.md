# Predictor-Primary Mambino — a Predictive-Coding LM (PC-LM): design memo

**Audience:** a fresh implementing agent with NO context of this session. Self-contained.
**Status:** design only. ZERO code. This is a *re-wiring* of Mambino's existing blocks, not a new task.

---

## 0. TL;DR

On char-LM, standard Mambino (predictor as an *auxiliary* to a main-SSM LM) **loses to a plain
S5 at iso-param** — because on a *prediction* task the auxiliary predictor is **redundant** with
the main SSM (both predict the next token; they split one budget). This memo proposes flipping the
roles: **make the predictor the primary output (the LM), and repurpose the main SSM as a
surprise-gated memory.** Same building blocks (predictor, SSM, W_ε, surprise gate), different
connections, different task setup. The goal is a *predictive-coding language model* whose own
prediction error (a) gates what enters memory (adaptive compute) and (b) can drive inference-time
learning (the continual-learner north star) — not "lower BPC than S5," which is likely not where
the win is (see §6).

---

## 1. Why — the redundancy diagnosis (with the data)

Iso-param (~795K, 50k steps, seed 42, causal, enwik8) char-LM, **lower BPC = better**:
- **Pure S5 (gelu, state 160): Test 1.798** ← winner
- Mambino 2.0 (gelu, state 64 + gate): ~1.85
- Mambino gateless (gelu, state 64): 1.864

Standard Mambino wiring:
```
x → main SSM → y → head → predict next token        ← the LM lives in the main SSM
x → predictor → x̂ → ε → gates the write into main SSM ← a SECOND predictor
```
On **classification/reasoning** (ListOps) the main task ≠ prediction, so the predictor's ε is an
*orthogonal* signal and it HELPS (config4 gateless 0.5993 > config5 pure-S5 0.5917). On **char-LM
the main task IS prediction**, so the predictor duplicates the main SSM's job; splitting the budget
into two predictors loses to one big predictor (Pure S5). **This is the finding**: predictive-coding
augmentation helps when main-task ≠ prediction, and is redundant when main-task = prediction.

The fix is not tuning and not "asymmetric predictor" (a patch). It is to stop having two predictors.

---

## 2. Design principle — predictor primary

One prediction mechanism. The **predictor emits the next-token distribution (the LM output).** The
**main SSM becomes a memory** that the predictor reads. The **surprise ε = x − x̂ does real work**:
it gates what is written to memory (write surprising, skip predictable → adaptive compute), and
optionally drives inference-time updates. No redundant second predictor; the model *is* a
predictive coder.

---

## 3. Building blocks to REUSE (all already in `s5/mambino_ssm.py` / the char-LM harness)

- Diagonal SSM scan (`jax.lax.associative_scan(binary_operator, (Λ, u))`) — the memory.
- `W_eps_bar`, `B_bar`, `C_tilde`, `D`, the predictor SSM params (`Λ_s, B_s, C_s`) — repurpose.
- The surprise gate `_surprise_gate(eps_seq)`: `z = EMA-norm ||ε||`, `g = tanh/sigmoid(κz+b)`.
- The char-LM path: `LMModel`/`BatchLMModel` (per-position head, causal), `lm_train_epoch`,
  `lm_validate` (BPC), `prep_lm_batch`, `--task=lm`, step-based schedule (`--lm_steps`), the
  `bin/gilbreth_enwik8.sh` launcher. Kill-switch discipline is already the house style.
- Cluster B fast-weight kernel (`[[cluster_b_fastweight_memo]]`) for Formulation C.

**Nothing new to invent at the primitive level.** This is a `MambinoSSM.__call__` rewrite + a flag.

---

## 4. Formulations (implement A first; B, C are extensions)

Notation per position t; `emb(x_t)` = token embedding (H-dim); `h` = memory state.

### A — Self-Predictive Surprise-Gated SSM (SP-SSM). START HERE.
Collapse to ONE SSM that predicts itself and gates its own write:
```
logits(t) = Head( C̃ · h(t-1) )               # predict token t from memory-BEFORE-t  (this IS the LM)
ε(t)      = surprise(logits(t), x(t))         # e.g. 1 - p(x_t), or emb-space residual (see D2)
g(t)      = sigmoid(κ·z(t) + b)               # z = causal-EMA-norm ||ε||   (reuse _surprise_gate)
h(t)      = Λ̄ · h(t-1) + g(t)·( B̄ · emb(x_t) ) # SURPRISE-GATED write: skip predictable tokens
loss      = mean_t CE(logits(t), x_t)
```
This is essentially **Pure S5 + a surprise-gated input write** (~16 extra params). Minimal, clean,
and it directly tests "does self-surprise-gating help / enable adaptive compute." Kill-switch:
`g≡1` ⇒ exactly Pure S5.

### B — Predictor + Memory (uses the second SSM as a richer readout)
Keep a separate, more expressive predictor that reads the memory (linear C̃ may be too weak):
```
h(t)      = Λ̄ · h(t-1) + g(t)·( B̄ · emb(x_t) )      # surprise-gated memory (main SSM)
x̂(t)      = Predictor( C_s · h(t-1) )  → logits(t)   # predictor SSM/MLP reads memory → next token
ε(t)      = surprise(logits(t), x(t)) ;  g(t)=gate(||ε||)
loss      = mean_t CE(logits(t), x_t)
```
Now the predictor is the primary output and the main SSM is pure memory — the redundancy is gone
(one prediction path). Extra params go to the predictor's expressiveness, not a duplicate LM.

### C — + Inference-time learning (the north star; reuse Cluster B)
Let ε drive **fast-weight updates to the predictor at inference** (`M_t = γM_{t-1}+η·s·(v⊗k)`,
read `o_t = M_t q_t`, added into the predictor). The model then *adapts as it reads* — an
unsupervised, self-supervised continual learner. This is the fundamentally new capability; scope it
after A/B work. See `[[cluster_b_fastweight_memo]]` for the locked mechanism and the JAX chunk-scan.

---

## 5. Key design decisions (raise with the user / Fable-verify)

- **D1 — one SSM (A) or two (B)?** A is the cleanest test and the honest baseline for "does the gate
  help LM at all." B is where the predictor can specialize. Do A first; only add B if A shows the
  gate is neutral-or-helpful and you want headroom.
- **D2 — what is ε for an LM?** Options: `1 − p(x_t)` (probability the model assigned to the truth),
  the CE loss itself, or an embedding-space residual `emb(x_t) − x̂_emb`. The gate needs a *magnitude*;
  `1 − p(x_t)` is bounded [0,1] and interpretable. Pick one and be consistent; the EMA-normaliser
  makes the scale forgiving.
- **D3 — read h(t-1) (prior) vs h(t) (posterior) for the LM output?** Predicting token t must read
  memory that does NOT contain x_t → **h(t-1)** (causal, no leak). This is the whole point (predict
  before seeing). Double-check no path leaks x_t into logits(t).
- **D4 — gate polarity.** Write-strength is a *magnitude* ⇒ unsigned `sigmoid` is natural (0=skip,
  1=write). A signed gate (erase) is possible but less obviously meaningful for a write. Start unsigned.
- **D5 — per-layer or global memory?** Start per-layer (matches the stacked SSM). Hierarchical
  predictive coding (each layer predicts the next layer's input) is a richer variant — later.

---

## 6. Success criteria — HONEST (do NOT set "beat Pure S5 BPC" as the bar)

Formulation A is ~Pure S5 + a gate, so it may **not** lower BPC — and that's fine. The wins to
measure, in priority order:
1. **Adaptive compute:** the gate lets you *skip the write (and downstream compute) on predictable
   tokens*. Headline metric = **BPC vs. fraction-of-tokens-written** (or FLOPs/token). If you can hit
   ~Pure-S5 BPC while writing/computing on far fewer tokens, that is the result (and it is the Sony
   on-sensor angle in a generative setting).
2. **Interpretability:** the surprise = the model's own uncertainty; show it spikes at word/morpheme
   boundaries (cleaner than ListOps' frequency-confounded operators).
3. **Inference-time adaptation (C):** BPC on a shifted corpus, before vs after on-device adaptation.
4. **BPC parity or better** is a *bonus*, not the thesis. If A *does* beat Pure S5, great; if it ties
   at much lower compute, better.

Pre-register this framing so a null BPC result is "adaptive-compute at parity," not a failure.

---

## 7. Minimal experiment (mirror the current char-LM Stage-1)

Same harness, iso-param ~795K, d=256, 8 layers, L=1024, causal, 50k steps, seed 42 (then 8 seeds):
- **Baseline:** Pure S5 (already have: Test 1.798).
- **A:** SP-SSM (Pure S5 + surprise-gated write). Report BPC AND the BPC-vs-write-fraction curve.
- Kill-switch check: `g≡1` reproduces Pure S5 bit-for-bit.
- Ablate: gated-write ON vs OFF; the write-fraction knob (threshold on g).
Then B, then C. Each new task/size needs fresh 8-seed baselines (paired stats, as in ListOps).

---

## 8. Kill-switch / fallback / reuse discipline (house rule — non-negotiable)

- CLI flag (e.g. `--pc_lm=off|A|B|C`); `off` ⇒ byte-identical to the current Pure-S5/`LMModel` path.
- `g≡1` in A ⇒ byte-identical Pure S5 (byte-equivalence test committed).
- Per-component disable (gate off, memory-only, predictor-off) + checkpoint compat both ways.
- Fable-verify the causality (D3, no x_t leak) and the gradient path BEFORE training.
- Build on a worktree off `v2/surprise-gate` (e.g. `v2/pc-lm`); push from the local box (Gilbreth
  has no GitHub auth); env `/scratch/gilbreth/raisul/envs/s5m`.

---

## 9. Why this matters (the bigger picture)

- **Turns the char-LM "loss" into the motivation.** The redundancy finding *is* a contribution, and
  the flip is a principled, brain-aligned response — a predictive-coding LM where prediction, surprise,
  and memory are explicit and interacting, vs. a standard LM where prediction is implicit.
- **Feeds the north star** (`[[user_long_term_vision]]`): unsupervised/self-supervised learning by
  prediction + inference-time adaptation = autonomous continual learner.
- **Feeds the Sony proposal** (`[[sony_award_proposal]]`): surprise-gated *write/compute* is the
  adaptive-compute mechanism in a generative setting — the same "spend compute on the surprising"
  story, now with an explicit self-generated surprise signal.
- **Position vs prior art:** predictive-coding networks / PredNet-style models exist; the novelty is
  the **SSM + self-surprise-gate + inference-time-learning** instantiation. Cite, don't claim first PC-LM.

---

## 10. Code pointers
- `s5/mambino_ssm.py` — `MambinoSSM.__call__` (~L560), `_apply_main_scan_with_additive_pc` (~L483),
  `_surprise_gate` (~L357), the predictor scan. This is what you re-wire.
- `s5/seq_model.py` — `LMModel`/`BatchLMModel` (the causal per-position head).
- `s5/train_helpers.py` — `lm_train_epoch`/`lm_validate`/`lm_cross_entropy`/`prep_lm_batch`,
  `make_warmup_cosine` (step schedule).
- `s5/train.py` — the `--task=lm` branch. `run_train.py` — LM CLI flags. `bin/gilbreth_enwik8.sh` — launcher.
- Char-LM results + iso-param design context: `[[charlm_plan]]`, `[[mambino2p0_ppac_result]]`.

## 11. First actions
1. Re-read §1–§6. Hand Fable Formulation A's equations + the causality claim (D3) for a go/no-go.
2. Add `--pc_lm` flag + Formulation A in a `MambinoSSM`-style module; byte-equivalence (`g≡1`==Pure S5) test green first.
3. Run A vs Pure S5 at iso-param; report **BPC and BPC-vs-write-fraction**, not BPC alone.
4. Only then B, then C.

Related: [[charlm_plan]], [[mambino2p0_ppac_result]], [[cluster_b_fastweight_memo]], [[cluster_a_surprise_gate]], [[sony_award_proposal]], [[user_long_term_vision]], [[feedback_always_have_fallback]], [[feedback_lean_by_default]].
