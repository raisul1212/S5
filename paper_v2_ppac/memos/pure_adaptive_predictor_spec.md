# Pure Adaptive Predictor (PAP) — design spec

**Audience:** the implementing agent. Self-contained. **Status:** design; the pivot after the Mambino-LM
hierarchy lost on enwik8. Supersedes the hierarchy direction of `mambino_lm_spec.md` for the LM line.

---

## 0. TL;DR
The divvied predictive-coding hierarchy LOST to Pure S5 on enwik8 (iso-param 795K: Pure S5 **1.8194**
beats every Mambino-LM variant ≥ +0.08 BPC, 2.5× faster). **PAP** is the pivot: take the Mambino
mechanisms and make them the **primary, undivided, FULL-capacity** model — the analog of Pure S5, at the
**same P = 80**, but as a **pure error-based predictor** with the two anchors below. Not a base + bolt-on;
not divvied. The bet: PAP is competitive on stationary text *and* adapts on a non-stationary stream, both
from the same undivided predictor.

## 1. The two anchors (non-negotiable)
1. **Surprise-gated integration** (in the recurrence):
   `h(t) = A·h(t-1) + g(‖ε(t)‖)·B·ε(t)` — the state moves only when surprised. ε≈0 ⇒ the state coasts ⇒
   **compute ∝ surprise**, intrinsic to the recurrence (not bolted on). `g` = the Mambino surprise gate
   `g=σ(κ·z+b)`, `z` = EMA-normalized ‖ε‖ (reuse Cluster-A `_surprise_gate`). **This gate is a
   NONLINEARITY**, so a surprise-gated error-integrator is NOT an S5 — it can compete on stationary text,
   unlike a linear predictor (which is algebraically S5).
2. **Error-driven inference-time learning**:
   ε drives a gradient update to a **small fast-weight** — the model **learns as it reads**. Always on
   (`|ΔW| ∝ ε` ⇒ intrinsically surprise-proportional, no separate gate to collapse). Reuse the Stage-0
   `grad_ce` pattern / [[cluster_b_fastweight_memo]].

## 2. Why (chain of results)
- char-LM + enwik8: PC-augmentation is redundant-to-harmful when main-task = prediction; the divvied
  hierarchy/predictor splits the budget and loses to one big S5 ([[mambino_lm_stage0_result]]).
- A *linear* pure predictor is algebraically an S5 (`h_t=(A−BC)h_{t-1}+Bx_t`) → capacity isn't the win.
- **Anchor 1's `g(‖ε‖)` supplies the nonlinearity** that lifts a FULL-capacity (P=80) pure predictor out
  of the S5 class; **Anchor 2** supplies stream adaptation. Undivided + full-size + both anchors is the
  design the negative results point to.

## 3. Architecture (per token t; causal LM: predict x_{t+1}). P = 80, undivided.
```
x̂_t      = C · h_{t-1}                          # predict current token (embedding) from state
ε_t      = embed(x_t) − x̂_t                     # prediction error (feature space)
z_t      = EMA_norm(‖ε_t‖)                       # normalized surprise (causal EMA)
g_t      = σ(κ·z_t + b)                           # surprise gate            [ANCHOR 1]
h_t      = a ⊙ h_{t-1} + g_t · (B · ε_t)         # surprise-GATED error integration, state P=80
logits_t = (W + W_fw_t) · h_t                    # readout; W_fw = the adapting fast-weight  [ANCHOR 2]
# after x_{t+1} is revealed (chunked), always-on:
W_fw ← W_fw − η · ∇_{W_fw} CE(logits_t, x_{t+1}) # error-driven inference-time learning
```
- Backbone (`A=a`, `B`, `C`, `W`, gate κ,b) pretrained offline on English; **W_fw resets per stream** and
  is the only thing that moves at inference (v0). Same conj-sym diagonal SSM machinery as S5 → P=80.
- Kill-switches: `g≡1` ⇒ ungated error-integrator; `W_fw≡0, η=0` ⇒ frozen predictor; `--pap=off` ⇒
  existing LM path bit-identical.

## 4. The two questions PAP must answer
- **T1 — stationary English (does anchor 1 make a full predictor competitive with S5?):** PAP with
  adaptation OFF vs Pure S5, iso-param P=80, enwik8 BPC. Hypothesis: the `g(‖ε‖)` nonlinearity ⇒ PAP ≈ or
  > S5 at equal capacity (and coasts cheaply when confident). If PAP << S5 here, anchor 1 isn't pulling
  weight — diagnose before streaming.
- **T2 — cross-lingual stream (does anchor 2 adapt?):** `English → L2 → English` with known boundaries.
  Frozen big S5 must crater at L2 and never recover; always-adapt PAP must recover. Headline = the
  **post-shift recovery curve** (BPC vs position after each boundary).

## 5. Baselines & metric
- **Frozen Pure S5** (checkpoint 11456233, P=80×8, 795K, 1.8194 English) — control; craters at L2, no recovery.
- **PAP, adaptation off** (anchor 1 only) — isolates the surprise-gated predictor.
- **PAP, full** (both anchors) — the proposal.
- Metrics: stationary BPC (T1); post-shift recovery curve + per-segment BPC (T2); ‖ΔW_fw‖ vs surprise and
  the coasting rate (mean g) = the compute-∝-surprise efficiency plot.

## 6. Staging (each gates the next)
- **A — data + streaming harness + control.** Build the L2 byte corpus + a streaming-eval loop (long
  contiguous stream, state carried across chunks, BPC-vs-position); run frozen big S5 → confirm it craters
  at L2, no recovery. (Reuse the checkpoint + LM path; no PAP code yet.)
- **B — PAP backbone (anchors 1), offline.** Implement the surprise-gated error-integrating predictor as a
  drop-in SSM (`--pap`, reuse the S5 conj-sym diagonal machinery + `_surprise_gate`); pretrain on English;
  **T1** vs Pure S5 at P=80 iso-param. Kill-switch + byte-equivalence green first.
- **C — anchor 2 (adaptation).** Add the always-on error-driven W_fw update (chunked); **T2** recovery
  curve vs frozen S5. Headline.
- **D — richer fast-weight** (delta-rule/Titans), longer/multi-shift streams, efficiency plots.

## 7. House rules
- Reuse: S5 conj-sym diagonal SSM + `--task=lm` harness; Cluster-A `_surprise_gate`; the frozen-S5
  checkpoint; Stage-0 `grad_ce`; Cluster-B chunk-scan (v1).
- Kill-switch `--pap=off` ⇒ existing LM path bit-identical; `g≡1` and `η=0` disable each anchor; byte-eq
  test; Fable-verify causality + the inference-update gradient BEFORE any cluster run
  ([[feedback_always_have_fallback]]).
- Causality: `x̂_t` reads `h_{t-1}` (no leak); the W_fw update at t uses only realized x_{≤t+1}.
- `S5-sgate` / `v2/surprise-gate`; env `/scratch/gilbreth/raisul/envs/s5m`; push from local.

## 8. Open knobs (defaults; revisit)
- L2 language: pick a strong, clean byte shift in Stage A. Update cadence: 128-byte chunks. η: small,
  tuned on a val stream. What-adapts: W_fw readout (v0). Backbone: P=80, layers to match S5's 795K budget.

## 9. Honest risks
- If T1 shows PAP << S5, anchor 1 (surprise-gating) doesn't add expressivity at capacity → the "not an S5"
  argument is only formal; reconsider before streaming.
- Online W_fw update instability → small η + chunking + adapt only W_fw in v0.
- "Adaptation beats frozen" may need STRONG shifts (cross-lingual) — characterize the shift-magnitude
  threshold; that's the finding, not a failure.

Related: [[mambino_lm_stage0_result]], [[mambino_lm_spec]] (superseded hierarchy), [[cluster_a_surprise_gate]]
(the reused gate), [[cluster_b_fastweight_memo]] (the fast-weight), [[user_long_term_vision]],
[[sony_award_proposal]], [[feedback_always_have_fallback]].
