# Char-LM: plan + exposition (Mambino 2.0 → multi-task, the ICLR extension)

**Purpose.** Add a character-level LM (text8) as the second task, turning the single-task
surprise-gate result into a multi-task ICLR submission. This memo is the end-to-end plan
(what to build, in what order, with what controls) AND the exposition (how it slots into the
paper narrative). Grounded in the actual S5-fork code (`~/dev/ssm-baselines/S5`).

**Honest framing — this is a genuine test, not a formality.** The gate MIGHT not transfer:
ListOps is discrete symbolic reasoning; char-LM is statistical next-symbol prediction. Both
outcomes are publishable — transfer ⇒ "the mechanism generalizes"; no transfer ⇒ "reasoning-
specific," an honest narrowing of the claim. Design so BOTH are clean and interpretable.

---

## 0. What the char-LM buys the paper (the reviewer gaps it closes)

The current story (Mambino 2.0 on LRA-ListOps): a ~16-param predictive-coding gate that reaches
statistical parity with the heavier S5-Dense at fewer params, beats Pure-S5-gelu by +1.33 pp
(7/8 seeds), with a seed-stable, vocab-verified surprise-separation mechanism. Reviewer rejections
this directly answers:
- **"Single task."** text8 is a different modality (generative, causal, next-token) → generalization.
- **"Interpretability is confounded"** (operators are 2× rarer than digits, so "surprise = rarity?").
  Char-LM gives a CLEANER, frequency-controllable surprise story (Section 6).
- **"Where's the causal mechanism?"** The char-LM erase-ablation (Section 6) is the money figure.

---

## 1. Task / data / metric

- **Primary: enwik8** — 100 MB raw Wikipedia, **byte-level ~205-symbol vocab** (markup + case).
  Standard split **90M / 5M / 5M** (train/val/test). THE recognized char-LM architecture benchmark.
  **Why enwik8 not The Pile:** The Pile (825 GB) is pretraining-scale — comparable numbers need
  125M–2.8B-param models on ~300B tokens across 8+ A100s for weeks. On our **two A30s** that is
  infeasible AND the result would not be comparable to the literature. enwik8's 100M bytes fit the
  budget: 3–4 configs × 8 seeds of 1–4M-param models ≈ a few GPU-hours each → ~1–3 days on 2×A30.
- **Optional secondary: text8** — same data cleaned to lowercase a–z + space (**27-symbol vocab**).
  NOT the benchmark; keep only as a cleaner substrate for the word-boundary interpretability panel
  (no markup noise). Cheap once the plumbing exists. If skipped, run the boundary analysis on the
  letter-run spans of enwik8.
- **Metric: bits-per-character (BPC)** = test cross-entropy (nats) / ln 2, **val-BPC-selected**.
  Report **mean ± sd over the SAME 8 seeds** as ListOps (6554595/42/12345/271828/314159/1/2/3) so the
  matched-seed, paired-test rigor carries over.
- **Sequence length L:** start **1024** (fast smoke), then **2048** (matches ListOps), stretch 4096/8192
  (SSMs shine at long context). Non-overlapping chunks; **stateless chunks** at first (no state carried
  across chunks) — simplest and standard for an ablation.

---

## 2. Architectural delta from ListOps (this is the real engineering)

The classification path POOLS over time → one label/seq (`seq_model.py:209-221`). Char-LM needs a
**per-position causal head**:

1. **No pooling.** New `LMModel`: `StackedEncoderModel` → per-position `nn.Dense(vocab)` → log_softmax
   at EVERY t (drop `masked_meanpool`).
2. **Causal.** Set `bidirectional=False` (ListOps used True). MambinoSSM is then fully causal:
   predictor `x_hat=C_s@s(t-1)` is causal; the forward `associative_scan` is causal; the surprise-gate
   causal-EMA is causal. **The gate needs ZERO changes — it is already an online/causal operator.**
   (This is a genuine narrative plus: brains predict causally and online; bidirectional attention does
   not. The gate is natively autoregressive.)
3. **Norm: prefer layernorm (`batchnorm=False`).** S5's batchnorm normalizes per-feature over the batch
   (and possibly time); if it sees the whole chunk it leaks future statistics into each position at
   train time (a causal leak + train/test mismatch). Layernorm is per-position, causal-safe.
   **Decision D1:** verify `SequenceLayer`'s norm axis in `s5/layers.py`; default to layernorm for LM.
4. **Loss: per-position next-char CE.** `target[t]=input[t+1]`; predict from `output[t]`; mean CE over t
   (drop the last position). `BPC = mean_CE / ln 2`. New loss fn — the current `cross_entropy_loss`
   (`train_helpers.py:381`) is pooled/per-seq (`logits.shape[0]=n_classes`, scalar label).
5. **Embedding:** char id → one-hot(27) → `nn.Dense(d_model)` (reuses the existing `encoder` Dense) OR
   `nn.Embed`. One-hot+Dense reuses current code with zero new machinery.

---

## 3. Build list (minimal, kill-switched, all in git)

| # | file | change |
|---|---|---|
| A | `s5/dataloading.py` | `create_enwik8_lm_dataset` (+optional text8): download/cache, byte→id, split 90/5/5, chunk to L, return (x, y=x shifted +1). Register `"enwik8-lm"` (+`"text8-lm"`) in `Datasets`. padded=False, no lengths. |
| B | `s5/seq_model.py` | `LMModel` (+`BatchLMModel` vmap): encoder(no pool) + per-position decoder. |
| C | `s5/train_helpers.py` | `lm_cross_entropy` (per-position) + `validate_lm` → BPC. |
| D | `s5/train.py` | `--task=lm` branch: model_cls=BatchLMModel; loss=LM CE; metric/select on val BPC. Reuse HiPPO, opt (BfastandCdecay), MambinoSSM init. |
| E | `s5/mambino_ssm.py` | **UNCHANGED** — same gate, same `surprise_gate` flag; pass `bidirectional=False`. |
| F | `bin/gilbreth_text8_mambino_gelu_sgate.sh` | mirror the ListOps launch script; `SEED`/`GATE_RANGE` env. |

**Kill-switch discipline (hard rule):** `--surprise_gate=False` ⇒ byte-identical to gateless causal
Mambino; `--use_mambino_ssm=False` ⇒ plain causal S5. Ship a byte-equivalence test + checkpoint-compat
both ways, exactly as Cluster A did.

---

## 4. Ablation matrix (mirror ListOps exactly)

Same gelu head, causal, iso-param, 8 matched seeds:
- **S5-gelu (causal)** = config5 analog (vanilla S5 LM).
- **Mambino-gelu gateless (causal)** = config4 analog (+predictive coding, no gate).
- **Mambino 2.0 (causal, signed gate)** = config4 + 16-param gate.
- **+ unsigned gate** (does the erase/pop matter in the causal regime too?).

Primary comparisons (paired, per-seed): gate vs gateless (does +16 params lower BPC?); Mambino vs
vanilla S5; signed vs unsigned. Same statistics as ListOps (paired t, ddof=1, report p).

---

## 5. Baselines / positioning

**NOT a SOTA-BPC chase.** The internal iso-param, matched-seed ablation IS the result. Cite small-model
text8/enwik8 BPC for calibration only (a ~1–3M-param model should land ~1.3–1.5 BPC on text8 — sane, not
SOTA). Claim = "the gate transfers to generative LM," measured by the gate delta at iso-param.

---

## 6. Surprise interpretability on char-LM (the cleaner, confound-fixed story)

This is where char-LM BEATS ListOps as an interpretability substrate:
- **eps at linguistic structure:** char-level prediction error naturally spikes at WORD BOUNDARIES
  (space → first letter), rare characters (q, z, x), morpheme starts; and drops mid-word (predictable
  continuations). Measure per-position z; correlate with {is-word-start, char unigram freq, position-in-word}.
- **Frequency control (fixes the confound Fable flagged on ListOps):** regress z on char unigram
  frequency; show the RESIDUAL still tracks structure (word-start) ⇒ surprise ≠ mere rarity. Char-LM has
  many characters at each frequency, so this control is clean (impossible on ListOps' operator/digit split).
- **Causal erase-ablation (the money figure):** eval/retrain with `g` clamped ≥ 0 (kill the pop/erase) →
  does BPC rise? If yes, the SIGNED erase *causally* helps prediction ⇒ mechanism, not correlation. Run on
  BOTH text8 and ListOps for a two-task causal result.
- **Gate-behavior sweep** (erase% per layer, z-sd, seed-stability) reusing the `eps_diag.py`/`eps_sweep.py`
  machinery, adapted to per-position LM (no pooling, causal). Expect the causal gate to differ from the
  bidirectional ListOps gate — study it, don't assume it replicates.

---

## 7. Compute / timeline (Gilbreth)

enwik8 L=1024, d_model=128, n_layers=6–8, bsz ~32–64: ~a few GPU-hours/seed on an A30. Ablation = 3 configs
× 8 seeds × (signed + gateless) ≈ 24–48 runs; **two A30 nodes in parallel → ~1–3 days** wall-clock batched.
Env `/scratch/gilbreth/raisul/envs/s5m`; worktree-isolated; push from the local box (Gilbreth has no GitHub auth).

---

## 8. Open decisions / risks

- **D1** layernorm vs batchnorm (causal leak) → default layernorm; verify norm axis in `layers.py`.
- **D2** stateful vs stateless chunks → start stateless (simplest; matches the ablation's intent).
- **D3** L = 1024 first, then 2048.
- **D4 (scientific risk)** the gate may NOT transfer — it was validated bidirectional on reasoning; the
  causal/generative regime is genuinely different. Both outcomes are designed to be clean.
- **Risk** batchnorm running-stats + causal LM = subtle train/test mismatch; layernorm removes it.

---

## 9. Exposition — the paper narrative

**Title direction:** *"Prediction-Error Gating for State-Space Models: a 16-parameter, brain-inspired
mechanism for reasoning and language."*

**Arc.**
1. **Motivation (brain-inspired):** biological systems gate what they write to memory by *surprise*
   (prediction error) — write the surprising, discount the predictable. Formalize the roles surprise can
   play (the 4-roles lemma: signed first-moment → state/learning; unsigned second-moment → gain/segmentation;
   one scalar can't serve all four → a *signed* gate is the minimal object that does state+gain+segmentation).
   **Tighten this into a precise, provable statement — it is the paper's theoretical spine, not decoration.**
2. **Mechanism:** the ~16-param signed surprise gate on a predictive-coding SSM (MambinoSSM). Causal, online,
   negligible params/FLOPs. Kill-switch to the exact baseline.
3. **Result 1 — reasoning (LRA-ListOps):** parity with heavier S5-Dense at fewer params (statistically
   indistinguishable, p≈0.23); +1.33 pp over vanilla S5-gelu (7/8 seeds); signed > unsigned 8/8.
4. **Result 2 — language (text8, THIS memo):** the gate transfers (or the honest narrowing if not).
5. **Why it works (interpretability):** surprise separates informative from predictable symbols
   (frequency-controlled), and the causal erase-ablation shows the signed pop is load-bearing.
6. **Kicker:** the mechanism is hardware-cheap (ONE PPAC figure: +19% acc/mJ, 2× acc/area vs S5-Dense) —
   a bonus, not the thesis.

**What makes it ICLR (vs workshop):** multi-task (3+4) + a causal mechanism figure (6) + rigorous 4-roles
math (1). Those three are the delta from the current single-task state.

---

## 10. First actions (in order)

1. Build A–D (text8 dataloader, `LMModel`, LM loss, train.py `--task=lm`) with the kill-switch +
   byte-equivalence test. **Verify D1 (norm) first.**
2. Smoke: 1 seed, L=1024, **gateless causal S5** on an enwik8 slice → BPC descending toward ~1.3–1.5
   (small model). Proves the LM plumbing before touching Mambino.
3. Gateless causal Mambino, then Mambino 2.0 (signed), 1 seed → confirm the gate runs causally + BPC
   delta direction.
4. Scale to 8 seeds × {S5-gelu, Mambino-gateless, Mambino-2.0-signed, unsigned}.
5. Surprise diagnostics (adapt `eps_diag.py`) + the causal erase-ablation.

Related: [[cluster_a_surprise_gate]], [[mambino2p0_ppac_result]], [[cluster_b_fastweight_memo]]
(role-2 is orthogonal — could later stack on the LM too), [[mambino_paper_positioning]].
