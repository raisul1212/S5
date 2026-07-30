# Cluster B — Implementation Plan (role 2: inference-time learning / fast weights)

**Written:** 2026-07-30. **Spec it implements:** `cluster_b_fastweight_role2.md` (same directory).
**Base:** branch `v2/surprise-gate` @ `36eebe3` (Cluster A committed + 8-seed ACC landed).
**Status:** PLAN, Fable-audited, consolidated. No code written yet.

---

## 1. What we are doing

Adding a **surprise-gated fast-weight associative memory** to `MambinoSSM` as a *parallel
additive branch*, so the layer can **learn within a single forward pass** — write key→value
associations into a matrix state `M_t` during inference and read them back later in the same
sequence, with both the *content* and the *write strength* driven by the prediction error the
predictor branch already computes.

The recurrence (B0, the first build) — post-audit, theory-pure form:

```
k_t = norm(W_k x_t),  q_t = norm(W_q x_t)            H -> d   shared address space
v_t = norm(W_v eps_t)                                H -> d   SIGNED content (the first moment)
s_t = sigmoid(kappa_B z_t + b_B)                              UNSIGNED write strength (magnitude)
M_t = gamma * M_{t-1} + (1 - gamma) * s_t * (v_t k_t^T)       M in R^{d x d}, M_0 = 0
o_t = M_{t-1} q_t                                             EXCLUSIVE read (read before write)
y   = ys + D*x + W_o o_t                             attach in __call__, NOT in the scan
```

`M_t` is **state, not parameters** — created and destroyed within one forward pass. That is the
whole point: the weights that matter at token `t` were written by tokens `1..t-1` of *this*
sequence, not by SGD.

**Why the recurrence looks like this** — five decisions, each load-bearing:

1. **`v` from `eps`, `k`/`q` from `x`.** The 4-roles theorem gives role 2 the *signed first
   moment* for its **direction** and the *magnitude* for its **write strength**. Sourcing `v`
   from eps supplies the first; the gate supplies the second. `k`/`q` stay on `x` so the write
   address and the read query live in a **shared, anchored space** — if both came from eps
   (the memo's literal reading), lookups would only hit when surprise *patterns* recur, which is
   a much weaker retrieval condition. Semantically the memory then holds: *at surprising moments,
   store the residual, addressed by the current input; when a similar input recurs, retrieve what
   surprised you then.*
2. **`v` is L2-normalized too, not just `q`/`k`.** Without this, `||v|| ∝ ||eps||`, so surprise
   would enter the write **twice** — once through the gate and once through the content magnitude.
   That would blunt the load-bearing ablation (arm 3 freezes the gate's z-dependence, but the
   write would still be surprise-modulated through `||v||`). Normalizing `v` puts the *direction*
   of eps in the content and the *magnitude* entirely in the gate — exactly the theorem's
   decomposition, cleanly separated. It also makes `||M|| <= 1` structural rather than hoped-for.
3. **`(1-gamma)` coupling instead of a free `eta`.** Makes `M` a bounded EMA, decoupling memory
   length from output magnitude, and removes the `eta` x `W_o` scale redundancy that interacts
   badly with zero-init. One scale on the path, not three. Mirrors the `a`/`(1-a)` convention
   `_surprise_gate`'s EMA already uses.
4. **Exclusive read `M_{t-1}`.** With an inclusive read the token reads its own write, and the
   layer can learn `W_q ≈ W_k` so `k_t·q_t ≈ 1`, degenerating into a surprise-gated *instantaneous*
   rank-d path that uses no memory at all. Still causal, so no test would have caught it — but an
   arm-2 win would have been a gated local feature, not learning. Read-before-write removes the
   shortcut, matches the delta-rule convention, and mirrors the predictor's existing causal shift
   (`s_fwd_shifted`, line 446).
5. **Attached in `__call__`, not folded into `Bu_elements`.** Keeps the fast weight a *sibling*
   of the main scan rather than a predecessor — see §1b.

B1 (later, only after B0 runs end-to-end) upgrades the Hebbian write to the **delta rule**
`M_t = M_{t-1}(I - beta_t k_t k_t^T) + beta_t (s_t v_t) k_t^T`, an error-correcting write that
removes whatever was previously stored along `k_t` before writing the new value.

---

## 1b. Block diagram (post-Cluster-B layer)

```
 input (L,H) ──► [DAC] ──► x ═══════════════════════════════════════════╗
                            ║                                    (x bus)║
 ┌──────────────────────────▼──────────────────────────────────────┐    ║
 │ STAGE 1   PREDICTOR                              consumes: x    │    ║
 │   Bs_bar·x → <SCAN 1> s(t)=Ls_bar·s(t-1)+Bs_bar·x(t) [fwd only] │    ║
 │   causal shift s(t-1) → Cs → x_hat                              │    ║
 └──────────────────────────┬──────────────────────────────────────┘    ║
                            ▼                                           ║
            eps = x - x_hat  (L,H) ══════════════════════════╗          ║
                            │                        (eps bus)║         ║
              ┌─────────────▼──────────────┐                  ║         ║
              │ z(t) = (||eps|| - mu)/sigma│  EMA, 2 tiny scans║        ║
              └──────┬───────────────┬─────┘                  ║         ║
                     │               │                        ║         ║
        g = tanh(kA·z + bA)     s = sigmoid(kB·z + bB)         ║         ║
          SIGNED - roles 3+4        UNSIGNED - role-2          ║         ║
          on the role-1 write       write strength             ║         ║
                     │               │                        ║         ║
 ┌───────────────────▼───────────┐  ┌▼────────────────────────▼─────────▼───┐
 │ STAGE 2a  MAIN SSM            │  │ STAGE 2b  FAST WEIGHT       * NEW *   │
 │  consumes: x, eps, g          │  │  consumes: x, eps, s                  │
 │                               │  │                                       │
 │  ROLE 1  =  Pi·eps, Pi = Weps │  │  ROLE 2                               │
 │  u = B_bar·x + g·(Weps·eps)   │  │   k = norm(W_k·x)   ┐ shared address   │
 │  <SCAN 2> fwd  <SCAN 3> bwd   │  │   q = norm(W_q·x)   ┘ space + query    │
 │  ys = C_tilde·[h_fwd | h_bwd] │  │   v = norm(W_v·eps) ← SIGNED content   │
 │                               │  │                                       │
 │                               │  │  M_t = g·M_{t-1} + (1-g)·s·(v k^T)    │
 │                               │  │        <SCAN 4>      state d x d      │
 │                               │  │  o_t = M_{t-1}·q_t   (exclusive read) │
 └───────────────┬───────────────┘  └────┬──────────────────────────────────┘
                 │ ys (L,H)              │ o (L,d) ──► W_o ──► (L,H)
                 └───────────┬───────────┘
                             ▼
                    ys + D·x + W_o·o  ──► [ADC] ──► output (L,H)
```

`eps` fans out **three** ways: the `z` statistic (both gates), the main scan's `W_eps·eps` term
(role 1, present since v1), and now `W_v` (role 2 content). `x` feeds the predictor, `B_bar·x`,
the k/q addresses, and the `D·x` feedthrough.

**Scan inventory.** Today (bidirectional main, gate on, noise off) the layer already runs
**3 substantive scans** — predictor fwd, main fwd, main bwd — plus 2 scalar-width EMA scans
inside `_surprise_gate`. Cluster B makes it **4 substantive**. All four are the *same monoid*
(`binary_operator`, `(a,b) -> (a_j a_i, a_j b_i + b_j)`); they differ only in width and in
whether the transition is complex-diagonal (`Lambda_bar`, P lanes) or real-scalar (`gamma`,
d^2 lanes).

**Dependency depth is UNCHANGED — the fast weight is a sibling of the main scan, not a
successor.** Both STAGE 2a and 2b consume `(x, eps)`; neither consumes the other; their outputs
meet only at the final sum. The critical path stays two scan-stages deep:

```
    STAGE 1 (predictor)  ──►  { SCAN 2 || SCAN 3 || SCAN 4 }  ──►  sum
```

Cluster B therefore adds **width, not depth**: +22% MACs at d=8 that can be hidden behind the
main scan given the silicon, or time-multiplexed onto it for zero extra area at +22% latency —
the same serial-vs-concurrent knob `ppac_v5_scheduler.barrier_point(..., concurrent_predictor=)`
already models for the predictor.

---

## 2. Why we are doing it

**Scientific reason (the theorem).** Surprise enters the update math in exactly two algebraic
forms: a signed first moment `Pi * eps` (a direction) and an unsigned second moment `eps^2`
(a magnitude). Cluster A's scalar gate multiplies the write, so it can express magnitude
(roles 3 = adaptive gain, 4 = segmentation) and a degenerate 1-D push/pop (role 1 = state write).
It **provably cannot** express role 2, because role 2 requires *accumulating an outer product
into a matrix state that is read back later by a query* — a rank-growing associative memory. No
per-token scalar creates that. Role 2 is unreachable by tuning Cluster A.

Note what is already present: `W_eps·eps` in the main scan input **is** `Pi·eps`, the signed
first moment — role 1 proper, there since v1. Cluster A added roles 3+4 on top of it. Role 2 is
the only one still missing its vector-valued surprise path.

**Strategic reason (the north star).** Inference-time learning is the project's stated long-term
goal — KV-cache-free LLMs and autonomous continual learners. Role 2 is the first concrete,
measurable instance of it in this codebase. Everything so far is train-time learning with a
clever inductive bias.

**Empirical reason (there is headroom).** Cluster A's 8-seed result closed most but not all of
the gap:

| model | params | acc (test@peakval) | n |
|---|---:|---:|---:|
| Config 4 Mambino gelu (gateless baseline) | 105,738 | 0.5993 +- 0.0072 | 8 |
| **Mambino 2.0** (= Config 4 + signed surprise gate) | 105,754 | **0.6050 +- 0.0042** | 8 |
| Mambino 2.0, unsigned gate | 105,754 | 0.6015 | 8 |
| Corner 1 Pure S5 dense (our matched-recipe repro) | 188,490 | 0.6089 +- 0.0053 | 8 |
| Corner 3' Mambino low-rank | 188,682 | 0.6138 +- 0.0027 | 8 |

Cluster A bought **+0.57 pp for 16 parameters**, and is still **0.39 pp short** of the 188K
Pure-S5 baseline. Cluster B's job is to close and overshoot that gap — at ~124K.

---

## 2b. The parameter-efficiency framing (and an honest FYI on external counts)

### Where Cluster B lands relative to our own baselines

| variant | added | **total** | vs Corner 1 (188,490) |
|---|---:|---:|---:|
| shared d=4 | 8,600 | 114,338 | **-39.3%** |
| **shared d=8** (default) | 17,944 | **123,682** | **-34.4%** |
| shared d=16 | 38,936 | 144,674 | -23.2% |
| separate d=16 | 65,560 | 171,298 | -9.1% |
| shared d=32 | 90,136 | 195,874 | +3.9% — breaks it |

Everything through d=16, both projection variants, stays **under** S5-Dense. Only d=32 crosses.
So the "+17% parameter penalty" is a penalty relative to **Config 4**, not relative to the
comparison baseline — against Corner 1 we are 34% *smaller*.

**But the two comparisons must not launder each other.** Being under S5-Dense makes the
*positioning* work. The *mechanism* question is against Config 4 at 105,738, and that is where
the +17% actually lives. It is settled by the controls in §8 (arms 3, 4, 5, 5b), not by the
table above.

### VERIFIED — the S5 authors' own LRA configs (computed 2026-07-30)

The S5 paper **reports no parameter counts for LRA** — only accuracies (confirmed against the
full text; the appendix has hyperparameters, no params column). But this repo is a **fork of the
S5 repo**, and the authors' own launch scripts ship in `bin/run_experiments/`. Parameter counts
below were computed by instantiating those exact configs
(`scratchpad/count_s5_lra_params.py`, `in_dim`/`n_classes` from `s5/dataloading.py`, complex
leaves counted as 2 reals per the repo's own `fn_is_complex` convention).

| S5 author config | d_model | layers | P | L | **params (computed)** | **acc (paper)** |
|---|---:|---:|---:|---:|---:|---:|
| `run_lra_listops.sh` | 128 | 8 | 8 | 2048 | **188,490** | 62.15% |
| `run_lra_imdb.sh` (Text) | 256 | 6 | 96 | 4096 | **1,321,154** | 89.31% |
| `run_lra_cifar.sh` (Image/sCIFAR) | 512 | 6 | 192 | 1024 | **5,133,706** | 88.00% |

Other reported S5 LRA accuracies: Retrieval 91.40%, Pathfinder 95.33%, Path-X 98.58%, avg 87.46%.

**The ListOps number is an exact hit: 188,490 = our Corner 1, to the parameter.** Corner 1 is not
a shrunken re-implementation — it *is* the S5 authors' own ListOps config, run under our
8-seed protocol. Two consequences:

- **The "our advantage is really just a smaller recipe" objection is dead.** An earlier draft of
  this section worried that Corner 1's 0.188 M vs a believed ~0.9 M S5 config meant part of any
  parameter win came from us choosing `d_model=128`. Verified false — 128 is *their* choice on
  this task. Every ListOps comparison in the paper is architecture-vs-architecture at the
  authors' own settings.
- **Cluster B at 123,682 is 34.4% smaller than the S5 authors' own ListOps model**, with no
  recipe confound to explain away.

**The stale landscape table is now known-unreliable and should not be cited.**
`~/dev/Mambino-journal-1/docs/lra_listops_landscape.md` §2 lists S5's ListOps config as
`d_model=192, ~0.9 M` — off by ~5x against the authors' actual script. Its other rows (Mega
~0.5 M, S4 ~0.7-1.0 M, LRU ~1 M, S7 ~1.1 M, Transformer ~6 M, Mamba ~10 M) carry the same
"approximate, from each paper's reported configs" caveat and are now **suspect by association**.
Keep them as rough orientation only; **any external parameter number that reaches paper text must
be re-derived from that paper's own config or repo**, exactly as was just done for S5.

### Reproduction note — the 62.15% is a single-seed, best-epoch number

Their script also publishes `--jax_seed=6554595` and `--epochs=40`, and 6554595 is the first seed
in our set — so we have a matched-seed reproduction of their exact config, at their own epoch
budget.

| Corner 1 (= their config) | at **their** seed 6554595 | best of our 8 seeds | 8-seed mean |
|---|---:|---:|---:|
| `test@peakval` (val-selected) | 0.6080 | 0.6200 (seed 42) | 0.6089 +- 0.0053 |
| `test_max` (best epoch) | 0.6115 | **0.6235** (seed 42) | 0.6159 +- 0.0046 |
| **S5 reported** | — | — | **0.6215** |

At their own published seed we land 1.0-1.35 pp short, but their number sits comfortably inside
our seed-42 range — our best-seed `test_max` (0.6235) exceeds it and our best-seed val-selected
(0.6200) is within 0.15 pp. That is consistent with a **single-seed, best-epoch** figure.

**This is an INFERENCE from the shape of the numbers, and it cannot be confirmed: S5 does not
document its reporting protocol** (checked; no protocol statement in the paper). Paper text must
therefore say "appears consistent with" and never assert what they did.

**Frame it as a reporting-convention difference, NOT as cherry-picking.** Single-seed best-epoch
reporting was the LRA norm in 2022-23, and there is a hard counterpoint to any accusation:
**Amos et al. 2024 independently retrained S5 and got 62.75%, above the paper's own 62.15%** — an
inflated number should not be beatable by an independent retraining. The likelier reading is a
convention difference plus our 40-epoch run of their script being under-trained relative to what
they and Amos achieved. An accusation in paper text would be attackable on exactly that point,
and the S5 authors are plausible reviewers.

**The stronger use of the same observation.** Under *their* convention (best seed, best epoch)
Corner 1 reports 62.35% and Corner 3' reports 62.35% — **exactly tied**. Under *ours*
(val-selected, 8-seed paired) Corner 3' 0.6138 beats Corner 1 0.6089 at p=0.022. The loose
convention **erases** a real architectural difference that the rigorous one recovers. That makes
the case for multi-seed val-selected reporting far more forcefully than any accusation, costs
nothing in reviewer goodwill, and is a standalone methodological contribution for an ICLR-style
paper. **State it about this specific pair, not as a general property** — see the next block,
where the same comparison is convention-*robust*.

### Cluster A under both conventions (pulled from Gilbreth 2026-07-30)

Per-seed signed-gate numbers, parsed from `run.log` across
`S5-sgate/checkpoints/...11423338` (seed 6554595) and `S5-sgate2/checkpoints/...s{42,12345,271828,314159,1,2,3}`.
Parse validated: reproduces the committed mean (0.6050) and sd (0.0042) exactly.

| seed | test@peakval | test_max |
|---|---:|---:|
| 6554595 *(S5's published seed)* | 0.6075 | 0.6075 |
| 42 | 0.5985 | 0.6070 |
| 12345 | 0.6015 | 0.6075 |
| 271828 | 0.6025 | 0.6035 |
| 314159 | 0.6115 | **0.6175** |
| 1 | 0.6085 | 0.6125 |
| 2 | 0.6045 | 0.6090 |
| 3 | 0.6055 | 0.6105 |
| **mean** | **0.6050 +- 0.0042** | 0.6094 |

| | val-selected 8-seed | best-seed best-epoch | params |
|---|---:|---:|---:|
| **Mambino 2.0 (Cluster A)** | 0.6050 | **0.6175** | 105,754 |
| Corner 1 (= S5's own config) | 0.6089 | 0.6235 | 188,490 |
| **S5 reported** | — | **0.6215** | 188,490 |
| Corner 3' (Mambino 188K) | 0.6138 | 0.6235 | 188,682 |

**Cluster A is 0.40 pp below S5's published 62.15% at 44% fewer parameters**, and its gap to
Corner 1 is **-0.39 pp val-selected / -0.40 pp best-seed-best-epoch** — essentially identical
under both conventions. So convention-sensitivity is **comparison-specific**: it flips the
Corner 1 vs Corner 3' verdict but leaves Cluster A vs Corner 1 unchanged. Do not over-generalize
the methodological point above.

**This gives Cluster B a convention-independent target: close ~0.4 pp.** Doing so puts Mambino at
parity with the S5 authors' own ListOps recipe at **123,682 vs 188,490 params (-34%)**, under
either reporting convention — a far more robust headline than one holding only under our
preferred metric.

None of this touches our claims: every load-bearing comparison (Corner 3' vs Corner 1, vs
Corner 2, Cluster A/B vs Config 4) is internal, matched-seed and paired. The external 62.15% is
context, never a comparator.

Two further caveats that survive verification:

1. **The LRA landscape is contested.** Amos et al. 2024 retrained the field and got Transformer to
   62.90% and S5 to 62.75%, arguing LRA is less SSM-favouring than originally assumed. Any
   LRA-based claim must engage with that rather than cite the 2021-2023 tables.
2. **The paper's locked positioning is "SOTA-close accuracy + real hardware efficiency," not
   SOTA-chase**, and §5g deliberately has no parameter column. A parameter-superlative is a
   **new claim surface** that would have to be defended against exactly the models the current
   framing sidesteps as chip-unfriendly.

### The ICLR read

The framing "**on-par accuracy at 4-7x fewer parameters than the competitive SSM band**" is
genuinely compelling and stands **without any chip content** — which makes a venue like ICLR
viable on the architecture story alone, with the PPAC work as a separate contribution rather
than a load-bearing one. What that route requires, and none of it is free:

- **verified external parameter counts** (caveat 1) — a real literature pass, not this table;
- **more than one task** — hence IMDB in §8; a single-benchmark parameter claim will not survive
  review;
- **mechanism attribution** — arms 3/4/5/5b, or the result reads as "adding params helps";
- **engagement with Amos 2024** (caveat 3).

Note also the upside case the current plan deliberately does *not* assume: **the accuracy jump
is unknown and could be much larger than budgeted.** The design is sized conservatively (d=8,
+17%) on lean-by-default grounds. If arm 2 shows a large effect, d=16 (144,674, still -23% vs
Corner 1) is one flag away and still under S5-Dense — so there is headroom to spend *without*
losing the parameter-efficiency framing. Do not lock d=8 before seeing arm 2.

---

## 3. What will change — file by file

Everything defaults **OFF**. With `fast_weight=False` the param tree and the forward pass are
byte-identical to today's `MambinoSSM`.

### 3.1 `s5/mambino_ssm.py` (the substance)

| Change | Where | Note |
|---|---|---|
| **Refactor** `_surprise_gate` -> extract pure `_surprise_z(eps_seq)` | ~line 357 | Pure code move: the EMA / bias-correction / z block becomes a helper; `_surprise_gate` then applies `tanh(kappa*z+bias)` on top. Traced computation unchanged => Cluster A stays bit-identical. Needed so the fast-weight gate reuses `z` with its **own** kappa/bias instead of hijacking Cluster A's. |
| New module fields `fast_weight` + `fw_*` | ~line 159, beside `surprise_gate` | All default to the no-op configuration. |
| Guarded param block `if self.fast_weight:` | `setup()`, ~line 347 | Mirrors the existing `if self.surprise_gate:` pattern so the tree is byte-identical when off. |
| New `_fw_gate(z)` | new | `sigmoid(fw_kappa*z + fw_bias)`; `fw_kappa` frozen at 0 in the control arm (§8). |
| New `_apply_fastweight(x_seq, eps_seq) -> (L,H)` | new | q/k/v from their configured sources, normalization, write gate, dispatch to an impl, exclusive shift, `W_o`. |
| New `_fw_seq / _fw_scan / _fw_chunk` | new | Three implementations of one recurrence — §5. |
| Hook into `__call__` | line 601 | `output = ys + Du` becomes `output = ys + Du + o_fw`, `o_fw = 0.0` when off. **After** the main scan and **outside** it. |
| `init_MambinoSSM` factory | line 616 | Thread every `fw_*` kwarg, defaults matching the module. |

### 3.2 `s5/train.py`

- Thread `fw_*` through **both** `init_MambinoSSM` call sites (~line 119, ~line 564) via
  `getattr(args, ..., default)`, exactly as the `gate_*` kwargs are threaded today.
- **Guardrails** beside the existing `gate_detach`/`lambda_pc` guard (line 104):
  - hard-error on `fw_rule="delta"` + `fw_impl="scan"` — the delta rule's data-dependent matrix
    transition is *not* an associative scan, and silently running it would compute a wrong answer;
  - hard-error on the compound frozen-predictor case (Fable Q4c): q/k/v sourced from a **detached**
    eps **and** the `W_eps` path disabled **and** `lambda_pc=0`.

### 3.3 `s5/train_helpers.py` — optimizer grouping

In the `BfastandCdecay` branch (line 299), the `map_nested_fn` key lists at lines 317-318 and
324-325 gain the fast-weight **scalars** — `fw_gamma_logit`, `fw_kappa`, `fw_bias` — so they join
`gate_kappa`/`gate_bias` in the `'ssm'` group (Adam, no weight decay, ssm_lr). The **projection
matrices** `W_q/W_k/W_v/W_o` are left unlisted and fall through to `'regular'` (AdamW,
weight-decayed, faster lr) — the same treatment `W_eps`, `B_s`, `C_s` already get.

### 3.4 `run_train.py`

New argparse flags mirroring `--surprise_gate`/`--gate_*` (lines 164-175):
`--fast_weight --fw_dim --fw_heads --fw_proj --fw_rule --fw_kq_source --fw_v_source --fw_impl
--fw_chunk --fw_gate_mode --fw_kappa_init --fw_bias_init --fw_gamma_init --fw_norm_qkv
--fw_out_init --fw_read`.

### 3.5 New files

- **`bin/test_cluster_b_fastweight.py`** — correctness gate (§7b), modeled on `bin/test_v2_gate.py`.
- **`bin/gilbreth_lra_listops_mambino_gelu_fw.sh`** — SLURM launcher modeled on
  `bin/gilbreth_lra_listops_mambino_gelu_sgate.sh`, with the `SEED` env parametrization already
  proven there (commit `f9c3808`).
- **`bin/gilbreth_lra_imdb_mambino_gelu_fw.sh`** — same for task 2 (§8).
- **this file**.

### 3.6 Explicitly NOT changed

`s5/ssm.py` (Pure S5), `s5/layers.py`, `s5/seq_model.py`, the PPAC pipeline
(`paper_v2_ppac/chip/*`), any v1 artifact, any frozen number.

---

## 4. Design decisions

**D1 — gate polarity: UNSIGNED (`sigmoid`) for the fast-weight write.** Role 2's write strength
is a *magnitude*; a negative write strength has no associative-memory meaning (storing a negated
value is the delta rule's job). Own `fw_kappa`/`fw_bias`, independent of Cluster A's signed
`tanh`. Empirical note: for the *state write* (Cluster A) signed beat unsigned, 0.6050 vs 0.6015
— evidence about role 1, not role 2, and the theorem predicts the roles want opposite polarities.

**D2 — q/k/v source: TWO orthogonal flags, defaulting to the theory-pure split.** The original
`fw_source` enum is replaced by `--fw_kq_source={x,eps}` (default **x**) and
`--fw_v_source={x,eps}` (default **eps**), so all four combinations are explicit and nothing is
hidden behind an opaque name. Rationale in §1 point 1. `fw_v_source=x` is retained as the
ablation that isolates whether surprise-as-content matters at all, beyond the gate.

**D3 — projection cost.** Dense `W_k/W_q/W_v` at 3H^2 = 49K/layer is ~4x the whole model — a
non-starter. Two lean variants behind `--fw_proj`:

- `separate`: `W_q, W_k, W_v` each `H x d`, plus `W_o` `d x H` = **4Hd** + 3 scalars
- `shared`: one down-projection `P` (`H x d`) reused for q/k/v via three `d x d` mixers, plus
  `W_o` (`d x H`) = **2Hd + 3d^2** + 3 scalars

| d | separate /layer | x8 | vs 105,738 | shared /layer | x8 | vs 105,738 |
|---:|---:|---:|---:|---:|---:|---:|
| 4  |  2,051 | 16,408 | +15.5% | 1,075 |  8,600 |  **+8.1%** |
| 8  |  4,099 | 32,792 | +31.0% | 2,243 | 17,944 | **+17.0%** |
| 16 |  8,195 | 65,560 | +62.0% | 4,867 | 38,936 | +36.8% |
| 32 | 16,387 |131,096 |+124.0% |11,267 | 90,136 | +85.2% |

**Default `fw_proj=shared`, `fw_dim=8`** (+17.0%, 123,682 total), with **d=4 (+8.1%)** as the lean
control. Fable's judgment: d=16 separate (+62%) and anything at d=32 forfeit the "cheap add-on"
framing; shared d ∈ {4, 8} is the defensible band. **But see §2b — d=16 shared (144,674) is still
23% under Corner 1, so it stays available if arm 2 shows a large effect. Do not lock d=8 before
seeing arm 2.**

**D4 — where to build B1: JAX, here.** The STP v4 Triton kernel is PyTorch and is an *algorithmic*
reference for the WY chunked form, not a drop-in. Paper numbers must come from this JAX line, and
a `jax.lax.scan` over chunks with dense intra-chunk matmuls is correct with no custom kernel.
Prove the science first; a Pallas port is optimization. B1 does not start until B0 runs
end-to-end.

**D5 — direction: forward-only.** The main scan is bidirectional but the predictor is not, and
"learning during inference" is inherently causal. Forward-only, matching the predictor.

**D6 — read convention: EXCLUSIVE, `o_t = M_{t-1} q_t`.** §1 point 4. Exposed as
`--fw_read=exclusive|inclusive` so the shortcut hypothesis is directly testable, but exclusive is
the default and the one the claim rests on.

**D7 — normalization: `q`, `k` AND `v`.** §1 point 2. `--fw_norm_qkv` (default True).

---

## 5. The scan problem, and why there are three implementations

B0's transition coefficient `gamma` is a **scalar**, constant in `t` and identical for every entry
of `M`. So `M_t = gamma*M_{t-1} + W_t` decomposes into `d^2` independent scalar linear recurrences
— the same primitive the SSM scan already uses. Flattening `M` to length `d^2` and calling
`jax.lax.associative_scan(binary_operator, (A, W))` with `A` filled with `gamma` therefore gives
exactly `M_t`. **Audit verdict: CORRECT**, with associativity verified explicitly and two caveats:
it relies on `M_0 = 0` (a carried initial state would need `gamma^t * M_0` injected manually —
matters if streaming inference is ever added), and the scan is inclusive, so the exclusive read is
produced by *shifting the output*, never by changing the scan.

| impl | what | cost | role |
|---|---|---|---|
| `seq` | `jax.lax.scan`, one step at a time | O(L) sequential, O(d^2) memory | **ground truth**; reference for the equivalence test |
| `scan` | `associative_scan` over flattened `M` | parallel, **O(L*d^2) memory** | simplest parallel version; fine at small d |
| `chunk` | chunk L into blocks of C, explicit `C x C` decay matrix, `lax.scan` over chunks carrying `S` | parallel, O(L*d + N*d^2) memory | **the one that scales**, required for B1 |

Memory is why `chunk` exists. At batch 50, L=2048, `scan` materializes `50*2048*d^2*4` bytes per
layer per array: 26 MB/layer at d=8, 105 MB/layer at d=16 (~839 MB across 8 layers for one copy,
several GB once autodiff holds its tape). At L=4096 (IMDB) both double.

The chunked decay matrix is built **explicitly** as `D[j,j'] = gamma^(j-j')`, masked. The usual
"rescale k by `gamma^(-j')`" trick is **banned** — the audit quantified it: fp32 precision
collapses once `gamma^(-C) > 1/eps`, i.e. at **gamma < 0.878 for C=128** (gamma < 0.771 at C=64),
and in bf16 it breaks at gamma < 0.963 at C=128 — essentially always.

**Exclusive-read correction to the chunked form:** the D matrix is *not* reusable unmodified.
Inclusive uses `D[j,j'] = gamma^(j-j')` for `j >= j'` (unit diagonal) with carry exponent `j`;
exclusive uses **zero diagonal**, `j > j'`, and carry exponent **`j-1`**. Getting this wrong is a
silent off-by-one that leaks the current token's own write — precisely the shortcut D6 exists to
remove. Tests 2 and 6 (§7b) catch it.

B1's delta rule has a **data-dependent matrix** transition and is NOT an associative scan — hence
the `train.py` guardrail.

---

## 6. Budgets

**Fast-weight state** (`M`, per layer, single head) — activation memory, not parameters:

| d | entries | FP32/layer | x8 layers FP32 | INT8 x8 | fits 64 KB state SRAM? |
|---:|---:|---:|---:|---:|:--:|
| 4   |     16 |    64 B |  512 B | 128 B | yes |
| 8   |     64 |   256 B |   2 KB | 512 B | yes |
| 16  |    256 | 1,024 B |   8 KB |   2 KB | yes |
| 32  |  1,024 |   4 KB  |  32 KB |   8 KB | yes (tight alongside SSM state) |
| 128 (un-projected H x H) | 16,384 | 64 KB | 512 KB | 128 KB | **no — 8x over budget** |

State is a non-issue at any sane `d`. It is the **parameters** that constrain the design. The last
row is the quantitative reason an un-projected `H x H` fast weight was never an option.

**Added compute**, per token per layer: `4Hd + 2d^2` MACs. Against Config 4's 307.2 M real MACs
for a whole L=2048 x 8-layer inference:

| d | MACs/token/layer | added MACs | vs 307.2 M baseline |
|---:|---:|---:|---:|
| 4  | 2,080 |  34.1 M | +11.1% |
| 8  | 4,224 |  69.2 M | **+22.5%** |
| 16 | 8,704 | 142.6 M | +46.4% |

**State this plainly in any writeup:** Cluster A was +0.57 pp at *zero* silicon. Cluster B is not
— at d=8 it is roughly +22% MACs on top of +17% parameters. PPAC is sequenced last because there
is no point costing a mechanism that has not yet earned its accuracy.

---

## 7. Fallback and kill-switch discipline (hard rule)

1. **CLI kill-switch.** `--fast_weight=False` (default) => param tree and forward pass
   **byte-identical** to today. Asserted by test, not inspection.
2. **Soft kill-switch.** `W_o` is **zero-initialized**, so even with `fast_weight=True` the branch
   contributes exactly 0 at step 0 and training *starts* at the baseline, then learns its way in.
   The audit confirmed this is not a dead-gradient trap: `dL/dW_o != 0` provided `o_t` is not
   identically zero, so `W_o` escapes on the first update and every other gradient turns on behind
   it. It also identified a real double-zero trap — zero-init `W_o` *and* a zero-init `eta` would
   be permanently dead — which deleting `eta` (§1 point 3) removes structurally.
3. **Per-component disable.** `--fw_gate_mode=off|const|surprise`; `--fw_rule=hebb`;
   `--fw_read=inclusive`; `--fw_norm_qkv=False`; `--fw_v_source=x`; `--fw_heads=1`; `--fw_impl`
   switches the scan without changing semantics; `--fw_gamma_init=0` gives the no-memory control.
4. **Checkpoint compatibility both directions.** An existing Cluster-A 106K checkpoint must load
   into a `fast_weight=True` model (missing `fw_*` initialized to no-op), and a `fast_weight` model
   with the branch off must load back into the baseline.
5. **Guardrails.** Hard-error on incoherent combos (§3.2).

### 7b. Test plan — `bin/test_cluster_b_fastweight.py`

| # | Test | Asserts |
|---|---|---|
| 1 | **Byte-equivalence** | `fast_weight=False` gives logits bit-identical to baseline AND an identical param pytree, same seed |
| 2 | **Three-impl equivalence** | `seq` vs `scan` vs `chunk` agree to `allclose` across several `gamma`, `C`, `d`, both read conventions, and **L not divisible by C** |
| 3 | **Param counts** | measured `fw_*` totals match the §4 D3 table exactly, per `fw_proj` x `fw_dim` |
| 4 | **Gradient reachability** | after one optimizer step, `jax.grad` non-zero for `W_q,W_k,W_v,W_o,fw_gamma_logit,fw_kappa,fw_bias` |
| 5 | **Checkpoint compat** | round-trip both directions per §7.4 |
| 6 | **Causality** | perturbing `x_t` changes `o_s` for `s > t` only — under the exclusive read, **never for `s = t`** |
| 7 | **Boundedness** | `\|\|M\|\| <= 1` and no NaN/Inf as `gamma -> 1` and `gamma -> 0` |
| 8 | **Determinism** | same seed => identical logits |

Tests 6 and 7 protect the *result*, not just the code.

---

## 8. Experiment plan

### 8a. Tasks

**Task 1 — LRA-ListOps (L=2048), primary.** Matched to Cluster A so the comparison is paired:
blocks=8, d_model=128, ssm_size_base=16 (P=8), bidirectional, epochs=40, bsz=50,
opt=BfastandCdecay, lr_factor=3, ssm_lr_base=0.001, weight_decay=0.04, activation=gelu,
batchnorm=True, lambda_pc=0.

**Task 2 — LRA-Text / IMDB byte-level (L=4096), `imdb-classification`. CONFIRMED, and the
verified configs make the case much stronger than the original intuition did.**

Three reasons, now with numbers behind them:

1. **Length is the axis Cluster B should differentiate on.** A fast weight with decay `gamma` has
   characteristic memory ~`1/(1-gamma)`; beating a *bounded* SSM state is the premise, so L=4096
   (2x ListOps) is where it should pay off. This makes task 2 a test of a *prediction*, not just
   a generalization check.
2. **The parameter headroom is far larger than on ListOps.** The S5 authors need **1,321,154**
   params for IMDB versus 188,490 for ListOps — 7x more. On ListOps our ceiling for a
   parameter-efficiency claim is bounded by their already-lean 0.188 M; on IMDB there is room for
   a multiple-x result.
3. **Cheapest of the realistic options**: 35 epochs at 1.32 M, versus sCIFAR's 250 epochs at
   5.13 M.

Same plumbing (`BatchClassificationModel`, single input), so the launcher is a near-copy.
AAN/retrieval is the best *pure* mechanism fit but needs the `RetrievalModel` head and two encoder
passes — deferred.

**Task-2 methodology.** Mirror the ListOps protocol rather than inventing one: pick a reduced
parameter budget, run **Pure S5 and Mambino+Cluster B both at that budget**, 8 matched seeds,
paired tests — the direct analogue of Corner 1 vs Corner 3'. Cite the authors' 1.32 M / 89.31% as
the reference point, not as a comparator (their config, our budget, no paired test possible).
Choosing our own Mambino config *and* leaving S5 at theirs would invite the obvious "you tuned
yours and not theirs" objection.

**Task 3 (optional) — sCIFAR (L=1024), `lra-cifar-classification`, as a LENGTH CONTROL.** At
L=1024 the prediction is that Cluster B helps *less* than at 2048. If it helps *equally*, that is
evidence the gain is capacity rather than memory. All three tasks give a length trend at
1024 / 2048 / 4096, which is a stronger story than two unrelated benchmarks.

**Correction to an earlier assumption: sCIFAR is NOT the cheap option.** At the authors' config it
is **5,133,706 params and 250 epochs** — by a wide margin the most expensive of the three, despite
the shortest sequence. A scaled-down sCIFAR would be cheaper but then is not comparable to their
88.00%. It is also the weakest mechanism fit (a smooth-signal task where key-value recall has
least to offer), and the project's prior sCIFAR result (v0.4 add-DSPC, 0.8174) is from the **NCB
torch line** — different architecture, nothing transfers as a baseline. Run it only if arms 2/3
land on both ListOps and IMDB and the length-trend argument is worth the compute.

**Cost warning:** no baselines exist on tasks 2 or 3. Config 4 gateless and ideally Cluster A both
need fresh 8-seed runs there, so each new task is ~32 runs, not ~16.

### 8b. Seeds

**Reuse v1's 8, no new ones:** `6554595, 42, 12345, 271828, 314159, 1, 2, 3`. On ListOps the
baselines already exist per seed; do not re-run them.

### 8c. Arms

| # | arm | flags | what it isolates | priority |
|---|---|---|---|---|
| 1 | baseline | `fast_weight=False` | reference; have 0.5993 on ListOps | have it |
| 2 | **B0 theory-pure** | `fw_v_source=eps, fw_gate_mode=surprise` | the role-2 hypothesis | **first** |
| 3 | **constant-gate control** | `fw_gate_mode=const` (`fw_kappa` **frozen at 0**, `fw_bias` learned) | **load-bearing ablation** — truly iso-param with arm 2, isolates *z-dependence* rather than z-dependence + gate magnitude jointly | **first** |
| 4 | no-memory control | arm 2 + `fw_gamma_init=0` frozen | gated *memory* vs gated *local feature* | second |
| 5 | static-M control | learned static `d x d`, no write | content-dependent memory vs a static rank-d channel | second |
| 5b | **width-matched control** | gateless Config 4 **widened to ~123,682 params** (via `d_model` / `ssm_size_base` / `glu_rank`) | **"the mechanism, not the capacity"** — no config exists between 106K and 188K, so this must be built | second |
| 6 | content ablation | arm 2 + `fw_v_source=x` | does surprise-as-*content* matter beyond the gate? | second |
| 7 | lean | arm 2 at `fw_dim=4` | is the capacity even needed? | third |
| 8 | stacked | arm 2 + `surprise_gate=True` | does role 2 add on top of roles 1+3+4? | third |
| 9 | capacity headroom | arm 2 at `fw_dim=16` | only if arm 2 shows a large effect (§2b) | conditional |
| 10 | permuted-z | arm 2, gate driven by shuffled z | surprise-*specificity*, cheap | optional |
| 11 | B1 delta | `fw_rule=delta` | error-correcting write | after B0 |

Why arm 3 is defined this way: with `fw_kappa` initialized to 0 the gate is a *constant*
`sigmoid(b_fw)`, so a naive "replace the gate with 1" control would differ from arm 2 not only in
z-dependence but in effective write rate. Freezing `fw_kappa` at 0 while letting `b_fw` learn keeps
the parameter count identical and the write rate free, so the only remaining difference is the
thing under test.

### 8d. Success criteria — stated before the runs

- **Primary:** does arm 2 beat **0.6050** (Cluster A, 8-seed) by more than seed noise, and clear
  **0.6089** (Pure S5 188K) at ~124K params?
- **Load-bearing:** **arm 2 must beat arm 3.** If they tie, the honest conclusion is "the fast
  weight helped, surprise gating did not" — reported as such.
- **Mechanism:** arm 2 must beat arms 4, 5 and **5b**. If arm 4 (gamma=0) matches, there is no
  inference-time *learning*, only a gated local feature. If arm 5 or 5b matches, the gain is
  capacity, not content-dependent memory.
- **Theory:** arm 2 vs arm 6 tests whether surprise-as-content earns its place, or the gate alone
  suffices.
- **Generalization:** the ListOps result must replicate on IMDB (§8a), ideally with a *larger*
  margin at L=4096.
- **Statistics.** Analyze **paired** (paired t / Wilcoxon on per-seed deltas), never as two
  independent means: unpaired SE at n=8 is ~0.0035, giving only ~0.6 power for a 0.8 pp effect,
  whereas paired (rho ~ 0.5) gives SE ~0.0025 and ~0.9 power. **Pre-registered:** effects below
  ~0.5 pp are unresolvable at n=8 and will be reported as inconclusive, not null. The comparison
  against Pure S5's 0.6089 is a separate, *unpaired* contrast with worse power.

### 8e. Chip/PPAC — deliberately last

Only once the science lands: extend the JAXPR extractor and register the config exactly as
`mambino2p0` was (`paper_v2_ppac/chip/`, commits `c6ef30b`, `84ca964`), honoring the hard invariant
`sum(emitted FLOPs) == JAXPR walker total`. Expect a cost, per §6. Per §2b, the architecture story
may stand on its own without this.

---

## 9. Fable audit — COMPLETE, verdict folded in

Dispatched before any code was written, per the standing rule that anything touching the scan or
the gradient path is Fable-verified first.

**Verified correct:** the flattened-`M` associative scan (monoid associativity shown explicitly,
inclusive-prefix convention confirmed, `M_0 = 0` dependency flagged); the read's closed form
`o_t = sum_i gamma^(t-i) s_i (k_i·q_t) v_i` as decayed linear attention with a per-token scalar
write gate; and the chunked recurrence's exponents term by term.

| # | Finding | Change made |
|---|---|---|
| 1 | Inclusive read lets the layer learn `W_q ≈ W_k` and degenerate into a gated *instantaneous* rank-d path — causal, so no test would catch it | **Exclusive read `M_{t-1}`** (D6); chunked D matrix corrected to zero-diagonal, carry exponent `j-1` |
| 2 | The `gamma^(-j')` rescale trick breaks fp32 below `gamma ~ 0.878` at C=128, and bf16 essentially always | **Banned**; explicit `C x C` decay matrix only |
| 3 | `eta` and `W_o` are a redundant double scale that interacts badly with zero-init; `eta=0` + `W_o=0` is permanently dead. Separately `\|\|M\|\|` scales as `1/(1-gamma)` and q/k normalization alone does not fix it | **`eta` deleted**; `(1-gamma)` coupling; and **`v` normalized too** (D7), which additionally prevents surprise entering the write twice |
| 4 | d=16 separate (+62%) and all d=32 variants forfeit the "cheap add-on" framing | Default moved to **shared d=8** (+17.0%), d=4 as lean control, d=16 kept as conditional headroom (§2b) |
| 5 | arm2-vs-arm3 was not iso-param, was confounded by effective-write-rate, and did not test "beyond capacity" at all | Arm 3 redefined as **`fw_kappa` frozen at 0, `b_fw` learned**; added **gamma=0**, **static-M**, and **width-matched (5b)** controls; paired-seed analysis and a pre-registered 0.5 pp resolution floor |

**Noted, not adopted:** this is the unnormalized RetNet/GLA-family form (no `sum(k·q)`
denominator), which couples output magnitude to accumulated history mass. The `(1-gamma)` coupling
plus full q/k/v normalization addresses the same concern more cheaply; if instability appears in
practice, a denominator is the first thing to revisit.

---

## 10. Order of work

1. ~~Fable verdict~~ — **done**, folded in above.
2. Worktree `v2/cluster-b-fastweight` off `v2/surprise-gate`, isolated so nothing running is disturbed.
3. Scaffold: fields + `setup()` guard + `_surprise_z` refactor + `__call__` hook + `_fw_seq` only.
4. **Tests 1 (byte-equivalence) and 4 (gradients) green before anything else.**
5. Add `_fw_scan`, `_fw_chunk`; tests 2, 6, 7 green (2 and 6 catch the exclusive-read off-by-one).
6. Thread `train.py` / `train_helpers.py` / `run_train.py`; tests 3, 5, 8 green.
7. Launch script; 1-epoch smoke on one seed; check `gn` and loss are sane.
8. Arms 2 + 3 on ListOps across 8 seeds. **Read the ablation before spending anything else.**
9. If it holds: arms 4, 5, 5b, 6 -> then IMDB baselines + arms 2/3 -> then sCIFAR length control.
10. Commit everything — code, launcher, tests, results — per the keep-it-all-in-git rule.

## 11. Risks

| risk | mitigation |
|---|---|
| `scan` impl OOMs at larger d, batch 50, L=4096 | `chunk` impl exists from the start; d=8 default is comfortable |
| exclusive-read off-by-one silently reintroduces the shortcut | tests 2 + 6 exist specifically for this |
| arm 2 == arm 3 (surprise gating does nothing) | that IS the result; report it |
| arm 2 == arm 4 / 5 / 5b (no real memory, or static capacity suffices) | all three controls run in the second wave, before any writeup |
| parameter-superlative claim collapses under review | §2b caveats; verify external counts before any paper text |
| result is ListOps-specific | IMDB is task 2, not optional |
| role 2 wins on accuracy but loses on chip | expected and pre-stated (§6); §2b notes the architecture story may stand alone |
| Cluster A and B interact badly when stacked | arm 8 tests exactly that; arms 2/3 are independent of it |
