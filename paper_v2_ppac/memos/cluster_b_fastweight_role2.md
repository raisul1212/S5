# Cluster B — Role 2: Surprise-Gated Fast-Weight / Inference-Time Learning

**Audience:** a fresh implementing agent with NO context of the Cluster A session.
Read this top-to-bottom; it is self-contained. Everything you need to start coding
independently is here. **Do the Fable math check (Section 2) BEFORE writing code.**

**Status when written:** Cluster A (the surprise *gate*) is BUILT, run, and validated
(see Section 1). Cluster B has NOT been started. This memo specifies it.

---

## 0. TL;DR

Cluster A added a ~16-parameter **surprise gate** that *modulates* the predictive-coding
error write inside `MambinoSSM`. By a formal theorem (Section 1), a scalar gate can serve
predictive-coding roles **1 (state write) + 3 (gain) + 4 (segmentation)** but provably
**cannot** serve **role 2 (learning)**. Role 2 = *inference-time learning* = a **fast-weight
memory** `M_t` (a matrix that accumulates key→value associations within a sequence), whose
**write strength is gated by surprise**. High surprise ⇒ learn this new association strongly;
low surprise (predictable) ⇒ don't bother. This is the project's "north star": a network that
*learns during inference*, not just at train time.

**What you build:** a parallel additive branch in `MambinoSSM.__call__`:
`output = ys (SSM) + D·x (feedthrough) + o_fw (fast-weight read)`, where `o_fw[t] = M_t q_t`
and `M_t = γ M_{t-1} + η·s_t·(v_t k_t^T)` (B0), with `s_t` = the surprise signal reused from
Cluster A. Then upgrade the plain outer-product write to the **delta rule** (B1), which needs a
chunked scan (Section 4).

**Non-negotiables:** CLI kill-switch to byte-identical baseline; Fable-verify the math first;
reuse v1's 8 seeds; isolate in a git worktree so nothing else is disturbed.

---

## 1. Why this exists — the 4-roles theorem and what Cluster A already did

**The substrate.** `MambinoSSM` (in `s5/mambino_ssm.py`) is a predictive-coding state-space
layer built on the S5 fork (`~/dev/ssm-baselines/S5`). Its forward pass (grounded, see the
actual code around lines 560–613):

1. `x = DAC(input)` — quantize/analog-in.
2. **Predictor:** `x̂(t) = C_s @ s(t−1)` (a small SSM predicts the next input). [`_apply_predictor_scan`]
3. **Surprise:** `eps = x − x̂`.  (This is THE prediction error. All of Cluster A/B ride on it.)
4. **Main scan (additive PC):** `u(t) = B̄·x(t) + W̄_ε·eps(t)`; run diagonal SSM →`ys`. [`_apply_main_scan_with_additive_pc`, ~line 483]
5. **Feedthrough:** `y = ys + D·x`.
6. `output = ADC(y)`.
   Intrinsic loss `mean(||eps||²)` is sown to `intermediates` (weighted by `lambda_pc`; we run `lambda_pc=0`).

**Cluster A (DONE).** Inside step 4, the error write became
`u = B̄·x + g(t)·(W̄_ε·eps)`, with `g(t) = tanh(κ·z(t) + bias)` (signed: push+/hold0/pop−) where
`z(t)=(||eps(t)|| − μ_t)/σ_t`, μ/σ = causal EMA of `||eps||` (Adam bias-corrected so z(0)=0).
Per-block learned scalars `κ, bias` (+2/layer ≈ 16 total). Kill-switch `surprise_gate=False` ⇒
byte-identical. Result (LRA-ListOps, seed 6554595): signed gate **0.6075** vs gateless **0.5993**
vs S5-Dense **0.6089** — i.e. ~16 params buy back the Mambino↔S5 accuracy gap, and the PPAC then
shows Mambino 2.0 hits S5-Dense accuracy at **+19% acc/mJ, ~4.6× acc/W (serial peak), 2× acc/area**.
That is the "efficiency at the same accuracy" headline. **Cluster B is the orthogonal upgrade:
add *learning* (role 2) on top of, or instead of, the gate.**

**The theorem (Fable-formalized; re-derive if you want conviction).** Surprise enters the update
math in exactly **two algebraic forms**:
- **Signed first moment** `Π·eps` (a *direction* / innovation) → drives **role 1 (state write)** and
  **role 2 (learning / fast-weight write)**.
- **Unsigned second moment** `eps²` (a *magnitude* / precision) → drives **role 3 (adaptive gain)**
  and **role 4 (segmentation / event boundary)**.

A single scalar `g` multiplying the write can encode magnitude (roles 3,4) and, if signed, a 1-D
push/pop (a degenerate role 1). It **cannot** implement role 2, because role 2 requires
*accumulating an outer product into a matrix state* `M` that is **read back later by a query** —
a rank-growing associative memory, not a per-token scalar. No scalar gate can create that memory.
**Therefore role 2 is a genuinely new mechanism, not reachable by tuning Cluster A.** That is the
whole justification for a separate Cluster B.

---

## 2. The mechanism (math) — verify with Fable BEFORE coding

Let the layer see, at each step `t`, its input `x_t ∈ R^H` (H=128) and surprise `eps_t ∈ R^H`.
Project keys/queries/values (learned):
```
k_t = W_k x_t,  q_t = W_q x_t,  v_t = W_v x_t        # or project from eps_t — see decision D2
```
(optionally L2-normalize k_t, q_t; optionally multi-head with head dim d_h so M is block-diagonal.)

**Surprise write-gate** (reuse Cluster A's signal — do NOT reinvent it):
```
s_t = σ(a·z_t + b)   ∈ (0,1)      # z_t = EMA-normalized ||eps_t||, exactly as in _surprise_gate
```
`s_t≈1` on surprising tokens (write hard), `s_t≈0` on predictable tokens (skip the write).

### B0 — gated linear-attention fast weight (BUILD THIS FIRST)
```
M_t = γ · M_{t−1} + η · s_t · (v_t ⊗ k_t)          # M ∈ R^{H×H} (or per-head d_h×d_h)
o_t = M_t q_t                                       # the read, added to the block output
```
`γ ∈ (0,1)` decay (learned scalar or per-head), `η` write rate (learned). **This recurrence is
still associative** (transition is scalar `γ`), so it can be done with `jax.lax.associative_scan`
over matrix-valued state OR a chunked scan for memory (Section 4). It is the cheapest correct
role-2: a surprise-gated Hebbian memory.

### B1 — delta rule / DeltaNet (UPGRADE after B0 works)
Replace the pure Hebbian write with an *error-correcting* write that first removes the old value
stored along `k_t`, then writes the new one:
```
M_t = M_{t−1}(I − β_t k_t k_t^T) + β_t (s_t v_t) k_t^T ,   β_t ∈ (0,1)   # β_t may itself be s_t-gated
o_t = M_t q_t
```
The data-dependent **matrix** transition `(I − β_t k_t k_t^T)` is what breaks the naive diagonal
scan and needs the chunked DeltaNet algorithm (Section 4). B1 is the "real" inference-time learner
(it does gradient-descent-like updates on an associative memory); B0 is the warm-up that de-risks
the plumbing.

**Fable checklist (ask before coding):**
1. Confirm B0's `M_t` recurrence is associative in `γ` and that `associative_scan` over `(γ·1, W_t)`
   with `W_t = η s_t v_t k_t^T` yields the same `M_t` as the sequential loop. Confirm the read
   `o_t = M_t q_t` is what you get by materializing `M_t`.
2. Confirm B1's chunked form (WY representation) reproduces the sequential delta rule exactly.
3. Confirm the surprise gate `s_t` is the *only* coupling to Cluster A, and that with the fast-weight
   branch OFF the layer is byte-identical to the current forward (kill-switch).
4. Sanity: gradient of task loss flows into `W_k,W_q,W_v,γ,η` (and `β`) — nothing is accidentally
   `stop_gradient`'d the way Cluster A's `gate_detach` froze the predictor. (That was a real bug in A.)
5. Parameter/FLOP budget: `W_k,W_q,W_v` are 3·H² = ~49K params/layer *if dense* — that dwarfs the
   16-param gate and even the model. **Use low-rank or multi-head-small-d_h projections** so role 2
   stays cheap (decision D3). Have Fable bound the added params/FLOPs vs config4's 105,738.

---

## 3. Where it goes in the code (exact hooks)

File: `s5/mambino_ssm.py` (class `MambinoSSM`, Flax `nn.Module`).

- **Fields** (near line 159, alongside `surprise_gate: bool = False`): add
  `fast_weight: bool = False`, plus `fw_rank: int`, `fw_heads: int`, `fw_gamma_init`,
  `fw_eta_init`, `fw_rule: str = "hebb"|"delta"`, `fw_source: str = "x"|"eps"`. All default to the
  OFF/no-op configuration.
- **`setup()`** (line 166; note the existing pattern at ~line 347 `if self.surprise_gate:` that
  guards param creation so the tree is byte-identical when off): add
  `if self.fast_weight:` block creating `W_k, W_q, W_v` (low-rank), `fw_gamma`, `fw_eta` (and
  `fw_beta` for delta). **Guard everything** — off ⇒ no new params.
- **New method** `_apply_fastweight(self, x_seq, eps_seq) -> (L, H)`: computes k/q/v, the surprise
  gate `s_t` (call the existing `self._surprise_gate(eps_seq)` — but note it returns the *signed*
  `g`; for a write *magnitude* you likely want the **unsigned** `sigmoid` form, so either add an
  unsigned path or take `abs`/`relu` — decision D1), runs the B0/B1 recurrence (Section 4), returns
  `o_fw` at shape (L,H).
- **`__call__`** (line 560): after step 4 (`ys = self._apply_main_scan_with_additive_pc(x, eps)`,
  line 597) and before/at step 5, add:
  ```python
  o_fw = self._apply_fastweight(x, eps) if self.fast_weight else 0.0
  output = ys + Du + o_fw          # Du is the existing feedthrough term
  ```
  Attaching in `__call__` (not inside the SSM scan) keeps the diagonal FFT scan UNTOUCHED — the SSM
  branch keeps its efficiency and its kill-switch; role 2 is a clean additive memory. **Do not** fold
  it into `Bu_elements` — that would entangle it with the associative scan and the chip workload.
- **Factory** `init_MambinoSSM` (line 616) and the two call sites in `s5/train.py` (~lines 107, 546):
  thread the new `fw_*` kwargs, exactly as the `gate_*` kwargs were threaded for Cluster A.
- **Optimizer:** put `fw_gamma, fw_eta, fw_beta` (the scalar dynamics) in the no-weight-decay `ssm`
  group in the `BfastandCdecay` opt config (same place `gate_kappa/gate_bias` went). The projection
  matrices `W_k/W_q/W_v` go in the normal decayed group.
- **CLI:** add `--fast_weight`, `--fw_rank`, `--fw_heads`, `--fw_rule`, `--fw_source`, inits — in
  `run_train.py` argparse, mirroring `--surprise_gate` / `--gate_*`. Model the launch script on
  `bin/gilbreth_lra_listops_mambino_gelu_sgate.sh`.

---

## 4. The scan problem (this is the crux)

The SSM branch uses `jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))` — a
**diagonal** linear recurrence, cheap and FFT/parallel-friendly. Role 2's `M_t` is **matrix-valued**:

- **B0 (Hebbian, scalar `γ`)** *is* associative: state = (γ·1, W_t=η s_t v_t k_tᵀ), same
  `binary_operator` shape but over H×H (or heads × d_h×d_h) matrices. Two options:
  - (a) `associative_scan` over matrix state — simplest, but O(L·H²) memory. Fine for a first run at
    H=128, L=2048 *if* you use **heads** (block-diagonal M): heads=h, d_h=H/h ⇒ memory O(L·h·d_h²).
  - (b) **Chunked scan** (recommended, and required for B1): split L into chunks of size C; inside a
    chunk compute the contribution with parallel matmuls; carry `M` across chunks with a short
    `jax.lax.scan`. This is the standard "chunked linear attention" recipe.
- **B1 (delta rule)** has a *data-dependent matrix* transition ⇒ NOT a simple associative scan. It
  needs the **DeltaNet chunked (WY-representation) algorithm**: within-chunk you solve a small
  triangular system to get the corrected writes, then carry `M` across chunks. Reference
  implementation exists in the **STP project's fused Triton kernel** (memory `[[stp_v4_triton_kernel]]`:
  fused fwd+bwd, per-token physics, all 15 grads verified, `scan_impl="triton"`). **Caveat:** that
  kernel is **PyTorch/Triton**; this repo is **JAX/Flax**. So it is an **algorithmic reference, not a
  drop-in**. Your JAX options, cheapest first:
  1. `jax.lax.scan` over chunks with dense intra-chunk matmuls (no custom kernel) — correct, good
     enough to get the science; optimize later.
  2. A **Pallas** kernel (JAX's Triton analog) porting the STP WY algorithm — only if step 1 is too slow.
  Start with option 1. Prove the science before you optimize the kernel.

**Decision:** if the JAX chunked delta rule proves heavy, it is acceptable to first prototype B1 in
the **STP torch line** (where the kernel already exists) purely to confirm the *science* (does a
surprise-gated delta memory help ListOps-style reasoning), then port to the JAX Mambino for the
paper numbers. Discuss with the user which line to spend effort on.

---

## 5. Kill-switch / fallback discipline (HARD RULE — the user enforces this)

For any new mechanism, ALL of the following are part of the design, not afterthoughts:
- **CLI kill-switch** back to the known-good baseline: `--fast_weight=False` ⇒ forward + param tree
  **byte-identical** to current `MambinoSSM`. Add a test that asserts this (a `jnp.allclose` on
  outputs AND an identical pytree structure, exactly like Cluster A's byte-equivalence check).
- **Per-component disable:** independent flags so you can turn off the delta correction (fall back to
  B0), the surprise gate on the write (fall back to ungated fast weight), heads, etc.
- **Checkpoint compatibility both directions:** an existing 106K/189K Cluster-A checkpoint must load
  into the fast_weight model (missing `fw_*` params initialized to no-op), and a fast_weight model
  with the branch off must load back into the baseline. Cluster A verified this; match it.
- **Guardrails:** hard-error on incoherent flag combos (Cluster A added a `train.py` guard for the
  `gate_detach=True & lambda_pc=0` frozen-predictor trap — add the analogous guard if you introduce
  any `stop_gradient` on the fast-weight path).

---

## 6. Experiment plan

- **Task:** LRA-ListOps (`--dataset=listops-classification`), the reasoning benchmark where the gate
  paid off. Same recipe as Cluster A: `blocks=8, d_model=128, ssm_size_base=16 (P=8), bidirectional,
  epochs=40, bsz=50, opt=BfastandCdecay, lr_factor=3, ssm_lr_base=0.001, weight_decay=0.04,
  activation=gelu, batchnorm=True, lambda_pc=0`.
- **Seeds — REUSE v1's 8:** `6554595, 42, 12345, 271828, 314159, 1, 2, 3`. Do NOT invent new seeds;
  the whole comparison is matched-seed. Reuse existing per-seed baselines rather than re-running them
  (S5-Dense and Mambino-gelu numbers already exist per seed).
- **Arms (run each across the 8 seeds):**
  1. baseline: `fast_weight=False` (= Cluster A gateless, ≈0.599) — for byte-equivalence + reference.
  2. `fast_weight=True, fw_rule=hebb` (B0), surprise-gated write.
  3. `fast_weight=True, fw_rule=hebb`, **ungated** write (ablation: does the *surprise* gating matter,
     or is it just extra capacity?).
  4. `fast_weight=True, fw_rule=delta` (B1), surprise-gated.
  5. **Stacked:** Cluster A gate ON + fast_weight ON (does role 2 add on top of roles 1+3+4?).
- **Success criteria (state honestly):**
  - Primary: does a surprise-gated fast weight lift accuracy *beyond* Cluster A's 0.6075 / past
    S5-Dense 0.6089, at a modest param/FLOP cost? (The role-2 hypothesis.)
  - Load-bearing ablation: arm 2 vs arm 3 — **the surprise gating must beat the ungated fast weight**,
    else you've only shown "more params help," not "inference-time learning gated by surprise helps."
  - Watch param/FLOP budget (Section 2.5): report accuracy-per-added-param vs just widening config4.
- **Chip/PPAC (later, not first pass):** role 2 introduces dense `M q` matmuls + outer products — a
  *new* compute primitive vs the SSM's diagonal scan. When the science lands, extend the JAXPR
  workload extractor + PPAC exactly as `mambino2p0` was registered (see `paper_v2_ppac/chip/` and the
  `mambino2p0` commits on branch `v2/surprise-gate`). Expect role 2 to *cost* chip efficiency — the
  question is whether the accuracy gain justifies it. Don't pre-judge.

---

## 7. Repo / env / worktree setup

- **Repo:** `~/dev/ssm-baselines/S5` (JAX/Flax; NOT the nc-block torch line). Origin
  `github.com/raisul1212/S5`. **Note:** pushes must originate from a machine with GitHub auth
  (the local dev box), NOT from Gilbreth (the cluster has no GitHub credentials — pushes there fail).
- **Branch:** build Cluster B on top of Cluster A. Base off `v2/surprise-gate` (Cluster A is committed
  there). Create `v2/cluster-b-fastweight`. **Isolate in a git worktree** so the main checkout and any
  running jobs are untouched (Cluster A used `~/dev/ssm-baselines/S5-surprise-gate` locally and a
  separate worktree on Gilbreth). Use `git worktree add`; on Gilbreth, if `origin/<branch>` refuses,
  fetch first and use `FETCH_HEAD` *from inside the worktree* (a cross-worktree `FETCH_HEAD` checkout
  fails — learned the hard way).
- **Gilbreth env:** `conda activate /scratch/gilbreth/raisul/envs/s5m` (jax 0.4.30, flax 0.8.5).
  SLURM: partition `a30`, account `raisul`, 1×GPU, model the script on
  `bin/gilbreth_lra_listops_mambino_gelu_sgate.sh`. SSH host alias `gilbreth` (key-only).
- **Keep everything in git** (explicit user instruction): commit code, launch scripts, AND generated
  artifacts (workload YAMLs, result CSVs) so the work is publishable/reproducible. Paths in
  `paper_v2_ppac/chip/*` are now repo-relative (`__file__`-based) — keep new paths repo-relative too.

---

## 8. Verification discipline

- **Fable audits the math FIRST** (Section 2 checklist) and again after the chunked scan is written
  (does chunked == sequential?). The user's standing rule: verify with Fable, especially anything
  touching the scan or the gradient path. Fable is a reasoning auditor — give it the exact recurrences
  and code; it does not need cluster access.
- **Byte-equivalence test** committed alongside the code (kill-switch).
- **Chunked-vs-sequential equivalence test** (`allclose` on a random input) committed for B0 and B1.
- Report failures honestly (frozen gradients, NaNs from unnormalized keys, exploding `M` when γ→1).

---

## 9. Open decisions for the implementing agent (raise with the user)

- **D1 — gate polarity for the write:** role 2's write is a *magnitude* (learn-strength), so the
  *unsigned* surprise (`sigmoid`, or `|z|`) is the natural coupling, whereas Cluster A's headline gate
  is *signed* (push/pop). Decide whether to add an unsigned path in `_surprise_gate` or derive `s_t`
  fresh. (The 4-roles theorem says role 2 uses the *signed first moment* `Π·eps` for the *direction*
  v_t/k_t, and the *magnitude* for the write strength — so you may want both: signed content, unsigned
  gate.)
- **D2 — project k/q/v from `x` or from `eps`?** From `x` = standard fast weight over the input stream;
  from `eps` = "learn the surprising residual" (more aligned with the PC story). Try both; `x` is the
  safer default.
- **D3 — projection cost:** dense `W_k/W_q/W_v` (3H²≈49K/layer) is too many params. Use low-rank
  (`fw_rank`≈16–32) or multi-head with small `d_h`. This keeps the "cheap add-on" framing intact and
  makes arm-2-vs-arm-3 a fair capacity comparison.
- **D4 — which line to prototype B1 in** (JAX chunked scan vs STP torch kernel) — Section 4.

---

## 10. Pointers

- Cluster A memory: `[[cluster_a_surprise_gate]]` (the gate mechanism, the premise validation, the
  jobs). This memo is its role-2 sibling.
- Fast-weight kernel reference: `[[stp_v4_triton_kernel]]` (DeltaNet chunk-scan, torch/Triton — port,
  don't import) and `[[stp_v3_gpt2_fair_result]]`.
- North-star framing: `[[user_long_term_vision]]` (KV-cache-free LLMs + autonomous continual learners
  + inference-time learning — role 2 IS the inference-time-learning piece), `[[mambino_strategic_juncture_multidomain]]`.
- Paper positioning / PPAC: `[[mambino_paper_positioning]]`, `[[feedback_ppac_from_jaxpr]]`
  (the hard invariant for any future role-2 chip workload), `[[feedback_always_have_fallback]]`
  (the kill-switch rule), `[[feedback_lean_by_default]]` (don't over-build; clear the accuracy floor
  cheaply).

**First action for the implementing agent:** re-read Section 2, hand Fable the B0 recurrence +
`s_t` definition + the `__call__` hook, get the go/no-go on the math and the param budget, THEN scaffold
the flags + kill-switch test. Do not write the delta-rule chunk scan until B0 runs end-to-end.
