# Pre-registration: LRA-Text (IMDB) second task

**Written 2026-08-04, before any IMDB run has been launched.** Nothing in this
file may be revised after the first IMDB result lands. If something here turns
out to be wrong, the correction goes in a *new*, separately dated section at the
bottom, with the original text left intact.

The point of this document is narrow. On ListOps our primary metric
(`test@peakval`) favours the paper's central claim far more than the
alternative (`test_max`): +0.76 pp on 7/8 seeds versus +0.04 pp on 3/8. We
believe `test@peakval` is the correct rule for reasons given in §4, but that
belief is worth nothing if the metric is chosen after seeing the numbers. So it
is fixed here, in advance, for the second task.

## 1. Task

LRA-Text: binary sentiment classification on IMDB movie reviews at
**character level**, `l_max = 4096`, vocabulary 135 (`min_freq=15`,
`append_eos=True`), 2 classes. Published S5 accuracy 89.31%.

## 2. Configurations

Derived from the exact parameter formulas (reals; one complex = two reals;
bidirectional, so two C matrices), which reproduce S5's published IMDB total of
1,321,154 exactly:

    main_ssm(P,H) = 6PH + H + 3P      predictor(P,H) = 6PH + 3P
    gate(H)       = H^2 + H           norm(H) = 2H
    encoder = 34,816   decoder = 514   H = 256   n_layers = 6

| config | P | ssm_size_base | blocks | output | expected params |
|---|---:|---:|---:|---|---:|
| S5-Dense  | 96 | 192 | 12 | half_glu2 | 1,321,154 |
| S5-0      | 138 | 276 | 12 | gelu | 1,314,230 |
| Mambino-0 | 69 | 138 | 6 | gelu + predictor | 1,314,230 |
| Mambino-G | 69 | 138 | 6 | gelu + predictor + surprise gate | 1,314,242 |

`blocks` is chosen so `block_size = 23` for both gateless arms. S5-0 and
Mambino-0 are **exactly** iso-parameter, by the same structural identity that
holds on ListOps: the predictor at P costs precisely what doubling P costs,
independent of H. **These counts are assertions, not estimates.** If the built
models do not hit them, the configuration is wrong and no run proceeds.

## 3. Training protocol

S5's own IMDB recipe, unchanged, for all four configurations: `epochs=35`,
`bsz=50`, `d_model=256`, `n_layers=6`, `p_dropout=0.1`, `weight_decay=0.07`,
`ssm_lr_base=0.001`, `lr_factor=4`, `warmup_end=0`, `opt_config=standard`,
`dt_global=True`, `C_init=lecun_normal`, `batchnorm=True`, `bidirectional=True`,
`lambda_pc=0`. Only architecture flags vary between configurations.

Seeds: 6554595, 42, 12345, 271828, 314159, 1, 2, 3 — the same eight as ListOps.

**One deviation from S5, declared here:** `--val_split=0.1`. S5's loader ships
no validation split for IMDB and their code assigns the test set to the
validation role (`s5/dataloaders/lra.py:139`, comment: *"Use test set as val
set, as done in the LRA paper"*). We carve 10% of train instead, giving
22,500 train / 2,500 val / 25,000 test. The split is drawn with the dataset's
own fixed seed, not `jax_seed`, so every run and every configuration selects
against the identical validation set. Consequence accepted in advance: our
models see 10% less training data than the published runs, uniformly across all
four configurations.

## 4. Metrics — fixed in advance

**Primary: `test@peakval`.** Test accuracy at the epoch with the highest
validation accuracy. Identical in meaning to the ListOps protocol, which is the
entire reason for §3's deviation. All comparisons between configurations use
this and only this.

Justification, which does not depend on the outcome: selection by maximum test
accuracy is sensitive to training-curve variance, so it systematically flatters
the noisiest configuration. On ListOps the inflation runs from +0.45 pp
(Mambino-0) to +1.17 pp (S5-0), and S5-0 is the highest-variance configuration
at sigma = 0.0081. A selection rule whose bias depends on the model under test
cannot arbitrate between models.

**Secondary, reported but never used for comparison: `test_max`.** Best test
accuracy at any epoch. Reported because it is the rule the published 89.31% was
produced under, so it is the only column directly comparable to it.

**Tertiary, reported: last-epoch test accuracy.** No selection of any kind.
Included as a selection-free arbiter.

Statistics: 8-seed mean ± sample standard deviation (ddof=1). Paired t-test over
matched seeds; exact sign test alongside. Two-tailed p reported.

## 5. Claims under test, and what would falsify them

- **C1 (predictor).** Mambino-0 > S5-0 on `test@peakval`. Predicted direction
  positive, by analogy with ListOps (+0.76 pp). Falsified if the point estimate
  is negative, or if it is positive but the sign test is worse than 5/8.
- **C2 (gate).** Mambino-G > Mambino-0 on `test@peakval`. Predicted positive
  (ListOps: +0.58 pp). Same falsification rule.
- **C3 (reference).** Mambino-G versus S5-Dense. **No direction is predicted.**
  On ListOps this was −0.39 pp, and IMDB's gate holds only 29.9% of parameters
  against ListOps' 70.1%, so we have no basis for a prediction here.

We commit to reporting all three regardless of sign, and to reporting C1 and C2
even if the ListOps result fails to replicate.

## 6. Chip-side expectation, recorded in advance

The area advantage is expected to **shrink** relative to ListOps, and may
invert against S5-0. The gate is 29.9% of S5-Dense's parameters here versus
70.1% on ListOps, and at P=69 the predictor costs 1.61x the gate it displaces
(the mechanism is parameter-favourable only when H > 6P; here H/P = 3.7 for
S5-0 and 3.4 for the Mambino arms, versus 16 on ListOps).

Recording this now so that a weaker chip result reads as a confirmed prediction
about the mechanism's regime of validity rather than as a disappointment
discovered afterwards.

## 7. Addendum, 2026-08-04 (still before any IMDB run)

Recorded per the rule at the top of this file. **No number in §2 changes.**

Environment findings from preparing the dataset on the Gilbreth login node:

1. `load_dataset("imdb")` fails outright on the cluster env (`datasets` 5.0.0):
   `huggingface_hub` now requires a namespaced id and raises `HfUriError` on the
   bare name. The loader now falls back to `stanfordnlp/imdb`, same data.
   ListOps never exposed this because it reads local TSVs and never touches the
   Hub.
2. `torchtext` is unavailable in this environment, so the vocabulary is built by
   this repo's pure-Python `_build_vocab_from_iterator` fallback. Its `min_freq`
   boundary (`cnt >= min_freq`) matches torchtext's, so the builder is not the
   difference.
3. **The full-train vocabulary is 134, not the hardcoded `IN_DIM = 135`.** The
   current Hub parquet conversion of IMDB differs slightly from the 2023
   script-based version, and one borderline character now falls below
   `min_freq=15`. Highest token id observed is 133.

Consequences, stated plainly: 135 is an upper bound, so no token id can exceed
the encoder width; the encoder retains one unused input column; the encoder
parameter count stays 34,816 and therefore **every total in §2 is unaffected**.
One rare character now maps to `<unk>` instead of carrying its own id, which we
regard as immaterial to accuracy at this frequency. An assertion now fails
loudly if the vocabulary ever exceeds `IN_DIM`, which is the direction that
would genuinely corrupt a run.

This does mean our IMDB preprocessing is not bit-identical to the one behind the
published 89.31%. The paper should say so rather than claim an exact
reproduction of their input pipeline.

## 8. Addendum, 2026-08-05: pilot 1 result and why it is void

**Pilot 1 (jobs 11459360-11459365) ran to completion and produced a large
negative result for C1.** Recorded here in full before any re-run, because a
pre-registered test that fails is not something to quietly repeat until it
passes.

| metric | S5-0 | Mambino-0 | delta |
|---|---:|---:|---:|
| test@peakval | 0.8771 | 0.7817 | **-9.54 pp**, 0/3 |
| test_max | 0.8795 | 0.7822 | -9.73 pp, 0/3 |
| last-epoch | 0.8790 | 0.7082 | -17.08 pp, 0/3 |

By the falsification rule in §5, C1 fails on this evidence.

**However, the runs are not a valid test of the mechanism, for a reason
identified in the code and not inferred from the numbers.** Mambino-0 seed
6554595 trained normally for eleven epochs, reaching 0.7343 test accuracy, and
then diverged:

    Train Loss:  0.54044  ...  Test Accuracy: 0.7343
    Train Loss: 19.24254  ...  Test Accuracy: 0.5020
    Train Loss:  0.71826  ...  Test Accuracy: 0.5356   (chance, for 24 epochs)

S5-0 on the identical seed trained cleanly throughout (loss 0.71 -> 0.01,
accuracy 0.65 -> 0.88). Mambino-0's three seeds spanned 0.7343/0.7732/0.8376,
an 11-point range, against S5-0's 0.8763-0.8779.

**Cause.** `opt_config=standard`, which S5's IMDB recipe uses, had no parameter
grouping for MambinoSSM. Only `BfastandCdecay` (which ListOps uses) was ever
extended for it. So every Mambino-specific parameter -- including the
predictor's `Lambda_s_re`/`Lambda_s_im` -- fell through to the `regular` group:
AdamW at `lr_factor x ssm_lr` = 4x the SSM learning rate, **with weight decay
0.07 applied to the state-transition eigenvalues**. That destabilises the
recurrence. IMDB is simply the first Mambino run to use `standard`.

This is an execution defect, not evidence about the predictor. The measurement
never took place.

**Disposition.** The grouping is fixed (`standard` extended to mirror
`BfastandCdecay`, plus a guard that now raises if any opt_config lacking a
Mambino grouping is used with Mambino parameters present). Pilot 1 stands on the
record as reported above. Pilot 2 will be run on the fixed code and reported
alongside it, not in place of it. **§§1-7 are unchanged**: same configs, same
seeds, same metrics, same falsification rules. Nothing was revised in light of
the numbers.

If pilot 2 also comes back negative on healthy training curves, C1 fails on
IMDB and that is the finding.

## 9. Addendum, 2026-08-05: reduced-budget arm ADDED (not substituted)

**This is a declared addition, made after pilot 1 and before any of these runs.**
It does not revise §§1-7. C1, C2 and C3 stand exactly as written, on exactly the
four configurations named there, and will be reported whatever they show. The
arm below tests a **different claim** and is reported separately.

**Motivation.** §§1-7 hold parameters constant: all four configs sit within
0.52% of S5-Dense, because the gate's budget is *reallocated* into state or
predictor. That design measures mechanism at fixed capacity, which is what C1
and C2 are about. It cannot measure parameter reduction, and on ListOps the
headline chip claim is a reduction (Mambino-G at 56% of S5-Dense's parameters).
IMDB currently has no analogue of that claim.

**The arm.** Bank the gate instead of reallocating it:

| config | P | ssm_size_base | blocks | params | vs S5-Dense |
|---|---:|---:|---:|---:|---:|
| S5-0-R | 96 | 192 | 12 | 926,402 | 70.1% |
| Mambino-0-R | 48 | 96 | 6 | **926,402** | 70.1% |
| Mambino-G-R | 48 | 96 | 6 | 926,414 | 70.1% |

A 29.9% parameter reduction. Mambino-0-R is exactly iso-parameter with S5-0-R by
the same structural identity (predictor at P costs what doubling P costs), and
`block_size = 16` throughout, matching S5-Dense's own blocking. Same recipe,
same eight seeds, same `--val_split=0.1`, same metrics as §4.

**Claim under test (C4), fixed in advance.** Mambino-G-R reaches within 1.0 pp
of S5-Dense on `test@peakval` at 70.1% of its parameters. Falsified if the
shortfall exceeds 1.0 pp. **No direction is predicted for Mambino-G-R against
Mambino-G**, the iso-parameter config: fewer parameters should cost accuracy,
and the question is how much.

**Why this arm should carry the chip result.** In the iso-parameter arm the
64x64 gate array is deleted and the saving is spent on a larger state, buying
silicon back. Here the saving is kept: `main(48) + predictor(48) = main(96)`, so
the SSM silicon matches S5-Dense's while the gate array disappears entirely.
PPAC for these three configs is not yet run and no chip number is predicted here.

**Ordering.** This arm runs only after pilot 2 establishes that the mechanism
survives on IMDB at all under the fixed optimizer grouping. If pilot 2 is
negative, C4 is moot and the arm will not be run.

## 10. Addendum, 2026-08-05: seed set corrected to IMDB's published seed

**Defect.** §3 fixed the seed set as "6554595, 42, 12345, 271828, 314159, 1, 2, 3
-- the same eight as ListOps". Reusing the ListOps set wholesale was wrong.
6554595 is the seed the S5 authors published for **ListOps**
(`run_lra_listops.sh`); it carries no special status on IMDB. The seed they
published for IMDB is **8825365** (`run_lra_imdb.sh`), and it was absent.

**Correction.** The IMDB seed set is

    8825365, 42, 12345, 271828, 314159, 1, 2, 3

i.e. 6554595 is replaced by 8825365; the other seven are unchanged. Having the
task's own published seed in the set is what makes "at their published seed we
obtain X" a statement we can make on IMDB, as we already can on ListOps.

**Why this is not results-driven.** The change is motivated by which seed the
authors published for this task, a fact fixed before any run and independent of
any outcome. The three pilot-1 runs that used 6554595 are void regardless (§8,
optimizer divergence), so no valid result is being discarded. Everything else in
§§1-7 stands.

## 11. Addendum, 2026-08-05: arm priority and seed allocation

**Two arms, different claims, different n.** Recorded before any of these runs.

| arm | configs | params | claim | seeds |
|---|---|---:|---|---:|
| **-R** (reduced) | S5-0-R, Mambino-0-R, Mambino-G-R | 926,402 | **C4**, parameter reduction | **8** |
| **-iso** | S5-0-iso, Mambino-0-iso, Mambino-G-iso | ~1,314,230 | C1/C2, mechanism at fixed params | **4** |
| reference | S5-Dense | 1,321,154 | shared by both | 8 |

Rationale: the paper's headline on IMDB is the efficiency claim, so the arm that
carries it gets the statistical power. The bare names `S5-0` / `Mambino-0` /
`Mambino-G` remain aliases of the `-iso` arm so pilot-1 job names still resolve.

**Consequence that must be stated in the paper, not buried.** At n=4 the
mechanism comparisons on IMDB are **descriptive, not inferential**. For
calibration: on ListOps, C1 was t=2.20 at n=8 and needed n=16 to reach t=3.93.
At n=4 nothing short of an enormous effect will clear any significance bar, and
C1/C2 on IMDB should be reported as point estimates with their seed counts and
no significance language. The pre-registered falsification rules in §5 still
apply to the direction and the sign test; they simply carry less weight at this
n, and that is a deliberate resource-allocation choice rather than a discovery
made after the fact.

If the -iso point estimates come back contradicting ListOps, that is worth
knowing and reporting even at n=4, and would justify spending the extra four
seeds to settle it.

## 12. Addendum, 2026-08-05: blocks corrected in §2 (no parameter change)

**§2's `blocks` column and its "block_size = 23" sentence are wrong** and are
superseded here, per the rule at the top of this file. §2 itself is left intact.

`train.py:93` computes `block_size = ssm_size_base / blocks`, line 145 halves it
under conjugate symmetry, line 154 tiles `blocks x block_size`. So
`ssm_size_base / blocks` must be **EVEN**. The values in §2 give 23, which is
odd, so the halving truncates:

    276 / 12 = 23  ->  23 // 2 = 11  ->  11 * 12 = 132 state entries, 138 needed
    TypeError: mul got incompatible shapes for broadcasting: (132,), (138,)

Confirmed empirically: of the four §2 configurations only S5-Dense (192/12 = 16,
even) built at all. Corrected values:

| config | ssm_size_base | blocks §2 | blocks ACTUAL | block_size |
|---|---:|---:|---:|---:|
| S5-Dense | 192 | 12 | 12 (unchanged) | 16 |
| S5-0 | 276 | 12 | **6** | 46 |
| Mambino-0 | 138 | 6 | **3** | 46 |
| Mambino-G | 138 | 6 | **3** | 46 |

**No parameter count changes.** `blocks` only tiles the HiPPO initialisation;
`Lambda` ends up `ssm_size/2` however it is factored. All four §2 totals were
subsequently verified against the real `run_train.py`: 1,314,230 / 1,314,230 /
1,314,242 / 1,321,154, exactly as §2 asserts. So §2's *claims* stand; only its
`blocks` column was unbuildable.

The constraint is documented nowhere in the repo and every pre-existing S5
config satisfies it accidentally by using powers of two. Now noted in the
launcher header and in `mambino-paper/CLAUDE.md` §4.

## 13. Kill-switch

`--val_split` defaults to `0.0`, which reproduces the released loader
byte-for-byte. Only the IMDB launcher sets it. Every other dataset factory is
called with an identical argument list to before, verified by signature
introspection in `s5/train.py`.
