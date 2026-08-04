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

## 7. Kill-switch

`--val_split` defaults to `0.0`, which reproduces the released loader
byte-for-byte. Only the IMDB launcher sets it. Every other dataset factory is
called with an identical argument list to before, verified by signature
introspection in `s5/train.py`.
