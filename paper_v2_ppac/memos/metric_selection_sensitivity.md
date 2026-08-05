# Metric selection sensitivity on LRA-ListOps

> **UPDATE 2026-08-05: the n=16 extension substantially refutes the Claim 1
> fragility described below.** On the eight new pre-registered seeds all three
> selection rules agree (+0.74 / +0.74 / +1.03 pp), so the `test_max` anomaly was
> a property of the original eight seeds, not of the mechanism. Pooled over all
> sixteen, Claim 1 is positive under every rule and significant under two,
> including the selection-free one:
>
> | metric | orig 8 | new 8 | all 16 |
> |---|---:|---:|---:|
> | test@peakval | +0.76 (7/8) | +0.74 (6/8) | **+0.75 pp, t=3.93, 13/16** |
> | test_max | +0.04 (3/8) | +0.74 (7/8) | +0.39 pp, t=1.94, 10/16 |
> | last-epoch | +0.30 (5/8) | +1.03 (7/8) | **+0.66 pp, t=2.73, 12/16** |
>
> n=16 means: S5-0 0.5933 +- 0.0071, Mambino-0 0.6008 +- 0.0065.
>
> The §"What this shows" conclusion that "the gate is the better-supported
> claim" was true at n=8 and is **no longer true**. Claim 1 now has the stronger
> statistics. Claim 2 is still at n=8 pending Mambino-G's extension; revisit
> when it lands. Everything below is retained as the n=8 record.

**Computed 2026-08-04**, from the eight matched-seed run logs, before the n=16
extension landed. Recorded so that the paper's metric choice is documented as
having been made *with* the alternatives in hand.

**Decision (author, 2026-08-04, final): `test@peakval` is the primary metric.**
All comparisons between configurations use it and only it.

## The three metrics

All computed from the same 40-epoch curves in the same run logs, so the runs are
identical and only the selection rule differs.

- `test@peakval` — test accuracy at the highest-validation epoch. Selection on
  held-out validation data.
- `test_max` — best test accuracy at any epoch. Selection on test.
- `last-epoch` — test accuracy at epoch 40. No selection of any kind.

| config | test@peakval | test_max | last-epoch |
|---|---:|---:|---:|
| S5-0 | 0.5917 ± 0.0081 | 0.6034 | 0.5970 |
| Mambino-0 | 0.5993 ± 0.0072 | 0.6038 | 0.6000 |
| Mambino-G | 0.6050 ± 0.0042 | 0.6094 | 0.6039 |
| S5-Dense | 0.6089 ± 0.0051 | 0.6159 | 0.6091 |

| contrast | test@peakval | test_max | last-epoch |
|---|---|---|---|
| Mambino-0 − S5-0 (predictor) | **+0.76** t=2.20 7/8 | +0.04 t=0.13 3/8 | **+0.30** t=0.84 5/8 |
| Mambino-G − Mambino-0 (gate) | +0.58 t=1.84 6/8 | +0.56 t=3.48 7/8 | +0.39 t=1.35 4/8 |
| Mambino-G − S5-Dense | −0.39 | −0.65 | −0.52 |

## What this shows

**Claim 1 (predictor) is magnitude-sensitive to the selection rule.** Positive
under all three, but +0.76 / +0.04 / +0.30 is a 19x spread. The reported +0.76
is the most favourable of the three.

**Claim 2 (gate) is robust.** +0.58 / +0.56 / +0.39, positive under all three,
and strongest under `test_max` (t = 3.48), the rule that most disadvantages it.
By this evidence the sixteen-scalar gate is the better-supported of the paper's
two mechanism claims, which is the reverse of the current emphasis.

**Claim 3 is stably negative** under all three rules, between −0.39 and −0.65.

## Mechanism: it is variance, acting in both directions

| config | last-epoch − test@peakval | cross-seed sigma |
|---|---:|---:|
| S5-0 | **+0.53 pp** | 0.0081 |
| Mambino-0 | +0.07 | 0.0072 |
| Mambino-G | −0.11 | 0.0042 |
| S5-Dense | +0.02 | 0.0051 |

S5-0's validation-selected checkpoint is half a point *worse* than simply taking
epoch 40. ListOps ships 2,000 validation examples, so validation accuracy carries
roughly +-1.1 pp of standard error; against S5-0's noisy training curve that is
enough to select a poor epoch. The same variance that makes max-over-test pick a
lucky epoch for S5-0 makes max-over-validation pick an unlucky one. Both are
selection artifacts of the same underlying noise, pulling opposite ways;
last-epoch is subject to neither.

So `test@peakval` does not merely fail to flatter S5-0. It penalises it, and
that penalty inflates Claim 1.

## Rationale for the decision, stated in full

`test@peakval` is retained as primary because selecting a checkpoint on the test
set is not a defensible protocol at any venue, whatever it shows, and because a
rule whose bias depends on the model under test cannot arbitrate between models.
That argument does not depend on which way the numbers fall.

The counter-consideration, recorded honestly: on this task the chosen rule is
also the one most favourable to Claim 1. Both facts are true simultaneously.

## Consequences to carry into the paper

1. The n=16 extension matters more for Claim 1 than for Claim 2. Claim 2 is
   already robust; Claim 1's magnitude is what needs pinning down.
2. Consider whether the abstract's emphasis should shift toward the gate, which
   the evidence supports more strongly than the predictor.
3. **Author decision, 2026-08-04: this nuance stays in this memo. The paper
   reports `test@peakval` only.** The alternative considered and declined was a
   footnote or appendix table carrying all three. Residual risk, recorded so it
   is a known and accepted one rather than an oversight: the run logs are
   released, so a reviewer can recompute `test_max` and `last-epoch`
   independently and may ask why Claim 1 moves under them.
4. The IMDB pre-registration already commits to reporting all three metrics for
   that task. That commitment predates this analysis. It is left standing:
   pre-registered commitments are not revised in light of results, which is what
   makes them worth anything.
