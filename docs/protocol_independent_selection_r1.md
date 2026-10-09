# R1: independent-seed and disjoint-item selection analysis

Declared October 9, 2026 before examining E2 control outcomes. This separate
analysis consumes the existing eight-parent E2 reliability sample. It changes
no model, optimizer, correction, evaluation, candidate or continuation budget.
No new GPU runs, fitting, intervention or method selection are authorized here.
Original E2 outputs and analysis remain intact. Failed parents are not replaced;
incomplete coverage produces an incomplete review rather than a full comparison.

## Primary comparison

For each K=3 parent, select the future candidate using fresh continuation offset
+50,000,000 and score that fixed selection using offset +100,000,000. Select the
immediate comparator using original M1. Use equal weights over tied maxima,
rounded to twelve decimal places. Compare future utility of the two selections
on the scoring continuation. Uniform random selection uses all three candidates;
holding parameters has zero utility relative to the unchanged parent.

Selection and scoring use disjoint evaluation item halves in each task role.
Sort item identities by SHA256 of the fixed namespace
`third-eye-independent-selection-r1`, role, item ID and prompt hash. The first
half selects, the other scores; swap halves and average the two estimates
within each parent. Parent subtraction uses the scoring half's actual outcomes.
Equal weighting of target/OOD/retention heads matches frozen A2 utility. The
primary is **independent future selector minus immediate selector**, evaluated
on another continuation seed and other items. It may be negative. It is not
regret against a true oracle and is not clipped to zero.

This avoids choosing and scoring the future maximum from the same realized
outcomes. It does not establish the true best candidate, an independent
population evaluation, or an unbiased regret estimate against an unknown oracle.
The shared finite item pools, trained M1s, verifier and small cohort still limit
inference; half-sized selection sets also have additional noise. Terminal
continuations remain M2=M1 and must not be removed or resampled.

## Secondary comparisons and practitioner evidence

Report the same +50M-to-+100M comparison using full shared items, explicitly
labeled as sharing evaluation items. Also report all six ordered selection/
scoring pairs among original M2 and the two fresh continuations, with disjoint
item folds. None becomes a selected primary after results are seen.

Use the three continuation outcomes to compare selection based on one versus
two continuations. For each held-out continuation, select using each of the
other two individually and average their scored utilities; compare with selection
using their mean. Score only on the held-out continuation's scoring fold. Average
three held-out choices and both fold orientations within each parent. This is
an observational selection comparison, not a new training intervention. It can
support a bounded cost/value discussion for one versus two continuation samples;
it cannot identify a universally sufficient number of repeats or reliable
selection of a population-optimal candidate. Report greedy, random and holding
parameters alongside it, without claiming equivalence from nonsignificance.

Retain the existing paired noise comparisons: original H1/H2 strict-winner
reversals, fixed-candidate optimizer variation, fixed-continuation optimizer
variation, and fresh-continuation variation. Paired differences are descriptive;
the controls are different stochastic processes, not a universal interchangeable
noise floor. A positive raw reversal rate alone supports no excess-noise claim.

## Units, execution and stopping

Require eight distinct trajectories with two parents per four streams. Bootstrap
5,000 times within stream over parent trajectories using seed 20261009; aggregate
with equal stream weights. Fold orientations and continuation pairs are repeated
measurements, not extra independent observations. Main intervals are conditional
on the fixed evaluation item pools; the existing paired-item uncertainty remains
a separate sensitivity analysis. No new acceptance or significance gate is added.

Validate code and contracts in a scheduled CPU allocation before enabling the
analysis controller. All real statistics and archives use CPU Slurm jobs. A
standard-library login watcher reads only scheduler and receipt metadata, waits
for the core and known noise jobs to end, then submits one bounded CPU review.
The review records incomplete coverage, preserves failures and archives source,
input hashes and outputs. No automatic architectural/scaling decision follows.

A versioned login relay also reconciles already-terminal `afterany` review
dependencies, which Slurm may purge before the late review is submitted. It
requires accounting for every job, retains active dependencies, preserves original
and normalized requests, and accepts only the known E2 control jobs. Failed
dependencies remain failed evidence and lead to incomplete scientific review.
Frozen original relay and experiment sources remain unchanged.

SmolLM's four early states remain a limited sanity check, excluded from this
noise/regret cohort. Mechanism analyses remain correlational. No causal
intervention or broader-model result is inferred from this supplement.
