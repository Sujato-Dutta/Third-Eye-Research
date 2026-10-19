# Third Eye: When to Commit Recursive Parameter Updates

Research direction revised October 18, 2026. This is the active forward plan;
historical protocols, failures and results remain immutable. The study combines
controlled measurement with one prospective update-versus-hold policy test.
Successful forecasting, useful admission control and a causal explanation remain
hypotheses. This document does not launch experiments.

## 1. Question, contribution and claim boundaries

**Question:** When does committing a verified correction update improve future
utility, rather than merely repair the triggering failure, and can an inexpensive
rule recognize those cases before the update becomes persistent?

The intended contributions are:

1. A reproducible estimate of decision uncertainty in recursive **weight** updates,
   separating identifiable sources of variation and selection optimism.
2. A clean replication of update-versus-hold outcomes and admission headroom;
   one pooling sensitivity only if stable interactions warrant it.
3. A frozen admission rule evaluated against immediate selection, protected-case
   replay and holding parameters, including independent future scoring and cost.

The contribution must be the resulting finding and decision benefit. Variance
decomposition, abstention and confidence intervals are established tools.
Do not rename a failed predictor as a successful controller. Do not claim that
self-updates generally fail, future value is inherently unpredictable, or the
same phenomenon occurs in other systems without testing them.

## 2. Starting evidence

Completed infrastructure covers verified corrections, K=3 candidates, persistent
LoRA updates, rollback, H1/H2 labels, forecasting, isolated code verification,
item-level evaluation, provenance, Slurm dependencies and bounded accounting.

Completed collections include 36 original T=5 trajectories (180 states/540
labels), eight balanced T=5 confirmation trajectories (40 states/120 labels),
eight matched reliability parents, eight later development parents with five
continuations per candidate, a half-LR intervention, four early SmolLM3 states,
and the corrected 216-cell crossed experiment with 72 exact historical replays.
Collections overlap; their counts are not independent sample sizes.

Key results, in aggregate-utility percentage points with 95% intervals:

| Comparison | Result | Meaning |
| --- | --- | --- |
| Frozen learned H2 versus matched H1, balanced confirmation | Approximately 0.00 [-0.60, +0.63] | No demonstrated forecasting advantage. |
| Independently scored observed-future versus immediate selection | +0.34 [-0.55, +1.24] | Lookahead benefit unresolved. |
| Observed-future versus random, secondary | +1.61 [+0.88, +2.34] | Privileged future observations may help; this is not learned prediction. |
| Forced immediate-selected candidate plus continuation versus holding parent | -1.96 [-3.59, -0.33] | Historical motivation for admission control, not a general single-update harm estimate. |
| Half-LR primary selected-candidate effect | -0.24 [-0.97, +0.37] | No confirmed intervention benefit. |
| Corrected native model/correction pairing surplus | -0.19 [-0.61, +0.16] | No established aggregate native-pairing benefit. |
| Corrected native-future versus immediate selection | -0.65 [-1.38, +0.07] | Latest data do not support positive lookahead selection. |

Historical H1/H2 winner disagreement was 62.5%; fresh-continuation disagreement
was 68.75%. These are different comparisons, and excess horizon disagreement
was not established. Neither is a universal noise floor.

Llama's template inserted the current date. Restoring historical rendering
reproduced the investigated adapter exactly; corrected experiments freeze the
context. Historical cross-day effects, including the harm estimate above, need
fresh confirmation. This defect does not establish why earlier forecasts failed.

Evidence: [core](docs/evidence_review_20261011.md),
[forecasting](docs/forecasting_review_20261013.md),
[LR intervention](docs/mechanism_review_20261014.md),
[corrected matrix](docs/crossed_results_review_20261018.md).

## 3. Position relative to existing work

- [Self-Healing Harness](https://arxiv.org/html/2609.24130v1) already gates
  persistent instruction changes using triggering/protected cases while keeping
  model weights fixed. Its 55% statistic concerns replay-decided rejections,
  not all proposals. Our distinction must be parameter changes, future
  continuation consequences and independently measured decision value.
- [Rehearse](https://arxiv.org/html/2607.27687v1) ranks proposed code modifications
  using outcome memory. [Predict Before You Modify](https://arxiv.org/html/2610.13991v2)
  forecasts agent modifications; Appendix A.3 explicitly fixes underlying LLM
  weights. Predictive selection itself is already established.
- [TuneAhead](https://arxiv.org/abs/2606.17660) and
  [target-independent micro-interventions](https://arxiv.org/abs/2609.08618)
  predict training response. Cheap probes are not automatically novel.
- [SEAL](https://arxiv.org/abs/2506.10943) and
  [self-modification/consolidation](https://arxiv.org/abs/2606.03979) already study
  persistent adaptation. Do not conflate similarly titled sleep papers or assume
  their learning settings equal our two-example correction regime.
- [LoRA intruder dimensions](https://arxiv.org/abs/2410.21228),
  [SLAO](https://arxiv.org/abs/2512.23017) and
  [ReCoLoRA](https://arxiv.org/abs/2607.07719) cover spectral effects and continual
  adaptation. A spectral constraint is a baseline or sensitivity, not our default
  new method. [Selection optimism](https://www.jmlr.org/papers/v11/cawley10a.html)
  is also established.

The specific gap to test is **independently validated admission of endogenous
parameter corrections, including whether future labels add value beyond immediate
protected-case replay**. Useful admission and useful H2 supervision are separate
questions. Neither distinction is a first-ever claim.

## 4. Fixed core protocol and decision

Primary scope: Qwen3-4B and Llama-3.2-3B-Instruct, each on math and code. Retain
K=3, two unique verified corrections per batch, LoRA rank 16, LR 1e-4, 50 optimizer
steps and accumulation 8. Candidate batches may overlap but must differ. Keep
eight harvest attempts and fixed train portions GSM8K 1,024/MBPP 306. Scarcity is
reported; terminal continuation has M2=M1 and zero executed steps.

Utility is the equal-weight target/OOD/retention accuracy delta. Freeze separate
training, selection/calibration and sealed scoring manifests; preserve the
historical 64/64/256 evaluation sizes for comparability where independent item
availability permits. Audit overlap before collection; never silently shrink a
set or reuse development items as sealed confirmation.

Separate **which candidate** from **whether to commit**. First choose the
immediate-gain candidate using selection items and the fixed tie rule. Third Eye
then admits that candidate or preserves the parent. It does not search for a
different winning candidate after seeing future outcomes.

Compare two regularized linear gates with the same feature family: immediate
multi-head gain, correction/probe loss change, retention interference and effective
update norm. Train/calibrate one using H1 outcomes only and the other using H2
outcomes; neither uses spectral features. Freeze a development-only selection
criterion, breaking practical ties toward the cheaper H1 gate, then select one
primary gate before confirmation. Freeze all normalization and thresholds. Disclose
future labels used for development selection; H1-only fitting is not zero-cost
method development. Preserve a simple target-improvement/protected-replay gate
as a distinct baseline, not a synonym for either learned gate.

This is **before persistent commitment, after disposable candidate training**.
Candidate training, evaluation and gradients are not zero-cost features.
No confidence guarantee follows merely from naming a score a lower bound.
Future labels and scoring items are forbidden inputs. Existing code is improved
in place; no architecture sweep or separately branded successor is planned.

## 5. Experiments in order

### A. Saved-data measurement and development decision

Within two working days, run one declared compute-node analysis. Existing data
are already inspected: every new analysis of them is exploratory, and earlier
primary endpoints remain unchanged.

1. Estimate parent, within-parent candidate, continuation and item variability
   where replication identifies them. Fixed-batch optimizer repeats are separate
   controls. Harvest and optimizer changes coupled in a draw cannot be assigned
   separate variance components. Report uncertainty and non-identifiability;
   do not pool incompatible historical/corrected conditions into one model.
2. Test cross-draw recipient/donor residual agreement using disjoint item folds.
   A zero native-pairing mean does not exclude heterogeneous interactions, and
   squared interactions alone do not establish stable signal beyond noise.
3. Quantify same-outcome versus independent-scoring selection optimism for
   K=1,2,3 and predeclared item subsamples. Average subsets within parents.
   Larger-K curves are synthetic only; calibrate both null and recoverable-signal
   controls. The standard maximum-selection argument is not a new theorem.
4. Recompute forced-immediate-versus-hold utility on the corrected diagonal and
   matched parent scores. Replicate it on B's 16 fresh parents. A neutral or
   positive mean does not exclude useful admission if conditional gains differ.
5. Estimate admission headroom. For the fixed immediate-selected candidate, let
   mu_p be its expected future gain. Ideal admission value is mean(max(0,mu_p));
   improvement over the better constant action is that value minus
   max(0,mean(mu_p)). Never substitute max(0,a noisy scored gain). Use disjoint
   draws/items to choose admit/hold and score it, averaging rotations within
   parents. Label this a privileged independent-admission proxy, not the true
   oracle or a guaranteed upper bound. Report value versus both constant actions.
6. Fit the admission rules using whole-parent development folds. Report utility
   against immediate, replay-gate and no-update baselines, risk–coverage curves,
   accepted/rejected utility and false rejection of beneficial updates.

Produce a substantive descriptive precision table for 3–5 related studies:
independent units, repeats, item counts, K, selection/scoring separation, reported
gain/interval and any MDE computable from their own evidence. Contrast their
reported resolution with our design's sensitivity, marking missing information.
Our variance cannot invalidate their gains; different processes are not a common
noise floor. No spectral decompositions, new constraints or unrelated probes.

Calculate MDE/power for joint baseline comparisons, with pilot-variance sensitivity.
Compare buying more disjoint scoring items, more parents or more draws. Additional
items reduce item uncertainty, not between-parent/harvest variation; inference
must be benchmarked and charged, and finite item availability checked.

**Decision:** require clean heterogeneous benefit/harm and promising independent
admission headroom before C. Use 0.5 pp over the better constant action as the
practical target. If corrected and fresh development data do not demonstrate
credible headroom of that scale, stop before C; unresolved uncertainty permits
only a declared precision check within A/B's caps. A weak proxy is not proof
that ideal admission is impossible. Do not base the study on historical -1.96 pp.

### B. Fresh development and conditional pooling sensitivity

If A warrants it, collect 16 new initial parents: four collision-checked seeds
per stream, three continuation draws per candidate. Keep historical outcomes
development-only. New seed/item manifests are frozen before collection.

Only if A shows stable interaction, use eight preselected parents, two per stream,
to compare standard K=3 size-two selection with a separate K=1 pooled control.
Pool the exact union of that parent's existing candidate batches: 3–6 unique
examples, not a newly required size-eight pool. Deduplicate by problem ID using
a frozen correction tie rule. Include every designated parent; preserve original
scarcity stops. Do not restrict analysis to parents yielding six corrections.

Keep rank, LR and optimizer steps fixed per update; audit token exposure and all
costs. Pooling changes both data composition and search: call this a strategy
comparison, not a pure batch-size intervention. The core K=3 protocol is unchanged.
Do not assume six disjoint examples were previously required. No extra harvesting,
duplicates, spectral method or pooled-rule promotion into C after inspecting C.

Freeze the primary gate, both horizon controls, simple replay baseline and exact
confirmation design after B. Require useful discrimination and feasible power;
reject apparent gains from holding everything or exploiting one stream. Preserve
all negative results. Synthetic recoverability does not establish real-data power.

### C. Untouched admission confirmation

Prioritize 48–64 fresh parents, 12–16 per stream, with three continuation draws
per candidate. Use A/B to choose fixed N and scoring-item counts for at least
80% planned joint power for the two superiority tests at a 0.5 pp effect over
the better constant action. Separately evaluate retention noninferiority and the
practical point-estimate threshold; do not call their combined pass probability
80%. Include larger-effect and uncertain-variance sensitivities. The historical
40-state cohort had eight trajectory units, not 40 independent parents. Neither
32 nor 64 is automatically sufficient.
If item noise dominates, consider larger disjoint evaluation sets before more
training. Freeze allocation and N before C; no resizing from sealed outcomes.
If powered comparisons cannot fit, stop or narrow the claim before collecting C.

Select using calibration items and score on sealed items and independent
continuation seeds. Offline labels measure the candidate-plus-standardized-
continuation outcome; hold has zero parameter-change utility. This estimand is
not the value of an arbitrary adaptive recursive policy; stage D tests that.

Use a hierarchy fixed before C:

1. **Admission test:** the single frozen primary gate versus forced immediate
   selection and hold, with simultaneous 95% intervals. Require positive lower
   bounds for both and at least 0.5 pp estimated gain over the better constant
   action. Require the one-sided 95% retention lower bound versus hold above
   -0.5 pp. A rule that only matches hold shows no useful adaptation.
2. **Horizon-label test:** H2-trained versus H1-trained gating. This tests the
   extra value of future supervision, not whether admission works at all. A tie
   does not invalidate successful admission. Nonsignificance is not equivalence;
   any equivalence claim needs a predeclared +/-0.5 pp margin and adequate power.
3. **Novelty test:** compare calibrated gates with simple protected replay. A
   future-specific or improved-controller claim requires its own positive,
   independently scored advantage. If replay matches the method at adequate
   precision, report the practical result without claiming a novel superior gate.

Report all heads, harmful-update rates, coverage, calibration, no-update and
resource costs. Drop a separate TuneAhead adaptation; discuss its different
information/budget setting transparently. Adjust the declared horizon/replay
secondary family for multiplicity. Infer over independent parents, not candidates,
folds, model-fitting seeds or continuation draws. No repeated significance peeks,
favorable seed replacement or endpoint substitution.

Do not adopt whichever gate looks best on C. No H2 superiority means no restored
forecasting claim, even if calibrated H1 admission succeeds.

### D. Recursive decision value

This is secondary/descriptive, not a substitute for C. Only after useful C results
and measured affordability, plan 16 paired T=5 initial conditions, four seeds per
stream, with two evolving policies: the already frozen gate and the strongest
non-hold baseline selected using B, not favorable C outcomes. This is 32 rollouts,
not 32 independent pairs. Score hold from the same initial checkpoints.

Omit D if the minimum balanced design cannot fit its cap; do not buy eight weak
trajectories instead. Larger N requires power/cost justification, not adapter
counts alone. Use five **offered update opportunities**; rejection retains the
parent but counts its proposal cost. No confirmation claim from generation count.

Use matched initial conditions and equal total compute allowances. Histories
diverge after decisions; do not count generations as independent replicates.
Freeze primary final utility and retention endpoints; report cumulative utility,
coverage, accepted-update count, terminal states, tokens, SUs and elapsed time.
No full H2 branch-label expansion is needed online. Any savings must include
rejected candidate training, harvesting and diagnostics; do not promise 50% savings.

### Release and fixed scope

Remove the third-backbone, H=3 and spectral extensions. Scope stays two backbones,
two task families, LoRA, K=3 and H=2; T=5 evidence is conditional/descriptive.
The four historical SmolLM states remain a sanity check, not replication.

Release permitted outcome data, frozen manifests, variance/MDE and optimism
tools, checkpoint/template hash verification, the date-drift regression check,
source versions and full cost/failure records. Remove secrets and verify dataset
and model redistribution permissions. Preserve failed forecasters and nulls.

## 6. Budget and execution boundaries

**Hard cap: 1,200 SUs for the existing bounded-additions ledger and all work below.**
It already records 87.6311 SUs; remaining headroom is 1,112.3689 SUs. Earlier original
campaign costs must be disclosed separately; do not claim this cap retroactively
covers the entire historical project. Account availability does not raise the cap.

| Stage | Maximum additional SUs |
| --- | ---: |
| A: headroom, variance, precision audit and release analysis | 8 |
| B: fresh development, conditional pooled control | 192 |
| C: powered independent confirmation and scoring precision | 560 |
| D: optional paired recursive check | 250 |
| Failed jobs, retries and contingency | 78 |
| **New reservations maximum** | **1,088** |
| **Existing plus planned maximum** | **1,175.6311** |

Before fresh B outcomes, transfer 72 SUs from contingency to B for measured
runtime headroom; the total ceiling is unchanged. B uses selection-half H1
gains and actual update norms; archive probes but exclude them from gates
because their retention anchors overlap scoring items.

These are ceilings, not proven cost forecasts. Benchmark a representative job
before each GPU stage. Reserve every array element, validation, CPU minimum and
retry before submission. Reallocate between stages only before their outcomes
and within the hard cap; drop D before weakening C. If essential powered
comparisons cannot fit, stop rather than silently reduce controls. Never auto-renew
failed jobs.

Count candidate training as well as continuations: C's 48–64 parents imply
576–768 updates before terminal skips, not just 432–576 continuation updates.
D's two-policy design permits up to 480 candidate updates. Harvesting, token
length, evaluation and queue delay prevent converting these counts directly to
SUs or elapsed time. No stage proceeds from an unbenchmarked cost assertion.

All training, inference and statistical analysis runs in Slurm allocations;
login nodes only stage files and inspect scheduler/metadata. Use up to 20 allocated
Vista GH200s, prioritizing complete independent units. Queue delay is unbounded.

Target A within two working days, B within the following week, C within the next
week, D over the following two weeks; these are conditional milestones. Freeze
evidence by late November and reserve December for manuscript/reproducibility
review. No stage launches automatically after a scientific review stop.

**Completion decision:** retain the chosen title and measurement-led scope. Make
admission effectiveness the headline only after fresh utility and retention
criteria pass. Successful admission without an H2-label advantage is a distinct
result, not an automatic failure; matching simple replay does not establish a
new algorithmic contribution. If clean headroom, power or practical superiority
is absent, stop expansion and assess the narrower empirical result honestly.
Further novelty requires a justified finding, not a stronger-sounding title.
