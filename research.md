# Third Eye: When to Commit Recursive Parameter Updates

Research direction revised October 22, 2026. This is the active forward plan;
historical protocols, failures and results remain immutable. The study combines
controlled measurement with one prospective update-versus-hold policy test.
Historical forecasting recovery is closed. Useful admission and a causal
explanation remain unproven; the final study tests update-effect reliability.

October24 closeout: the [fixed32-parent final cohort](docs/final_confirmation_results_20261024.md)
and automatic analysis completed successfully. Forced future utility was-1.76pp
versus hold, joint95% interval[-3.11,-0.42]. The Qwen-minus-Llama difference
was+2.80pp[+0.27,+5.33], but positive Qwen benefit and useful admission did not
replicate. The fixed rule did not outperform hold or replay. Validation passed209
tests; no forecasters were fitted. The bounded ledger is542.5736 SUs used/reserved;
complete C cost was309.0019 SUs. Freeze evidence and write the scoped empirical
paper. No further experiment is queued or launched automatically.

October24 review follow-up: [the revision plan](revision_plan.md) uses the new
500-additional-SU allowance. The16-run SmolLM3 extension is approved for immediate
execution; other revision controls remain planned. Qwen3-8B and Llama-3.1-8B
are future DGX extensions when access returns. The eventual five-model study
would span family and scale; unrun models are not counted as results.
The original confirmation and its endpoints remain unchanged.

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

October 21 outcome addendum: the [fresh 16-parent B collection](docs/admission_results_20261021.md)
completed cleanly, with no terminal draws. Forced candidate-plus-continuation
utility was -2.31 pp versus hold; both separate parent/item intervals exclude
zero. Learned H1/H2 gates were +0.16/+0.13 pp versus hold, with intervals crossing
zero and no demonstrated replay or forecasting advantage. Privileged admission
headroom was +0.84 pp, not learned prediction. The declared development rule
provisionally prefers H1. Confirmation remains unreleased: useful learned
admission and the planned power requirement are unresolved. This addendum
records outcomes; the predeclared protocol and endpoints below remain unchanged.

The [exploratory CPU audit](docs/admission_decision_audit_results_20261021.md)
reproduced B fits. A training-only H2 backbone mean achieved +1.14 pp versus
hold in both CV schemes; crossed parent/item uncertainty remains [-0.01,+2.36].
It admits Qwen and holds Llama; candidate features add no consistent benefit.
This is context calibration, not restored forecasting. C remains stopped for
independent-item precision, provenance and measured-cost review.

October 22: [four inference benchmarks and checked fixture classification](docs/scoring_precision_results_20261022.md)
completed. No recorded research exposure remained in the searched artifacts;
this is not an absence-of-exposure guarantee. Measured scoring costs now inform
design, but a powered, fully priced confirmation remains unfrozen.

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

Historical learned H1/H2 gates failed to establish practical or forecasting
advantage. They remain negative evidence, not a requirement for further fitting.
The final cohort compares only frozen hold, forced, replay and model-family
rules. New final outcomes never select a rule, threshold or candidate.

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

### C. Final empirical confirmation

The October 22 design audit did not establish an affordable, powered learned-
admission confirmation. Preserve that decision. The active final experiment is
now the separately declared [32-parent empirical confirmation](docs/final_confirmation_20261022.md),
with no new forecaster fitting or forecasting comparison. H2 remains an observed
future consequence, not a prediction model. This replaces the unexecuted C/D
proposal below the completed development work; historical declarations stay intact.

Use eight independent fresh seeds per model/task stream, the same K=3 size-two
A2 update budget and three native continuation draws. Freeze 128 target,128 OOD,
256 retention scoring items per task independently of selection. Exclude the
eight reserved code cost-calibration items; disclose HumanEval final OOD versus
MBPP development OOD. Freeze source, identities, seeds and rules before outcomes.

Primary: Qwen-minus-Llama future forced-update utility, macro-averaging math/code,
with joint parent/shared-item two-sided95% intervals. This tests an interaction
for two fixed models, not universal harm or a causal model-identity effect.
Pooled harm is secondary: heterogeneous benefit can exist when its mean is zero.
Require conditional planned power at least80% at75% pilot effect and1.25 SD
inflation before collection. Preserve the failed joint two-endpoint calculation.

Compare hold, forced update, protected replay, and a transparent development-
derived fixed rule that commits Qwen and holds Llama. The latter is a scoped
baseline, not a novel learned controller. No test-time fitting. Save actions
before final outcomes. Score parent, three M1 and nine M2 checkpoints per parent.
Retain terminal M2=M1 and all failures; never replace unfavorable seeds.

Secondary family: family-rule minus hold/forced/replay; replay minus hold/forced;
forced future minus forced immediate. Report Holm adjustments and simultaneous
intervals, retention, coverage, all heads/streams, scarcity and resource costs.
Useful family-rule admission requires positive simultaneous comparisons, at least
0.5pp above the better constant and retention lower bound above-0.5pp. A positive
model gap alone does not prove a benefit/harm crossover or useful admission.

### D. Evidence freeze and writing

Do not launch additional recursion, scaling, architecture sweeps or forecasting
recovery for this paper. Existing T=5 results remain clearly labeled historical
or development evidence. After the single fixed cohort, produce verified JSON,
CSV, findings text, uncertainty plots and provenance/cost records automatically.
Write the supported empirical paper whether the interaction replicates or not;
no automatic follow-up or result-dependent reframing into method success.

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
At declaration it recorded 87.6311 SUs. Through October 22 provenance/cost closeout,
the ledger records 233.4067 SUs; remaining cap headroom is 966.5933 SUs. Earlier original
campaign costs must be disclosed separately; do not claim this cap retroactively
covers the entire historical project. Account availability does not raise the cap.

| Stage | Maximum additional SUs |
| --- | ---: |
| A: headroom, variance, precision audit and release analysis | 8 |
| B: fresh development, conditional pooled control | 192 |
| C: powered independent confirmation and scoring precision | 560 |
| D: no new recursive campaign for this paper | 0 |
| Failed jobs, retries and contingency | 78 |
| **New reservations maximum** | **838** |
| **Existing plus planned maximum** | **925.6311** |

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

Count candidate training as well as continuations: the fixed32-parent C cohort
allows384 updates before terminal skips and416 checkpoint evaluations. Its
conservative complete reservation after the documented runtime correction is
533.355 SUs, within C's560-SU ceiling.
No new recursive or forecasting campaign is part of paper completion. Harvesting,
output length, verification and scheduling prevent promising an exact finish.

All training, inference and statistical analysis runs in Slurm allocations;
login nodes only stage files and inspect scheduler/metadata. Use up to 20 allocated
Vista GH200s, prioritizing complete independent units. Queue delay is unbounded.

A/B are complete. Queue the one fixed C cohort with validated dependencies and
up to20 concurrent allocated GPUs, then freeze evidence and write. No new study
is launched automatically after that cohort. Scheduler delay remains unbounded.

**Completion decision:** retain the chosen title and measurement-led scope. Make
admission effectiveness the headline only after fresh utility and retention
criteria pass. A useful fixed baseline is a decision result, not restored forecasting; matching
simple replay does not establish a new algorithmic contribution. If clean headroom, power or practical superiority
is absent, stop expansion and assess the narrower empirical result honestly.
Further novelty requires a justified finding, not a stronger-sounding title.
