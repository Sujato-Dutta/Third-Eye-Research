# Empirical strengthening package: October 11 feasibility

October 8 assessment; recommendation only. No protocol adoption or new
scientific jobs are triggered by this document. Assume the deadline means
the end of October 11, 23:59 IST, leaving approximately eighty hours.
If it means 00:00 at the start of October 11, approximately fifty-six hours
remain and the GPU package is substantially less reliable.

## Judgment

Priorities 1–5 can fit a bounded empirical study if GPU allocation starts
promptly. They cannot be guaranteed with the current scheduler state. The
three checkpoint recoveries remain pending, with no start estimate, at
16:21 IST on October 8. The allocation reports 2,470 available SUs. Additional
model breadth is lower priority than measurement reliability and replication.
The pasted acceptance probabilities are subjective and are not evidence.

## Minimum package and timing

| Work | Bounded scope | Planning effort/runtime |
| --- | --- | --- |
| Selection versus random/oracles/holding the model | Use all cached outcomes; pair by trajectory; report future value as well as regret, harm and head trade-offs | 1–3 hours implementation/review; short CPU compute |
| Noise and stochastic continuation sensitivity | Predeclare 12 parent states, three per stream, spread over generation and distinct trajectories; keep fixed candidates; collect aligned item outcomes; fixed-batch optimizer repeats plus two additional continuation seeds | 4–8 hours implementation/validation; roughly 8–18 hours after GPU allocation; provisionally 80–140 GPU node-hours |
| Balanced replication | Two new T5 trajectories for each of four streams; eight independent trajectories / forty complete K=3 states; frozen methods and analysis, no selection from new outcomes | Approximately 174 GPU node-hours; 24–40 hours after allocation, with possible longer tails |
| Mechanism associations | Predeclare a few feature groups: probe/update magnitude, gradient/retention interference and data diversity; examine within-parent differences and generation/task interactions | 4–8 hours implementation/analysis; mostly cached data and CPU |
| Detectability and prior-work comparator | Known-signal synthetic positive controls, grouped uncertainty and existing learning curves; fixed TuneAhead-style tabular comparator using available pre-commit inputs | 2–6 hours implementation/analysis; CPU once dependencies are ready |

These are planning estimates, not measured runtimes for the new noise pipeline.
Noise costs are provisionally inferred from existing parent-harvest and full
label costs, retaining K=3, fixed update budgets and verified corrections.
Profile a representative early wave before promising its completion time.
The GPU package is roughly **254–314 additional node-hours**, plus existing
recovery, validation and any extra evaluation required by a measured noise
problem. Cumulative campaign usage, actual SUs and running/reserved jobs must
be rechecked. The twenty-GPU cap includes existing recovery jobs.

The independent jobs can overlap, but each five-generation trajectory is
sequential. Aggregate GPU-hours divided by twenty is not an honest wall-time
estimate. Production queue waits have no reliable upper bound. Do not fill
twenty slots by adding unplanned experiments.

## Measurement distinctions

Evaluation uses greedy decoding and conditional-likelihood multiple-choice
scoring with deterministic execution. Changing only the evaluation RNG seed
may reproduce identical predictions. That proves numerical repeatability,
not statistical precision on sixty-four-item development sets.

Current labels store aggregate accuracy, not complete aligned item outcomes.
Paired item-level resampling therefore requires saved-checkpoint evaluations
that preserve item IDs and correctness at parent/M1/M2. Report winner
stability, interval-valued pairwise margins and oracle-selection optimism.
Do not manufacture an independent-binomial null that ignores shared items.

Keep three factors separate: finite evaluation sampling uncertainty,
optimization-seed variation with a fixed correction batch, and new
continuation-harvest/batch variation from a fixed M1. The last factor changes
the actual future process; it is not just measurement error. Compare the
original H1 ranking to seed-averaged H2 outcomes descriptively without
rewriting the frozen single-continuation A2 labels or Gate 1.

For K=3, independent random unique-winner rankings disagree two-thirds of
the time. Thus 57.5% observed disagreement is not by itself evidence of
excess instability. The relevant null should be estimated from matched
same-update repeats and paired evaluation uncertainty under the actual
protocol; unrelated random rankings are only a reference.

Regret against a privileged future oracle is nonnegative by construction.
It does not establish a deployable improvement. Compare observed H1-greedy,
uniform random, future oracle and holding parameters fixed over both steps,
including net future utility and harm. Distinguish observed-data oracle
advantages from repeat-stable differences in expected value.

## Forecasting and mechanism limits

Synthetic positive controls can show that a pipeline recovers a prescribed
signal. They do not increase the independent sample size of the real negative
result or prove intrinsic unpredictability. Report detectable effect sizes
and uncertainty at trajectory level. Three fitting seeds do not triple the
number of independent test trajectories.

Observed immediate gains and observed candidate margins are legitimate
post-update/oracle comparison signals, with their measurement cost stated.
They cannot be inserted into a predictor advertised as using only pre-commit
information. Keep those information budgets explicit in every baseline.

The old test results have been examined. Any newly developed comparator must
be frozen before the fresh balanced cohort is evaluated. Preserve old results
and do not rearrange the old split after discovering missing Llama-code.
Do not choose a feature subset, regularizer, training size or seed from the
new replication outcome.

TuneAhead combines static descriptors and standardized probe measurements
with LightGBM. A cached-feature adaptation is feasible, but it must be
identified as an adaptation: its original probe has one hundred steps,
whereas this protocol's pre-commit probe has ten and a full candidate update
has fifty. A full faithful reproduction would require additional features
and an incompatible probing budget; it is outside this deadline package.
Source: [TuneAhead paper](https://arxiv.org/html/2606.17660v1).

Feature associations can support an explanation of conditions under which
rankings change; they do not identify causal mechanisms by themselves. Use
within-parent contrasts, trajectory-aware intervals and correction for the
few predeclared tests. A stronger causal claim requires an intervention.

## Schedule and stopping decisions

1. October 8 evening: freeze noise/replication/comparator scope, validate the
   operational extension, submit the first noise wave, and compute cached
   regret/null/power analyses. Begin stable paper sections immediately.
2. October 9 morning: review the preliminary noise results. If the observed
   utility differences are largely explained by measurement or stochastic
   variation, revise the headline or stop the positive-instability story.
   Preserve the planned/completed sample and failures; no favorable-state
   selection or silent truncation.
3. Aim to start balanced replication by **October 9 noon**, prioritizing its
   longer math jobs. The noise and replication tasks share a rolling cap of
   twenty GPU nodes including original recovery. A start later than October
   10 morning leaves inadequate completion and review margin.
4. Aim to finish GPU outcomes by October 10 night / October 11 morning.
   Use October 11 for final checks, statistics, plots and claim freeze.
   Incomplete replication must be reported and narrows the supported claims.
5. Write throughout. Reserving all writing until October 11 midnight risks
   the October 12 submission even if experiments finish.

The core eligibility decision is scientific, not just a date: meaningful
repeat-stable value differences and balanced replication would strengthen
the empirical paper. Passing noise checks alone does not make replication,
power or explanation merely cosmetic, and none guarantees acceptance.

Do not add a third/8B model or longer-horizon experiments to the critical
path. The existing no-8B-before-Gate-2 restriction still applies. Claims remain
about H=2 forecasting inside five-generation recursive chains; an extra model
does not substitute for evidence at longer forecast horizons.

The content fits the **Interpretability and Analysis of Models for NLP** area,
which ARR lists explicitly. Track fit is about the contribution, not a lower
acceptance standard: [ARR areas](https://aclrollingreview.org/areas).
