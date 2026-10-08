# Forecasting recovery and paper-route assessment

Prepared October 8, 2026. This is a recommendation and feasibility assessment,
not an adopted protocol, a new experiment submission, or a change to any gate.
The supplied proposal was checked against the research document, F2 artifacts,
the live recovery queue and observed label-generation costs.

## Recommendation

Stop architecture redesign. Begin the training/validation signal diagnostic
on the existing audited snapshot while the three recovery jobs wait. Start
writing the stable protocol and empirical-analysis sections in parallel.
The full proposed expansion to 250–400 training states, followed by recursive
policy experiments, is not a credible October 10 completion plan.

An empirical paper on the gap between immediate and continuation-dependent
update value is the more defensible deadline route. It requires explanations,
controls, uncertainty and replication, rather than a list of failed models.
Keep the architectural route conditional on a frozen method passing independent
evaluation and demonstrating actual recursive-policy benefit. Neither route
guarantees acceptance or an oral presentation.

## What the results establish

F2 passed 126 tests and all fifteen fits completed. The full H=2 model's mean
validation Spearman across seeds 42/43/44 is 0.16473. The independent H=2
control averages 0.26667. Neither primary passes the frozen Gate 2 criterion.

The independent control's own-horizon scores are:

| Training/evaluation horizon | Mean Spearman | Mean informative top-1 |
| --- | ---: | ---: |
| H=1 / H=1 | 0.41027 | 0.46667 |
| H=2 / H=2 | 0.26667 | 0.40833 |

These are three training seeds on the same forty validation states from eight
trajectories, not three independent validation populations. They motivate
investigating the horizon gap; confidence intervals must account for shared
states and trajectory dependence. Forecasting H=1 well does not satisfy an
H=2 gate or demonstrate recursive improvement.

Three qualifications to the supplied proposal matter:

1. F2 has **7,135 parameters**, whereas the independent GRU has **50,691**.
   Its weaker performance does not establish over-parameterization. Different
   capacity and inductive assumptions confound the architectural comparison.
2. Low forecast correlation does not establish intrinsically noisy or
   impossible-to-predict continuation outcomes. Limited samples, missing
   pre-commit information, fitting choices and evaluation resolution remain
   alternatives. A ceiling claim needs additional evidence.
3. All 531 candidate branches in the audited snapshot have continuation
   available. A terminal/non-terminal comparison is currently unavailable;
   earlier scarcity cases have different protocol provenance and cannot be
   pooled silently into the main A2 sample.

## Feasibility and timing

At 15:36 IST, the three one-generation recoveries were still pending for
Priority, without scheduler start estimates. Their dependent CPU audit/forecast
job is 1057272. Each recovery preserves two completed branches and recomputes
the unfinished branch. Its 48-hour reservation is a limit, not a runtime
estimate. The current snapshot remains usable for independent diagnostics.

The metadata-only timing check reports:

| Stream | Published states | Mean GPU node-hours per H=2 state |
| --- | ---: | ---: |
| Llama code | 45 | 3.169 |
| Llama math | 42 | 5.831 |
| Qwen code | 45 | 3.190 |
| Qwen math | 45 | 5.214 |

Balanced planning mean: **4.351 GPU node-hours per state**. The Llama-math
estimate excludes unfinished final-state work, so it may underestimate that
stream. Observed completed five-state trajectories take roughly 11–33 hours;
three other trajectories reached their 36-hour timeout. No twenty-node
allocation is guaranteed. Available project balance was 2,470 SUs; this is
allocation headroom, not a completion-time guarantee.

After existing recovery, training will contain one hundred states. Keeping
the old development and test partitions separate, a balanced expansion costs:

| Eventual training states | New training trajectories | Fresh validation trajectories | New states | GPU node-hours proxy | Ideal hours at 20 nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 260 | 32 | 8 | 200 | 870 | 43.5 |
| 400 | 60 | 8 | 340 | 1,479 | 74.0 |

Each trajectory has five states; the new validation population has forty
states. Counts are balanced across four model/task streams. With the current
97-state prefix these totals would be 257/397 until recovery completes.
The projections exclude loading, retries, final evaluations and queue gaps.
Long sequential trajectories can extend wall time beyond aggregate work
divided by twenty.

The previous core policy/robustness projection was 36–60 hours after Gate 2.
Combining it with expansion gives roughly **80–134 hours before additional
queues and broader confirmations**. That extends beyond the October 10 target.
The full proposed architectural rescue should therefore not be deadline-critical.

Planning effort for a smaller route:

- Existing-data decomposition, margin/tie analysis and initial report:
  approximately 2–4 hours of implementation/review after CPU access. The
  arithmetic itself should take minutes; the pipeline is not yet implemented.
- Fixed simple controls, whole-trajectory learning curves, uncertainty and
  model/task/head breakdowns: approximately 4–8 additional hours for validated
  implementation and interpretation, with compact fits on CPU compute nodes.
  The observed fifteen-fit F2 job took 62 seconds including its full suite,
  so CPU fitting is not the dominant computational cost.
- Optional independent forty-state validation: eight new balanced five-state
  trajectories, about 174 GPU node-hours. Allow roughly 24–40 hours after
  allocation, plus queue delays and possible long tails. Only eight GPU jobs
  exist in this scope; using twenty slots does not accelerate a single
  trajectory's sequential generations.
- Tables, plots, claim audit and a complete paper draft: budget a further
  12–24 hours of work, overlapping writing with analysis and any replication.
  This is an effort estimate, not a promise of a submission-ready paper.

## Recommended bounded scientific sequence

1. Freeze F2 and its negative results. Preserve recovery and original data.
   Run diagnostics on training/development data now; finish the full-data
   audit when recoveries complete. Do not wait for three additional training
   states to begin useful analysis.
2. Decompose H=2 labels into immediate and continuation increments. Report
   absolute and within-parent variation, H1/H2 rank agreement, strict reversals,
   winner margins, regret from choosing the immediate-best candidate, ties and
   per-head/model/task/generation breakdowns. Use all prescribed utility heads.
3. Audit pre-commit feature availability and within-parent variation. Several
   verifier features can be constant because only passing corrections survive.
   Existing loss, gradient, probe and relative-feature measurements should be
   assessed before collecting another expensive feature group. Relative
   features alone are not a new contribution; F2 already uses them.
4. Freeze a small comparison using the existing independent GRU and simple
   linear/heuristic controls, matched H=1/H=2 and continuation-residual targets.
   Use nested whole-trajectory training subsets and all three fitting seeds.
   Learning curves and grouped uncertainty can test data limitation with
   cached labels before committing to hundreds of new states. Predeclare
   selection and report the complete comparison, not its winning seed.
5. If claiming stochastic instability, measure it with a bounded, balanced
   continuation-repeat experiment on fixed saved M1 checkpoints. If claiming
   evaluation noise, use paired item-level uncertainty where outputs exist;
   aggregate sixty-four-example scores alone cannot establish a noise ceiling.
   Keep the frozen gate calculation and report sensitivity descriptively.
6. For a changed forecaster, use a fresh validation population; old validation
   has already informed development. Keep the held-out test out of fitting
   and selection. Before opening test results, freeze methods, features,
   claims, metrics and scope. A confirmatory finding also needs independent
   evidence, even when Gate 2 is not used as an architectural success gate.
7. Set an October 9 decision: pursue a positive architecture claim only with
   valid independent forecasting evidence and enough time for policy outcomes.
   Otherwise freeze the empirical-analysis route and its scoped claims, or
   defer submission if the mechanism/replication evidence remains inadequate.
   Broad 8B, PD and additional architecture work remain ineligible.

## Publication interpretation

An empirical/negative-results contribution can be considered for ACL Main;
it is not automatically a Findings publication. ARR explicitly recognizes
negative results and analysis contributions. Main versus Findings is a venue
decision, with soundness and reproducibility required for both and distinct
contribution/impact considerations for Main:
[reviewer guidance](https://aclrollingreview.org/reviewerguidelines),
[area-chair guidance](https://aclrollingreview.org/acguidelines).

A defensible claim would be that immediate update quality transfers imperfectly
to standardized H=2 value under the tested small-model LoRA protocol, with
identified conditions and evidence. Universal forecasting impossibility, a
general self-improvement ceiling, or an effective new architecture is not
supported by the current results. Any harmful-update claim must distinguish
the frozen any-negative-component definition from practically substantial
and statistically supported deterioration.

Evidence: `runs/deployment/evidence/forecast_route_feasibility_20261008.json`,
the original/F2 preserved archives and monitor responses 391/392. No additional
experiments were submitted during this assessment.
