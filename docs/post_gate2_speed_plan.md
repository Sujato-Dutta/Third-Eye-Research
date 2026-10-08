# Post-Gate-2 execution priorities

Prepared October 7, 2026. This is a prospective execution recommendation, not
a protocol amendment, scope adoption, benchmark result, or launch authorization.
The current A2 runs and Gate 2 review stop remain unchanged.

## Preserve the decisive evidence

Keep the 96 core five-generation policy trajectories: two core backbones, math
and code, three seeds, and all eight policies. Preserve the 18 order-stability
trajectories, matched H=1 comparison, greedy baseline, Direct versus Dynamics,
target/OOD/retention evaluation, confidence intervals, and complete outcome logs.
Keep K=3, unique examples within batches, candidate size two, eight correction
attempts, the fixed harvest portions, fifty-step updates, and all scarcity rules.
Core recursive improvement is still unproven even if Gate 2 passes.

## Execution savings, in priority order

1. Use a rolling controller capped at twenty GPU nodes and the live Slurm
   submission limits. Prioritize longer math trajectories to reduce the final
   scheduling tail. Release only dependencies that have actually passed.
2. Run each endpoint's frozen final evaluation as soon as that trajectory is
   ready, ideally in its existing GPU allocation. Do not wait for every policy
   trajectory to finish. Final-test results cannot affect subsequent choices,
   forecaster selection, or experimental scope.
3. Interleave the predeclared robustness trajectories with core comparisons.
   Keep statistics, reporting, and compact fits on scheduled CPU allocations.
   Bootstrap work uses the frozen evaluation results. Draft stable paper
   sections while compute jobs run.
4. Consider caching identical computations across policies. Cache eligibility
   must bind the exact base-model revision and adapter hash, source/execution
   profile, manifests, ordered prompts, sampling seeds and generation settings,
   training configuration, and batch composition as applicable. Never reuse a
   correction pool from a different evolved model. Rebuild policy-specific
   history/context, and keep post-update outcomes unavailable to selectors.
5. Validate any caching implementation on compute nodes against the uncached
   execution: completions, verifier outcomes, ordered batches, encoded features,
   policy choices, adapter hashes and evaluation results must agree. Cache
   validation must not delay the main launch while an unvalidated optimization
   is being developed. The existing validated runner remains the fallback.
6. Record actual shared compute separately from each policy's standalone
   computational cost. Cached execution must not manufacture a favorable
   compute-benefit comparison or additional independent seeds.

The next online stage already uses one parent harvest per generation rather
than the four harvests used for H=2 labels. That saving is already included in
the current timing estimate; it is not a newly achieved optimization.

## Quantitative planning bounds

The estimate derived from 112 audited A2 states gives about 476 GPU-hours for
core updating generations and 121 for order stress. Together these require
about thirty ideal hours at twenty nodes, before no-update trajectories, final
tests, loading overhead, scheduling gaps, and CPU analysis. These are component
projections, not measured online runtimes.

For each model/task/seed, seven updating policies start from the same parent.
Sharing only the first parent harvest could avoid six duplicate harvests per
group. Across the twelve core groups, the observed mean harvest durations give
an opportunity of approximately 53 GPU-hours. Stress has a further estimated
11-hour opportunity. This is roughly 11% of the combined updating/stress proxy;
cache-hit equivalence and actual savings remain unmeasured. Additional shared
prefixes, probes or identical endpoint evaluations may save more, but are not
included in a promised speedup.

Retain the current 36-60-hour post-gate core planning range until the first
online wave measures actual costs and scheduling overlap. Do not promise an
unbenchmarked factor-of-two speedup.

Evidence: `runs/deployment/evidence/post_gate2_timing_proxy_20261007.json`.

## Conditional scope priorities

If a deadline-driven scope decision is necessary, preserve the entire core
study. Prioritize one additional-family confirmation (Gemma-3-4B-IT) and one
same-family scale confirmation (Qwen3-8B). Llama-3.1-8B is the lower-priority
additional confirmation. Deferring it would reduce conditional confirmation
labels from nine to six trajectories and policy comparisons from twenty-seven
to eighteen, without changing any core seed, policy or task. It would remove
evidence about that particular 8B backbone; claims must narrow accordingly.
This recommendation has not been adopted, and no confirmation jobs are released.
Any scope decision should be recorded before inspecting final-test results.
Keep H=3 and any unactivated PD extension outside the deadline-critical work.

ARR states there is no fixed sufficient count of models or datasets for every
claim; evidence and scope must agree. This supports an explicit, well-supported
scope rather than an acceptance guarantee:
[ARR author guidance](https://aclrollingreview.org/authors),
[ARR reviewer guidance](https://aclrollingreview.org/reviewerguidelines).
