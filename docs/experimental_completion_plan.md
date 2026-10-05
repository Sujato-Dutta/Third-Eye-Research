# Experimental completion planning, October 3, 2026

This is a planning assessment, not a protocol amendment or launch authorization.
The active A1 calibration, its four-GPU cap, review stops, scientific budgets,
source, and the holds on 20-GPU expansion and 8B experiments remain unchanged.

Update: the subsequent October 4 authorization supersedes the original resource
and scheduling holds. The implemented controller handoff, allocation, and
review boundaries are recorded in [deadline execution](deadline_execution.md).
The four-GPU and 500-hour descriptions in this assessment are historical.

## Measured cost

The four completed seed-1042 pilots provide the following measurements.
Harvesting includes the parent pool once and each of the three continuation
pools once; the parent statistics repeated in candidate records are not summed
three times. Training includes all three candidate and three continuation
updates. Elapsed time is the corresponding completed Slurm task duration.

| Stream | Elapsed | Harvesting fraction | Optimizer training fraction |
| --- | --- | --- | --- |
| Qwen math | 4h 40m 50s | 87.9% | 5.1% |
| Qwen code | 3h 01m 40s | 84.3% | 7.9% |
| Llama math | 3h 14m 14s | 88.0% | 5.1% |
| Llama code | 3h 13m 51s | 88.7% | 5.0% |

Mean cost is 3.5441 GPU node-hours per complete matched state. The weighted
harvesting fraction is 0.8734205. Generation dominates harvesting; the logged
verifier CPU time is small, so additional verifier workers alone will not
remove this bottleneck. These measurements cover four root states, not later
recursive generations, larger models, or a full online policy comparison.

Local evidence: `runs/deployment/evidence/a1-*-labels*.jsonl`, including
`a1-qwen-math-complete-labels.jsonl`, and the completed pilot inspection described
in [A1 execution](a1_execution.md).

## Throughput scenarios

Holding other work fixed, an s-fold harvesting improvement gives total runtime
factor `(1 - 0.8734205) + 0.8734205 / s`.

| Harvesting speedup | Overall speedup | GPU hours per state | 180 label states |
| --- | --- | --- | --- |
| 1x, measured baseline | 1.00x | 3.54 | 638 hours |
| 2x, unmeasured target | 1.78x | 2.00 | 359 hours |
| 3x, unmeasured target | 2.39x | 1.48 | 266 hours |

The default study planner produces up to 180 core label states: two backbones,
two task families, three seeds, three label trajectories, and five generations.
This is an extrapolation of that planner, not an approved post-A1 launch plan.
These totals exclude previous spending, online policy trajectories, transfer,
profiling, and final evaluation. Even the 359-hour case needs a full remaining
budget calculation before submission. Speedup targets are not achieved results.

### October 4 measurement update

All four seed-2042 calibrations completed successfully. Together with the first
pilots, the eight complete states average 3.7648 GPU hours per state, with
87.71% of elapsed time spent harvesting. The same 180-state extrapolation is
now about 678 GPU hours, rather than the earlier 638-hour four-pilot estimate.
This remains a core-label-only projection, not the complete five-model budget.
Tracked GPU spending is 50.3094 hours; the twelve remaining calibrations are
still covered by the active controller's unchanged reservations and ceiling.

### Scheduling opportunity

The active A1 controller uses four-task barriers: it waits for every task in a
wave before submitting the next seed's wave. In seed 2042, Qwen math took
6h 09m 47s while the other streams took approximately three hours. Subsequent
fixed independent states therefore could not fill the finished tasks' slots.
For the next reviewed execution plan, use rolling arrays capped at the approved
concurrency, with exactly the same task grid, review stops, accounting, and
per-task allocation guards. The current controller and submitted jobs are not
modified. Any scheduler savings depend on actual allocations and queue order;
GPU-hours per scientific experiment do not decrease from this change alone.

### Prepared profiling diagnostic

[`tools/profile_harvesting.py`](../tools/profile_harvesting.py) profiles the
existing seeded generation path, without optimizer updates or research labels.
It records actual attention implementation, uninstrumented throughput, peak
CUDA memory, aggregate sampling events, and CPU/GPU operator timings. It checks
that instrumentation preserves completions and leaves adapter weights unchanged.
It requires a visible GPU and a Slurm compute allocation and rejects unallocated
execution before loading a model. CLI, syntax, lint, and that rejection path
have been checked locally; a GPU run has not been submitted or validated.

Its first measurement target is the existing per-row multinomial sampling and
full-vocabulary filtering/sorting, alongside model forward work. These are
profiling hypotheses, not demonstrated bottlenecks or achieved speedups. Use a
reviewed vacant compute slot inside the resource ceiling, without exceeding
the active calibration cap or bypassing its review stop. The diagnostic lives
outside the frozen scientific source inventory. It uses a short diagnostic
token count, not a replacement for any 512-token scientific harvest.

Example command inside an allocated environment, with actual config and
manifest paths supplied:

```bash
python tools/profile_harvesting.py --repository "$PWD" \
  --config "$PROFILE_CONFIG" --manifest "$PROFILE_MANIFEST" \
  --output runs/profiling/harvest_diagnostic.json \
  --trace runs/profiling/harvest_diagnostic_trace.json
```

Keep the diagnostic separate from candidate labels and scientific claims.

## Recommended order

1. Finish the unchanged 20-state A1 calibration and review Gate 1. Preserve all
   pre-A1 evidence and the four-pilot inspection. Do not replace unfavorable
   states or introduce new starvation thresholds.
2. Prepare the Sprint 2 dataset manifest, grouped trajectory splits, training
   commands, and report validation while calibration is queued. Run preparation
   and scientific analysis on scheduled CPU allocations. Launch the scientific
   Sprint 2 work only following its Gate 1 decision.
3. Benchmark generation overhead on a compute allocation before proposing any
   throughput deployment. Inspect attention implementation, independent seeded
   sampling overhead, padding, tokenization, and repeated parent computations.
   Preserve A1 prompts, order, attempt limits, verifier, within-batch uniqueness,
   K=3, training budgets, and terminal semantics. Validate outputs, verified
   pools, labels, checkpoint reload, and rollback for any implementation change.
   A backend or batching change that changes scientific outputs cannot be
   silently mixed into the current dataset; record the execution profile and
   obtain a separate review before adoption. No mid-campaign engine swap.
4. Generate each expensive H=1/H=2 label and feature record once. Reuse the same
   eligible, leakage-free dataset and split for Direct, Dynamics, the matched
   H=1 baseline, and feature/scalar ablations. This reuse is already supported.
   Schedule compact forecaster fits, bootstrap statistics, and reports on CPU
   compute nodes where appropriate, preserving their training budgets.
5. Review Gate 2 before releasing broader parallel GPU work. The implementation
   requires at least 30 nonconstant held-out validation states, in addition to
   rho >= 0.30 and informative top-1 >= 0.45. Twenty Gate 1 calibration states
   do not satisfy this requirement. Keep whole-trajectory splits and the final
   test set separate from fitting and model selection.
6. After the required reviews, parallelize independent seeds, streams, and
   policy trajectories within verified live queue limits and the reviewed
   resource budget. Generations inside a trajectory retain their dependency.
   More concurrent GPUs reduce elapsed time, not billed node-hours.

## Evidence to protect

Protect the H=2 versus matched H=1 comparison, greedy and other planned core
baselines, persistent recursive updates, both task families, both core
backbones, at least three independent seeds, retention/OOD evaluation, harmful
update analysis, confidence intervals, and reproducible split/checkpoint records.
Direct and Dynamics share labels; evaluate Dynamics under the planned fixed
budget and report an unhelpful result honestly rather than expanding its search.

Use the research document's existing conditional rule for the PD extension:
omit it if its activation conditions are unmet by October 7. Keep 8B and
cross-family confirmation behind Gate 2. Prioritize the predeclared confirmation
comparisons over an exploratory hyperparameter grid. Reducing planned model
coverage would narrow the supported generalization claim; it is not a free
substitute for that evidence and is not approved by this assessment.

## Resource and deadline decision

The current campaign ceiling is 500 GPU node-hours, including previous runs.
The unoptimized default label dataset alone exceeds it. Preserve scientific
budgets by obtaining measured throughput gains or reviewing a higher resource
ceiling against the actual remaining TACC allocation. This document changes
neither the ceiling nor concurrency. Check the live allocation before promising
additional hours.

October 10 remains the requested experimental freeze date. A reliable full-study
ETA needs the Gate 1 decision, the complete Sprint 2 manifest, benchmarked
throughput, available allocation, queue starts, and measured recursive policy
costs. The four successful pilots establish feasibility, not Gate 1, Gate 2,
full recursive success, or an acceptance prediction.

ARR's guidance evaluates soundness, reproducibility, and evidence supporting
the paper's scoped claims; extra experiments are required when those claims
depend on them. This supports prioritizing decisive comparisons, but cannot
guarantee ACL Main acceptance: [ARR reviewer guidelines](https://aclrollingreview.org/reviewerguidelines).
Vista bills allocated node time and publishes queue limits that must be checked
with `qlimits`: [Vista running jobs](https://docs.tacc.utexas.edu/hpc/vista/running/).
