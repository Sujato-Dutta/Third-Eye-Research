# Third Eye experiment inventory and compute accounting

Verified against the research document, frozen Amendment A1, study planner,
audited dataset archive, and completed-run evidence on October 4, 2026.
This inventory changes no scientific protocol or launches. Later stages are
planned scope, not experiments already submitted or completed.

## Models and seeds

| Backbone | Role | Tasks in the current planner |
| --- | --- | --- |
| Qwen3-4B | Core labels, forecasting and recursive comparisons | Math and code |
| Llama-3.2-3B-Instruct | Core labels, forecasting and recursive comparisons | Math and code |
| Gemma-3-4B-IT | Conditional cross-family confirmation | Math |
| Qwen3-8B | Conditional cross-scale confirmation | Math |
| Llama-3.1-8B-Instruct | Conditional cross-scale confirmation | Math |

There are two active core backbones and five planned backbones total. All three
confirmation backbones remain behind Gate 2. Secondary-task confirmation is not
included in the current planner counts.

Core policy comparisons and main forecaster fits use three seeds: 42, 43, 44.
Core labels use three trajectories per base seed per model/task, with actual
seeds 1042–1044, 2042–2044, and 3042–3044. Transfer labels use 900042–900044;
stress runs use 100042–100044. Gate 1 is a separate five-seed calibration grid:
1042, 2042, 3042, 4042, 5042 on each of four streams, giving twenty one-state
runs. It is not the main three-seed recursive experiment.

## Experiment counts

| Stage | Construction | Planned count |
| --- | --- | --- |
| A1 calibration / Gate 1 | 2 models × 2 tasks × 5 seeds | 20 one-state runs; 60 candidate records |
| Core H=1/H=2 labels | 2 models × 2 tasks × 3 seeds × 3 trajectories | 36 T=5 trajectories; up to 180 states / 540 candidates |
| Main forecasters | 4 variants × 3 seeds | 12 fits |
| Feature / scalar ablations | 6 variants × seed 42 | 6 additional fits |
| Core recursive comparisons | 2 models × 2 tasks × 3 seeds × 8 policies | 96 T=5 trajectories; 420 updating generations plus 60 no-update generations |
| Confirmation labels | 3 models × 3 seeds | 9 T=5 trajectories; up to 45 states / 135 candidates |
| Confirmation forecasting | 3 models × 3 existing forecasters | 9 evaluations; no new forecaster training |
| Confirmation recursive comparisons | 3 models × 3 seeds × 3 policies | 27 T=5 trajectories; 90 updating generations plus 45 no-update generations |
| Shuffled-order stability stress | 2 core models × 3 seeds × 3 policies, math | 18 T=5 trajectories; 90 updating generations |
| Final held-out evaluation | Core and confirmation policy endpoints | 96 core + 27 confirmation = 123 evaluations |

With all confirmation models, there are 45 five-generation label trajectories
and 123 main/confirmation policy trajectories; stress adds 18 trajectories.
Counts are maxima when all generations produce complete states. A1 scarcity can
end trajectories; no missing state is fabricated. Calibration is separate and
its labels are not silently pooled into the main label-training manifest.

The eight core policies are no update, random, confidence/verifier heuristic,
greedy immediate selection, H=1 MLP, matched temporal H=1, Third Eye-Direct H=2,
and Third Eye-Dynamics H=2. Confirmation uses no update, greedy, and Direct.
Stress uses greedy, H=1 MLP, and Direct.

E0/E1 cover pipeline sanity and immediate/future ranking disagreement; E2/E3/E9
cover forecasting accuracy and matched H=1/H=2 comparisons; E4 covers recursive
policy comparisons; E5 retention/regression; E6/E7 family/scale transfer; E8
feature ablations; E10 seed/order stability; E11 compute–benefit; E12 separate
target/OOD/retention forecasts; E13 Direct versus Dynamics. These analyses share
the datasets and trajectories above rather than multiplying them per figure.

E14 primal-dual selection is separately gated by Gate 2, usable retention
forecasts, and the predeclared October 7 activation decision. It has no committed
run count or cost yet. H=3 is an optional appendix extension and is not scheduled.
The planner marks stress as an optional execution stage, while E10 in the
research document requires order-stability evidence. The eighteen listed stress
runs should remain in the completion inventory when claiming that requirement.

## Ablations and forecaster dataset

The six explicit ablations remove gradient features, reversible-probe features,
history, data diversity, or retention features individually, or replace
multi-head forecasting with scalar-only forecasting. They each currently have
one fit at seed 42; the four main forecasters each have three fits. Direct versus
Dynamics and H=1 versus H=2 are additional architectural/horizon comparisons.
All eighteen fits reuse the same core labels and split; ablations do not launch
new LLM label trajectories.

Splitting groups entire trajectories and merges groups sharing state IDs. The
nominal split is 60/20/20, with validation and test group counts rounded upward.
For 36 independent complete trajectories, the implementation gives 20 train,
8 validation, and 8 test trajectories: 100/40/40 states or 300/120/120 candidate
records. Actual audited splits can differ with shared-state groups or scarcity.
Gate 2 requires at least thirty nonconstant validation states, not thirty
candidate rows. Transfer backbones are held out from core forecaster training.

## Audited task dataset sizes

| Role | Math | Code |
| --- | --- | --- |
| A1 correction-harvest pool | 1,024 GSM8K-train prompts | 306 MBPP-train prompts |
| Target development | 64 GSM8K | 64 MBPP |
| OOD development | 64 MATH-train problems, evaluation only | 64 MBPP-validation problems, development proxy |
| Retention development | 256 MMLU-validation items | Same 256 items |
| Target final test | 1,319 GSM8K-test | 499 MBPP-test after overlap audit |
| OOD final test | 500 MATH-500 | 164 HumanEval |
| Retention final test | 256 MMLU-test items | Same 256 items |

A1 CPU proof establishes the enlarged training counts. Selection and final
suite counts come from the original audited data archive; A1 requires those
development/final files to remain byte-identical. These are shared fixed suites,
not fresh dataset samples multiplied by model or seed. Target and OOD final
tests are not used for update construction or candidate selection. HumanEval is
final evaluation only; MBPP validation is the code OOD-development proxy.

At each nonterminal label state, K=3 distinct candidate batches each contain two
unique verified corrections. Candidates may overlap across batches. The fixed
harvest allows at most sixteen revisions per initially failed problem, stopping
at the first passing revision, with 512 new tokens per completion. Pool size is
an observed outcome, not a guaranteed training-set size. Each full candidate or
continuation update uses fifty optimizer steps, rank sixteen, and LR 1e-4.
Labels require one parent harvest and three branch continuation harvests, with
up to six full updates per state. Online selection uses one parent harvest and
one selected update, except greedy, which trains all three candidates.

## GPU allocation and cost measurement

Independent eligible trajectories can run on up to twenty Vista GPU nodes,
one node per trajectory. Generations within a trajectory stay sequential.
Gate 1 and Gate 2 retain their review boundaries. CPU-suitable forecaster fits,
verification, bootstrap statistics and reporting are intended for scheduled CPU
nodes. The generic planner currently emits CUDA forecaster commands, so CPU
placement must be explicit in the reviewed execution plan; it is not already a
completed migration. No training, inference or scientific analysis runs on
login nodes. Vista GPUs observed in the allocation are GH200.

| Cost category | Current basis |
| --- | --- |
| Previously reconciled GPU usage | 50.3094 hours before the seed-3042 wave; includes pre-A1 failures and validation, not a live total |
| Twelve remaining calibrations | 96 hours reserved at an eight-hour limit per task; reservation is not actual spending |
| Completed profiling job 1047005 | 0.25 billed GPU hours, applying the minimum charge |
| Completed batching diagnostic 1047102 | 19m 46s = about 0.3294 GPU hours |
| Core labels | 180 × measured mean 3.7648 = about 677.7 GPU hours; extrapolation |
| Core online comparisons | About 471 GPU hours from a parent-harvest/training/overhead proxy; not measured online cost |
| Optional order stress stage | About 102 GPU hours using the same proxy; preserve required E10 coverage |
| Confirmation models and final evaluation | Not yet benchmarked; no defensible committed total |

Each allocation record is reconciled against Slurm job/array-task accounting,
including failures, cancellations and diagnostics. Pending reservations are
tracked separately and released/replaced with actual usage after completion.
Count each allocation once; do not sum both parent array jobs and child tasks,
or add component timers on top of the same billed wall time. The parent harvest
statistics repeated across three candidate records count once.

Vista's published formula is nodes × max(elapsed hours, 0.25) × queue charge
rate. GPU queues gh/gh-dev charge 1 SU per node-hour; CPU queue gg charges 0.33.
CPU service units belong in the total allocation ledger separately from GPU
hours. Queue waits cost elapsed deadline time but not running node-hours.
Reports should include per-update harvesting/training/probe/evaluation timers,
peak memory, candidate training steps, measured GPU minutes per accepted update,
and accuracy/retention gain per GPU hour. Component timers explain utilization;
Slurm/allocation accounting establishes billed cost.

The planning target is 1,000 cumulative GPU hours; 3,000 is the active upper
envelope, not required expenditure. The checked October 4 allocation snapshot
was 3,333 available service units. The original research document's roughly
100–160 A100-hour estimate is a historical assumption, not a hardware conversion
or validated cost for the enlarged A1 harvest. Core labels plus the online proxy
already total approximately 1,149 hours, before calibration and additional
stages, so the complete baseline is not yet demonstrated to fit the target.

About 87.71% of measured label runtime is harvesting. Hypothetical safe twofold
harvesting throughput would reduce core label hours to about 380.5, but no such
gain is deployed. Job 1047102 tested larger inference batches and found changed
completions in every larger-batch case; those speedups cannot be counted as
validated savings for the frozen runs. Twenty nodes reduce wall-clock duration,
not GPU-hours. A dollar price has not been established; cost is reported in
allocated service units rather than an invented monetary rate.

Sources: [study planner](../experiments/plan_study.py),
[A1 protocol](protocol_amendment_a1.md),
[execution and cost assumptions](deadline_execution.md),
[Vista charging policy](https://docs.tacc.utexas.edu/hpc/vista/running/).
