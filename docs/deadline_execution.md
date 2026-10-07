# October 4 deadline execution override

This records the resource and scheduling change authorized on October 4 to
prioritize experimental completion by October 10. It is an operational override,
not a new scientific amendment. The frozen A1 document and scientific source
remain byte-identical. The earlier four-GPU cap, 500-hour campaign ceiling, and
four-task scheduling waves are historical settings superseded by this override.

## Resource envelope

The live Vista allocation check at 04:25 UTC (09:55 IST) reported 3,333 available
service units in IRI23021. The reviewed execution envelope is up to 20 simultaneous
GPU nodes and 3,000 cumulative GPU node-hours, including the 50.3094 hours already
tracked before the latest running wave. The allocation retains headroom for CPU
jobs and accounting changes. This is an upper limit, not a request to spend it
all. The actual allocation and aggregate scientific/diagnostic usage must be
rechecked before later stages.

The live queue limits allow 20 jobs and 96 nodes per user in the production GPU
queue. Allocation timing depends on shared-cluster scheduling; enabling twenty
concurrent jobs does not promise twenty immediately allocated nodes.

## Validated A1 handoff

All 107 tests passed on the existing CPU compute allocation in 32.62 seconds:
101 research tests plus six scheduling-handoff tests. Additional operational
validation checked the unchanged source and amendment inventories, the unchanged
twenty-state grid, reviewed four-pilot evidence, task non-duplication, existing
reservations, remaining allocation, deadline, and eight GB of project storage
headroom. The operational tool and launcher hashes bind the handoff to that proof.

The first auxiliary validation launcher failed before Python started because its
shared-library path was missing. The corrected launcher explicitly resolves the
virtual environment's interpreter library directory. The failed log is preserved;
the successful full validation preceded every scheduling handoff action. No GPU
experiment failed or restarted because of this launcher issue.

CPU controller 1045577 was intentionally cancelled after its validated
replacement, 1046983, was submitted with `afterany:1045577`. Original GPU array
1046534 was preserved, including its running task. The replacement started after
the original CPU job left the allocation. The archived original campaign and
submission records remain available; the legacy controller must not be resumed
because its phased scheduling could repeat tasks now covered by the replacement.

The replacement adopted the existing four seed-3042 tasks and submitted exactly
the eight previously unsubmitted seed-4042/5042 tasks. It submits nothing outside
the frozen calibration grid. At 04:41 UTC (10:11 IST):

| Job | Work | Observed state |
| --- | --- | --- |
| 1046983 | Replacement CPU controller | Running |
| 1046534 | Four seed-3042 calibrations | Four GPU tasks running |
| 1046997 | Eight seed-4042/5042 calibrations | Pending for Priority |
| 1047005 | Isolated generation profiling | One GPU running |

The eight already completed states remain unchanged. All twelve remaining
calibrations are now submitted, without per-seed wave barriers. There are twelve
remaining scientific tasks, so filling twenty GPUs with additional unplanned
calibrations would change the sample. The remaining capacity is available for
eligible later work following scientific review.

The active operational ledger is
`runs/campaign_a1_20261003/parallel_resume_20261004/campaign.json`; the campaign's
`status.json` links to it. The original `campaign.json` remains historical. The
replacement will analyze all available complete calibration states on its CPU
node, back up artifacts, and stop at `gate1_review_required`. It does not launch
forecasters or scaling automatically, even if the gate passes.

## Performance work

One isolated GPU profiling job runs the existing seeded generation method for
Qwen and Llama sequentially. Its half-hour reservation is additional to the
96 hours reserved for the twelve calibrations and belongs in the resource
ledger. It performs no optimizer updates and publishes no research labels.
It checks completion parity with the uninstrumented path and unchanged adapters,
and measures sampling operators, model computation, throughput, and memory.
No generation optimization is deployed until its measurements and output
equivalence checks justify it. Batch size, correction attempts, token budgets,
K=3, learning rate, rank, optimizer steps, and terminal rules remain unchanged.

Profiling job 1047005 completed successfully in 4m 06s, with a minimum-billing
cost of 0.25 GPU hours. Both models passed completion parity and unchanged
adapter checks; both already use SDPA attention. The short 16-row, 64-token
measurements gave approximately 239 decoded tokens/second for Qwen and 333 for
Llama. These are diagnostic prompt timings, not full-harvest throughput or
claimed speedups. Operator reports show substantial matrix-operation/kernel
dispatch activity alongside sampling work. Raw diagnostic traces were losslessly
compressed on the scheduled CPU node from about 1.2 GB to 113 MB; their original
SHA-256 hashes are preserved.

Isolated job 1047102 benchmarks sampled inference batches 16/32/64 on 64 training
prompts from each task family, using both root and saved trained adapters for
both models. It keeps the 512-token maximum and independent seeds, uses no
optimizer updates, and preserves every scientific run/config. Its one-hour
reservation belongs in the same resource envelope. Matching completions on
these finite prompt cases would qualify a batching choice for further complete
harvest/label equivalence checks, not automatic deployment.

The initial benchmark submission referenced `afterok:1047005`, which Slurm
rejected after that completed job retired from its live job table. No diagnostic
or scientific experiment launched from the rejected submission. Successful
COMPLETED/exit-zero accounting was verified before resubmitting without the
retired live dependency; the failed attempt and claim are retained.

## Next-stage order

After the twenty-state Gate 1 evidence is reviewed, a passing decision makes
small-model Sprint 2 label generation eligible for a rolling twenty-GPU plan.
Keep the planned seeds, trajectories, five-generation depth, grouped splits,
and all core comparisons. Reuse each label/feature dataset across forecasters
and ablations. Run compact forecaster fits and statistical analysis on scheduled
CPU compute nodes where appropriate. Every generation preserves its predecessor
checkpoint dependency; only independent trajectories run concurrently.

Gate 2 remains the next scientific decision and retains its evidence minimum
and forecasting thresholds. Held-out family/8B confirmation remains behind
Gate 2. Do not shrink training, harvesting, sample-size requirements, retention
evaluation, or confidence intervals to manufacture a timely passing result.

Using the eight measured states, the default core label dataset projects to
approximately 678 GPU node-hours. At a sustained twenty nodes this is about
34 hours of ideal aggregate execution, with queue waits and task imbalance
adding time. It excludes online policy comparisons and later-model confirmation.
This calculation motivates parallel scheduling; it is not a full-study ETA.
October 10 remains the experimental freeze target, contingent on queue starts,
complete scientific evidence, and the research gates.

## Clarification: 1,000-hour planning target

The 3,000-hour envelope is an allocation ceiling, not a measured requirement or
a forecast of spending. Use 1,000 cumulative GPU node-hours as an optimization
target when preparing the next-stage plan. This target is not yet demonstrated
for the complete five-model study and does not change the active controller's
ceiling or release any scientific stage.

Eight completed calibration states give the following baseline decomposition:

- Mean label-state elapsed time: 3.7648 hours.
- Mean total harvesting: 3.3021 hours (87.71% of elapsed time).
- Mean parent-pool harvesting alone: 0.8216 hours.
- Mean six-update training time: 0.2060 hours.
- Mean remaining label-state overhead: 0.2566 hours.

The 180-state core label projection is 677.7 hours. Online trajectories harvest
one parent pool per generation, rather than the parent plus three continuation
pools used for labels. They must not be charged four harvests or six training
updates per generation. The default core planner has 420 updating generations:
360 under policies training one selected candidate and 60 under greedy H1,
which trains all three. No-update trajectories skip harvesting and training.

As a planning proxy, applying the measured parent-pool time, candidate training
time, and the entire remaining label-state overhead to those generations gives
about 471 hours. That overhead is not a measured online cost, and later
generations can have different failure rates. Labels plus this online proxy
already total about 1,149 hours, before calibration expenditure, final evaluation,
conditional transfer, or optional stress experiments. Thus the unchanged full
baseline cannot currently be promised within 1,000 hours.

If harvesting alone were safely twice as fast, the label projection would be
about 380.5 hours; at three times, about 281.4 hours. These are sensitivity
calculations, not achieved savings. Forecasters and feature ablations already
reuse the same labels, and CPU-suitable fits and analysis should use scheduled
CPU allocations. Keep scientific budgets, seeds, gates, and evaluation coverage.
Twenty allocated GPU nodes would consume 1,000 node-hours in 50 hours of ideal
aggregate execution; concurrency reduces elapsed time, not this billed cost.

At 05:54 UTC (11:24 IST), diagnostic job 1047102 was COMPLETED, exit zero, with
19m 46s elapsed. Batches 32/64 achieved case-specific speedups of approximately
1.37–3.46 times, but every larger-batch task/adapter case changed some completions
relative to batch 16. Both reports have an empty
`matching_batches_on_tested_prompts` list and unchanged adapter weights. This is
a completed diagnostic with a failed equivalence requirement for deployment,
not a science-job failure. Larger batching is not applied to the frozen runs.
Reports are preserved locally in
`runs/deployment/evidence/batch_benchmark_1047102_{qwen3_4b,llama3_2_3b}.json`
and in the compute campaign's profiling directory. Any future throughput gain
must pass the appropriate equivalence checks before contributing to a committed
1,000-hour execution forecast.

The authorized eight-attempt assessment is now recorded separately in
[retry budget diagnostic](retry_budget_diagnostic.md). CPU validation passed
116 tests; an audit of twelve complete A1 states retained 87.61% of verified
corrections with 47.36% fewer retry attempts. No parent or original-checkpoint
continuation loses the required correction supply, but candidate batches change
in ten states, so old labels cannot represent the shorter protocol. Read-only
GPU replay array 1047590 reserves six additional GPU hours, capped at four
concurrent tasks. The active A1 calibration remains at sixteen attempts and
stops at its original Gate 1 review boundary.

## Gate 1 completed and reviewed, October 4 evening

All twenty fixed A1 calibration tasks and controller 1046983 completed with
exit zero. The controller stopped at `gate1_review_required` at 13:57 UTC
(19:27 IST), as required. Independent artifact review verified all sixty label
records against the report hashes, three distinct candidate hashes per state,
candidate size two, matched fifty-step budgets, valid target/OOD/retention
metrics and exact consequence deltas. There were zero terminal continuations.
The review is bound to the Gate 1 report SHA-256 and is preserved as
`gate1/evidence_review.json` remotely and
`runs/deployment/evidence/a1_gate1_evidence_review_20261004.json` locally.

Gate 1 passed both criteria: seventeen of twenty states (85%) had disjoint
immediate/H=2 best-candidate sets; forty-three of sixty candidates (71.67%)
reduced at least one target/OOD/retention component immediately. State-bootstrap
95% intervals are 65–100% for disagreement and 53.33–86.67% for harmful-component
frequency. These are phenomenon-screening results, not evidence that a trained
forecaster works or that recursive policy selection improves final performance.

Original/pre-A1 and all twenty A1 calibration allocations total 94.5103 GPU
hours. Including profiling, batching and the first eight-attempt replay array
brings reconciled GPU usage to approximately 97.0950 hours. CPU service units
remain separate; later reservations are not counted as completed spending.

At 15:57 UTC (21:27 IST), CPU preparation job 1047869 and paired retry-control
array 1047870 were submitted. The CPU job reruns the 116-test suite, checks
unchanged scientific source and reviewed evidence, and materializes the core
plan with 36 label trajectories, 18 CPU forecaster fits, and 96 core policy
trajectories. Its original emitted plan is retained; a separate execution plan
records CPU forecaster placement and resource settings. It submits no training.

The two GPU controls depend on successful CPU preparation and reserve four
GPU hours maximum. They mirror original development evaluation and compare
eight and sixteen retries within the same initialized Llama process, reporting
historical reproducibility separately from the retry-cap effect. No optimizer
updates, scientific labels, automatic eight-attempt adoption or 8B jobs are
released. The expensive core dataset awaits retry-budget review rather than
silently mixing old sixteen-attempt labels with a different protocol.

At 15:59 UTC (21:29 IST), CPU job 1047869 completed successfully in 42 seconds.
All 116 tests passed in 28.80 seconds, and the core plan was prepared under
`runs/sprint2_a1_20261004/`. Paired control array 1047870 had cleared its success
dependency and was pending for Priority. No large scientific label or forecaster
training job had been submitted.

## Live review, October 5 morning

At 04:19 UTC (09:49 IST), both paired Llama controls in array 1047870
were COMPLETED with exit zero. Math used 49m 56s and code 1h 16m 10s.
The live Slurm queue was empty. No large label generation, forecasting fit,
online policy trajectory, or 8B job had been submitted.

Both same-session eight-attempt prefixes and attempt-accounting checks passed,
and adapters and original research evidence were unchanged. The shorter
harvest retained 74/80 corrections for math and 27/36 for code, taking 23.43%
and 45.21% less harvesting time, respectively. These are two root-state
paired timings, not full-study savings or evidence of equal recursive outcomes.
The short run always preceded the full run, so timing can include order effects.

Fresh sixteen-attempt Llama runs still differ from historical runs. Root math
target accuracy was 60/64 versus 62/64 historically, and OOD 38/64 versus 36/64;
root code target was 31/64 versus 32/64. Retention was identical at 121/256.
Within-session agreement separates this discrepancy from retry truncation;
it does not identify its cause or prove cross-launch reproducibility.
Original and diagnostic results remain preserved.

Reconciled GPU expenditure is approximately 99.1967 hours, excluding CPU SUs.
Gate 1 remains passed and reviewed; Gate 2 is untested. Scientific A1 remains
at sixteen attempts. Eight-attempt adoption and subsequent campaign execution
remain uncommitted, so different protocol labels are not silently mixed.
Detailed control evidence is preserved in
`runs/deployment/evidence/paired_retry_control_1047870_llama_{math,code}.json`,
and the timestamped review in `runs/deployment/evidence/progress_20261005.json`.

## Eight-attempt execution authorized and queued, October 5

Following explicit authorization to use eight attempts for new harvesting,
[Amendment A2](protocol_amendment_a2.md) is frozen and staged in its own source
and result directory. A1 evidence remains archived unchanged. See the
[A2 execution record](a2_execution.md) for the concrete dependency chain:
CPU validation 1049531, dual-node CUDA controls 1049532, comparison 1049533,
36 label trajectories capped at twenty concurrent GPUs in 1049534, and CPU
forecasters/Gate 2 reporting in 1049535. All scientific execution occurs on
Slurm compute nodes; downstream jobs require successful predecessors. No online,
8B or transfer job is released before Gate 2 review.

## October 5 evening: validation queue reroute

At 12:38 UTC / 18:08 IST, CPU validation remained passed, but GPU array
1049532 was still pending for Priority after nearly eight hours. No CUDA
control had begun and no label task had started. A read-only queue check
confirmed that gh-dev permitted a single two-node job, ninety minutes was
within its two-hour limit, and two development nodes were idle.

Pending validation was moved to one two-node development allocation, job
1049914. Its two Slurm process ranks execute the same frozen CUDA verification
and terminal-test commands once each on different scheduler-assigned nodes.
Separate step-host metadata records the actual hosts and ranks; the new CPU
reviewer compares the same source/plan/control hashes, model outputs, verifier
results, repeated fifty-step training and software versions. It uses actual
step hosts for the distinct-node check because one allocation's Slurm node-list
string is shared by both ranks. It records the new operational launcher and
reviewer hashes without modifying the frozen scientific source or plan.

Review job 1049915 depends on successful 1049914 completion. The existing
label array 1049534 was updated to depend on successful 1049915. It retains
all 36 tasks and a maximum of twenty concurrent GPU nodes. CPU forecaster job
1049535 still depends on that same label array. Original pending validation
1049532 and review 1049533 were canceled before running, after the replacement
chain and label dependency were recorded. This is an operational queue change,
not a restart of scientific trajectories or an additional protocol amendment.

At 12:44 UTC / 18:14 IST, 1049914 was RUNNING on two automatically allocated
development nodes, and both Llama model instances had loaded. Review, labels
and CPU forecasting were pending on their success dependencies. No execution
failure was reported. Reproducibility is not yet declared passed. The replacement
reserves the same three maximum GPU-hours as the unstarted original controls;
queued reservations are not completed spending.

Evidence:
`runs/deployment/evidence/a2_development_reroute_20261005.json` and
`runs/deployment/evidence/a2_dev_reroute_files_20261005.json` locally;
`runs/a2/validation/development_reroute.json` and `step_host_{0,1}.json`
remotely. Frozen A2 source SHA remains
`917c86b78482c73797619d17feab423f91d3775026db086e15f7f529e4fcf988`.

## October 5 evening: validation metadata defect repaired

At 13:40 UTC / 19:10 IST, development validation 1049914 was FAILED after
36m 02s (Slurm exit 15:0 after srun terminated its second rank). The actual
Python exception was a metadata assumption at the end of Qwen's model checks:
`b.source_proof["proof_sha256"]` attempted to index None. Directly loaded Qwen
uses a pinned Hugging Face source and legitimately has no mirror-proof record.
Both Llama instances had logged their within-node checks as passed. No complete
cross-node proof was published, so no reproducibility pass is claimed.

This was a validation-script implementation defect. It did not alter A2
candidate construction, training, harvest attempts or scientific results. The
failed log and existing implementation-test checkpoints are preserved. Label
array 1049534 and CPU forecasting 1049535 remained blocked and never started.
The failed two-node allocation consumed 1.2011 GPU-hours, bringing reconciled
GPU expenditure to about 100.3978 hours, excluding CPU SUs.

Validation revision V2 records direct pinned-source identity separately from
verified mirror-proof identity. Six regression tests cover direct-source None,
verified mirrors, unpinned/mismatching revisions, and invalid mirror proofs.
Completed model controls are now persisted independently before the next
model begins, avoiding loss of earlier diagnostics if another check fails.
New verification artifacts use `runs/a2/validation/v2/`; failed artifacts are
not overwritten. Frozen A2 scientific source, configuration, plan and original
operational files remain unchanged; V2 tools have separate recorded hashes.

The repaired success chain is CPU validation 1049976 -> two-node development
CUDA validation 1049977 -> CPU comparison 1049978 -> the same 36-task label
array 1049534, capped at twenty concurrent GPUs -> existing CPU forecasting
1049535. Unstartable review 1049915 was canceled only after the replacement
label dependency was successfully updated. No scientific trajectory was rerun.

At 13:47 UTC / 19:17 IST, CPU validation 1049976 completed in 53 seconds,
exit zero: 132 tests passed in 39.00 seconds, and V2 was bound to the unchanged
frozen A2 source and plan. The scheduler still showed the CPU job completing
while the GPU success dependency cleared. CUDA verification remains required;
no cross-node reproducibility pass or Gate 2 result is asserted yet.

Evidence: `runs/deployment/evidence/a2_validation_v2_submission_20261005.json`,
`a2_validation_v2_files_20261005.json`, and
`a2_cuda_dev_1049914_failed.log` locally; `runs/a2/validation/v2/cpu_v2.json`
and `submission.json` remotely. Scientific source SHA remains
`917c86b78482c73797619d17feab423f91d3775026db086e15f7f529e4fcf988`.

## October 5 night: A2 validation passed; labels eligible

At 16:00 UTC / 21:30 IST, metadata-repaired CUDA validation 1049977 was
COMPLETED, exit zero, in 36m 15s on two development nodes. Both Qwen and Llama
passed the tested decoding, verifier, repeated fifty-step training and checkpoint
controls. Both nodes passed their two terminal-continuation tests. CPU review
1049978 completed with exit zero in nine seconds and published
`runs/a2/validation/gpu_review.json` with `passed=true`.

The two CUDA reports have identical byte hashes and matching tested model
control digests, while independent step-host metadata confirms different
allocated nodes. Downloaded reports and their hashes were independently
checked against the review evidence. This establishes parity of tested A2
computations in the pinned GH200 environment. It does not assign a root cause
to historical A1 output differences or claim determinism across untested
software/hardware conditions.

At 16:02 UTC / 21:32 IST, all 36 label tasks in array 1049534 were PENDING for
Priority, with their success dependency cleared and array throttle twenty.
No label task was running or completed. The scheduler reported StartTime Unknown
and all non-drained/reserved production GPU nodes allocated. CPU forecasting
1049535 remains queued after the label array. Current limitation is production
GPU availability/priority, rather than a validation error. No reliable allocation
start estimate is available; the projected 30-48 hours to Gate 2 assumes sustained
near-twenty-node concurrency after allocation, plus inter-stage queue waits.

Reconciled GPU expenditure is about 101.6061 hours, excluding CPU charges and
unspent reservations. Gate 1 remains reviewed and passed; A2 Gate 2 remains
untested. Evidence: `runs/deployment/evidence/a2_gpu_review_1049978.json`,
`a2_cuda_v2_1049977_rank{0,1}.json`, and
`a2_validation_passed_progress_20261005.json` locally. The existing successful
validation -> 36 labels -> CPU forecasting chain proceeds without a new protocol
change, without login-node execution, and stops again for Gate 2 review.

## October 6 morning: label generation started

At 04:29 UTC / 09:59 IST, label tasks 1049534_0 and 1049534_1 were RUNNING
on production compute nodes, with 34 tasks pending for Priority. By 04:30 UTC /
10:00 IST, tasks 0 through 6 were RUNNING: seven allocated GPUs, 29 pending,
and no completed trajectories or reported failures. Array throttle remains
20. Allocations are increasing, but twenty simultaneous GPUs are not guaranteed.

The first two Qwen-math trajectories (seeds 1042 and 2042) loaded their pinned
models and completed initial development evaluations. Both reported 57/64
target, 28/64 OOD, and 165/256 retention. These are starting-model checks, not
update gains or Gate 2 results. Their first recursive states are underway;
no complete-state ledger records were present in the downloaded metadata
snapshot. CPU forecaster job 1049535 remains pending on the full label array.
No new source, protocol, budget, or data change was made for this progress check.

Evidence: `runs/deployment/evidence/a2_progress_20261006_morning.json` and
`a2_progress_metadata_20261006.tar.gz`. The 30-48-hour Gate 2 planning estimate
still assumes near-twenty-node concurrency after allocation; actual timings
will be revised after complete A2 state/trajectory measurements are available.

## October 6 late morning: twenty GPUs allocated

At 06:11 UTC / 11:41 IST, twenty label trajectories were RUNNING in array
1049534; the remaining sixteen were pending for JobArrayTaskLimit. This is
full use of the authorized twenty-node concurrency cap. Pending trajectories
become eligible as slots open, subject to the scheduler allocating nodes.
CPU forecasting 1049535 remains pending on the full label array.

No job failures, completed trajectories, or complete-state ledger records were
present in this snapshot. Jobs were approximately 1h 37m to 1h 55m into their
first states. Representative Qwen-math/code and Llama-math logs show actual
candidate updates, development evaluations and standardized continuation
harvesting underway at eight attempts. Initial Qwen-code and Llama-math
metrics match their saved CUDA control baselines. Subsequent branch metrics
are not aggregated as science results before complete K=3 labels exist.

The October 7-8 Gate 2 planning window remains conditional on maintained
parallel allocations and successful runs; first complete-state timings are
still needed to refine it. Evidence:
`runs/deployment/evidence/a2_full_parallel_progress_20261006.json` and
`a2_progress_metadata_20261006_259.tar.gz`. Live cumulative GPU-hours are
recorded from Slurm elapsed time in the progress evidence, separately from
CPU SUs and unspent reservations. No scientific or execution-code change was
made in this progress check.

## October 6 early afternoon: first complete states audited

At 07:27 UTC / 12:57 IST, twenty trajectories remained RUNNING and sixteen
were pending for the twenty-task array cap. No full T=5 trajectory had completed
and no job failure was reported. CPU forecaster job 1049535 remained dependent
on the label array. The downloaded metadata then contained five complete Qwen-
code states and their accepted t+1 checkpoints; these runs had entered the next
generation. Accepted ledger entries are commitments, not scarcity events.

A subsequent immutable-label snapshot contained nine complete Qwen-code states,
with twenty-seven candidate records. An independent local CPU audit passed
all twenty-seven: K=3, distinct unique-example batches, verified-pool membership,
eight-attempt limits/seed stride, matched fifty-step budgets, valid development
metrics, and exact H=1/H=2 deltas. All twenty-seven continuations were available;
zero terminal candidates were present in this snapshot. These are data-quality
checks on published states, not a Gate 1/2 decision or a full-study conclusion.
No scientific source or queued command changed.

The first five measured Qwen-code states took 2.628-2.885 hours, median 2.771.
Other streams and later generations still need complete-state measurements;
extrapolating one stream's first generation to the whole campaign is not an
established runtime forecast. The October 7-8 Gate 2 window remains conditional
on sustained allocations and successful runs. Evidence:
`runs/deployment/evidence/a2_progress_270.json`,
`a2_published_label_health_269.json`, and `a2_progress_metadata_267.tar.gz`.

## October 6 afternoon: twenty complete states audited

At 15:24 IST, all twenty GPU slots were running label trajectories; sixteen
additional trajectories were queued. No full T=5 trajectory had completed
and no job failure was reported. CPU forecasting 1049535 remains queued on
the complete label array and will stop for Gate 2 review after its checks.

Twenty complete states / sixty candidate labels passed the published-artifact
health audit: seven Qwen math, eleven Qwen code, and two Llama math states.
All had distinct K=3 candidate compositions, unique examples within batches,
matched training budgets, and valid target/OOD/retention and H=2 labels. No
terminal candidate or parent scarcity event was recorded in this snapshot.
Llama code remains queued; two Qwen-code states have reached generation one.

Whole-state medians are 2.813 hours for Qwen code, 4.848 for Qwen math, and
4.155 for Llama math. These measurements do not yet establish later-generation
or Llama-code runtimes. Gate 2 is estimated for late October 7 to October 8,
conditional on successful runs, sustained allocations, and sufficient usable
held-out states. The scheduler snapshot gives 209.18 cumulative GPU-hours,
excluding separate CPU SUs and unspent reservations. Source, protocol, seeds,
data, and queued commands are unchanged.

Evidence: `runs/deployment/evidence/a2_progress_278.json` and
`a2_published_label_health_277.json`.

## October 6 evening: forty-nine complete states audited

At 20:58:58 IST, twenty GPU trajectories were RUNNING and sixteen remained
queued. No full trajectory or scheduler failure was recorded. CPU forecasting
1049535 remains pending on the full label-array dependency.

Forty-nine complete states / 147 candidate labels passed the independent
published-artifact audit: Qwen math sixteen, Qwen code twenty-nine, and Llama
math four. No terminal continuation or parent scarcity event was recorded.
Two Qwen-code trajectories have completed four of five generations. Llama
code remains queued; all four streams must finish or record their predeclared
scarcity outcomes before the queued CPU stage can evaluate the gates.

Whole-state medians are 2.874 hours for Qwen code, 4.845 for Qwen math, and
4.264 for Llama math. The Gate 2 planning window remains late October 7 to
October 8, conditional on successful runs, sustained allocations, and enough
usable held-out states. Cumulative GPU elapsed time is 320.719 hours, with CPU
SUs and unspent reservations excluded. No source, protocol, data, seed, or
queued execution command changed.

Evidence: `runs/deployment/evidence/a2_progress_285.json` and
`a2_published_label_health_284.json`.

## October 7 morning: fourteen full trajectories completed

At 11:10:58 IST, fourteen of thirty-six label trajectories had completed all
five generations successfully. Twenty GPU trajectories remained running and
only two were queued. All nine Qwen-code trajectories are complete. CPU
forecasting 1049535 remains pending on the complete label-array dependency.

The metadata snapshot contains 112 complete states / 336 candidate labels
across all four streams. No scheduler failure or parent scarcity event was
recorded. The full new-label artifact audit is pending because SSH disconnected
during its archive transfer; the earlier 147-label audit passed. No new
336-label health-pass claim is made before the transfer and audit finish.

Gate 2 review is estimated around October 8, conditional on successful
remaining trajectories, sustained allocations, and sufficient usable held-out
states. Cumulative GPU elapsed time was 604.527 hours, excluding CPU SUs and
unspent reservations. Monitoring disconnection does not interrupt Slurm jobs.
No scientific or Slurm execution command changed.

Evidence: `runs/deployment/evidence/a2_progress_292.json` and
`a2_progress_metadata_289.tar.gz`.
