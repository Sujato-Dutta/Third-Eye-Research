# A2 execution: eight-attempt harvesting

Submitted October 5, 2026, 04:40 UTC / 10:10 IST, following authorization
to use eight harvesting attempts for new experiments.

The frozen [A2 amendment](protocol_amendment_a2.md) preserves A1 candidate,
training, continuation, data and science-gate rules. New source and results are
isolated under `/scratch/11617/sujato_ts/third_eye_research/amendments/a2`.
Archived A1 source and results remain under `amendments/a1` and are not pooled
into A2 forecaster datasets. Historical configuration fingerprints remain
compatible; A2 includes its own protocol identity and seed stride.

| Job | Resource | Work | Success dependency |
| --- | --- | --- | --- |
| 1049531 | CPU `gg` | Full suite, A2 tests, data/provenance checks, plan preparation | None |
| 1049532 | GPU `gh`, two concurrent tasks | Independent-node decoding, verification, repeated 50-step training, rollback/checkpoint and terminal controls | 1049531 |
| 1049533 | CPU `gg` | Compare CUDA control evidence; require exact tested outputs and different nodes | 1049532 |
| 1049534 | GPU `gh`, 36 tasks, maximum 20 concurrent | Two core models, math/code, seeds 42/43/44 with three repetitions, up to T=5 | 1049533 |
| 1049535 | CPU `gg` | Audit all labels, separate A2 phenomenon report, 18 forecaster fits and Gate 2 reports | 1049534 |

The CPU suite includes tests of original/A1 configuration compatibility,
native eight-attempt retry prefixes, A2 terminal states, scarcity exits, and
rejection of mixed A1/A2 label datasets. CUDA controls use diagnostic archived
training corrections only as implementation-test fixtures; they produce no
scientific trajectory labels. New scientific harvesting uses only A2 generation.

The label audit verifies complete K=3 states, distinct example compositions,
unique within-batch samples, verifier-pool membership, eight-attempt limits,
matched fifty-step update budgets, finite target/OOD/retention metrics, exact
consequence deltas, and M2=M1 adapter hashes for terminal continuations. Parent
starvation remains recorded, and no replacement trajectory is selected from
outcomes. The same available A2 label files and trajectory split feed every fit.

All expensive stages are success-dependent. A CUDA comparison failure blocks
label generation. A failed/insufficient A2 phenomenon check stops CPU fitting.
After forecasting, the workflow stops at `gate2_review_required`; no online,
8B, transfer or broader scaling job is submitted in this chain. A1's passed
Gate 1 remains screening evidence, not proof of A2 forecasting or improvement.

The label jobs request 36-hour maximum wall limits to avoid assuming root-state
timings extrapolate to every recursive generation. Maximum new GPU reservations
are 1,299 hours: 36*36 for labels and two*1.5 for CUDA controls. This is a ceiling
reservation, not expected spending or a measured full-study estimate. Jobs
release nodes on completion. Prior reconciled GPU usage was 99.1967 hours;
CPU service units are recorded separately. The planning target remains 1,000
GPU-hours and the total ceiling 3,000. No assertion that all remaining work will
fit within 1,000 hours has been established from root controls alone.

Only Slurm compute nodes execute tests, inference, training and scientific
analysis. Login work is limited to source staging, Slurm submission and status
inspection. The job chain persists independently of the local IDE or SSH session.
No specific GPU nodes, `--gres`, or Slurm environment-export override is requested.

Submission evidence:
`runs/deployment/evidence/a2_submission_20261005.json`.
Source archive proof:
`runs/deployment/evidence/a2_source_upload_20261005.json`.
Remote plan: `runs/a2/execution_plan.json`.
Validation evidence: `runs/a2/validation/`.
Final stage status: `runs/a2/status.json`.

At 04:41 UTC / 10:11 IST, CPU validation was running; CUDA controls, review,
all 36 label tasks and CPU forecasters were pending on their declared dependencies.
Reproducibility repair is awaiting allocated-node evidence; the historical A1
Llama replay discrepancy has not been assigned an unsupported root cause.

At 04:42 UTC / 10:12 IST, CPU job 1049531 was COMPLETED, exit zero,
in 57 seconds: 126 tests passed in 41.68 seconds, the A2 plan was prepared,
and archived A1 evidence/source checks passed. GPU controls remained pending;
reproducibility is awaiting their comparison, not yet declared resolved.

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
