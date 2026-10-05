# A1 execution

Current operational settings are recorded in the
[October 4 deadline execution override](deadline_execution.md): up to twenty
GPU nodes, a 3,000-hour cumulative GPU envelope, and validated parallel scheduling
of all remaining fixed calibrations. Earlier four-GPU/500-hour/wave settings below
describe the original execution. The frozen A1 science and review stops remain.

The four-stream A1 calibration implements
[Protocol Amendment A1](protocol_amendment_a1.md). Its source is deployed to
`amendments/a1` beneath the original scratch project, keeping original source,
Qwen partial runs and the completed Llama trajectories unchanged. The original
environment/model cache are reused by absolute path; no credentials are copied.

CPU setup waits for the original trajectory array to finish, runs the complete
suite, and builds immutable expanded training manifests at the original dataset
commits. Evaluation files remain byte-identical. A short scheduled CUDA check
then verifies actual adapter training, terminal H=2 identity, checkpoint reload,
and source integrity. Only after both checks pass can A1 pilots be submitted.

The controller first submits four fresh seed-1042 pilots, one per model/task
stream, using one full matched K=3 H=2 state each. The existing Gate 1 minimum
is 20 complete states; sixteen additional one-state calibrations use the fixed
seed grid in A1. Arrays are limited to four concurrent GPUs throughout. They
are calibration work, not the broad 20-GPU study. After the first four pilots,
the controller writes `pilot_inspection.json` and stops at
`pilot_review_required` before submitting the other sixteen. Inspection checks
complete K=3/H=2 labels, unique within-batch IDs and distinct compositions,
verified-pool membership, fixed optimizer budgets, continuation availability,
terminal identity, and populated target/OOD/retention metrics. It reports
terminal frequency per stream and overall, including fully terminal streams.
No new scientific threshold is applied to terminal frequency.

A healthy four-pilot report must be inspected before a compute controller is
resumed with `--resume --reviewed-pilots <inspection-file-sha256>` (or the job
environment variables `THIRD_EYE_A1_RESUME=1` and
`THIRD_EYE_A1_REVIEWED_PILOTS=<inspection-file-sha256>`). The hash binds review
to the exact evidence; invalid or modified pilot artifacts cannot release the
remaining calibrations. This operational checkpoint leaves the frozen A1
harvesting, sampling, training and terminal-state rules unchanged. Both CPU and
CUDA proofs must explicitly pass and match the deployed source before launch.

Wall-time reservations are
eight hours per task and counts/spend are checked against the existing global
500 H200 node-hour ceiling, including the original campaign's GPU costs.

The controller stops at `gate1_review_required` even when the gate passes.
At that review, PASS makes small-model Sprint 2 labels and H=1 versus H=2
Third Eye-Direct forecasting the next priority. INSUFFICIENT EVIDENCE calls for
more predeclared small-model states under the same A1 rules. FAIL calls for a
recorded candidate-construction revision and fresh validation rather than
scaling. Gate 2 requires held-out H=2 Spearman >=0.30 and informative top-1
>=45%, with the existing evidence minimum and tie-adjusted chance baseline.
8B/scaling eligibility requires Gate 2, and no scaling is automatically released.
Insufficient complete states are reported as `insufficient_evidence`. There is
no automatic forecaster, transfer, or 20-GPU expansion. Source and amendment
hashes, original accounting, submitted IDs, per-task status, complete labels,
correction-scarcity records and Gate 1 evidence remain available for review.

CPU setup produces `runs/a1_cpu_passed.json`; CUDA verification produces
`runs/a1_gpu_passed.json`. Calibration writes
`runs/campaign_a1_20261003/campaign.json`, `status.json`, and `gate1/gate1.json`.
Completed phases stage A1 sources, manifest data and results to
`$WORK/third_eye_results/campaign_a1_20261003`, excluding the shared model cache
and environment. The restricted login service only relays scheduler submissions;
all installation, testing, dataset preparation, inference, training, analysis,
and result staging run on compute nodes.

The controller has a 48-hour allocation. Its phase/job ledger supports a reviewed
`--resume` in a fresh CPU allocation with the same source/amendment, retaining
submitted IDs rather than duplicating runs. Failed compute jobs are not silently
restarted. Pending reservations, deadline, storage headroom, source identity and
the hard expansion hold remain enforced on continuation.

Local validation: the initial 93 tests passed in 71.39 seconds. After the
operational inspection checkpoint was added, all 101 tests passed in 120.14
seconds, including eight inspection-integrity tests. Lint and shell syntax
checks passed. The isolated A1 base source archive SHA-256 is
`498125cba82513c11bce7ca9ceea87b4e12d05c4be719ce656661f538b3fdf59`.
The frozen amendment SHA-256 on Vista is
`48eb888333f2dc838f4b5a9454375ec81f74f9f703fe8de69ccd7016a1e933c3`.
The operational checkpoint is separately versioned as
`vista_a1_inspection_control_20261003_v1.tar.gz`; deployment records retain its
digest and per-file hashes alongside the unchanged base archive and amendment.
It modifies the controller, inspection utility and tests only; the scientific
labeling/harvesting/training implementation remains the validated A1 source.

At October 3, 2026, 06:17 UTC (11:47 IST), both validation jobs have passed:

| Job | Purpose | Dependency | Observed state |
| --- | --- | --- | --- |
| 1044310 | CPU setup and full test suite | afterany:1042902, satisfied | Completed, 00:00:57 on i617-141 |
| 1044311 | Scheduled CUDA terminal-state verification | afterok:1044310, satisfied | Completed, 00:00:36 on c642-071 |
| 1044312 | Four-stream A1 pilot controller | afterok:1044311, satisfied | Running on i614-021; submitted only the first four pilots |

The original Llama math task `1042902_18` completed successfully in 05:08:36;
Llama code `1042902_27` completed successfully in 06:14:44. The original Qwen
failures and partial artifacts are retained. The dependency waited for the
whole original array to finish, including its failed tasks, without restarting
or changing those runs. CPU setup and the controller were temporarily held
while the operational inspection checkpoint was deployed, then released with
their original dependencies and IDs intact. CPU validation passed all 101 tests
in 36.27 seconds and prepared 1,024 GSM8K training examples and 306 MBPP training
examples after the frozen overlap audit. The CUDA adapter/terminal identity
test passed in 16.10 seconds on an allocated NVIDIA GH200 node. Both jobs exited
with code 0; their source inventories match SHA-256
`f06285d80e3a248a8141448d51fbac55a095016cfef6036351f18741984abcaa`.
Pilot submission follows this successfully verified chain. The
controller's four-GPU cap and the 20-GPU expansion hold apply throughout. The
scheduler submission record is preserved in
`runs/deployment/evidence/a1_deployment_20261003.json` locally and
`amendments/a1/experiments/archive_control/a1_deployment.json` on Vista.
The checkpoint deployment record is preserved locally in
`runs/deployment/evidence/inspection-deployment-evidence_20261003.json` and in
`amendments/a1/experiments/archive_control/inspection_control_deployment.json`
on Vista. Later execution-note updates do not change the frozen amendment or
the verified source inventory.

At 06:19 UTC (11:49 IST), GPU verification is still in Slurm's `COMPLETING`
cleanup phase despite its successful accounting record and exit code. The
controller remains `PENDING (Dependency)` until that cleanup finishes; no A1
pilot array has been submitted yet. This preserves the success dependency and
does not bypass validation. The remaining sixteen calibrations cannot be
submitted by this first controller invocation because it stops for inspection.

At 07:36 UTC (13:06 IST), CUDA verification has fully left `COMPLETING`.
Controller `1044312` is running and has submitted exactly one pilot array,
`1044550`, with indices `0,5,10,15` and a four-task concurrency cap. Its source
and amendment hashes still match the validation proofs. Prior GPU usage is
20.1914 H200 node-hours, including original runs and both GPU verification
stages; the four new tasks reserve 32 node-hours at eight hours per task.

| Array task | Stream | Seed | Observed state |
| --- | --- | --- | --- |
| 1044550_0 | Qwen math | 1042 | Pending, Priority |
| 1044550_5 | Qwen code | 1042 | Pending, Priority |
| 1044550_10 | Llama math | 1042 | Pending, Priority |
| 1044550_15 | Llama code | 1042 | Pending, Priority |

No A1 pilot labels or `pilot_inspection.json` are available yet. Starvation
frequency and Qwen-versus-Llama asymmetry therefore remain unmeasured under A1.
The four-pilot inspection must finish before releasing the remaining sixteen;
high terminal frequency is reviewed for meaningful H=2 comparisons without a
new numerical cutoff. The Gate 1 review stop, Gate 2 priority, and scaling/8B
holds remain in force. The controller has not submitted another array.

At 14:22 UTC (19:52 IST), three seed-1042 pilots have completed successfully:

| Array task | Stream | Runtime at inspection | State |
| --- | --- | --- | --- |
| 1044550_0 | Qwen math | 03:45:52 | Running |
| 1044550_5 | Qwen code | 03:01:40 | Completed, exit 0 |
| 1044550_10 | Llama math | 03:14:14 | Completed, exit 0 |
| 1044550_15 | Llama code | 03:13:51 | Completed, exit 0 |

Locally inspected published labels from the three completed streams each contain
one complete K=3 state. All nine continuations are available, with 50 candidate
optimizer steps and 50 continuation steps each; none is terminal. Verified
continuation-pool sizes are Qwen code 11/18/15, Llama math 149/53/74, and Llama
code 26/26/35. These are preliminary availability observations, not Gate 1 or
a full four-stream inspection result.

At 14:24 UTC, Qwen math has saved two of its three branch records and is still
working through the final branch. Its partial records remain diagnostic until
the complete matched state is published. `pilot_inspection.json` is not yet
available. Controller `1044312` continues to wait for this original four-task
array; no remaining calibration or scaling jobs have been submitted. No
scientific code, A1 rule, or terminal-frequency threshold was changed.

At 15:15 UTC (20:45 IST), Qwen math is still running, elapsed 04:38:03, with
the other three pilots completed successfully. Its third candidate has finished
the t+1 target/OOD/retention evaluation and is harvesting corrections for the
final standardized continuation (70 initially failed training problems, fixed
16-attempt budget). The four-pilot inspection report remains pending. No
remaining calibration jobs or scaling have been released; no protocol or code
changes were made during this progress check.

At 17:17 UTC (22:47 IST), all four seed-1042 pilots and controller `1044312`
have completed successfully. Qwen math finished in 04:40:50. The scheduled
inspection reports `status=valid`, no errors, four complete K=3 states, twelve
candidate branches and zero terminal continuations in every stream. All batches
have distinct compositions and unique within-batch examples; candidate and
continuation budgets remain 50 optimizer steps, and all target/OOD/retention
metrics and H=2 deltas are valid. Every branch has nonidentical t+1/t+2
evaluations. Terminal identity is not empirically exercised in these four
pilots because no branch is terminal; the CUDA validation already verified
that fallback.

The inspection SHA-256 is
`9c267eb55c3c9b42df1e476874f0cd840439bd36dfec6712e7cb1f35c67fba8b`.
The evidence was reviewed and recorded in `pilot_review.json`, with a local
copy at `runs/deployment/evidence/a1_pilot_review_20261003.json`. No starvation
asymmetry or terminal dominance prevents meaningful H=2 comparisons in this
pilot sample. This is a clean feasibility inspection, not a passed Gate 1:
only four of the required twenty complete A1 states are available.

| Stream | Verified continuation-pool sizes, k0/k1/k2 | Terminal branches |
| --- | --- | --- |
| Qwen math | 21 / 13 / 13 | 0 / 3 |
| Qwen code | 11 / 18 / 15 | 0 / 3 |
| Llama math | 149 / 53 / 74 | 0 / 3 |
| Llama code | 26 / 26 / 35 | 0 / 3 |

Qwen has smaller verified pools than Llama within each task family in this
one-seed sample, but all twelve pools supply the unchanged two-example
continuation. This descriptive supply difference is retained for later seeds;
it is not a terminal-rate threshold or a saturation claim.

At 17:23 UTC (22:53 IST), reviewed continuation controller `1045577` is queued
on `gg`, pending for Priority. It retains the original campaign ledger and
inspection hash, releases the sixteen fixed seed-2042/3042/4042/5042 states in
four waves capped at four concurrent GPUs, and stops at Gate 1 review. The
GPU waves have not yet been submitted while the CPU controller is pending.
Tracked GPU usage before these calibrations is 34.3678 H200 node-hours against
the unchanged global ceiling of 500.

The first resume submission with `afterok:1044312` was rejected by Slurm with
`Job dependency problem` because that completed job had retired from its live
job table. No experiment was launched. All CPU/CUDA/controller/four-pilot
accounting records were verified as completed with exit zero before resubmitting
the continuation without that stale live dependency. The failed scheduler
attempt and its claim remain archived; validation/source/review checks are
still enforced inside the compute controller. The submission record is
`calibration_resume_submission.json`, copied locally to
`runs/deployment/evidence/a1_calibration_resume_submission_20261003.json`.
No scientific source, A1 semantics, or terminal-frequency threshold changed.
The 20-GPU and 8B holds remain; forecasting/Gate 2 remains the next priority
only after sufficient Gate 1 evidence and a passing decision.

At 17:25 UTC (22:55 IST), continuation controller `1045577` is running on
compute node `i615-014`. It rechecked the unchanged four-pilot evidence and
has submitted only the first calibration wave: array `1045578`, indices
`1,6,11,16`, one seed-2042 state per stream, capped at four concurrent GPUs.
All four tasks are pending for Priority. The other twelve fixed calibrations
will follow as three four-task waves under the same reviewed controller. No
old pilot was rerun, no extra seed was selected, and Gate 1 remains unevaluated
at four complete states. The mandatory Gate 1 review stop and scaling holds
remain active.

At October 4, 03:51 UTC (09:21 IST), all four seed-2042 tasks in array
`1045578` have completed with exit zero. Their published files each contain a
complete K=3 state with distinct batch hashes, valid target/OOD/retention
metrics and H=1/H=2 metric deltas, and matched 50-step candidate/continuation
budgets. None of the new twelve branches is terminal. This checks published
labels; it is not a replacement for full pool/checkpoint inspection or Gate 1.

| Stream | Seed-2042 runtime | Continuation-pool sizes |
| --- | --- | --- |
| Qwen math | 06:09:47 | 28 / 22 / 17 |
| Qwen code | 03:22:37 | 12 / 12 / 13 |
| Llama math | 02:59:29 | 62 / 71 / 78 |
| Llama code | 03:24:37 | 37 / 34 / 34 |

Eight of twenty complete states are now available, with 24 candidate records
and zero terminal branches across those states. Seed-3042 array `1046534`,
indices `2,7,12,17`, is pending for Priority; controller `1045577` remains on
its scheduled CPU node. The other eight fixed calibrations have not yet been
submitted. Tracked global GPU spending is 50.3094 hours out of the unchanged
500-hour ceiling. Gate 1 is not yet evaluated, and 20-GPU/8B expansion remains
held. Local evidence is `runs/deployment/evidence/a1_eight_state_progress_20261004.json`
and the archived live Slurm snapshot. No active scientific source or job was
changed during this check.
