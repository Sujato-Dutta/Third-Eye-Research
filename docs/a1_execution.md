# A1 execution

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
| 1044312 | Four-stream A1 pilot controller | afterok:1044311 | Pending |

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
