# Vista campaign execution

The active campaign is consolidated on TACC Vista. DGX jobs were retired on
October 1, 2026; their artifacts remain calibration evidence and are excluded
from Vista research labels. All installation, tests, downloads, verification,
and training run in SLURM compute-node allocations.

## Deployment and accounting

| Item | Verified setting |
| --- | --- |
| Project directory | `/scratch/11617/sujato_ts/third_eye_research` |
| Durable results directory | `/work/11617/sujato_ts/vista/third_eye_results` |
| Allocation | `IRI23021`, 3,400 SUs available at inspection |
| GPU queue | `gh`: H200 nodes, up to 20 jobs and 96 nodes per user |
| Development queue | `gh-dev`: one job per user, two-hour limit |
| CPU queue | `gg`: up to 20 jobs per user |
| CPU setup | Job `1040276`, completed in 7 minutes 16 seconds |
| CUDA preflight | Job `1040277`, passed in 3 minutes 16 seconds |
| Submission relay validation | Job `1041749`, passed on a CPU node |
| Campaign controller | Job `1041757`, running on a CPU node |
| Operational workflow audit | Passed as a compute step in allocation `1041757` |
| Core pilots | All four completed successfully: `1041762`, `1041764`, `1041766`, `1041768` |
| Initial Gate 1 trajectories | Array `1042902`: Qwen tasks stopped for correction scarcity; both Llama tasks completed successfully by October 3 06:15 UTC |
| A1 verification chain | Jobs `1044310` CPU setup and `1044311` CUDA verification passed; `1044312` pilot controller follows the afterok chain |
| Current A1 parallelism | At most four one-node GPU tasks; broad 20-GPU expansion held |
| GPU spending ceiling | 500 H200 node-hours, including GPU preflight/pilots |
| Project storage ceiling | 50 GB, independently enforced |
| Completion target | October 10, 2026, 23:59:59 IST |

Concurrency limits do not imply immediately available nodes. Both GPU queues
were occupied at the last allocation check. CPU staging installs the pinned
environment, validates data and isolated verification, and downloads the two
core models while waiting for a short CUDA allocation. The initial combined
setup/controller jobs `1040201` and `1040212` were superseded before execution.
CPU setup `1040253` then failed before installation because the accounting
utility path is unavailable on compute nodes. The replacement uses a verified
allocation snapshot when that utility is absent. Its unsatisfied dependent
jobs `1040254` and `1040255` were retired, and their accounting is retained.

Vista charges whole nodes with a 15-minute minimum per job. GPU budget checks
sum all tasks, including queued reservations; concurrency reduces elapsed time,
not total node-hours. The larger ceiling reflects the added allocation and is
not an A100-equivalent efficiency claim. DGX calibration used 6.5553 A100
GPU-hours, recorded separately. CPU setup and the CPU controller consume SUs
at the verified `gg` charge rate and are separate from GPU accounting.

The staged deployment record is `vista_deployment_v5.json`. The immutable
research manifest, created after preflight, is
`experiments/campaign_20261001_vista_v1.json`. Its controller writes
`runs/campaign_20261001_vista_v1/status.json` and `events.jsonl`. Each submitted
array records its cluster, allocation, task indices, concurrency, and reserved
cost; a task cannot be submitted twice under the same plan. TACC wrapper
preambles are handled explicitly when parsing job IDs.
Vista accepts `sbatch` only from login nodes. Controller `1040278` failed on its
first direct compute-node submission and submitted no research jobs. The
replacement keeps all scientific processing on compute nodes and uses the
restricted scheduler relay in `cluster/vista_submission.py`. The login service
only submits permitted Vista job scripts under `IRI23021`; it does not import
research code, run tests, or load models. Requests and responses are retained in
`runs/vista_submission_queue`, with scheduler-relay hashes recorded separately
in `vista_submission_relay_sha256.txt`. Claimed submissions are never retried
automatically after a crash, preventing duplicate allocations.

The controller polls the scheduler every two minutes. Between stages and on
stops, it stages sources, fixed data, proofs and results to `$WORK`; model
caches and credentials are excluded from durable result staging.

## Protocol and validation

The Vista pilot templates retain K=3, H=2, candidate size two, optimizer steps,
LoRA settings, correction seeds, bf16 precision, and generation/scoring batches
from the calibrated protocol. Core model commits are pinned to the same
revisions as DGX. All four model/task pilots run again on Vista before protocol
freezing; hardware changes are not assumed to preserve generated tokens or
accuracy. Every run records source hashes, dependencies, allocated hardware,
and isolated-verifier configuration. Public Llama weights and tokenizer
vocabulary are verified against original hashes, with metadata differences
preserved in the source proof.

Vista uses Linux aarch64. The pinned PyTorch 2.8.0 CUDA 12.9 ARM wheel is
installed on compute nodes; the CUDA 12.6 index has no matching ARM wheel.
NF4 is optional and is not required by these bf16 LoRA experiments. The CPU
stage checks canonical code references, namespace isolation, timeouts, and
early exits. The GPU stage checks adapter updates, exact rollback/checkpoint
reload, generation, and repeatable sampled batching before publishing success.

The local suite passed 77 tests before the site-wrapper fix; 11 focused
scheduler/source/gate/deadline regressions then passed. Vista CPU setup completed
successfully: all 78 tests passed, all 252 canonical code references passed in
the isolated sandbox, and both core models downloaded with source verification.
The immutable CPU proof is `runs/vista_preflight/1040276/passed.json`.
CUDA preflight also passed on an allocated GH200 node: adapter training,
rollback, checkpoint reload, reproducibility and actual Qwen generation were
verified. The GPU proof is `runs/vista_preflight/1040277/passed.json`.
The scheduler relay passed seven additional tests and an actual compute-to-login
test-only submission in CPU job `1041749`. The operational audit checks generated
CLI arguments for core, transfer, final-evaluation and stress stages, and verifies
that the research source still matches the GPU proof. All 345 generated argument
sets across six entrypoints passed in the controller's existing CPU allocation;
the redundant pending audit job `1041780` was cancelled. The audit proof is
`runs/vista_operational_audit/1041757/passed.json`. Its fixture plans cannot be
used as research protocols or gate evidence. Research gates remain pending;
queued work is not a passing research result.

The four pilots were released after this audit. Their pending wall-time requests
were reduced to four hours, preserving margin over the approximately 49-minute
completed DGX pilot and improving backfill opportunities. The protocol is
unchanged, and controller reservations remain conservative at twelve hours per
pilot. A test-only `gh` submission accepted a 20-task array with `%20`
concurrency. Independent study batches use that concurrency after measured
protocol freezing and the required research gates; prerequisites initially use
four model/task pilots and four predetermined Gate 1 trajectories.

The separately hashed scheduler/audit source bundle and deployment evidence are
under `experiments/archive_control`, so ordinary campaign backups also preserve
them. The original validated research source inventory remains unchanged.

At the October 2, 22:18 IST progress check, all core pilots had completed with
exit code `0:0`. Qwen math/code took 1:09:07 and 1:09:39; Llama math/code took
0:55:54 and 1:03:52. Each completed a matched H=2 state and accepted one update.
The controller froze both protocols from these measurements and submitted
array `1042902` for the four predetermined T=5 trajectories needed to evaluate
Gate 1 on 20 complete states. Its requested task limit is 7 hours 10 minutes.
The array was pending for priority, with no scheduler start estimate. Accounted
GPU time was 4.5589 H200 node-hours including the preflight minimum; 495.4411
node-hours remained before new reservations. Neither research gate had yet
been evaluated, and pilot outcomes are calibration evidence.

The controller evaluates Gate 1 on the first predetermined 20 complete states
and Gate 2 on sufficient informative validation states. Failure, insufficient
corrections, missing evidence, exhausted budget, or an estimated stage extending
past the declared deadline stops progression. Final benchmarks stay sealed
until the appropriate evaluation stage. The controller has a 48-hour scheduled
lifetime; scheduler delays or a wall-time stop require a reviewed continuation
from retained artifacts, not an unrecorded rerun.

### October 3 protocol feasibility stop

The four successful pilots established one-state execution, not T=5 correction
availability. In the initial trajectory array, Qwen math task `0` stopped at
generation zero after two candidate branches completed: the remaining branch's
second update had only one verified correction, below the frozen size of two.
No partial matched state was published as research labels. Qwen code task `9`
completed two generations, then stopped at generation two with two verified
corrections. At batch size two, those corrections support only one distinct
batch; K=3 distinct matched candidates are impossible. Local replay of the
retained pool reproduced this rejection and confirmed that the sampler accepts
the feasible root math pool.

Llama math and code tasks `18` and `27` remained running, with four and three
complete states respectively at inspection. The array therefore had nine
complete matched states at that earlier inspection, and its failed tasks prevent the originally required
20-state Gate 1 evaluation. Gate 1 and Gate 2 have not been evaluated; the larger study
is not submitted. This is a correction-scarcity protocol edge case, not evidence of
forecasting success or failure. Any correction-generation budget or training
pool revision requires a separately recorded protocol and uniform repiloting;
the frozen campaign and its failure evidence remain unchanged. At 05:08 UTC,
Llama math completed successfully after 05:08:36; Llama code subsequently
completed successfully after 06:14:44. Both remain pre-A1 evidence.

The October 3 response is frozen in
[Protocol Amendment A1](protocol_amendment_a1.md), with implementation and
execution described in [A1 execution](a1_execution.md). A1 permits overlapping
distinct compositions, enlarges only correction harvesting, and records
identity terminal continuations. Its calibration is capped at four GPUs,
stops after the first four pilots for evidence inspection before the remaining
sixteen, and then stops for Gate 1 review; the 20-GPU expansion remains held.

## Timeline and monitoring

The planning target is main results October 4–6, validation and result review
October 7–9, and final reproducibility artifacts October 10, ahead of the
October 12 submission. This depends on queue starts, successful pilots, and
research gates. At sustained 20-node concurrency, 500 GPU node-hours correspond
to 25 GPU wall-clock hours; prerequisite stages and queue waits add time.

```bash
cd /scratch/11617/sujato_ts/third_eye_research
squeue -u "$USER" -o '%i %j %P %T %M %R'
sacct -X -j 1040276,1040277,1041749,1041757,1041780 --format=JobID,State,Elapsed,ExitCode
tail -n 40 third_eye_vista_cpu_setup_1040276.out
tail -n 40 third_eye_vista_preflight_1040277.out
tail -n 40 third_eye_vista_controller_1041757.out
tail -n 40 third_eye_vista_audit_controller_step_v4_1041757.out
cat runs/campaign_20261001_vista_v1/status.json
```

Source archive SHA-256:
`a0610be992a6d1e06420b5c5695355fc2ff780314f11f2ede39e621991d46a63`.
Audited data archive SHA-256:
`b4a69f292b3532300c4a48b27e76055c338e9210c8480f11c50742b56740347e`.
Credentials remain local and are excluded from source and data bundles.

Queue/storage conventions are documented in the
[Vista user guide](https://docs.tacc.utexas.edu/hpc/vista/) and the local
[setup notes](tacc%20setup.md).
The login-only submission restriction is explicit in
[Vista job management](https://docs.tacc.utexas.edu/hpc/vista/jobmanagement/).
