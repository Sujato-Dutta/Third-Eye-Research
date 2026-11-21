# Third Eye: current execution progress

## November 20, 09:58 IST

- **Longer trajectories: 16/16 complete.** DGX Llama-8B scoring jobs `38278`
  and `39390` passed, then both archive jobs and the Llama-8B model review
  passed. The two Qwen-8B archives on Lonestar6 were also verified. All four
  models are ready for the combined trajectory review, queued as `3597817`.
- **Confirmation: 19/20 verified.** DGX parent 12 completed in 15:33:09.
  Lonestar6 CPU job `3594910` verified and exported all six recovered
  parents. DGX parent 13 (`38040_13`) failed at the storage admission check
  before training or writing a run directory. The same fixed seed is queued
  on Lonestar6 as `3597813`, using the pinned scientific runner and existing
  validated inputs. The scheduler has not given a start time.
- DGX CPU jobs `42014` and `42015` inspected and exported the data needed for
  the two reviews. Lonestar6 has the exported per-item contrasts and their
  checksum receipts. The full 20-parent review (`3597819`) waits for a
  successful exit from `3597813` and applies the frozen analysis. The smaller
  review inputs avoid copying adapter tensors into the DGX home quota.

## November 19, 10:25 IST

- **Longer trajectories: 14/16 runs complete across both clusters.** Both
  transferred Qwen3-8B trajectories finished on Lonestar6 and their result
  archives passed verification. The complete cohort review is still pending.
  Both remaining Llama-8B seeds finished training on DGX and failed during
  scoring with CUDA out of memory. Their checkpoint audits passed and
  scoring-only jobs `39390` and `38278` are awaiting GPUs.
- **Confirmation: 12/20 previously complete, six more scored and awaiting
  final archive verification.** All six Lonestar6 recoveries (parents
  14–19) wrote their complete scores and passed the scientific artifact
  checks. The two bundle jobs exited with errors because scratch rejected
  tokenizer hard links during archival. CPU job `3594910` is verifying
  the existing raw or compressed tensors against their training seals and
  exporting the completed results without hard links. It has started the
  first export. No training or scoring is repeated.
- **DGX confirmation:** parent 12 (`38039_12`) is running with about 14 hours
  elapsed. Parent 13 (`38040_13`) waits for parent 12. Full-cohort analyses
  and cross-cluster consolidation remain pending.

## Latest transfer and scheduler check: November 18, approximately 17:14 IST

**All six saved-state bundles are uploaded to Lonestar6.** The ownership
manifest is uploaded and ownership job `3588143` has been released into its
dependency chain. Production `3588144`/`3588145` is pending; it has not started.

Routing CPU setup **`3592177`** is running. Routing CUDA preflight
**`3592178`** waits for that setup. Both production jobs explicitly depend on
the new CUDA job and ownership verification; ownership verification depends on
the new CPU job. Earlier passed validation receipts are not a report that
these replacement jobs have passed. The local handoff committed ownership
before the replacement validations finished, but the scheduler dependencies
still prevent production from starting early. Future coordinator runs now
also require exact current job IDs in completed readiness proofs.

The handoff assigns two Qwen8B trajectories and six sealed-state scoring
recoveries to Lonestar6. DGX retains its running Llama8B index-2 trajectory
and audited index-3 scoring recovery. Only the two pending Qwen8B DGX
fallbacks were newly held. The transferred inputs, ownership and scheduler
dependencies are now remote: these submitted jobs can continue without the
local transfer process. Subsequent result collection/consolidation still
requires a connection.

Scientific counts remain **12/20 confirmation parents** and **12/16 longer
trajectory seeds** complete. Primary retries `38039`/`38040` are still pending;
Llama8B `36444` is running (about 3:41 elapsed at 17:14), and scoring recovery
`38278` is pending. No new complete-cohort scientific result is claimed.

## Latest afternoon check: November 18, approximately 16:30 IST

- **Confirmation: 12/20 complete.** Fixed-seed retries `38039`/`38040` are
  pending GPU priority/dependency, respectively. Neither has started. The
  10:32 storage check passed at 47.402/50 GB after recovery of about 945 MB;
  that is a timestamped storage measurement, not a new afternoon scan.
- **Longer trajectories: 12/16 complete.** Llama8B index 3 (`36443`) failed
  after 16 hours with CUDA OOM during final scoring on a 40-GB A100. It is
  not counted complete. Llama8B index 2 (`36444`) is running, approximately
  2 hours 36 minutes elapsed at the 16:07 check. The two Qwen8B seeds remain
  pending for the verified TACC handoff.
- **Training preservation:** CPU audit `38276` passed for all ten sealed
  Llama8B index-3 checkpoints and donor commitments. Scoring-only recovery
  **`38278`** is queued on DGX; training and harvesting are not repeated.
  Eight frozen scoring checkpoints are evaluated with a fresh backend process
  for each, unchanged configuration, exact restore and a free-GPU-memory
  check before loading. Partial scoring evidence remains intact. Archive
  `35094` now waits for recovery `38278`.
- **Automatic follow-up:** CPU admission `38277` waits for any exit of the
  running Llama8B index-2 job. It exits after inspection if the seed succeeded.
  If scoring failed, it may queue recovery only after a complete training seal
  passes its integrity audit. It does not repeat unfinished training.
  Initial long CPU-controller requests `38274`/`38275` were cancelled before
  execution; the replacement CPU admissions request 20 minutes and release
  the CPU after scheduling, rather than holding it throughout a GPU wait.
- **Transfer: 5/6 archives acknowledged on Lonestar6.** Archive 19 is uploading
  (approximately 112/502 MB at 16:33). The earlier SSH transport stopped with
  archive 18 incomplete. A fresh MFA code restored access, and the transport
  now runs as a separate hidden background process, like the handoff
  coordinator. Both require an awake, connected local machine until staging
  and handoff finish. Temporary-account access expired and is not used again.
- **TACC production has not started.** Initial CPU `3588141` and CUDA
  `3588142` validations passed. An operational routing extension explicitly
  retains a pending, audited DGX scoring recovery without duplicating it on
  TACC. Repeat CPU setup and CUDA preflight are prepared for submission after
  the current upload, before ownership release. The same production node
  jobs `3588144`/`3588145` remain queued. Seeds, training budgets, data,
  endpoints and frozen scientific sources are unchanged.

No new complete-cohort scientific result is claimed this afternoon. The
latest work repairs execution and preserves already completed training.

## Earlier morning snapshot

Last research-job snapshot: **November 18, 2026, approximately 10:33 IST**.
Lonestar6 preparation/queue updates below were checked this morning. The
full-device DGX inventory retains its
earlier timestamps; they have not been relabelled as new measurements.
Counts describe the current confirmation and longer-trajectory round. They are
not a percentage completion estimate for the entire paper.

## Lonestar6 migration is queued

Access and allocation were verified: `IRI23021` initially showed **5,940 SUs**
available before this round's charges. Yesterday's CPU preparation `3586053`
and readiness `3586076` failed because the parent directory for pytest's
temporary path had not been created. Their dependent GPU requests were
cancelled before GPU execution. The operational fix creates that directory;
the frozen scientific sources and protocol remain unchanged.

Replacement CPU preparation `3588141` **completed successfully**, including
the two frozen test invocations (5 and 15 tests). Three-model full-workload
CUDA validation `3588142` also **completed successfully** in 10 minutes 49
seconds. Held ownership verification `3588143`, production
`3588144`/`3588145`, and final archive checking `3588146` are submitted with
verified dependencies. Production is still blocked by input/ownership
verification, not a currently failed CPU or CUDA validation.
The production requests pack independent tasks onto **six full A100s across
two nodes**, with a total new reservation below **197 SUs**.

The task grid contains four remaining 8B trajectories and scoring-only recovery
of confirmation parents 14--19. A coordinator leaves DGX work eligible while
TACC prepares; at handoff, only pending DGX fallbacks are held. Running/completed
DGX seeds stay on DGX and are skipped on TACC. Original running jobs
are untouched. Saved-state transfer and ownership activation are still in
progress; this is not a report that production GPUs have started.

See [Lonestar6 execution](lonestar6_execution_20261116.md) for the exact input
bindings, budget, preparation status and local transport dependency. No new
GPU research is submitted to the temporary DGX account. All six sealed
checkpoint bundles reached the local machine. Four reached Lonestar6;
the final two uploads have resumed. Keep this machine connected until
transfer and ownership handoff complete. The authenticated TACC connection
was restored using the saved MFA code this morning. The previous connection
ended with a partial parent-18 upload. Uploads now resume from verified boundary
bytes, record progress, and use a bounded channel timeout; full archive and
member checksums remain mandatory on the receiving compute node. The handoff
coordinator has restarted. The temporary account's access window has ended;
no further login to that account is used.

## Completed and running experiments

### Main confirmation: 12 of 20 parents complete

The fixed Qwen3-4B math confirmation uses eight unique corrections, 25 optimizer
steps and three continuation draws per parent. Target, OOD and retention are
scored on the frozen enlarged evaluation pools.

- Parents 0--11 completed, including their dependent inspection/archive jobs.
- Parent 11 completed in 13 hours 42 minutes. Parents 12--13 and the original
  six fallback attempts refused to start research because whole-home storage
  left less than the fixed 2-GB scratch reserve. No quota breach occurred.
- Same-seed replacement arrays `38039` (12) and `38040` (13) are bound to the
  successful CPU headroom check `38037`; parent 13 also depends on parent 12.
  Parent 12 is pending GPU priority, not blocked by storage validation.
  The six migration fallbacks
  below remain held for exclusive Lonestar6 handoff. Original failure logs and
  scientific outcomes remain preserved; no completed parent is restarted.
- Six temporary-account MIG attempts, parents 14--19, failed during final
  scoring. These are operational failures, not completed scientific results.
- Final review `35054` had become dependency-blocked by failed attempts. Its
  dependency registry now refers to the replacement jobs; strict cohort review
  still checks all 20 fixed parents, including completed original parents.

| Parent index | Failed MIG job | Held replacement fallback array job |
| --- | --- | --- |
| 14 | 35465 | 38023 |
| 15 | 35466 | 38024 |
| 16 | 35467 | 38025 |
| 17 | 35468 | 38026 |
| 18 | 35469 | 38027 |
| 19 | 35471 | 38028 |

### Longer trajectories: 12 of 16 paired seeds complete

Each paired trajectory compares native continuation with half-LR continuation
through T=5, with frozen T=3/T=5 scoring points.

| Backbone | Complete | Running | Pending |
| --- | ---: | ---: | ---: |
| Llama-3.2-3B-Instruct | 4 | 0 | 0 |
| Qwen3-8B | 2 | 0 | 2 |
| Llama-3.1-8B-Instruct | 2 | 1 | 1 |
| Qwen3-4B | 4 | 0 | 0 |
| **Total** | **12** | **1** | **3** |

The completed trajectories passed their dependent inspection/archive jobs.
Qwen4B's final seed (`35102`) completed in 8 hours 48 minutes; its archive
and complete-cohort review `35104` also passed. The four temporary-account 8B requests returned
through their owning-account handoffs. Registered original-account fallbacks
are `36443`, `36444`, `36445`, `36446`. Llama8B seed index 3 (`36443`) is running
on a full A100, approximately 12 hours 48 minutes elapsed at the morning check.
The other three remain eligible until the verified TACC handoff. Running or
completed seeds will be skipped on TACC. Model reviews and integrated review
`35105` still require their complete fixed cohorts.

The completed four-seed Qwen4B math cohort has immediate gain +1.060 pp,
native T=3 gain +2.525 pp (simultaneous 95% interval [0.647, 4.542]), and
native T=5 gain +1.747 pp ([-0.242, 3.731]). These are exploratory shared-item,
parent-level intervals with the frozen 32-endpoint simultaneous family.
The T=5 decline from T=3 is uncertain: -0.778 pp ([-2.672, 1.182]).
Unlike the completed Llama3B cohort, Qwen4B does not show an aggregate decline
below its parent here. Results support investigating backbone dependence;
they do not establish universal downstream harm. No final four-backbone
comparison is available until both 8B cohorts finish.

The latest storage snapshots report approximately 47.402 GB allocated on the
primary 50.000-GB account and 44.755 GB on the second account. Primary storage is
below the hard quota and passes the unchanged 2-GB scratch-reserve check.
Consolidation must check headroom before any large result import.
CPU inventory `38019` completed. CPU maintenance `38034` completed
lossless inode sharing of byte-identical artifacts in the twelve
completed, owning-account parent folders. Delegated failure artifacts and
unrelated project data are excluded. It recovered 708,788,224 allocated bytes.
CPU job `38037` retired 236,650,496 bytes of regenerable pip, Hugging Face Xet
download-chunk and Tectonic build caches. All model weights, environments,
research outputs and original failure evidence remain intact. Its fresh
whole-home snapshot was 47,401,684,992 bytes and its storage check passed.
The interim dependency-held requests `38021`/`38022` were replaced by
`38039`/`38040` before research execution; no completed seed was restarted.

## Full A100 availability and account allowances

Live Slurm associations were checked directly:

| Login | Slurm account | GPU ceiling | CPU ceiling | RAM ceiling |
| --- | --- | ---: | ---: | ---: |
| `muef2320` | faculty | 2 | 64 | 512 GB |
| `se23uari167` | student | 1 | 32 | 256 GB |
| `se23ucse205` | student | 1 | 32 | 256 GB |

These ceilings are separate account allowances, not reserved devices.
`gpu_faculty` and `gpu_student` both use nodes `dgxa,dgxb` and report the same
seven full-A100 resources. Faculty's partition QoS priority is 100 versus 50
for student; both partitions have priority tier 1 and preemption is off.

| Node | Full A100 resources | Allocated full A100 resources |
| --- | ---: | ---: |
| dgxa | 2 | 2 |
| dgxb | 5 | 5 |
| **Total** | **7** | **7** |

At the approximately 11:00 inventory check, two full A100s were running our
original experiments and five were allocated to other users; none was free.
The faculty request was already in `gpu_faculty`, with an estimated start of
November 17 at 14:11 IST. This was not a reservation. The four 32-hour
delegated trajectories subsequently returned to their original queue because
their protective runtimes no longer fit the temporary lease. At 17:56 there
were no GPU research jobs left on the temporary account. The TACC migration
replaces that temporary capacity rather than interrupting healthy DGX work.

## MIG failures and checkpoint preservation

The earlier short compatibility test passed on a 20-GB slice, including all
25 training steps and development-set scoring. It did not cover the full
memory demands of final scoring. All six released MIG attempts ran for
approximately 6--7 hours, then failed with CUDA OOM during M0 retention
scoring. The first failure tried to allocate another 1.23 GiB on a 19.50-GiB
slice. No MIG research jobs remain running or queued.

Partial outputs were preserved under `failure_evidence`; no failed run is
counted as a scientific completion. Read-only CPU audit `36281` verified all
**66 sealed checkpoint files** (11 per parent) against their stored SHA-256
commitments. Every attempt had finished saving its parameter updates before
the scoring failure. The checkpoints therefore remain available for a
validated scoring-only recovery, avoiding repeated training. Recovery is now
included in the dependency-blocked TACC production queue; no recovery GPU has
started yet. Completed outcomes and partial scoring evidence remain untouched.

## Scientific evidence available

The complete four-seed Llama3B math review reports equal-role utility changes:

| Endpoint | Change, percentage points |
| --- | ---: |
| First update versus unchanged parent | -0.21 |
| Native T=3 versus unchanged parent | -7.11 |
| Native T=5 versus unchanged parent | -16.08 |
| Half-LR T=5 versus native T=5 | +6.70 |

The simultaneous interval for native T=5 versus the parent is
[-20.07, -12.29] pp; half-LR versus native T=5 is [+3.22, +9.78] pp.
Half-LR reduces the loss but does not recover utility above holding the parent.
These are exploratory results for one backbone, one math stream and four
paired seeds. Cross-model robustness awaits the other complete cohorts.
No forecasting claim is required for this study.

## Storage and safeguards

| Account role | Whole-home usage | Limit |
| --- | ---: | ---: |
| Main confirmation, 10:45 snapshot | 46.88 GB | 50 GB |
| Longer trajectories, 10:45 snapshot | 44.25 GB | 50 GB |
| Temporary faculty, before transfer exports | 121.10 GB | 200 GB |

GB denotes one billion bytes. Main-account headroom is limited, especially
while failed raw checkpoints are preserved. Only verified lossless archival
or deduplication may reclaim research-artifact space. Owner data, datasets,
code and trained checkpoints on the faculty account remain preserved.

Original lease watchdog `35965` stopped after an accounting-query error.
Replacement `36280` passed four lease-admission boundary checks and used live
`squeue` with retries rather than the accounting database. It returned the
remaining pending delegated jobs when their runtimes no longer fit, then
finished with no delegated GPU jobs left. Quota guard `35966` also finished.
At 17:56 the temporary account retained only storage monitor `35967`.
Original-account monitors `35488` and `35489` remained active. CPU export
`36547` subsequently prepared the read-only migration bundles; their space is
additional to the faculty snapshot above and remains well below its 200-GB
limit.

Temporary access ends **November 17, 23:59:59 IST**. Pending work that cannot
fit its protective runtime returns through owning-account handoffs. The two
original accounts can continue beyond that date. All scientific computation,
validation, storage scans and archival run in Slurm compute allocations.
Login-node operations are limited to transport, scheduler queries/submissions
and small metadata reads.

## Remaining work

1. Complete eight confirmation parents and four paired trajectories, preserving
   all fixed seeds and protocol settings.
2. Finish TACC preparation, saved-state staging and exclusive seed handoff;
   execute the queued work and verify its returned artifacts.
3. Run the already connected complete-cohort inspections/statistical reviews.
4. Incorporate validated cross-model effects, continuation contrasts and
   mitigation results into manuscript figures/tables and rebuild the paper.

## Evidence

- Full-device inventory: `runs/deployment/dgx_parallel_faculty_requests_20261112/082-full-a100-availability-20261116.response.json`.
- Live account/partition/QoS associations and checkpoint audit: `runs/deployment/dgx_parallel_faculty_requests_20261112/089-faculty-quota-verification.response.json`.
- Replacement guard: `runs/deployment/dgx_parallel_faculty_requests_20261112/088-recovery-audit-status.response.json`.
- Cohort counts/storage: primary request `018-progress-20261116`, secondary
  request `021-progress-20261116`, faculty request `080-progress-20261116`
  under the corresponding deployment queues.
- [Confirmation execution](dgx_positive_regime_20261109.md).
- [Longer trajectories](long_horizon_execution_20261109.md).
- [Temporary-worker mapping and fallback](dgx_temporary_worker_20261112.md).
