# Third Eye: current execution progress

Last research-job snapshot: **November 17, 2026, approximately 09:15 IST**.
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

Replacement CPU preparation `3588141` is running on a compute node. CUDA
validation `3588142`, held ownership verification `3588143`, production
`3588144`/`3588145`, and final archive checking `3588146` are submitted with
verified dependencies. The first frozen CPU suite passed (5 tests); the
second suite was still running at the last log check. Full validation has
not yet passed.
The production requests pack independent tasks onto **six full A100s across
two nodes**, with a total new reservation below **197 SUs**.

The task grid contains four remaining 8B trajectories and scoring-only recovery
of confirmation parents 14--19. A coordinator leaves DGX work eligible while
TACC prepares; at handoff, only pending DGX fallbacks are held. Running/completed
DGX seeds stay on DGX and are skipped on TACC. The two original running jobs
are untouched. Saved-state transfer and ownership activation are still in
progress; this is not a report that production GPUs have started.

See [Lonestar6 execution](lonestar6_execution_20261116.md) for the exact input
bindings, budget, preparation status and local transport dependency. No new
GPU research is submitted to the temporary DGX account. All six sealed
checkpoint bundles reached the local machine. Four reached Lonestar6;
the final two uploads have resumed. Keep this machine connected until
transfer and ownership handoff complete. The authenticated TACC connection
was restored using the saved MFA code this morning.

## Completed and running experiments

### Main confirmation: 11 of 20 parents complete

The fixed Qwen3-4B math confirmation uses eight unique corrections, 25 optimizer
steps and three continuation draws per parent. Target, OOD and retention are
scored on the frozen enlarged evaluation pools.

- Parents 0--10 completed, including their dependent inspection/archive jobs.
- Parent 11 (`35053_11`) is running on a full A100; elapsed time at the
  morning check was approximately 6 hours 46 minutes. Parents 12--13 remain queued.
- Six temporary-account MIG attempts, parents 14--19, failed during final
  scoring. These are operational failures, not completed scientific results.
- Same-seed owning-account fallback jobs are pending. Final review `35054`
  remains connected to the fixed cohort through the dependency registry.

| Parent index | Failed MIG job | Pending fallback array job |
| --- | --- | --- |
| 14 | 35465 | 36073 |
| 15 | 35466 | 36096 |
| 16 | 35467 | 36211 |
| 17 | 35468 | 36209 |
| 18 | 35469 | 36245 |
| 19 | 35471 | 36247 |

### Longer trajectories: 12 of 16 paired seeds complete

Each paired trajectory compares native continuation with half-LR continuation
through T=5, with frozen T=3/T=5 scoring points.

| Backbone | Complete | Running | Pending |
| --- | ---: | ---: | ---: |
| Llama-3.2-3B-Instruct | 4 | 0 | 0 |
| Qwen3-8B | 2 | 0 | 2 |
| Llama-3.1-8B-Instruct | 2 | 0 | 2 |
| Qwen3-4B | 4 | 0 | 0 |
| **Total** | **12** | **0** | **4** |

The completed trajectories passed their dependent inspection/archive jobs.
Qwen4B's final seed (`35102`) completed in 8 hours 48 minutes; its archive
and complete-cohort review `35104` also passed. The four temporary-account 8B requests returned
through their owning-account handoffs. Registered original-account fallbacks
are `36443`, `36444`, `36445`, `36446`; all were pending this morning. They remain
eligible until the verified TACC handoff. Model reviews and integrated review
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

The morning storage monitor reports 48.413 GB allocated on the primary
50.000-GB account and 44.467 GB on the second account. Primary storage is
below the hard quota but above the monitor's safety threshold. No new large
result import is authorized by those snapshots; consolidation must check
headroom first. Scientific outputs and original failure evidence remain intact.

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

1. Complete 10 confirmation parents and five paired trajectories, preserving
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
