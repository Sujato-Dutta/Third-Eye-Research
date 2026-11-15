# Third Eye: current execution progress

Last verified: **November 15, 2026, approximately 10:17 IST**.
This report covers the current confirmation and longer-trajectory round, not
the percentage completion of the entire paper. Counts come from live Slurm
accounting; storage comes from compute-node monitoring receipts.

## Completed and running experiments

### Main confirmation: 8 of 20 parents complete

The fixed Qwen3-4B math confirmation uses eight unique corrections, 25 optimizer
steps and three continuation draws per parent. It scores target, OOD and
retention on the frozen enlarged evaluation pools.

- Parents 0--7 completed successfully, including inspection and lossless archival.
- Parent 8, job `35053_8`, is running; elapsed time was about 2 hours 53 minutes.
- Parents 9--13 remain queued on the original account.
- Parents 14--19 are delegated to the temporary account and await its GPU preflight.
- Final statistical review `35054` waits for the complete fixed cohort.

### Longer trajectories: 8 of 16 paired seeds complete

Each trajectory compares the native update policy with half-LR continuation
through T=5, with the frozen T=3/T=5 scoring points.

| Backbone | Complete | Running | Pending |
| --- | ---: | ---: | ---: |
| Llama-3.2-3B-Instruct | 4 | 0 | 0 |
| Qwen3-8B | 2 | 0 | 2 |
| Llama-3.1-8B-Instruct | 2 | 0 | 2 |
| Qwen3-4B | 0 | 1 | 3 |
| **Total** | **8** | **1** | **7** |

Both completed Qwen8B and both completed Llama8B trajectories passed their
dependent inspection/archive jobs. Qwen4B index 0, job `35096`, is running;
elapsed time was about 3 hours 33 minutes. The remaining two seeds of each 8B
backbone are delegated to the temporary account. Model-level reviews require
all four fixed seeds; the integrated review requires all four backbone cohorts.

## Scientific evidence available from this round

The complete four-seed Llama3B math review reports the following equal-role
utility changes:

| Endpoint | Change, percentage points |
| --- | ---: |
| First update versus unchanged parent | -0.21 |
| Native T=3 versus unchanged parent | -7.11 |
| Native T=5 versus unchanged parent | -16.08 |
| Half-LR T=5 versus native T=5 | +6.70 |

The simultaneous interval for native T=5 versus the parent is
[-20.07, -12.29] pp; the interval for half-LR versus native T=5 is
[+3.22, +9.78] pp. Half-LR reduces the loss but does not restore utility above
holding the parent. These are exploratory results for one backbone, one math
stream and four paired seeds. They support delayed degradation under the
tested recipe; cross-model robustness remains unestablished until the other
fixed cohorts finish. No forecasting claim is required for this study.

## Why the extra account has not received GPUs

The temporary account has an allowance of **up to two GPUs, 64 CPUs and
512 GB RAM**. This is a concurrency ceiling, not dedicated hardware.

The scheduler inspection at approximately 10:17 IST showed:

| Node | Configured full A100 resources | Allocated full A100 resources | Node state |
| --- | ---: | ---: | --- |
| `dgxa` | 2 | 2 | MIXED, online |
| `dgxb` | 5 | 5 | MIXED, online |
| **Total** | **7** | **7** | **All full A100s allocated** |

These are Slurm's full-device resource counts, not a count of every physical
GPU or every MIG slice. Smaller MIG resources also appear in the inventory;
they do not satisfy the jobs' explicit full-A100 request. The faculty,
student and scholar GPU partitions share the same two nodes. The faculty
partition is up, and preemption is disabled.

The preflight job `35441` is eligible and has no dependency or user hold. It
requests one full A100 for at most 30 minutes. The account's storage guards
report no block. The current scheduler reason is:

> Nodes required for job are DOWN, DRAINED or reserved for jobs in higher priority partitions.

That is a generic scheduler reason: the node inspection shows both nodes online
and all seven full A100 resources allocated. There are no explicit Slurm
reservations listed. The verified immediate constraint is full-GPU
availability; this is not evidence of an account ban or a storage-quota failure.

Slurm currently estimates preflight starting **November 16 at 12:30 IST**.
This is a prediction and has already shifted during earlier checks; it is
not a guaranteed allocation. The ten delegated experiments automatically
become eligible after successful preflight. Thus **two research GPUs are
currently running**, while the two additional slots remain unallocated.

## Storage and safeguards

| Account role | Whole-home usage | Operating limit |
| --- | ---: | ---: |
| Main confirmation account | 41.26 GB | 50 GB |
| Longer-trajectory account | 43.01 GB | 50 GB |
| Temporary account | 121.10 GB | 200 GB |

GB denotes one billion bytes. Whole-home monitoring includes existing account
contents. The owner's project data, datasets and trained checkpoints remain
preserved. All scientific computation, validation, storage scans and archival
run in Slurm compute allocations; login-node activity is limited to transport
and scheduler/metadata operations.

Storage monitors and quota guard `35502` are running. Lease watchdog `35507`
enforces the temporary-account cutoff of **November 17, 23:59:59 IST**. Delegated
jobs that no longer fit within the remaining protective runtime are returned
to their original queues through owning-account handoffs. Partial evidence
is preserved; fallback retains the same fixed seed and scientific settings.
Completed outcomes are retained regardless of whether they support the hypothesis.

## Remaining work

1. Complete the remaining 12 confirmation parents and eight paired trajectories.
2. Run the already queued cohort inspections and statistical reviews once their
   fixed dependencies complete. No additional architecture search is planned.
3. Review the cross-model effects, continuation contrast and mitigation results
   together, with their intervals and existing controls.
4. Incorporate the validated results into the manuscript, figures and tables,
   then rebuild and check the final paper/Overleaf package.

The temporary account has approximately 62 hours of authorized access left at
this check. Finishing the entire round by November 17 is unlikely if the extra
GPUs remain unavailable until the current predicted start. The original two
accounts continue independently beyond that temporary window.

## Evidence and execution details

- Live scheduler diagnosis:
  `runs/deployment/dgx_parallel_faculty_requests_20261112/042-allocation-diagnosis.response.json`.
- Current experiment and storage snapshots:
  `runs/deployment/dgx_parallel_primary_requests_20261112/017-progress-20261115.response.json`,
  `runs/deployment/dgx_parallel_secondary_requests_20261112/020-progress-20261115.response.json`,
  `runs/deployment/dgx_parallel_faculty_requests_20261112/041-progress-20261115.response.json`.
- [Main confirmation execution](dgx_positive_regime_20261109.md).
- [Longer-trajectory execution](long_horizon_execution_20261109.md).
- [Temporary worker, job mapping and fallback](dgx_temporary_worker_20261112.md).
