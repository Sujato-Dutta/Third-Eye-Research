# Temporary DGX worker, November 12--17, 2026

Account `muef2320`, UID 1189, home `/dgxb_home/muef2320`.
Credentials are read privately from the `DGX_Username3` block in the local
`.env`. The two other accounts continue their existing experiments.

## Verified resource limits

Slurm reports the faculty association, with aggregate limits of two GPUs,
64 CPUs and 512 GB RAM. Available partitions include `cpu_faculty` and
`gpu_faculty`. The cluster training guide, page 14, specifies 200 GB of SSD
storage for faculty accounts and the same compute limits. Its 30-day partition
walltime does not extend the temporary authorization ending November 17.

At connection, the account had no running or pending jobs. The allowance is a
resource limit; it does not guarantee immediate allocation of two full A100s.
Request the explicitly validated GPU resource type and retain Slurm's CUDA
visibility mask. The November 15 section records the approved 20-GB MIG
alternative alongside the original full-A100 requests.

## Owner data and cleanup boundary

Preserve the owner's project data, datasets, trained checkpoints, code,
environments, reports and configuration. Only clearly identified pretrained
model-weight downloads or caches may be removed if space is needed. Inventory
must precede any cleanup; record exact paths and recoverability before changing
an allowed cache. No broad project-folder reset is authorized.

All storage scans, software checks, downloads, training and evaluation require
Slurm compute allocations. Login-node use is confined to SSH/SFTP, scheduler
commands, and small staging or metadata operations. The storage limit applies
to the account's total usage, including existing owner data.

## Initial inspection

CPU job `35440` completed a read-only whole-account storage and hardware inventory
in 5:50 on `dgxa`.
It requests one CPU and 2 GB RAM for at most twenty minutes. Its outputs are
isolated in `/dgxb_home/muef2320/third_eye_account_audit_20261112`.
No model download, research update, or cleanup is part of that job.

Measured total usage is **121,100,177,408 bytes**, leaving **78,899,822,592 bytes**
under the 200-billion-byte operating limit. No cleanup is required or performed.
The account's `quota -s` reports no enforced filesystem quota; the guide's
200-GB allocation remains the operating limit. The node exposes A100-SXM4 GPUs
with 40 GB each. This metadata query does not itself allocate a GPU.

The original account's executable environment, pinned Qwen3-4B model index,
and frozen confirmation request are readable from this account. GPU job
`35441` requests one full A100, eight CPUs and 32 GB RAM for at most thirty
minutes to verify one optimizer step on eight training-only fixtures followed
by exact adapter rollback. It evaluates no final items and retains no research
update. All runtime caches are directed into the new audit directory, and the
scheduler's CUDA mask is preserved. Its scheduling deadline is November 16 at
18:00 cluster time, inside the temporary access window.

Local transport receipts:
`runs/deployment/dgx_temporary_requests_20261112/`.
Downloaded inventory and validation receipts:
`runs/deployment/dgx_temporary_20261112/`.
The SSH helper now selects credentials by username block, so duplicate generic
`Password` keys for the second and third accounts do not override each other.
It also retries a queue read when a Windows writer still holds the new file.

Use this temporary capacity for already declared pending experiment seeds.
Preserve running/completed experiments and their frozen settings. Any work
dispatched here must finish, be verified, and be handed back within the lease;
partition walltime and automatic retries must not run beyond the account's
authorized window.

## Parallel execution released, November 12

The temporary account now has ten delegated experiments queued, with at most
two full A100 allocations concurrently. The two original accounts retain their
one-GPU limits. Existing running jobs `35053_3` and `35078` were left running.
Completed results and the scientific protocol were not changed.

| Fixed experiment | Temporary GPU job | Owning-account handoff |
| --- | --- | --- |
| Qwen3-8B trajectory, seed index 2 | 35460 | 35474 |
| Llama-3.1-8B trajectory, seed index 2 | 35462 | 35475 |
| Qwen3-8B trajectory, seed index 3 | 35463 | 35476 |
| Llama-3.1-8B trajectory, seed index 3 | 35464 | 35477 |
| Qwen3-4B confirmation, parent index 14 | 35465 | 35478 |
| Qwen3-4B confirmation, parent index 15 | 35466 | 35479 |
| Qwen3-4B confirmation, parent index 16 | 35467 | 35480 |
| Qwen3-4B confirmation, parent index 17 | 35468 | 35481 |
| Qwen3-4B confirmation, parent index 18 | 35469 | 35482 |
| Qwen3-4B confirmation, parent index 19 | 35471 | 35483 |

The corresponding unstarted original jobs were cancelled only after their
downstream dependencies had been moved to the owning-account handoffs. No
running scientific job was cancelled. Primary review `35054` now waits for
uncompleted original indices 3--13 and all six confirmation handoffs. The four
trajectory archive jobs wait for their respective handoffs; subsequent model
and integrated reviews retain their existing dependency chains.

All delegated jobs require successful GPU preflight `35441` and operational
validation `35506`. The latter completed with **eight tests passed**, both
frozen request/source bindings checked, and **24 batch scripts verified**.
GPU preflight was still pending for priority at the release check: these are
queued experiments, not a claim that all four GPUs are already allocated.

### Quotas and preserved results

Results write directly to the original canonical student-account directories.
The temporary account has scoped access to the new experiment folders and
read-only access to the pinned weights and executable environment. Default
ACLs allow the owning student to inspect and archive delegated results after
the temporary lease. Sticky directories protect existing sibling results.
The owner's original projects, datasets and model weights remain untouched.

Latest observed whole-home use was approximately **38.74 GB / 50 GB** on the
primary account, **40.65 GB / 50 GB** on the second account, and **121.10 GB /
200 GB** on the temporary account. CPU job `35449` recovered 818,572,008 bytes
by removing only an identical transport archive with a verified local backup;
extracted research artifacts remain in place.

Student storage monitors `35488` and `35489` refresh whole-home measurements
every two minutes. Temporary-account monitor `35505` scans every twenty
minutes, with a fifteen-minute measurement timeout for its larger file tree.
Quota guard `35502` checks the snapshots every minute and cancels only named
delegated jobs if the two-GB student reserve is consumed or the student
measurements become stale. Its temporary-account guard reserves fifteen GB
below a 190-GB admission ceiling. It deletes no data and does not cancel the
original running jobs. The guard's four boundary/staleness checks passed on a
Slurm CPU allocation.

### Lease and automatic return

Lease watchdog `35507` enforces **November 17, 23:59:59 IST**. It cancels a
pending delegated job once its protective runtime plus thirty minutes can no
longer fit before that cutoff. A running delegated job cannot continue past
the cutoff. Scheduler deadlines and no-requeue settings provide an additional
boundary. The SSH transport also expires at the same time.

Each handoff runs after its delegated GPU job ends. Complete results are
inspected regardless of their scientific effect. An incomplete attempt is
preserved as failure evidence, and exactly the same fixed experiment is
resubmitted on its original account. Archive/review dependencies are updated
to that fallback. Existing submission receipts prevent duplicate fallback
jobs. Completion of the entire study by November 17 is not guaranteed: actual
queue delays and observed runtimes determine how much the temporary capacity
can finish.

### Transport and monitor repairs

Overlapping local SSH dispatchers caused duplicate setup submissions before
release. All old dispatchers were stopped, duplicate held jobs and duplicate
new storage monitors were removed, and isolated queues now use an OS-level
singleton lock plus atomic response files. Earlier failure receipts remain.
No delegated training ran during those setup failures.

Compute-node accounting initially tried `localhost:7031`, where this cluster
has no accounting daemon. The operational client now retries that specific
failure with a private, mode-0600 copy of the Slurm client configuration that
points accounting to `dgx-login01`. System configuration is unchanged. CPU
diagnostic `35504` verified real accounting queries with that client; test
coverage checks that submission commands do not receive this retry.

Canonical deployment metadata and test receipts are under
`runs/deployment/parallel_20261112/`. Active transport queues are
`runs/deployment/dgx_parallel_primary_requests_20261112/`,
`runs/deployment/dgx_parallel_secondary_requests_20261112/`, and
`runs/deployment/dgx_parallel_faculty_requests_20261112/`.
Do not restart dispatchers against the retired queues.

## November 15: 20-GB MIG dispatch

Full A100s remained occupied, so the available hardware was inspected under
the authorized alternative-resource request. Both nodes expose an unused
`gpu:a100_3g.20gb:1` allocation. Compatibility job `35960` completed on one
slice in 4:03 with the original BF16 Qwen4B model, all 25 optimizer steps,
exact rollback, batch-32 greedy inference, batch-16 sampled inference and
batch-32 conditional-likelihood retention scoring. Peak live tensors reached
16.16 GB; no sealed final items were evaluated. CPU job `35961` passed 11
operational tests and verified the 24 existing batch scripts.

Jobs `35465` and `35466` started at **November 15, 10:37:40 IST** on `dgxa`
and `dgxb`. Jobs `35467`, `35468`, `35469` and `35471` are also eligible for
the validated MIG profile, under the shared two-GPU account limit. All six
retain their canonical job IDs and original owning-account handoffs. Their
scheduler walltime is now 40 hours; training steps, LR, precision, batches,
evaluation, source hashes and seeds are unchanged. The per-job resource
overrides in `parallel/resource_plan.json` supersede the original full-A100
SBATCH resource/time comments; the frozen scientific runners are unchanged.

The 8B jobs retain their original full-A100 profile. The lease watchdog now
checks the relevant preflight for each resource profile: a failed or expired
full-A100 preflight cannot cancel a MIG task that passed its own validation.
The 40-hour MIG limit is used when deciding whether pending jobs still fit
before the cutoff. Complete/partial-result handling remains unchanged.

An initial five-second MIG probe (`35959`) exposed an empty-cache lookup,
before model loading or training. The preflight and positive worker now set
`HF_HUB_CACHE` to the existing shared pinned Qwen4 cache, while other runtime
caches stay in the audit directory. No weights are copied into the owner's
home. The failed receipt is preserved; `35960` is the successful replacement.

Activation `35964` completed after correcting two scheduler-only issues: the
original five-day monitor walltime no longer fit the remaining lease, and
completed dependency IDs had aged out of Slurm's active job memory. The
matching successful validation/preflight receipts are explicitly checked;
only already satisfied dependencies are cleared. The early CPU activation
failure `35962` and its progress receipt remain preserved. Research jobs
were held during these staging operations.

Active faculty safeguards are now deadline watchdog **35965**, quota guard
**35966**, and whole-home storage monitor **35967**. Their replacement time
limits fit the remaining authorization window. The original source-account
storage monitors remain active. Model hardware and resource-validation IDs
are recorded in the delegated result provenance.

See [current progress](current_progress.md) for counts and measured memory.

## November 16: full-workload MIG failures and full-A100 recheck

All six 20-GB research attempts failed with CUDA OOM during M0 final retention
scoring. The earlier functional preflight passed but did not establish full
final-workload memory compatibility. No failed attempt is counted as complete.
Owning-account handoffs preserved partial folders under `failure_evidence`
and registered same-seed fallback jobs; see the current-progress job map.

Read-only CPU job 36281 verified all 66 sealed adapter checkpoint files against
their original commitments. Training had finished before final scoring failed.
These checkpoints remain preserved for possible scoring-only recovery; no
such recovery has been released. The failed scoring outputs are preserved.

Direct scheduler inspection shows all seven full A100s allocated (dgxa 2/2,
dgxb 5/5). Faculty association muef2320 allows two GPUs, 64 CPUs and 512 GB RAM;
the original two student associations allow one GPU, 32 CPUs and 256 GB each.
Faculty and student GPU partitions share dgxa/dgxb and the same full devices.
Faculty partition QoS priority is 100 versus student 50; this is not a reserved
pool. Full-A100 preflight 35441 remains in the correct faculty partition.

Watchdog 35965 failed on a Slurm accounting query. Replacement CPU guard
36280 passed four boundary checks and is running using live squeue queries,
with retries for transient scheduler errors. It manages only the named
delegated jobs and has not cancelled any at this check. Quota guard 35966
and faculty storage monitor 35967 remain running. No original running job
or owner project was changed. The November 17 EOD access cutoff is unchanged.
