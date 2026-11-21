# Lonestar6 execution: remaining frozen experiments

Started November 16, 2026. All times discussed below are IST unless stated
otherwise. This is a change of execution site. It adds no seeds, recipes,
architectures or outcome-dependent selection.

## Verified access and resources

SSH authentication to `ls6.tacc.utexas.edu` succeeded for `sujato_ts`.
The initial live `taccinfo` snapshot showed project `IRI23021` with **5,940 SUs
available**, expiring April 30, 2027. This is the balance before the new jobs;
it is not an authorization to spend the whole allocation.

Live `qlimits` reports a maximum of 12 standard A100 nodes and eight GPU jobs
per user. A standard node provides three full 40-GB A100s. The queued production
requests use **two nodes and six GPUs**, packing independent frozen tasks onto
the devices. They do not distribute one model across the three GPUs.

The access check found 71 standard GPU nodes allocated, one reserved and one
drained. The small-A100 and H100 queues were also occupied. A larger pool does
not guarantee an earlier allocation; no start ETA has been established.

## Queued chain

| Job | Purpose | Resources / maximum duration |
| --- | --- | --- |
| 3586053 | Environment and pinned-model preparation | Development CPU node, 2 h |
| 3586076 | Final binding, package, source and protocol validation | Development CPU node, 30 min |
| 3586060 | Real CUDA checks for Qwen4B, Qwen8B and Llama8B | One standard A100 node, 30 min |
| 3586061 | Verify input archives and exclusive seed ownership | CPU, 10 min; release is coordinated |
| 3586062 | Production work queue, first node | Three full A100s, 32 h |
| 3586063 | Production work queue, second node | Three full A100s, 32 h |

A dependent CPU result-archive check is also being submitted; its job ID and
the latest states are recorded in the deployment queue receipt rather than
guessed here.

The final CPU validation follows preparation with `afterany`, so that a
staging-receipt failure can be checked against the actual installed assets.
CUDA validation requires this final CPU job to succeed. Production requires
both CUDA validation and the ownership job to succeed. No production GPU is
released by bypassing those dependencies.

Preparation was verified running on a compute node. It was moved from the
occupied `vm-small` CPU queue to an available development node, retaining its
job ID. The final CPU readiness pass also binds the final operational runtime
checksum before CUDA execution. Original submission output and intermediate
staging receipts remain preserved.

## Fixed work and ownership

The work queue contains ten existing tasks:

- Qwen3-8B: longer-trajectory indices 2 and 3, seeds 2026111102/2026111103.
- Llama-3.1-8B-Instruct: indices 2 and 3, seeds 2026111202/2026111203.
- Qwen3-4B confirmation: parents 14--19, seeds 2026110914--2026110919,
  recovering scoring from the eleven sealed checkpoints per parent.

The four 8B trajectories retain the original paired native/half-LR T=5
protocol and T=3/T=5 scoring. Recovery of the six confirmation parents repeats
no training or correction harvesting. All 66 saved adapter files were already
verified against their original seals on a DGX CPU allocation.

The two healthy original DGX runs continue. Their pending follow-on work stays
eligible while Lonestar6 prepares and waits for CUDA validation. At handoff,
the coordinator checks each registered DGX fallback:

1. A pending fallback is held and its hold is read back before admitting that
   seed on Lonestar6.
2. A running or completed DGX fallback remains on DGX and is skipped by the
   Lonestar6 queue. Its result is retained regardless of its scientific effect.
3. Unknown or conflicting ownership blocks release. It never causes a silent
   duplicate, changed seed or replacement based on an observed score.

The local coordinator stages the saved checkpoint exports and publishes the
ownership receipt after CUDA validation. Its status is
`runs/deployment/lonestar_20261116/handoff_coordinator_status.json`.
**Until staging and ownership transfer finish, this local transport must remain
connected.** Afterwards the released Slurm jobs execute independently. Cross-site
result import is still required before the existing DGX complete-cohort reviews.

## Budget reservation

The new migration chain has a **197-SU operating ceiling**, within 200 SUs.
No additional automatic GPU retry or expansion is authorized by this chain.

| Reservation | Maximum SUs |
| --- | ---: |
| Two production nodes, 32 h each, 3 SU/node-hour | 192.00 |
| One 30-minute standard-node CUDA check | 1.50 |
| Initial development CPU preparation, 2 h | 2.00 |
| Final development CPU readiness, 30 min | 0.50 |
| Ownership CPU check, including the 15-minute billing minimum | 0.25 |
| Result check, 30-minute `vm-small` allocation | 0.0715 |
| **Total reserved ceiling** | **196.3215** |

Actual charges depend on elapsed node time and include failed or short jobs;
the 15-minute billing minimum applies. Packing three workers onto each
standard node avoids paying for two idle GPUs for every independent task.
Allocation balance must be refreshed before any further expenditure.

## Frozen inputs and provenance

- Confirmation scientific source:
  `d5c3bc9fb65a2e416736ff42be7f7ebadf5a44943afc7277c52ef846fed0d1e0`.
- Trajectory source:
  `adb48501b796b2c20538a0e0142379b13f9f3f659338f6bff18c4200ebd72d2a`.
- Control export:
  `240d123c3b3d7309a6272aaf70ab14a7397e67b301f2e566703e17e480d923c8`.

Original request files, configurations, manifests, example files, donor data,
checkpoint metadata and adapter tensors retain their bytes. Separate path
bindings relocate inputs and record the exact original and bound identities.
The operational runtime does not patch the scientific source. Historical
zero-Vista-cost fields in the original declarations remain historical metadata;
new Lonestar6 expenditure is reported separately above.

Pinned 8B model files are downloaded from the same approved public sources and
checked against their original file hashes. Qwen4B retains its official pinned
repository/cache identity so its saved checkpoints can pass strict restoration
checks. Precision, batch sizes, optimizer budgets, seeds, chat date, verifier
and final item identities remain unchanged.

The CUDA check exercises all 25 training steps, exact adapter rollback,
batch-32 greedy generation, batch-16 sampling and the entire 1,024-item
retention scoring path. Retention answers from this compatibility check are
discarded; they cannot affect any scientific decision. This covers the final
workload geometry missed by the earlier MIG development-fixture check.

## Storage and remaining consolidation

All new dependencies, caches, models and working outputs are under
`/scratch/11617/sujato_ts/third_eye_ls6_20261116`. The existing Stockyard usage
was 941.5/1,024 GB; no existing project files were deleted. New environments
are not installed on `/work` or the login filesystem.

The temporary DGX account is used only to export our saved scientific inputs;
no replacement GPU work is submitted there. Its owner data remains untouched,
and its November 17 EOD access cutoff remains binding. Exported base-model
weights are not copied from that account: pinned models are prepared directly
on TACC compute nodes.

Completed Lonestar6 results require checksum-verified import or consolidated
review with the DGX results. DGX storage admission must be checked before
importing large archives. The complete-cohort reviews and manuscript updates
remain downstream of that consolidation; queue submission alone is not a
completed experiment or a supported cross-model claim.

## Evidence and policy

- Local scheduler and transport receipts:
  `runs/deployment/lonestar_requests_20261116/`.
- Local export and handoff receipts:
  `runs/deployment/lonestar_20261116/`.
- [TACC Lonestar6 guide](https://docs.tacc.utexas.edu/hpc/lonestar6/).
- [Lonestar6 charging policy](https://docs.tacc.utexas.edu/include/lonestar6-jobaccounting/).

All installation, model downloads, hashing, tests, evaluation and training run
in Slurm compute allocations. Login nodes handle authentication, SFTP,
scheduler submission/queries and small metadata reads. No specific compute
nodes are requested, and no GPU mask outside the allocated node is used.

## November 17 recovery update

Morning checks confirmed that CPU jobs `3586053` and `3586076` failed because
pytest's temporary parent directory did not exist. The dependent GPU jobs
were cancelled before execution; no scientific outcomes were produced by
those requests. The operational runtime now creates that directory before
the frozen tests. The source releases, configurations and seed grid remain
unchanged.

The saved MFA code restored the TACC connection. Replacement requests are:

| Purpose | Job | Last checked state |
| --- | --- | --- |
| Corrected CPU setup | 3588141 | Running on development compute node |
| Three-model CUDA preflight | 3588142 | Pending on CPU success |
| Input/ownership verification | 3588143 | Held pending safe handoff |
| Production node 0, three A100s | 3588144 | Pending on both validations |
| Production node 1, three A100s | 3588145 | Pending on both validations |
| Exported-result verification | 3588146 | Pending after both production requests |

The first frozen CPU suite passed (5 tests); the second was still executing
at the last log check. Full CPU validation and CUDA validation are not yet
complete. The resubmission script records each actual job ID immediately
and reuses it on a repeated invocation, avoiding duplicate submissions. The
handoff controller now reads current IDs from `submission_summary.json`
instead of hard-coding the cancelled ownership job.

All six saved-state archives are local. Four are acknowledged on Lonestar6;
the remaining two uploads have resumed. Neither transfer completion nor GPU
execution is claimed. Original DGX fallbacks remain eligible until a passed
CUDA preflight and scheduler-only ownership decision. The replacement
reservation, including prior CPU failures, remains below 200 additional SUs.

DGX has meanwhile completed 11/20 confirmation parents and 12/16 longer
trajectory seeds. Both four-seed smaller-backbone trajectory cohorts are
complete; the four remaining 8B seeds were pending at the morning check.
See `docs/current_progress.md` for the new Qwen4B results and storage limits.

## November 18 execution check

The authenticated SSH session and upload had ended before handoff. Four of six
sealed archives were acknowledged; parent 18 had only a partial upload and
parent 19 had not started. A fresh MFA code restored access. The upload helper
now resumes verified partial boundaries, records bytes transferred, publishes
responses atomically and bounds SFTP channel waits. Full receiving-side archive
and member hashes remain mandatory on compute nodes. The coordinator has
restarted and uses only the original two DGX accounts; temporary faculty access
expired November 17 EOD and is not used again.

Both preparation `3588141` (12:55) and CUDA validation `3588142` (10:49)
completed successfully. CPU test invocations passed 5 and 15 tests. Production
`3588144`/`3588145` remains dependency-blocked by held ownership verification
`3588143`, which awaits the remaining archives and exclusive seed handoff.
The fifth archive was approximately 317/502 MB uploaded at 10:32 IST; the
sixth remained pending. No production GPU execution is claimed.

DGX confirmation reached 12/20 complete. Subsequent original attempts refused
research execution because usage exceeded 48/50 GB, leaving less than the fixed
2-GB scratch reserve. Their logs are preserved. CPU-only, checked maintenance
recovered 945,438,720 bytes through lossless inode sharing and regenerable
download/build-cache retirement, reducing home usage to 47,401,684,992 bytes.
Model weights, environments, unrelated project data and all scientific results
remain preserved. Fixed-seed retries 12/13 are `38039`/`38040`, dependent on the
successful storage check and each other, respectively. Six held migration
fallbacks are now `38023`--`38028`; their registry and review dependencies were
updated while preserving earlier failed job IDs in history.

Longer trajectories remain 12/16 complete; Llama8B index 3 (`36443`) is running
on DGX and must be excluded from TACC execution at handoff. The other three
8B fallbacks remain eligible until ownership assignment. No new scientific
cohort-level result is claimed from incomplete collections.

## November 18 afternoon: durable transport and sealed-state recovery

The earlier tool-session SSH transport ended before completing archive 18.
The separate handoff coordinator survived, waiting for its response. A new
MFA code restored authentication through a separate hidden background SSH
process. Archive 18 finished; archive 19 was approximately 112/502 MB at
16:33 IST. Five of six staging acknowledgements now exist. The local machine
must stay awake and connected until the handoff receipt is committed.

DGX Llama8B index 3 (`36443`) failed after 16:00:50 with a 1.91-GiB CUDA
allocation failure during final scoring. All ten trajectory states had been
sealed beforehand. CPU audit `38276` passed all state/weight/donor identity
checks, preserving original request/source hashes. DGX GPU **`38278`** performs
scoring-only recovery in fresh per-checkpoint processes, retaining the frozen
batch geometry, precision, outputs, verifier and checkpoint restoration checks.
It uses expandable allocator segments and checks available allocated-GPU memory
before loading. No harvesting or optimizer update is repeated. Failed partial
scoring remains archived beside the run. Original archive `35094` now depends
on `38278`.

The running Llama8B index 2 is `36444`, approximately 2:36 elapsed at 16:07.
CPU follow-up `38277`, after any exit of that worker, inspects a completed
result or admits scoring recovery only if all ten trained states are sealed.
The two initially proposed long CPU controllers were cancelled before running;
short CPU admissions avoid occupying a CPU while a GPU is queued.

Pending, audited scoring recovery is explicitly retained on DGX at handoff.
The ownership validator now accepts `PENDING_RECOVERY` only with the complete
same-seed, original-source/request proof and `training_repeated=false`; it is
never treated as scientific completion. This changes operational ownership
routing, not the frozen experiments. Because the runtime hash changes, repeat
CPU setup and CUDA validation are queued for submission after the active
upload. The coordinator reads the resulting current registration and waits
for both before releasing ownership verification. Existing production job IDs
are retained, with updated dependencies. Failed validation still prevents
production.

Conservative completed-job accounting plus full remaining walltime reservations
gives **196.3215 SUs**, below the operating ceiling 197 and budget limit 200.
`reroute_budget_20261118.json` records the allocation. No new scientific seed,
architecture sweep, forecaster or outcome-dependent sample selection is added.

## November 18, 17:14: staging and scheduler handoff complete

All six archive uploads are acknowledged, the ownership manifest is uploaded,
and ownership verification `3588143` is released with dependency
`afterok:3592177`. Routing CPU setup **`3592177`** is running; replacement
CUDA preflight **`3592178`** waits for it. Production `3588144`/`3588145`
explicitly depends on `3592178` and `3588143`. No production GPU has started.
The manifest assigns eight tasks to TACC (two Qwen8B trajectories and six
confirmation scoring recoveries), retaining the running Llama8B trajectory
and audited scoring recovery on DGX.

The local coordinator encountered an earlier successful preflight receipt
while the replacement validation was running and committed ownership early.
No scheduler validation dependency was bypassed: CPU verification and GPU
production remain guarded by their current job IDs. This exact-ID readiness
check is now also enforced in the local coordinator for future handoffs.
The current remote dependency chain can run without the local machine;
fresh validation failures still stop production. Result return and complete
cohort consolidation remain downstream work.
