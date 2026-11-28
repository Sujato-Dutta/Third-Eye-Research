# Third Eye: current execution progress

## November 27, 18:40 IST

Lonestar6 has scored 53 of 94 checkpoint evaluations. The two seeds
without matched doses finished both baseline evaluations. Seven
matched seeds have scored seven of nine checkpoints each on nine A100s
across jobs **3615341–3615343**. The last three seeds, in job **3615344**,
are waiting on GPU priority without a scheduled start. The final review
**3615345** waits for all scoring jobs.

## November 27, 09:10 IST

All 12 Lonestar6 harvests completed successfully. CPU freeze **3615340**
sealed the batches at 22:27 IST on November 26. Ten seeds supplied all
three matched overlap levels. Seed 2026112506 had no budget-matched
medium batch, and seed 2026112507 had insufficient distinct matched
batches. Both remain in the intended cohort under the frozen scarcity rule.
The four scoring jobs **3615341–3615344** are waiting on scheduler priority.
The review **3615345** depends on them. Scoring has yet to begin.

## November 26, 20:38 IST

Lonestar6 preflight **3615346** completed successfully. All four harvest
jobs (**3615336–3615339**) are running on 12 A100s. Each seed has passed
model loading and reached verified-correction generation. The first
group started at 20:00 IST and the other groups at 20:13–20:14 IST.
The CPU freeze and all scoring and review jobs are queued behind the
harvests. No fresh seed has completed harvesting yet.

## November 26, DGX backup and storage

At 12:30 IST both workstation transfers were paused at the user's request.
DGX 2 automated cleanup was stopped before it submitted any jobs. Verified
files, manifests and partial downloads are available for a later resume.

DGX CPU inventories **43645** and **43646** found 50.19 GB on account 1
and 45.49 GB on account 2. Account 1 had 27.37 GB of Third Eye runs,
8.07 GB of model caches, and a 6.59 GB Python environment. Account 2
had 38.90 GB in downloaded model directories and 6.56 GB of research runs.

The [local backup](../runs/deployment/dgx_backup_20261126/README.md)
preserves Third Eye research artifacts with per-file checksums. CPU jobs
**43650** and **43651** published manifests for 27.55 GB and 44.98 GB
of unique content, respectively.
Account 2 transfer resumed after retrying a delayed manifest read.
Account 1 cleanup **43668** removed three pinned Qwen3-4B download shards
and verified that they were absent, reclaiming 8.04 GB. Cleanup on account 2 requires
matching SHA-256 checks for every archived shard of the selected model
and an idle account. Historical research runs are retained on both accounts.

## November 26, 08:26 IST

The overnight harvest jobs **3611193–3611196** exited before model
loading. The GPU launcher passed a literal `+` to the worker's argument
parser. No seed produced a harvest or started scoring. The old blocked
downstream jobs were cancelled and the original logs were retained.

The launcher was corrected in a new release with the same experiment
code, seeds and settings. A checksum mismatch in the newly added
preflight test was caught on the CPU and corrected before GPU release.
Preflight **3615346** passed the exact worker argument check for both
harvest and scoring across three lanes. Full input validation was running
at the last check.
Harvest jobs **3615336–3615339** depend on that preflight. Freeze
**3615340**, scoring **3615341–3615344**, and review **3615345** remain
queued in order.

## November 25, 11:21 IST

CPU validation **3611192** completed successfully. All four harvest jobs
are waiting on GPU queue priority, with tentative starts of 17:10, 20:40,
20:57 and 21:00 IST. No fresh harvest or scoring outputs have been
produced. The CPU freeze, scoring jobs and final review remain queued
behind their dependencies.

## November 24, 23:14 IST

The 12-seed prospective Qwen3-4B overlap experiment is queued on Lonestar6.
CPU validation **3611192** wrote a passing receipt with all six release
checksums verified and arithmetic profiles for 1,001 of 1,024 prompts.
The historical precision simulation estimated a 1.02 pp half-width at
12 seeds. Harvest jobs **3611193–3611196** then produce and retain the
training pools and initial adapters across up to 12 A100s. CPU job
**3611197** reproduces and seals the matched batches without using final
outcomes. Scoring jobs **3611198–3611201** wait for that seal and run the
paired updates and pooled control. CPU review **3611202** waits for all
scoring jobs.

The GPU requests total at most 138 SUs. Allocation `IRI23021` reported
5,126 available SUs at submission. All new artifacts are on scratch.
The [prospective design](../final_revision_plan.md#november-24-prospective-overlap-experiment)
specifies the overlap intervention, scarcity handling and primary contrast.

## November 24, 23:04 IST

All 20 archived Qwen3-4B replays and the two fresh timing seeds are complete.
The two final replay jobs (**3609764**, **3609802**) took 19 min 41 s and
17 min 13 s. CPU review **3609805** completed at 21:59 IST.
All 20 leave-one-seed-out folds chose a redundancy weight of zero from
the tested grid (0, 0.25, 0.5, 1). The penalty changed no selections, so
observed utility and retention differences from the single-checkpoint
probe were both exactly 0.00 pp. Each seed had six to nine first-step
fixes among its 128 target development items. The replay therefore gives
no empirical case for expanding this selector to fresh seeds. It used
the same archived candidates for both selectors. The causal effect of
deliberately varying skill overlap remains an open question.

## November 24, 21:16 IST

Fresh timing job **3607551** completed both seeds in 2 h 52 min 52 s
with two A100s running in parallel. Per seed, the two harvests took
122–125 minutes combined, one full 2,843-item checkpoint evaluation
took 41–43 minutes, and each training step sequence of 25 optimizer
steps took 81–86 seconds. Both seeds had 20 eligible corrections at
the initial checkpoint, with 22 and 11 after the first update. Each
had enough for one eight-example continuation. Matched-dose feasibility
and accuracy were outside the timing pilot.

Eight archived replays remain complete. The two six-seed jobs
**3609764** and **3609802** are pending with tentative 22:39 IST starts.
The pooled review **3609805** is queued behind both.

## November 24, 10:31 IST

All 12 remaining archives reached Lonestar6 scratch by 10:30 IST.
Both checksum-validation jobs (**3609763**, **3609801**) passed. Replay jobs
**3609764** and **3609802** cover six seeds each and are queued independently.
The first eight replays are complete. Slurm rejected the initial CPU review
submission, which included the historical replay job as a dependency.
Accounting confirmed that replay had completed successfully, and Slurm
accepted CPU review **3609805** with dependencies on the two pending replays.
Fresh timing job **3607551** remains pending on scheduler priority.

## November 24, 09:30 IST

Lonestar6 job **3607550** completed all eight archived Qwen3-4B probes
in 33 min 55 s on three A100s. The fresh two-seed timing job **3607551**
is still pending on scheduler priority.

The workstation-to-Lonestar6 transfer stopped after one archive and part
of the second. The new SSH session authenticated and the second archive
resumed at byte 9,895,936. The relay is running again. It will submit the
remaining replay jobs after each group of six archives lands and set up
the 20-seed review to run once all replay jobs succeed. The eight completed
seeds remain on scratch and will be reused.

## November 23, 22:42 IST

This round uses Lonestar6 for all scientific computation. CPU validation
job **3607549** passed in 4 min 24 s. Eight archived
Qwen3-4B probes (**3607550**) and two fresh timing seeds (**3607551**) were
released by validation and were waiting on GPU queue priority. The fresh seeds time
two full correction harvests, their updates when corrections are
available, and one full scoring pass per seed. They retain timings and
harvest counts.

All files for the other 12 archived parents reached the workstation
from DGX storage. The first archive reached Lonestar6 scratch, and the
second upload was in progress. The relay submits a checksum validation and one
three-GPU replay job for each six-parent group. A CPU analysis follows
the three replay jobs. It compares the paired-checkpoint selector with
the same two-step probe on M1 alone, using identical archived candidate
batches and disjoint development and final scoring items. Each of the
20 held-out scores uses a penalty weight chosen on the other seeds.
This is retrospective development evidence. The fresh confirmation
cohort has yet to be sized and submitted.

## November 23, 22:22 IST

Lonestar6 job **3606581 completed successfully** at 11:56 IST in 31 min
40 s. The CUDA probe verified exact rollback on an A100. Both archived
Qwen3-4B timing pilots ran concurrently and completed, including all
matched updates and pooled controls. Seeds 2026110913 and 2026110900 took
24 min 54 s and 26 min 28 s, respectively.

High overlap gave the best native increment on seed 2026110913 (0.00 pp
against −1.39 pp for low and medium). Medium overlap led on seed 2026110900
(+6.94 pp against +4.17 pp for low and +5.56 pp for high). The paired score
ranked low and medium first, respectively. Each pilot used 24 development
items per role in its scoring split and found one first-step fix in the
probe split. These observations are exploratory and the runtime estimates
apply to archived inputs and the compact development suites. A fresh
cohort requires pricing of new harvesting and full scoring. At this
snapshot the Lonestar6 queue is empty.

Verified reports are retained locally under
[revision_20261122](../runs/deployment/revision_20261122/).

## November 23, 11:11 IST

All remaining revision work moved to Lonestar6. DGX jobs 42795 and 42800
were cancelled before starting. The archived input for seed 2026110900
was transferred without base weights or final scores. Lonestar6 CPU job
3606571 verified the transfer checksums and all three overlap doses. The
pilots retain seeds 2026110900 and 2026110913.

The original Lonestar6 probe 3603934 expected a GPU-mask variable that the
full-node queue did not provide. Probe 3606547 discovered the three A100s and
then stopped at an import error caused by mixing two frozen source releases.
Both stopped before training. The blocked original timing job 3603948 was
cancelled. CPU job 3606576 verified the exact worker import order against a
single release, byte-identical core training and evaluation files, and
checksum-valid extraction of both archived adapters for each pilot seed.

Lonestar6 job **3606581** is pending for resources on the A100 development
queue, with a two-hour limit and no confirmed start time. The job runs the
CUDA probe and then launches both timing pilots on separate A100s within the
same allocation. A probe failure prevents pilot launch. The fresh-seed
cohort still depends on successful GPU validation and observed pilot cost.

## November 22, 11:38 IST

The paired-probe batch selector is implemented and the
[final revision plan](../final_revision_plan.md) includes its objective,
baseline comparisons and release sequence. The primary DGX account measured
50.54 GB against its 50 GB operating limit. It supplied a read-only CPU
export using node-local scratch. The second DGX account measured 45.49 GB
and passed the 2 GB reserve check. Lonestar6 reported 5,242 available SUs
at the start of this round. New files there are on scratch. Scientific
execution uses Slurm compute allocations.

The merged CPU review verified all 20 saved parents. On GSM8K, 85.2% of
the direct arm's fixes were already correct after the first update.
The corresponding overlap was 52.7% on MATH-500 and 47.8% on MMLU.
Strict GSM8K extraction failures fell from 8.87% to 3.69%. Held-out
retrospective checkpoint selection improved on always carrying by +0.03 pp
(95% CI [−0.20, +0.28]). This redirects the revision toward choosing the
next corrections. See the [saved audit](../runs/deployment/revision_20261122/saved_audit_review.json).

The CPU feasibility check produced matched low, medium and high doses in
56 of 60 archived draws with shared examples across batches and eight
unique examples within each. The earlier disjoint-dose variant was scarce.
The first CPU attempt encountered a token-label counting error and was
corrected and rerun. Lonestar6's first CPU audit request also needed the
queue's default memory specification. Both corrected workflows and the
per-site pilot input validations passed.

| Site | CPU admission | Pilot input validation | CUDA probe | Development pilot |
| --- | --- | --- | --- | --- |
| Lonestar6 | 3603913 passed | 3603946 passed | 3603934 pending priority | 3603948 waits for probe |
| DGX second account | 42794 passed | 42799 passed | 42795 pending priority | 42800 waits for probe |

Slurm estimated probe starts at 13:30 IST on November 22 for Lonestar6 and
19:52 IST on November 23 for DGX at this snapshot. Both can move. The probes
request 45 minutes and the timing pilots request 90 minutes. Fresh-seed
arrays await observed runtimes and successful GPU validation.

## November 22, 10:24 IST

The last confirmation parent (`3597813`) and the 20-parent review (`3597819`)
both completed successfully on November 21 IST. The GPU job took 11:35:04.
The confirmation includes all 20 Qwen3-4B math parents and three continuation
draws per parent, with no terminal draws. The four-model trajectory cohort
was already complete. Allocation `IRI23021` shows **5,242 available SUs**.

The first update improved mean utility by **+2.22 pp** (simultaneous 95% CI
[+1.05, +3.39]). The same continuation batch produced **2.03 pp less
incremental gain** when applied after that first update than when applied
directly to the parent ([-3.33, -0.77]). The sequential second update added
**+0.44 pp** ([-0.25, +1.14]), and its final advantage over the direct update
was **+0.19 pp** ([-0.35, +0.80]). Half-rate continuation changed utility by
**+0.02 pp** ([-0.29, +0.34]). The confirmation supports reduced returns to
the next update with the donor batch held fixed. A decline after the second
update was inconclusive in this positive-gain regime.

The verified review is saved at
`runs/deployment/lonestar_20261116/confirmation_review_20261122.json`.

## November 21, 08:52 IST

The last confirmation parent, Slurm job `3597813`, remains pending on GPU
priority. Slurm estimates a start at **19:43 IST on November 21**. At the
observed 14–18 hour runtime of comparable parents, that would put completion
around **10:00–14:00 IST on November 22**. Start estimates can move. The
20-parent review `3597819` remains queued after successful completion.
Allocation `IRI23021` shows **5,475 available SUs**.

The four-model trajectory review `3597817` passed. At update five, the
native-policy utility changes versus holding were −16.08 pp for Llama-3.2-3B
(simultaneous 95% CI [−20.07, −12.29]), −14.77 pp for Llama-3.1-8B
([−21.11, −10.07]), +1.75 pp for Qwen3-4B ([−0.24, +3.73]) and −3.64 pp
for Qwen3-8B ([−8.51, +2.61]). These are four paired seeds per backbone on
one math stream. The integrated review is saved locally at
`runs/deployment/lonestar_20261116/integrated_cross_cluster_review_20261121.json`.

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
