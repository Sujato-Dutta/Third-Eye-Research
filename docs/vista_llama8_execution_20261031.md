# Parallel Llama3.1-8B confirmation

**November5,10:54 IST:** all eight Llama parents are RUNNING on eight distinct
Vista GPU nodes: four code tasks1114001_0--3 and four math tasks1114002_4--7.
Code element0 has elapsed5h36m; the remaining seven have elapsed4h24m. No
research parent has completed or failed. Bounded tails of all eight allocated
job logs show active correction harvesting and development evaluation beyond
model initialization, with no traceback in those tails. These partial logs are
execution evidence, not sealed final results. Analysis1114003 remains pending
the successful completion of both arrays and will run automatically.

This clears the allocation blocker reported below. Other model cohorts remain
complete:56/64 planned fresh parents finished, with the final eight now active.
Scheduler and bounded-log evidence are saved in local queue responses089/090;
the verified status snapshot is in
`runs/deployment/evidence/vista_llama8_20261105/verified_status.json`.

**November4,10:57 IST live status:** all eight research parents remain pending
priority, with dependencies cleared and no start estimate. Their current
replacement chain has no failure: CPU1113999 and GPU timing1114000 completed;
arrays1114001/1114002 await allocation, and review1114003 awaits both arrays.
Both arrays retain their original submission and eligibility timestamps.

The completed GPU timing took1h30m54s. Its source/plan-bound readiness and
authorization checksums were verified after download on November3. Two full
1024-prompt math harvests took2476.835s and2748.561s on discarded parent/updated
fixtures. With the unchanged50% margin, the complete per-parent estimates are
25.620h/code and22.586h/math. These are pricing estimates, not observed research
parent runtimes.

On November3, only the pending scheduler walltime requests were shortened to
**28h/code and26h/math**, retaining additional headroom above those estimates.
No model/data/seed/update/evaluation setting, array index, job ID or dependency
changed, and no job was cancelled or resubmitted. Readback on November4 confirms
the new limits and four concurrent elements per array. The original ledger
reservations remain conservative upper bounds until final accounting.
The operation is recorded in
`runtime_adjustment_20261101_v2/scheduling_rightsize_20261103.json` remotely and
`runs/deployment/vista_dgx_requests_20261026/086-llama-backfill-rightsize.response.json`
locally. Readiness evidence is in
`runs/deployment/evidence/vista_llama8_20261103/`.

Qwen8B and its analysis are complete; four model cohorts and56/64 planned fresh
parents are finished. The remaining research execution is the eight Llama
parents and automatic complete-model analysis, followed by manuscript synthesis.

**November2,10:56 IST:** CPU supplement validation1113999 passed. GPU timing
1114000 is still pending priority; its expected start is unavailable. Code
array1114001, math array1114002 and analysis1114003 remain pending dependencies.
No replacement job has failed and no research parent has started. This is a
scheduler wait, not a new scientific decision. Live metadata shows the regular
and development GPU partitions are heavily allocated; idle node count alone
does not imply availability at this job's priority or walltime.

## November1 runtime recovery and spending authorization

The user removed the500-SU revision ceiling **for completion of the existing
eight Llama parents**. This does not authorize extra models, seeds, recipes or
other experimental arms. The live account reported1278 SUs available at the
November1 check. Reservation bookkeeping continues; actual runtime is billed.

Original source validation1112141 passed233 tests/lint, and preparation1112142
completed. GPU check1112143 finished both task compatibility paths but stopped
at its runtime guard: code required25.62h and math77.94h under the16-prompt
extrapolation, versus32h reserved. Original arrays1112144 were automatically
cancelled with zero allocated nodes. Preserve this as timing evidence, not a
failed research result. Vista's live QOS permits **48h maximum per job**.

The runtime supplement retains the original code estimate and measures two
complete1024-prompt math harvests, on discarded parent/50-step-fixture states.
Use the slower full-pool timing in the same ten-harvest,13-checkpoint, three-draw
formula with the original50% margin. Discard every calibration correction.
Authorize collection only if the complete estimate fits32h/code and48h/math.
No scientific data, seed, optimizer, inference or scoring budget changes.

The frozen source remains`8282f1d02158`; only an explicitly hash-bound
operational supplement is added. The prior failed16-prompt report is immutable.
A scheduler-ID parsing error was corrected while retaining held CPU job1113992;
its synthetic success fixture then failed because its code timing exceeded the
fixture's declared budget. Only that synthetic timing value was corrected.
All failed descendants had zero allocated GPU nodes. Those logs and accounting
remain in`runtime_adjustment_20261101`; no scientific parent was restarted.

The corrected supplement's three pure unit tests passed locally and are repeated
in the allocated CPU validation before calibration. **Accepted replacement
chain, November1,17:32 IST:**

| Stage | Job | Successful predecessor |
| --- | --- | --- |
| Runtime supplement/input checks |1113999| Scheduler allocation |
| Full-pool discarded GPU timing |1114000|1113999|
| Four code parents,0--3%4,32h each |1114001|1114000|
| Four math parents,4--7%4,48h each |1114002|1114000|
| Complete eight-parent analysis |1114003|1114001 and1114002|

Maximum new reservation is326.440 SUs:320 for parents,6 for calibration,
0.110 for CPU checks and0.330 for analysis. Actual revision charges before this
replacement are218.629444 SUs, including failed checks. Charges plus new
reservations are545.069444 SUs; the earlier500 ceiling is superseded only for
this Llama completion. Original32-parent, SmolLM and DGX scientific evidence
remain unchanged. Qwen continues independently on DGX.

Operation: `runtime_adjustment_20261101_v2` in the existing Vista campaign.
Local receipt: `runs/deployment/vista_llama8_20261101_v2/submissions.json`.
At17:34 IST,1113999 completed successfully in11 seconds with all three
supplement tests passing and frozen-input/source checks intact. Calibration
1114000 is pending priority; both four-parent arrays and analysis are pending
their declared dependencies. No research parent has started on Vista yet.
The October31 sections below document the original launch and historical cap.

October31,2026. Relocate the eight unstarted Llama3.1-8B parents to Vista,
while Qwen3-8B continues on DGX. This changes placement and scheduling only.
It does not add seeds, candidates, continuations, training steps or evaluation
items. No forecasting work is included.

## Budget

The last reconciled revision consumption is218.106944 SUs, leaving281.893056
under the authorized500-additional-SU cap. Read the live shared ledger under
an exclusive lock before reserving this launch; abort if other reservations
have consumed the required headroom. The account balance is a separate number.

| Stage | Partition | Maximum reservation |
| --- | --- | ---: |
| Source extraction, verification, full tests and lint | gg,20min |0.110SU|
| Pinned model/data preparation | gg,1h |0.330SU|
| Both-task real GPU and full-runtime check | gh-dev,1.5h |1.500SU|
| Eight complete parents, up to eight in parallel | gh,32h each |256.000SU|
| Complete-cohort verification/statistics | gg,1h |0.330SU|
| **Total new reservation** | | **258.270SU** |

At the last reconciled balance this makes476.376944 SUs charged plus reserved,
leaving23.623056 unreserved. Walltime reservations are protective ceilings,
not runtime estimates or actual charges. Failures and minimum billing count.
No automatic retry may consume additional SUs without a fresh ledger check.

## Frozen experiment

Use the same pinned public weight-equivalent Llama3.1-8B source, official shard
checksums and pinned metadata recorded in the original scale declaration.
Keep four code seeds2026110400--2026110403 and four math seeds
2026110300--2026110303. Preserve K=3, unique size-two batches, eight correction
attempts,50 AdamW steps, LoRA rank16, LR1e-4, bf16, three continuations per
candidate, the terminal identity rule and the frozen11Oct2026 rendering date.
All13 checkpoints per parent receive the original128/128/256 independent final
target/OOD/retention evaluation. Selection remains development-only.

The scale declaration's original DGX scheduling fields are historical;
`execution_placement` records the actual Vista placement in each new plan and
report. Scientific settings and the two-model97.5% primary interval procedure
remain fixed. Evaluate each parent's baseline and all descendants on the same
Vista runtime. Cross-model differences also involve hardware/runtime and cannot
isolate a causal size or family effect.

## Submission and duplicate prevention

All source checks, model downloads, tokenization, tests, training, inference,
statistics and large hashes run in Slurm allocations. Login activity is limited
to SFTP staging, metadata and scheduler/reservation bookkeeping. Reuse the
existing isolated verifier configuration and independent project environment.

Submit source validation, model preparation, both-task GPU verification, the
eight-way collection array and final review as an explicit afterok chain.
The GPU verification measures the complete protocol; collection does not start
if its conservative runtime estimate exceeds the32h per-parent reservation.
Do not reduce any scientific budget to force a timing check to pass.

After Vista accepts the complete chain, cancel only the unstarted DGX Llama
preparation/preflight/array/analysis and old cross-model finalizer:
29454,29455,29456,29457,29458. Preserve Qwen29452 and its analysis29453.
Verify that the targeted DGX jobs have not allocated nodes before cancellation.
Never run duplicate Llama parents across the two clusters. Retain their original
submission receipts and cancellation evidence. Verify the two complete reports
together after both sites finish.

## Submitted execution

**October31,20:17 IST:** Vista accepted the complete dependency chain. Its CPU
validation is pending scheduler priority; all successors are pending their
afterok dependencies. Array concurrency is verified as eight and each parent
has a32h ceiling. Queueing is confirmed; GPU execution has not begun.

| Stage | Vista job | Requires |
| --- | --- | --- |
| Full source validation |1112141| Scheduler allocation |
| Pinned model/data preparation |1112142|1112141|
| Both-task GPU verification/timing |1112143|1112142|
| Eight complete parents,0--7%8 |1112144|1112143|
| Complete-model analysis |1112145|1112144|

The immutable relocation source is`8282f1d02158`, full hash
`8282f1d02158f3489c2791d1c2f492040e12eff44c197cf424b61e7a499a0ef8`.
Its full suite already passed233 tests and lint in DGX CPU allocation29795;
Vista repeats validation on its own environment before model work. The earlier
DGX test-launch29794 failed because its fixture parent directory was missing;
its log is preserved. The corrected allocated launcher passed without changing
the frozen experiment. No Vista SUs were spent on these DGX checks.

Campaign: `/scratch/11617/sujato_ts/third_eye_research/runs/llama8_vista_20261031_8282f1d02158`.
Local submission receipt: `runs/deployment/vista_llama8_20261031/submissions.json`.
Verified shared-ledger reservation:258.270 SUs new,476.376944 total revision
charged plus reserved,23.623056 unreserved. Actual new charges remain unmeasured.

At14:47:16 UTC all12 scheduler entries in the old DGX Llama chain were verified
pending with zero allocated nodes, then jobs29454--29458 were cancelled.
Qwen29452 and its analysis29453 were preserved. The cancellation record is
`runs/deployment/evidence/vista_llama8_20261031/dgx_cancellation.json` locally,
and `llama_relocation_20261031.json` in the original DGX scale campaign.
