# Parallel Llama3.1-8B confirmation

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
