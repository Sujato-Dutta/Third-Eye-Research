# Recovery V4: reuse retained partial M1 weights

Operational correction, October8; no A2 scientific change. Exact reconstruction
of a pruned earlier M1 failed during E1 validation. All three timed-out final
states retain their third candidate's actual M1, with intact weight hashes and
complete fifty-step training logs. Reuse those saved weights instead of requiring
a newly retrained adapter to equal them bit for bit. Keep the old V3 release,
original artifacts and queued-job receipts intact.

Before replacement recovery submission, require CPU validation and a CUDA load
check on all three retained partial M1s. Require exact protocol, tensor and file
hashes, fifty consecutive finite completed steps, zero newly executed optimizer
steps during reuse, and exact rollback to each parent. Copy the producer training
log; record its hash. Keep the logical candidate budget at fifty steps already
executed. Report checkpoint-load time separately; the partial branch's original
producer timer is unavailable, rather than fabricated. Actual Slurm allocations
remain the GPU cost evidence.

The two complete branches, root correction pool and original accepted history
remain cached. The third branch's unfinished verified continuation runs through
the unchanged frozen A2 implementation. Candidate size2, K=3, correction attempts8,
steps50, probe10, learning rate, LoRA rank, manifests, utility and terminal rule
remain unchanged. Never edit the original TIMEOUT outputs or V3 implementation.

Stage V4 in `runs/a2/recovery_20261008_v4`. Do not launch a duplicate of an already
running or completed V3 job. If all three V3 GPU jobs remain pending after V4
validation, replace their queue entries with the tested V4 runtime and replace
the dependent CPU barrier. Retain cancellation/accounting evidence. No online
or scaling release follows the final full-data forecasting review.
