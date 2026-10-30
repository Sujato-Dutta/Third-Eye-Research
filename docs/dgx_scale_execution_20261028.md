# Two-model8B scale extension

**October29 evening status:** control analysis28974 passed. Lossless storage
rotation29449 reduced home usage from40,174,997,504 to21,710,651,392 bytes.
Qwen preparation29450 and real both-task GPU preflight29451 completed;
Qwen parent29452_0 is running and has finished label collection, with final
scoring underway. Llama's preparation and GPU array remain dependent on the
complete Qwen block. Parent reservations are protective ceilings, not measured
completion estimates; no full8B cohort result is available yet.

October28,2026. Execute on DGX after complete paired-control analysis28974.
Use the existing one-A100,32-CPU/256GB association limits,50GB home policy,
and compute-node-only execution. Vista's500-additional-SU cap is unchanged;
this extension consumes no Vista SUs. No forecasting development is included.

## Submitted jobs

Source release `8777e39cac29` passed all230 tests and lint in allocated CPU
job29448. Full source hash:
`8777e39cac291e8cdcee09d41e7a1b54d9d4ef1ccec78caf4429f71ad9ab7fde`.
Campaign: `~/third_eye_dgx_20261026/runs/scale_8777e39cac29`.

| Stage | Job | Dependency |
| --- | --- | --- |
| Full CPU validation |29448| Completed successfully |
| Lossless storage handoff |29449|29448 and current control analysis28974 |
| Qwen8B download/preparation |29450|29449 |
| Qwen8B real GPU compatibility/timing |29451|29450 |
| Qwen8B eight parents |29452, array0-7%1|29451 |
| Qwen8B complete-model analysis |29453|29452 |
| Llama8B cache swap/download/preparation |29454|29453 |
| Llama8B real GPU compatibility/timing |29455|29454 |
| Llama8B eight parents |29456, array0-7%1|29455 |
| Llama8B complete-model analysis |29457|29456 |
| Two-model completion verification |29458|29457 |

At submission the current control array was still running. Both8B arrays
were pending their explicit dependencies; the real8B GPU checks have not yet
run. Submission and CPU test success are not scientific model results.

## Frozen scope

- Qwen3-8B: `Qwen/Qwen3-8B`, revision
  `b968826d9c46dd6066d109eabc6255188de91218`.
- Llama3.1-8B-Instruct: official revision
  `0e9e39f249a16976918f6564b8830bc894c89659`, public weight-equivalent source
  `NousResearch/Meta-Llama-3.1-8B-Instruct` at
  `d10aef7999a2b5ba950ab3974312feeedbfe0b77`. Every weight shard matches the
  official LFS SHA256. Metadata is pinned to the public source and recorded
  explicitly; its tokenizer configuration is not claimed byte-identical to
  the official tokenizer configuration.
- Eight fresh parents per model: four code and four math. Qwen code seeds
  `2026110200..2026110203`, math `2026110100..2026110103`; Llama code
  `2026110400..2026110403`, math `2026110300..2026110303`.
- Same original development/final suites, K=3 distinct size-two batches,
  eight correction attempts, seed stride16,50 AdamW steps, LR1e-4,
  attention LoRA rank16/alpha32/dropout0, bf16 without quantization,
  microbatch1/accumulation8, sequence1024, greedy512-token evaluation,
  original generation batches, three continuations per candidate, and the
  terminal identity rule. Freeze the rendering date to11Oct2026.
- Final scoring remains128 target,128 OOD and256 retention items per
  checkpoint. Development-only immediate selection, uniform ties, hold and
  replay retain the existing definitions. No outcome-driven seed replacement.

This adds16 fresh parents,48 candidates,144 continuation draws and208 scored
checkpoints. Together with the original32 and SmolLM16, the intended study has
64 parents across five models. The8B blocks are balanced **scale sensitivity**
checks with four seeds/task, not equal-precision replacements for the main
eight-seed/task confirmations. Family, pretraining and instruction tuning
remain confounded with size; no causal scaling law is claimed.

Each model's declared primary is selected H2 minus selected H1 utility. Report
joint parent/shared-item bootstrap intervals with5000 draws, continuation draws
nested within parents, exact item-identity sharing, role decomposition and
selected/uniform endpoints. Use97.5% intervals for the two model-specific
primaries to control their joint95% coverage; other endpoints are exploratory.
Do not rewrite the original confirmation's primary tests or pool it with these
follow-ups as if every parent were identically sampled.

## Scheduled sequence

Full matching CPU tests and source validation precede execution. After both
that validation and28974 succeed, compact completed control adapters and evict
only their verified re-downloadable base-weight files. Then:

1. CPU download, exact-file verification and configuration/manifest preparation
   for Qwen8B.
2. Real allocated GPU compatibility check on both tasks:50-step updates,
   rollback, checkpoint reload, code/math verification, scoring timing and
   a discarded16-training-prompt correction harvest.
3. Eight parent jobs, serial array concurrency1. Each collects and scores its
   complete K=3/three-draw state before lossless adapter compaction.
4. Complete-model CPU verification and statistics.
5. Replace only the completed Qwen8B pretrained weight cache with Llama8B;
   repeat preparation, compatibility, eight serial parents and analysis.
6. Verify both scale reports and publish the completion receipt automatically.

GPU preflights reserve six hours; parent reservations use the seven-day site
maximum as a protective ceiling, **not a runtime estimate**. Compatibility
prices all13 development and13 final evaluations, ten potential harvests,
training/feature overhead and50% margin. A predicted overrun stops collection;
no scientific budget is reduced to fit. Actual timing is needed for a useful
completion estimate. All GPU stages depend on prior GPU-stage completion,
so the account never requests simultaneous GPU work across the two models.

The empirical CLI authorization binds the source, preflight, model, seed,
configuration and development manifest. It permits only one-generation label
collection under the frozen random commitment path, with item capture and
the original continuation offsets. It does not claim a forecasting gate passed
and does not enable learned policies or arbitrary8B scaling.

## Storage and artifact preservation

Allocated discovery29445 measured37,686,738,944 bytes used in the whole home.
Both8B weight sets cannot be retained together within50GB. Rotate them after
their dependency completes; keep model-source checksums and metadata for
exact re-download. Never remove an active model or a scientific result.

For completed parents only, gzip each adapter weight file losslessly, read it
back and verify the original SHA256, persist the archive checksum and original
path, then remove the redundant raw copy. Keep labels, batches, training logs,
checkpoint metadata and scored items immediately readable. Identical tokenizer
copies within a parent use identical-byte hardlinks. This changes neither
training nor evaluation. `restore_archived_checkpoint` in
`third_eye.experiments.scale` restores the original tensor bytes and rechecks
their hashes; run restoration only in an allocation with enough quota.

Require2.5GB whole-home headroom before each parent and verify the policy after
its compaction. Partial failures, original receipts and outputs remain preserved;
do not silently re-run a scientific parent into its old directory. Submitted
Slurm dependencies continue without the local machine or an MFA session.
