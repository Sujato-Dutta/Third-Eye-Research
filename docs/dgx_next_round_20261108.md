# DGX execution of the next experimental round

November 8, 2026. Following the allocation change, all new work uses DGX.
Vista array `1125744` was cancelled while pending with zero runtime. No model
checkpoint or project directory was transferred from Vista: only the frozen
inputs, two configuration files, source, and small provenance records.

## Submitted sequence

Campaign: `/dgxa_home/se23uari167/third_eye_dgx_20261026/runs/next_round_dgx_20261108`.

| Job | Work | Dependency |
| --- | --- | --- |
| 35035 | Initial setup; import-path failure retained | finished before GPU work |
| 35039 | Setup stopped at the 50-GB guard before model downloads | finished before GPU work |
| 35040 | Quota-safe CPU setup, model verification, full test suite | none |
| 35036, array 0--1 | First Llama and Qwen development parents | 35040 passes |
| 35037, array 0--5 | Remaining six fixed development parents | both initial pilots pass |
| 35038 | Complete eight-parent development review | both arrays pass |

The account permits **one full A100 at a time**. Arrays request
`gpu_student`, account `student`, one `gpu:a100`, 16 CPUs, and 64 GB host RAM;
their concurrency is one. The scheduler-provided CUDA device mask is preserved.
CPU work uses `cpu_student`. Login activity is restricted to staging and scheduler
operations. Scientific validation, downloads, hashing, quota scans, any cache
rotation, tokenization, training, compaction, and analysis run in allocations.

**November 8, approximately 11:03 IST:** setup `35040` completed successfully,
with all **253 tests** and lint passing. Both pinned model file sets and the
expanded manifests passed integrity checks. Whole-home usage after setup was
**44,195,692,544 bytes** (44.2 GB), leaving about 5.8 GB under the user limit.
Initial Llama task `35036_0` is running on an allocated A100 on `dgxa`;
Qwen waits for the one-GPU array slot. The six remaining parents and review
retain their dependencies. The Vista reservation was released after accounting
confirmed zero GPU runtime, and its SSH helper was closed.

**November 8, approximately 15:50 IST:** Llama parent `2026110600` completed
successfully. All four recipes were available, all three paired continuation
timing draws finished, and lossless retention passed. Development alone took
5,747 seconds (about 1 h 36 min); the pilot including continuations took 13,021
seconds (about 3 h 37 min), with Slurm reporting 3 h 37 min 35 s including
retention. Qwen task `35036_1` is running on `dgxa`, about 1 h 11 min elapsed at
inspection, still without complete development records. The six remaining
parents and final development review remain queued on their dependencies.
No pilot failure was reported. These are one-parent preliminary results;
the four-parent-per-model positive-regime screen has not been evaluated.

Initial setup found Python had cached the new release path before extraction.
The corrected launcher invalidates import caches and verifies already staged
files before resuming setup. Failed job logs and the original driver remain;
no scientific parent ran or was silently restarted.

The whole-home inventory found 46,087,098,368 bytes used and 14,470,511,048
missing model bytes. The quota guard stopped rather than download beyond the
limit. The completed scale model uses `models/scale/qwen3-8b`; the initial
cache lookup found no weights in the generic Hub location and removed nothing.
Setup now reads the scale model's verified source proof, checks every retired
weight hash, and rotates only that exact pretrained cache. The empty initial
rotation receipt and failed setup logs are retained.

## Frozen experiment

The scientific source remains
`149c4aac3ca35b07f9d3e4c333476f89a1282fa6e42a00c50e7d2ac83d99fa25`.
The source archive hash is
`5e0b082ac639bf8def530770b162b6717ef686ad9cbe341d2e038c25547f714f`.

Preserve the original model revisions, template/date, BF16, generation batches
32/16, eight correction attempts, training problems, four recipes, eight parent
seeds, and verified unique correction rules from
[the frozen plan](next_round_20261106.md). Imported expanded development and
evaluation manifests retain their original file checksums. Configuration files
are copied byte-for-byte; only deployment paths and model-source proof paths
are rebound.

All 32 recipe arms are development experiments: four recipes across four
parents per backbone. No final outcomes enter recipe choice. Optional fixed-r0
native/direct timing draws in the first two parents retain their original
six-hour timing cutoff and are explicitly marked incomplete if necessary.
The GPU reservation is 24 hours per parent to allow slower A100 execution;
this is a protective scheduler ceiling, not a runtime estimate or extra
optimizer budget. A100 timings cannot be interpreted as Vista SU measurements.

Native and matched direct arms run on the same hardware. Original completed
Vista outcomes are not substituted as paired DGX baselines. There is no
forecaster fit, architecture search, or automatic confirmatory expansion.

## Storage and validation

The stricter **50,000,000,000-byte whole-home limit** applies even if the server
does not expose an enforced quota command. Setup inventories the entire home,
prices missing model bytes, and requires four GB of experiment headroom.
It reuses verified model files wherever available. If necessary, it may remove
only recoverable Qwen-8B pretrained weight-cache files after checking that the
previous eight-parent scale cohort completed and no account GPU job is running.
Their hashes and metadata are retained; no research checkpoint, result, or
environment is deleted.

Every model file is checked against the existing frozen import registry. Model
proofs must preserve official identity, revisions, metadata, and weight hashes.
Setup runs the complete CPU test suite and lint before releasing GPU work.
Each completed development parent is compacted losslessly using existing
checksum-verified retention code, preserving labels and score files.

## Automatic stopping point

The complete-cohort CPU review writes `development_review.json`. It applies
the existing eligibility rule: mean equal-role development G1 at least +1 pp
and mean retention change at least -0.5 pp, across all four parents of a
model/recipe. This is a development screen, not confirmation of a positive effect.

If none qualifies, stop new collection. If one qualifies, freeze the fresh
confirmation design before collecting final outcomes. No confirmation or
mitigation jobs are automatically submitted. DGX execution consumes no Vista
SUs; DGX account and storage limits continue to govern execution.
