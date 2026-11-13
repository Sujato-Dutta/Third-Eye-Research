# DGX positive-regime execution — November 9, 2026

The [frozen confirmation design](positive_regime_confirmation_20261109.md)
tests the development-selected Qwen3-4B recipe on twenty fresh parent seeds.
All new execution is on DGX. No Vista jobs or SUs are required.

| Job | Work | Dependency | Verified status at submission |
| --- | --- | --- | --- |
| 35050 | Pinned-model and input checks, whole-home storage, full CPU suite | none | Completed, 0:0 |
| 35051, array 0–1 | First two fresh parents, three matched draws and mitigation each | 35050 succeeds | First parent running |
| 35052 | Timing and projected retained-storage check | Both 35051 tasks succeed | Pending dependency |
| 35053, array 2–19 | Remaining eighteen fixed parents | 35052 succeeds | Pending dependency |
| 35054 | Complete-cohort artifact checks and paired statistical analysis | All twenty parents succeed | Pending dependency |
| 35056 | Supplemental end-to-end scoring/sealing/reload tests | none; CPU only | Completed; all three scenarios passed |

Slurm initially assigned raw job ID `35055` to array task `35051_0`.
GPU arrays permit one concurrent task, matching the account's one-A100 limit.
The first two are included in the twenty; there are no additional efficacy
pilots or outcome-dependent replacements. The timing check reads elapsed time
and retained bytes, not scientific scores. Successful dependencies release
the next work automatically, without a local SSH session.

## Validated release

- Scientific source: `d5c3bc9fb65a2e416736ff42be7f7ebadf5a44943afc7277c52ef846fed0d1e0`.
- Source archive: `d0d4c464699b05835cad5a5d2260a0bdabc646715cdab48ed162f2899a667102`.
- CPU validation: **258 tests passed**, lint passed, on compute node `dgxa`.
- Whole-home usage after preparation: approximately **39.24 GB**, below the
  50,000,000,000-byte operating limit.
- Only completed Llama-3.2-3B pretrained weight-cache files were retired after
  ownership, path, source-hash and completed-cohort checks. All research
  adapters, outcomes, model metadata and environments remain retained.

Every parent first verifies real GPU training and exact adapter rollback with
a training-only compatibility fixture. It then seals all eleven research
checkpoints and their donor data before final scoring. Completed parents receive
independent artifact checks and lossless, checksum-verified adapter archives.
All scientific analysis, hashing and storage scans run inside Slurm allocations.
Login-node activity is limited to staging, small metadata reads and scheduling.

The first parent's real CUDA compatibility check passed: one optimizer step on
eight training-only fixtures, followed by exact rollback to the parent adapter.
The fixture consumed approximately 5.4 seconds and was not retained as a research
update. Parent `2026110900` is now collecting its research correction pool.

The supplementary CPU tests exercise complete scoring, immutable commitments,
independent artifact reload and tamper detection under nonterminal, initial-scarcity
and continuation-scarcity scenarios. Their separately hashed test file is retained
under campaign validation; the frozen scientific release is unchanged.

Campaign directory:
`/dgxa_home/se23uari167/third_eye_dgx_20261026/runs/positive_regime_20261109`.
Local release, submission and validation receipts:
`runs/deployment/positive_regime_20261109/`.

No confirmatory scientific result is available at launch. The earlier +2.132 pp
immediate gain is a development estimate. The larger-pool first pair will
establish a defensible runtime estimate. The old manuscript and Overleaf archive
remain unchanged until the complete confirmation is reviewed.

## November 11 operational check

Both fixed operational pilots completed, passed artifact inspection, and were
losslessly archived: `35051_0` took 15:52:19 and `35051_1` took 17:03:14.
All three continuation draws were available in the first pilot. These are
execution checks, not a cohort-level scientific conclusion.

The dependent storage check `35052` correctly stopped before releasing the
remaining eighteen parents: its projected whole-home requirement was
51,849,257,984 bytes, including a two-billion-byte scratch reservation. The
failed check and its report are retained as
`timing_review_storage_stop_35052.json`; no completed experiment is restarted.

CPU maintenance jobs `35330` and `35333` inspected storage. Job `35331`
inventoried identical historical artifacts; `35332` recovered 755,957,760
allocated bytes by sharing identical inodes. The broader follow-up inventory
`35335` identified additional byte-identical tokenizer/checkpoint copies in
completed historical work. Job `35336` applies that plan with SHA-256 checks
before and after every replacement. All scientific file contents, paths,
archival receipts, frozen sources, seeds and optimizer settings are preserved.
The unused narrower inventory `35334` changed no files.

The original timing script is resubmitted after successful maintenance. Only a
successful repeated storage check may release the existing `35053` array. Its
array task IDs must remain unchanged because the longer-horizon Qwen extensions
depend on those exact completed parents. All scans, hashing and inode sharing
run inside CPU Slurm allocations; login-node work remains scheduler/transport
activity only.

Maintenance completed successfully. The two deduplication applications recovered
approximately 1.97 GB without deleting any research path or changing its bytes.
Whole-home allocation is now 38,332,022,784 bytes. Replacement timing/storage job
`35337` passed: the projected completed cohort, including its two-GB scratch
reservation, is 49,877,738,496 bytes, below the unchanged 50-billion-byte limit.
The existing `35053` array dependency was repaired from failed `35052` to
successful `35337`; no array element or scientific setting was replaced.
`storage_repair_20261111.json` records the scheduler and storage repair.
The repaired array was verified running as `35053_2` (raw Slurm ID `35338`)
on `dgxb`; tasks 3--19 remain pending under the intended one-task array limit.

## November 12 progress

Three of twenty parents have completed successfully. The third (`35053_2`)
took 18:07:28, including inspection and archival. The fourth (`35053_3`, seed
`2026110903`) is running on `dgxb`, with eight saved checkpoint states reported
and final scoring still pending. Remaining tasks and the final review retain
their automatic dependencies. No new execution failure appears in the queue.

On November 12, pending indices 14--19 were delegated to the temporary account's
two-GPU pool. Original running index 3 and pending indices 4--13 stay on this
account. Review `35054` now waits for the retained original tasks and six
owning-account handoffs, with automatic same-seed fallback if a delegated job
cannot complete before November 17 EOD. See
[temporary-worker execution](dgx_temporary_worker_20261112.md) for job IDs,
validation, quota guards and cutoff handling. Scientific settings are unchanged.
