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
