# Scoring precision/cost check: execution record

Submitted October 21, 2026 under the frozen
[development declaration](scoring_precision_20261021.md). This campaign generates
development timing only and audits recorded final-item exposure. It does not
produce final accuracy results, retrain a policy or release confirmation.

Immutable source:
`2a4711afc18c2e6c0cc3797b75d0f4659a53023e45a4ea8d9ba8e3f565aa5335`.
Archive SHA256:
`e204a2ae53637e41ee809087d37387c452da7580cb8e714f0d5ec595f51d0bc3`.
Remote campaign:
`/scratch/11617/sujato_ts/third_eye_research/runs/scoring_precision_2a4711afc18c`.

The plan binds completed B's plan/report and the CPU decision audit's
report/completion to hashes. The source archive contains the declaration,
implementation, tests and Slurm scripts, excluding local credentials.
No historical evidence is overwritten. Saved parent and original-continuation
checkpoints are checked on CPUs before any GPU work.

| Job | Work | Dependency | Maximum SUs |
| --- | --- | --- | ---: |
| 1086306 | Full CPU tests and lint | None | 0.110 |
| 1086307 | Manifest/exposure audit and checkpoint preflight | afterok:1086306 | 0.330 |
| 1086308, elements 0–3 | Four inference timing streams, at most four GPUs | afterok:1086307 | 8.000 |
| 1086309 | CPU timing/provenance review | afterok:1086308 | 0.165 |

At **18:54:47 IST**, validation was RUNNING on `i615-073`; the remaining jobs
were PENDING for Dependency. These are submitted Slurm dependencies and continue
without the local SSH connection. Invalid dependencies cancel downstream jobs;
no automatic retry or scientific expansion is configured.

Maximum new reservation is **8.605 SUs**, charged to B development. B costs plus
this reservation are **152.8031 SUs**, below 192. The bounded-additions ledger is
**240.5992 SUs used/reserved**, leaving **959.4008 SUs** under 1,200. Actual final
charges, including minimum CPU billing and any failures, require reconciliation.
These are project-budget figures rather than TACC account balances.

Local lint passed. Four outcome-blindness/sample-selection synthetic controls
passed locally; two filesystem-fixture controls could not initialize because of
local temporary-directory permissions and are included in the full compute-node
suite. No full-suite pass is claimed at submission time.

Update at **18:56:07 IST**: validation **1086306 COMPLETED**, exit 0, in
75 seconds. The full allocated-node suite passed **192 tests in 61.23 seconds**,
plus lint, including all six new controls. This resolves the local fixture
verification limitation. Provenance was eligible/PENDING; GPU timing and review
still awaited their dependencies. No inference result is available yet.

All real artifact scans, suite checks, inference, timing and numerical review
run on allocated compute nodes. Login operations only uploaded/hashed the release,
reserved the ledger and called the scheduler. The GPU array performs zero training
steps and writes no final predictions. The review stops automatically for item
provenance, precision and cost assessment; C and recursion remain unreleased.

Local submission receipt:
`runs/deployment/scoring_precision_2a4711afc18c_submissions.json`.

## Evening progress check

Reconnected at **23:23 IST on October 21**. Provenance **1086307 COMPLETED**,
exit 0, in 622 seconds, having scanned **46,591 artifacts** with no unreadable
files and checked all eight saved benchmark checkpoints. Four timing elements
**1086308_0–3** remain PENDING for Priority; review **1086309** waits for their
completion. No timing job has run or failed. Preserve their IDs and queue age.

The raw exposure report is `recorded_exposure_unresolved`. Every flagged
filename is under automated validation fixtures: synthetic benchmark examples
reuse real-format item IDs, and mock final-score files name `fixture-final`.
There is no flagged research-run pathname in that report. This is a likely
fixture-classification false positive; it must not be reported as demonstrated
benchmark contamination or silently erased. The original report/hash remains
immutable.

Declared a [separate checked classification](scoring_fixture_classification_20261021.md),
requiring original artifact hashes, owning validation receipts and known mock
content rather than path names alone. Submitted full CPU validation **1087198**
then classification **1087199** with afterok dependency, at most **0.22 SU**.
Neither changes the four GPU jobs, policies, labels or final item pools.
Its source is
`353fc7447b004d4f72bc3da18533f2d659dca09b510f341744b0efe8cfc7e4fb`.

Reconciled finished original validation/provenance to minimum billing of
0.0825 SU each. With pending timing/review reservations and the separate
classification reservation, the bounded project ledger totals **240.5442 SUs**,
leaving **959.4558 SUs** under 1,200. No confirmation or recursion is submitted.

October 22 closeout: all original jobs and classifier jobs COMPLETED, exit 0.
The four GPU benchmarks ran in parallel at 11:53:50–11:57:41 IST; review finished
12:00:34 IST. Classifier validation passed 194 tests plus lint. All 180 flagged
artifacts were verified synthetic fixtures, with zero remaining flagged research
artifacts in the original search. See the
[completed review](scoring_precision_results_20261022.md) for interpretation,
measured timings and final accounting. Combined actual cost is 1.4125 SUs; the
project ledger is 233.4067 SUs with 966.5933 remaining under its cap. No new
scientific efficacy result or automatic confirmation release follows.
