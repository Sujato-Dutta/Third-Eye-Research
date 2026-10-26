# Approved third-family confirmation

**Current status, October26:** the original CPU validation and preparation
passed, but both CUDA preflights failed on an incorrect verifier API call.
All16 collection tasks were cancelled without allocation; no new scientific
outcomes were produced. The corrected immutable release `090fc9a12413` has been
submitted with the same16 seeds, model revision, training recipe, scoring suites
and analysis. The original release and logs are preserved. See the recovery
record below; the October24 submission table is historical.

**Latest operational update:** both replacement GPU checks passed. The measured
complete-runtime check exceeded the original14h allocation and correctly
cancelled the unstarted collection arrays. The unchanged validated release now
has increased reservations:23h/code parent,30h/math parent. See the runtime
supplement below. No scientific setting or data split was changed.

October24,2026. Execute16 fresh SmolLM3-3B initial-state runs, eight per task,
using the pinned official revision and existing independent scoring suites.
No new forecaster is trained. Qwen3-8B and Llama-3.1-8B-Instruct remain future
DGX extensions when access returns. The intended eventual five-model study
therefore spans three families and two size ranges; only completed models may
be counted in paper results. No Vista8B or DGX job is submitted by this launch.

Fresh seeds are2026102800--2026102807 for math and2026102900--2026102907 for
code. Preparation checks for collisions in existing collection records.
Keep K=3, size-two unique corrections, eight correction attempts,50 optimizer
steps, rank16, LR1e-4, and three native continuation draws per candidate.
Freeze date11Oct2026 and disable extended thinking. Final target/OOD/retention
sets remain128/128/256 with exact original identifiers, excluding the eight
code timing items. The declaration is in
`src/third_eye/experiments/smollm_confirmation.py`.

One primary extension endpoint is selected future-minus-immediate utility,
with joint run/shared-item95% intervals over16 runs stratified by task.
Report selected and uniform candidate averages, role and task summaries,
replay outcomes, correction scarcity and provenance. The original32-run primary
and adjusted secondary family remain unchanged. No family-ID admission rule
is transferred to the new model. Eight runs per task broaden family coverage;
they do not establish a universal backbone or model-size effect.

Reuse the existing confirmation collection routine after exposing its validated
single-task body as a helper. Each new parent still produces thirteen final
checkpoint evaluations. The new result reader checks immutable commitments,
item identities, adapter identities, source/config hashes and terminal equality.
Run the full CPU suite, then source-bound preparation, both-task real CUDA/
adapter/checkpoint/verifier checks and a measured complete-price check before
fresh collection. All scheduler dependencies are submitted together.

Historical SmolLM timings had only one native continuation per candidate.
Pricing explicitly includes six additional continuation collections as well as
all thirteen final checkpoints. Reserve14h per parent,224 SUs across16 tasks;
32 SUs move from revision contingency to this reservation. The full revision
allowance stays500 additional SUs. The other revision controls remain planned
and are not submitted in this SmolLM-only launch.

Both eight-element arrays can run concurrently, at most16 GPUs for this launch
and never above the global20-GPU ceiling. A source-bound automatic final CPU
review depends on both complete arrays. Jobs survive SSH/local-editor closure.
Allocated failures count against the budget and are preserved; no favorable
replacement seeds, automatic scientific changes or forecast/scaling expansion.
All scientific work and large-file hash verification runs on allocated nodes.

## Submitted execution

The immutable source release is `b13a585cdad5`. Its campaign is
`runs/smollm16_b13a585cdad5` on Vista. The submission receipt is
`runs/deployment/smollm16_b13a585cdad5_submissions.json` locally.

| Stage | Job | Required successful predecessor |
| --- | --- | --- |
| Full CPU validation |1096200| None |
| Pinned model/data preparation |1096201|1096200|
| Code GPU check |1096202|1096201|
| Math GPU check |1096203|1096201|
| Measured complete-runtime check |1096204|1096202 and1096203|
| Eight code parents |1096205, array0–7%8|1096204|
| Eight math parents |1096206, array0–7%8|1096204|
| Complete-cohort analysis |1096207| Both parent arrays |

At06:34 UTC on October24, validation was pending for scheduler priority and all
successors were pending on their declared dependencies. No failed job was
reported. Collection has been submitted, not yet completed or scored.

The maximum reservation, including validation/preparation/checks/analysis, is
226.215 SUs. Together with the previous bounded-work ledger baseline of
542.573611 SUs, charged plus reserved usage is768.788611 SUs. This reservation
uses226.215 of the500 additional-SU revision allowance; actual charges are
reconciled from accounting when jobs finish. The live account reported1531 SUs
available at06:04:41 UTC on October24; an allocation reservation is distinct
from that account balance. No8B job or other revision-control job was submitted.

## October26 implementation repair and resubmission

Jobs1096200 and1096201 completed successfully. CPU validation passed213 tests
and lint. Jobs1096202 and1096203 each failed after44 seconds because preflight
attempted to call `VerifierRegistry` as a function. It now uses the common
`verify_completions` API, as collection already did. Four added regression cases
exercise both task preflights through fixture verification, rejection before
training, checkpoint restoration and discarded-fixture timing. All four passed
locally; the replacement full suite is submitted on an allocated CPU node.

At07:20 UTC on October26, replacement validation1101336 completed successfully:
**217 tests passed and lint passed**. Preparation1101337 was running; both GPU
checks and the16 collection elements were waiting on their successful
predecessors. This is implementation validation, not a scientific result.
At07:20:58 UTC, accounting also confirmed preparation completed successfully;
the scheduler was releasing its node. Both GPU checks remained pending.

`sacct --array` confirms all16 old collection elements were cancelled with
zero allocated nodes. Complete accounting reconciled the original launch to
0.665 SUs, including the minimum charge for each allocated failure. The old
campaign's `failure_closeout.json` records the defect and replacement release.

The active campaign is `runs/smollm16_090fc9a12413`. Its immutable source hash is
`090fc9a124138c9538b7f0e1d35d38b85ebfa1224328afa021d97d7c1572f8b6`.

| Stage | Replacement job | Required successful predecessor |
| --- | --- | --- |
| Full CPU validation |1101336| None |
| Pinned model/data preparation |1101337|1101336|
| Code GPU check |1101338|1101337|
| Math GPU check |1101339|1101337|
| Complete-runtime check |1101340|1101338 and1101339|
| Eight code parents |1101341, array0–7%8|1101340|
| Eight math parents |1101342, array0–7%8|1101340|
| Complete-cohort analysis |1101343| Both replacement arrays |

The replacement reservation is226.215 SUs. Including0.665 SUs already charged,
the revision has226.880 SUs charged plus reserved, leaving273.120 SUs of its
500-SU allowance unreserved. The bounded-project ledger total is769.453611 SUs.
Vista reported1510 SUs available at07:17 UTC on October26. No8B, forecasting or
other revision-control jobs were launched. Submission is confirmed; completion
and successful GPU checks must be verified separately.

## October26 measured runtime reservation supplement

Both preflights1101338 and1101339 passed, including real50-step adapter training,
checkpoint restoration, verifier compatibility, timing and rollback. They used
discarded gold training fixtures and recorded timing rather than final accuracy.
The CPU runtime check1101340 then failed because its measured requirements were
79294.55s/code parent and104178.88s/math parent, including a25% margin; both
exceeded the original50400s reservation. Its failure and original cost report
are preserved. All16 parent tasks again had zero allocated nodes.

The operational supplement is
`tools/authorize_smollm_runtime.py`, copied into the campaign with its SHA256
bound by `runtime_adjustment_20261026/authorization.json`. It runs on an
allocated CPU node. It verifies the original source/plan/input identities,
both passed GPU checks, complete three-draw cost, all sixteen reservations and
the500-additional-SU ceiling. It writes a separately identified revised cost
report and an explicit runtime-authorization receipt. It does not rerun training
fixtures, modify the immutable release, overwrite the failed cost report or
change seeds, examples, optimizer steps, model revision, scoring or analysis.
The old declaration's14h/224-SU fields describe the original operational
reservation and are superseded only by this separately recorded supplement.

Scheduler `qlimits` confirmed the `gh` walltime limit is48h. Reserve23h/code
parent and30h/math parent, totaling424 SUs for16 parents. Collection scripts
receive these walltimes explicitly through `sbatch --time`.

| Stage | Active job | Dependency |
| --- | --- | --- |
| Increased-reservation runtime review |1101554| None; CPU, original checks reused |
| Eight code parents |1101555, array0–7%8|1101554|
| Eight math parents |1101556, array0–7%8|1101554|
| Complete16-run analysis |1101557| Both active arrays |

The revised chain reserves424.44 SUs. Prior failed/successful checks account for
1.4125 SUs, for425.8525 revision SUs charged plus reserved;74.1475 remain
unreserved from the500-SU allowance. The bounded-project ledger total is
968.426111 SUs, below its1200-SU limit. Other revision-control blocks are now
deferred because their original full designs cannot also be reserved within500.
The source experiment, all16 seeds and the original32-run results are preserved.
Source-bound automatic analysis follows both arrays; no additional model or
scientific expansion follows it automatically.

At10:55 UTC on October26, CPU runtime review1101554 completed successfully.
All16 elements of1101555/1101556 were pending for scheduler priority, with no
remaining prerequisite dependency. Analysis1101557 awaits both arrays. The
active operational allocation is also recorded separately in
`experiments/revision_20261024/runtime_20261026.json`; the original manifest is
preserved. No new SmolLM collection result was yet available at this check.
