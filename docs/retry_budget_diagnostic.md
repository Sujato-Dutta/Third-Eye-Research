# Eight-attempt correction-harvest diagnostic

Authorized October 4, 2026 to assess a shorter retry budget. This is a separate
diagnostic, not an amendment to active A1 experiments. Original experiments,
training budgets, seeds, candidate construction and Gate 1/2 stops stay frozen.
Eight retries retain the central research object: predicting H=2 consequences
of persistent, verified self-updates. The retry cap is a resource hyperparameter,
not a mathematical requirement of Third Eye. Nevertheless changing it can alter
correction supply, candidate composition and terminal frequency, so scientific
equivalence and acceptance cannot be asserted without evidence. Sixteen was the
explicit A1 response to observed correction scarcity, not a proven optimum.

## Predeclared diagnostic sequence

1. Validate the diagnostic implementation on an existing scheduled CPU node.
   Run the 101 research tests, six scheduling tests, and nine retry-prefix tests.
   Recheck the unchanged frozen scientific source hash.
2. Read every currently completed A1 calibration's parent and continuation
   correction pools. Retain corrections whose first successful attempt was at
   most eight. Reconcile observed retry accounting, verify immutable source
   hashes, report retained examples/difficulty, parent K=3 feasibility,
   continuation supply on original checkpoints, and changed candidate batches.
   This uses existing artifacts and spends no additional GPU inference.
3. If CPU validation and that audit succeed, replay the first eight attempts
   at the original seed-1042 root checkpoint on all four streams. Submit at most
   four diagnostic GPU tasks, each with a 90-minute reservation: six GPU hours
   maximum reserved. Use only training prompts, the original verifier, full
   1,024/306-prompt scan, original token limit/temperature/inference batches,
   and zero optimizer updates. Require the initialized adapter hash to match
   the historical parent. Compare pools to the historical first-eight subset
   and record actual inference runtime and unchanged adapter weights.
4. Review diagnostics before proposing any shorter-budget scientific campaign.
   Assess resource cost and structural feasibility, without choosing the cap
   based on favorable ranking disagreement or model-improvement results.
   A future scientific eight-attempt protocol must be separately declared and
   applied uniformly to compared methods; historical sixteen-attempt labels
   cannot silently become evidence for that new protocol.

## Random-seed control

The original collector uses `harvest_seed + 100000 + problem_index * 16 + attempt`.
Changing its `correction_attempts` setting to eight also changes that seed stride,
including the first attempts. The diagnostic keeps the original stride sixteen
through a small backend adapter while capping execution at eight. This isolates
the retry cap. It does not change the frozen collector or research configs.

## Scope and interpretation

Historical attempt savings are exact counts for a prefix of the original draws,
not measured GPU-hour savings. The prefix retains only the original checkpoint's
verified examples. If candidate sampling changes, original continuation pools
cannot establish the new branches' H=2 consequences. Root-only GPU replays do
not establish recursive performance, forecasting learnability, or Gate 1/2 for
an eight-attempt study. Comparing new runtime to a historical sixteen-attempt
runtime is informative but is not a same-session paired throughput benchmark.

The earlier aggregate of eight completed states/32 pools contained 58,267 retry
attempts. Of these, 54,240 (93.09%) were spent on problems never corrected within
sixteen tries. The 1,172 successes took 3.44 attempts on average. This motivates
the diagnostic but does not imply that every useful correction arrives early.

## Implementation and accounting

Tools: `tools/audit_retry_budget.py`, `tools/retry_prefix_backend.py`,
`tools/diagnose_eight_attempts.py`, and its Slurm launcher. Artifacts live under
`runs/retry_diagnostics/` outside scientific label datasets. CPU validation
produces a proof binding tool hashes to the unchanged scientific source.
Submission claims prevent duplicate GPU diagnostics. Reservations and actual
diagnostic allocation costs must count toward the cumulative resource envelope,
alongside the existing profiling/benchmark jobs, even though the diagnostics
produce no scientific labels. No inference or analysis runs on login nodes.

At 10:59 UTC (16:29 IST) on October 4, the seed-3042 GPU array had completed
all four tasks with exit zero. Twelve Gate 1 calibration states were available;
all eight seed-4042/5042 tasks were running. Their original A1 protocol was not
changed. CPU retry-diagnostic validation was in progress on controller allocation
1046983; no eight-attempt GPU diagnostic had yet been submitted.

## Validated historical result and GPU submission

The complete cluster validation passed all 116 tests in 39.36 seconds. Frozen
scientific source SHA-256 remains
`f06285d80e3a248a8141448d51fbac55a095016cfef6036351f18741984abcaa`.
The historical audit covered twelve complete states and their 48 pools:

| Stream | Verified corrections retained / original | Retry attempts with cap 8 / original |
| --- | --- | --- |
| Qwen math | 214 / 232 | 9,247 / 17,992 |
| Qwen code | 140 / 145 | 13,260 / 26,274 |
| Llama math | 896 / 1,034 | 9,069 / 15,204 |
| Llama code | 320 / 381 | 15,069 / 29,135 |
| Total | 1,570 / 1,792 (87.61%) | 46,645 / 88,605 (47.36% fewer) |

All twelve shortened parent pools can still form K=3 distinct size-two batches.
All 36 shortened continuation pools have at least two corrections on their
original checkpoints. However, shortened-pool sampling changes candidate
batches in ten of twelve states. This demonstrates structural feasibility on
historical checkpoints, not equivalent labels or unaffected recursive outcomes.

At 11:02 UTC (16:32 IST), GPU replay array 1047590 was submitted with indices
0–3, capped at four concurrent nodes and 90 minutes per task. Its separate
submission record reserves six GPU hours and links the validation proof. It
does not train adapters, replace A1 labels, or automatically adopt the shorter
cap. Local evidence is
`runs/deployment/evidence/retry_prefix8_audit_20261004.json` and
`runs/deployment/evidence/retry_prefix8_validation_20261004.json`; remote records
live under `runs/retry_diagnostics/`.

At 13:15 UTC (18:45 IST), eighteen calibration jobs had completed successfully,
with the two seed-4042/5042 Qwen-math jobs still running. Gate 1 remained held
until the full sample and controller review report. Two diagnostic tasks had
completed with exit zero: Qwen code and Llama math. Qwen code exactly matched
the historical first-eight correction pool and attempt accounting, retaining
eleven verified corrections. Its harvest took 1,288.36 seconds versus the
historical full-cap 2,196.26 seconds; this is an observed cross-run comparison,
not a same-session paired speedup.

Llama math retained 74 verified corrections, but did not reproduce the historical
prefix or accounting. It initially failed 101 problems rather than the original
109, so the discrepancy is already present before retry truncation. Adapter
weights were unchanged. This is a diagnostic equivalence discrepancy to
investigate, not a passed shorter-protocol validation or a failed Gate 1.
Original results and the discrepant replay are preserved. Qwen math and Llama
code diagnostics remained running; no shorter cap was adopted for the active
research experiments.

At 13:40 UTC (19:10 IST), nineteen calibration jobs had completed successfully;
only seed-4042 Qwen math remained running. All four diagnostic GPU tasks had
completed with exit zero. Both Qwen streams passed exact pool-prefix and attempt
accounting comparisons. Qwen math retained 21 corrections in 3,072.31 seconds
versus historical 4,692.98 seconds. Llama code retained 27 corrections in
1,479.76 seconds versus historical 2,712.53 seconds, but failed prefix and
accounting comparison, as did Llama math. Both Llama discrepancies need a
controlled explanation before historical-versus-replay differences are
attributed solely to the retry cap. All four adapter-unchanged checks passed.
The four diagnostic allocations consumed about 2.0053 GPU hours in total,
including model loading, rather than the six-hour reservation.

The next decision sequence is unchanged: obtain the twentieth complete A1 state,
review its Gate 1 report, resolve the Llama replay discrepancies before adopting
eight attempts, then prepare the core label/forecaster work under a separately
reviewed consistent execution protocol. No eight-attempt scientific labels,
forecasters, or 8B confirmation jobs have been launched.

## Live review, October 5 morning

At 04:19 UTC (09:49 IST), both paired Llama controls in array 1047870
were COMPLETED with exit zero. Math used 49m 56s and code 1h 16m 10s.
The live Slurm queue was empty. No large label generation, forecasting fit,
online policy trajectory, or 8B job had been submitted.

Both same-session eight-attempt prefixes and attempt-accounting checks passed,
and adapters and original research evidence were unchanged. The shorter
harvest retained 74/80 corrections for math and 27/36 for code, taking 23.43%
and 45.21% less harvesting time, respectively. These are two root-state
paired timings, not full-study savings or evidence of equal recursive outcomes.
The short run always preceded the full run, so timing can include order effects.

Fresh sixteen-attempt Llama runs still differ from historical runs. Root math
target accuracy was 60/64 versus 62/64 historically, and OOD 38/64 versus 36/64;
root code target was 31/64 versus 32/64. Retention was identical at 121/256.
Within-session agreement separates this discrepancy from retry truncation;
it does not identify its cause or prove cross-launch reproducibility.
Original and diagnostic results remain preserved.

Reconciled GPU expenditure is approximately 99.1967 hours, excluding CPU SUs.
Gate 1 remains passed and reviewed; Gate 2 is untested. Scientific A1 remains
at sixteen attempts. Eight-attempt adoption and subsequent campaign execution
remain uncommitted, so different protocol labels are not silently mixed.
Detailed control evidence is preserved in
`runs/deployment/evidence/paired_retry_control_1047870_llama_{math,code}.json`,
and the timestamped review in `runs/deployment/evidence/progress_20261005.json`.
