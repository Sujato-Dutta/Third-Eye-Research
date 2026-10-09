# Independent selection R1: execution record

Declared October 9 under [R1](protocol_independent_selection_r1.md), before
examining any E2 matched-control outcome. No E2 noise completion or noise
submission existed at the pre-declaration check, 09:37 IST.

The extension uses the existing eight-parent sample, original M2 and two fresh
continuations per candidate. It adds no GPU experiments or fitting. Primary
selection uses fresh +50M, scoring uses fresh +100M, with disjoint evaluation
halves and both fold orientations. The signed independently scored gap may be
negative; observed future-oracle regret remains a separate descriptive measure.
The secondary one-versus-two comparison uses held-out continuation outcomes.
There is no universal replicate-count, causal mechanism or acceptance claim.

## Validation and automatic execution

- Nine local checks passed, including negative independent gaps, tied winners,
  disjoint/stable item identities, shared-item selection artifacts, invalid seed/
  item coverage, trajectory-level sample counts and completed-dependency handling.
- CPU validation job **1059894** passed the same nine checks on Vista and the
  frozen source/method prerequisite checks in 28 seconds.
- The frozen six-file R1 release SHA256 is
  `67f4e81ef2ffd247811ad90f5604bcaac37042cdb0f85b1477fdd5b870e83dcf`.
- A scheduler-only watcher, initially PID **3585628**, waits for core trajectories
  and their known control jobs to end, then submits one bounded CPU analysis job.
  Failed or missing states produce an incomplete report; they are not replaced.
- The analysis writes source/input hashes and creates a separate durable work
  archive. Real analysis and archival execute on Slurm compute nodes exclusively.

The watcher is under
`runs/a2/empirical_v1/checkpoint_fallback_v3/independent_selection_r1/`.
`watch_status.json` carries its heartbeat; `submission.json` is created only when
the CPU review is submitted. No GPU is allocated for polling or waiting.

## Core review dependency repair

The earlier SmolLM failure showed that a completed validation could disappear
from Slurm's dependency lookup. The later E2 review also generates dependencies
on early controls, which may have ended long before review submission. Preserve
the frozen relay and use a versioned scheduler wrapper: validate the original
request, require complete accounting for known E2 control jobs, remove only
already-terminal `afterany` dependencies, retain active dependencies, and record
both argument lists and accounting. Failed controls remain failed scientific
evidence and lead to incomplete review.

After successful CPU validation and confirming no pending requests or unresolved
claims, the old relay PID1839477 was retired. The new validated scheduler relay,
initially PID **3585394**, serves the same queue with unchanged experiment scripts,
model/method identities, budgets, node cap and cutoff. Old relay provenance and
every dependency reconciliation are retained inside the R1 namespace.

At the service handoff, all eight core trajectories remained pending scheduler
priority and no reliability results existed. This is an analysis/control-flow
extension, not evidence that reversals exceed noise or forecasting generalizes.
