# Deadline-bounded empirical supplement: execution record

Authorized October 8, targeting a result freeze at October11 23:59 IST.
The checkpoint-based [E2 supplement](protocol_empirical_checkpoint_v3.md) was
adopted following authorization to execute the strongest feasible deadline
package. E1's failure remains evidence; E2 replaces its proposed execution
ordering and reliability sample, without changing A2 or old labels.
Follow [E1](protocol_empirical_v1.md), the [fixed comparator protocol](protocol_empirical_comparator_v1.md)
and [balanced replication protocol](protocol_empirical_replication_v1.md).
All scientific computation and dependency installation use Slurm compute nodes.
The login relay performs only bounded scheduler submissions and metadata checks.

## Current execution

| Stage | Job | Evidence/status |
| --- | --- | --- |
| Cached preparation, selection, mechanisms, ridge controls | 1057421 | Completed; 120 checks passed |
| Supplemental CPU validation | 1057437 | Completed; 127 checks passed |
| Four-stream exact replay verification | 1057438 | Stopped at first parent aggregate evaluation mismatch; original parent adapter hash matched |
| Twelve-noise-state controller | 1057439 | Canceled without launching measurements after its prerequisite failed |
| Fixed tabular comparator and nine GRU positive controls | 1057452 | Completed, isolated dependencies; all models reload exactly |
| Native-versus-instrumented evaluation diagnostic | 1057465 | Both evaluators agree; M1 replay hash differs from original |
| Freeze existing methods and ridge coefficients for new cohort | 1057466 | Completed; 42 existing neural fits, six ridge controls, three tabular fits frozen |
| Full native probe replay diagnostic | 1057477 | Same replayed M1 as without probe; differs from original |
| Identical-code verification diagnostic | 1057478 | Completed; all eight passes29/64, zero observed fluctuating items |
| Durable initial evidence archive | 1057487 | Completed; source, outcomes and failed validation preserved in Vista work storage |
| Actual inputs/backbone audit | 1057506 | Completed; all twelve original config/manifest identities match; Llama weight files match original signed proof |
| Recovery V4 CPU/CUDA validation | 1057520/1057521 | Completed; 137 CPU checks, exact loading/rollback of all three retained partial M1 checkpoints |
| V4 scheduler handoff verification | 1057558 | Completed; repaired dependencies after obsolete-job cleanup failure |
| E2 execution-v2 full CPU validation | 1057708 | Failed before collection: comparator interpreter lacks training packages; prediction-text variable collision |
| E2 execution-v2 CUDA verification | 1057709 | Canceled without execution after CPU failure |
| E2 eight-trajectory launch controller | 1057710 | Retired unused by scheduler-only prequeue handoff; validations unchanged |
| E2 execution-v2 blocked collection entries | 1057718–1057725 | Canceled automatically after CPU failure; no trajectory outputs or experiments |
| E2 execution-v2 dependent confirmation | 1057726 | Stopped before inference; preserved failed receipt |
| E2 execution-v3 full CPU validation | 1057729 | Completed in54s; all145 tests pass, matching passed receipt |
| E2 execution-v3 CUDA verification | 1057730 | Completed in15m51s; all four stream checks passed |
| E2 execution-v3 unused launch controller | 1057731 | Retired by bounded scheduler handoff |
| E2 eight balanced trajectories | 1057732–1057739 | Queued with `afterok` dependencies on both 1057729 and 1057730 |
| E2 original pending confirmation | 1057740 | Retired before execution with original pending graph |
| E2 launcher-path CPU checks | 1057803 | Completed;147 checks passed |
| E2 corrected eight balanced trajectories | 1057804?1057811 | Queued behind actual CUDA1057730 and CPU path-check1057803 |
| E2 corrected balanced confirmation | 1057812 | Queued after all eight replacement collections |
| Durable E2 source/methods/failure archive | 1057744 | Completed in20s; 6.8MB archive in Vista work storage |

No E1 twelve-state noise measurements were released. E2 balanced replication
GPU entries are now queued behind both validations. The replay verification failure, its logs and outputs
remain intact. A comparison diagnostic must establish whether the discrepancy
comes from instrumentation, runtime repeatability or verification. Do not
relax the exact-match requirement or replace the selected parent to get past it.

The original pending V3 queue entries1057197/1057269/1057271 and CPU1057272 were
retired only after V4 passed CPU/CUDA validation and replacements were submitted.
Their original artifacts and cancellation evidence remain intact. Replacements
**1057528/1057529/1057530** are pending for Priority at13:34 UTC /19:04 IST;
**1057531** waits for all three to complete successfully before full-data audit
and fixed forecasting fits. These recoveries reuse actual saved partial M1s,
preserving their fifty completed optimizer steps; they do not retrain a new
adapter and pretend its weights match the old one.

Controller1057522 failed after preparing the entire graph, when querying an
obsolete purged job1049535. A separate compute-node reconciliation1057558
verified CPU/CUDA success, all prepared inputs and the exact failure reason.
The scheduler-only relay rebound the graph to that successful verification.
No scientific validation was bypassed, and the failed controller is preserved.
The GPU ceiling is twenty including recovery, validation, noise and replication.

Completed branches were pruned in the original A2 study. Saved verified batches,
training seeds and adapter hashes allow deterministic reconstruction; the new
code requires matching ancestor, parent, M1 and M2 hashes. It rechecks saved
corrections with the original train manifest and verifier. It never reharvests
original pools or overwrites original labels/checkpoints. New measurements and
checkpoints use `runs/a2/empirical_v1`.

## Completed cached evidence

These numbers use the existing exposed held-out cohort, whose stream coverage
is uneven. They are not fresh balanced replication results. Utilities average
target/OOD/retention accuracy fractions; percentage-point values below multiply
by100. Oracles are privileged observed-data comparisons.

- Immediate-best minus uniform-random future utility: **+0.988 percentage points**,
  trajectory bootstrap95% CI **[+0.073,+2.130]**. The claim that greedy is near
  random is not supported by this result.
- Immediate-best absolute future utility: **+0.301 points**, CI **[-0.125,+0.859]**;
  42.5% of selected candidate utilities are negative.
- Uniform-random absolute future utility: **-0.687 points**, CI **[-1.465,+0.069]**.
- Observed future oracle: **+1.930 points**, CI **[+1.191,+2.712]**. Immediate-best
  regret versus this forced-update oracle: **1.629 points**, CI **[1.136,2.080]**.
  Holding parameters fixed over both steps has zero utility by definition.
- The four exploratory mechanism associations have intervals crossing zero
  on both training and development. No causal explanation has been established.

The fixed tabular adaptation has development utility Spearman **.3033** for H1,
**.1375** for H2 and **-.1650** for the continuation increment. Corresponding
top1 rates are45%,42.5%,22.5%. It has no fresh-cohort result yet and does not
change the original Gate 2 decision.

Independent-GRU positive controls recover the noiseless constructed signal at
development Spearman **.9375-.9500** across three fitting seeds; .6000-.6625
with noise.5 and .3500-.4750 with noise1. These are synthetic detectability
controls, not successes on real future-update labels or proofs of intrinsic
unpredictability. The original labels are untouched.

## Next barriers and scope

E1's exact historical replay remains stopped at its failed prerequisite;
the originally proposed twelve-parent measurements were not released. Do not
relax that requirement or present E2 as a successful E1 replay. E2 instead uses
actual retained weights in a separately declared eight-parent supplement,
following the adopted execution order and unchanged A2 semantics. Do not invent
a terminal-frequency or reversal threshold after observing results.

The full original feature/probe replay reproduced the same new adapter hash
as the replay without it; both differed from the original. Its cause remains
unresolved. Actual Llama backbone bytes, source proof, package versions,
configuration and manifest match. Repeating verification of the same generated
code eight times did not reproduce the original OOD28/64 total: every repeat
was29/64. This sample supplies no evidence that verifier variability explains
the observed difference.

A separate flag diagnostic found that the sandbox's Python isolation option
ignores its configured environment hash seed, with eight distinct hash values.
This matches [Python's documented isolation behavior](https://docs.python.org/3/using/cmdline.html#cmdoption-I).
No verifier flags have been changed. Do not attribute the adapter or accuracy
discrepancy to this finding without supporting evidence.

The [checkpoint-based fallback proposal](empirical_checkpoint_fallback_proposal.md)
would collect the same eight planned balanced trajectories, retain their actual
weights, and run reliability measurements from eight independent saved parents.
It changes E1's ordering and sample, while keeping A2 and old results unchanged.
E2 is now adopted. All eight GPU collection entries are already in the queue
to avoid delaying their scheduler eligibility; scientific execution still
requires both validations to pass. The [scheduler-only prequeue handoff](empirical_checkpoint_prequeue_v1.md)
replaces an unused pending CPU submission controller and preserves both validation
dependencies and runtime passed-receipt checks. Execution-v2 validation caught
two implementation defects; its blocked entries were canceled without collecting
outcomes. Execution-v3 fixes only the interpreter choice and colliding text
variable, with unchanged E2 scientific scope. At15:11 UTC /20:41 IST, CPU1057729 has completed successfully, with145
checks passed. CUDA1057730 is pending for Priority; all eight collections
still wait for both validation dependencies. The scheduler currently estimates
CUDA allocation at15:40 UTC /21:10 IST, but that estimate can change.
Job IDs are durable in `checkpoint_fallback_v3/trajectory_submissions.json`.

The new held-out cohort is eight T5 trajectories, two seeds6142/6143 per
Qwen/Llama math/code stream: forty complete states if no branch terminates
collection early. Keep all weights and failed/incomplete states. Methods are
frozen before any new outcome: all existing original/F2/three-horizon fits,
six fixed ridge controls, three tabular target fits and unchanged heuristics.
Stop after balanced confirmation for scientific review. No 8B or online scaling.

E2 planning estimate:227-267 additional GPU node-hours for reliability plus
balanced replication, with validation and existing recoveries accounted
separately. These are estimates, not measured new-pipeline runtimes. Queue delay,
replay failures and long trajectory tails can prevent meeting October11.

## Adopted deadline package

The fixed fresh cohort is eight trajectories: seeds6142/6143 in each of four
model/task streams. This provides forty states and120 labels if all complete.
All parent/M1/M2 weights remain on disk. Exactly eight independent reliability
parents are selected by metadata: generation0 for6142 and generation4 for6143
per stream. Their controls load actual retained weights, compare fixed-batch
optimizer repeats and two fresh continuation seeds, and repeat verification of
identical code. Native-versus-repeat metric differences are disclosed. No
success criterion depends on a new terminal-frequency threshold.

Early controls overlap later trajectory generations. Eight trajectories,
eight controls and three recoveries fit within nineteen GPU jobs; the relay
enforces a ceiling of twenty across all queued/running GPU jobs. CPU confirmation,
noise review and durable archival follow automatically. They stop at scientific
review, with no new architecture fits, online rollout or 8B scaling. GPU execution
has an October11 18:29:59 UTC deadline guard, including allocations received late.

The primary fresh comparison remains the frozen Direct-H2 versus matched-H1
future selection value, alongside all previously frozen methods and controls.
Main-paper scope should follow replicated effect sizes, reliability and mechanism
evidence. Architecture superiority and intrinsic unpredictability are unsupported
by the current evidence. Main or Oral acceptance cannot be guaranteed by this
experiment package.

Source A2 remains
`917c86b78482c73797619d17feab423f91d3775026db086e15f7f529e4fcf988`.
Source/scope/releases/input manifests and learned-method artifacts are hashed.
Local evidence is under `runs/deployment/evidence/empirical_*`; request/response
receipts are under `runs/deployment/vista_monitor_requests/`. Record actual
allocation durations separately from GPU wall-time reservations and CPU charges.

The [manuscript methods draft](manuscript_methods_draft.md) and
[paper evidence plan](acl_deadline_evidence_plan.md) are available for writing
while the fixed experiment runs. New result tables remain pending validation
and complete artifacts. Initial E2 archive SHA256 is
`df7d6dd2202bb307f6749fe2d85b0eff950c19fd63673f7ab699c89b0d9e5855`;
it preserves all frozen learned-method weights, execution sources, the failed
v2 evidence and the initial valid v3 queue, rather than claiming final outcomes.

Local verification of the initial archive passed: the whole-archive checksum
and all163 manifest-bound member checksums match. No new collection outcome
is included in this initial preservation bundle.

## Current core graph and third-backbone supplement

Original CUDA1057730 has completed successfully in15m51s. A generated-launcher
filename mismatch was found while all eight collection entries were still
pending. Preserve frozen V3 and its validated scientific code; the versioned
`empirical_checkpoint_path_repair_v1.py` wrapper changes only generated V2
launcher names to the V3 relay allowlist. Full147-check validation passed in
1057803. Replacement1057804?1057811 and confirmation1057812 are the current
core graph; retired pending entries produced no scientific outcomes. Receipts
are under `checkpoint_fallback_v3/path_repair_v1/submissions.json`.

The separately authorized [SmolLM3 supplement](smollm_execution.md) prepares
four early states across code/math and seeds6142/6143, preserving all A2 budgets.
Setup1057831 passed127 checks and staged pinned official weights; CUDA1057855
is queued. Optional work reserves future core control capacity within20 total
GPU nodes. It neither changes the original E2 primary population nor adds a
third-backbone noise-floor or full-T5 claim.
