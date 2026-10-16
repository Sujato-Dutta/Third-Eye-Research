# Fresh Third Eye forecasting pilots: execution record

This campaign implements the bounded development stage in
[the December research plan](december_research_plan.md). Submission was confirmed
on October11 at22:46 IST. It is part of the current Third Eye implementation.

| Job | Work | Scheduler dependency |
|---|---|---|
| 1065433 | CPU full suite and lint | None |
| 1065434 | CUDA adapter, five-draw/terminal and real-backbone preflight | afterok:1065433 |
| 1065435, array0–7, concurrency8 | Eight fresh one-state pilots | afterok:1065434 |
| 1065436 | Structural and continuation-reliability inspection | afterany:1065435 |

The final inspection runs even if pilots fail so missing/invalid states are
reported. It does not release any additional data collection automatically.
Validation failure blocks the dependent pilots. Slurm owns these jobs after
submission; an open editor or local SSH monitor is not required for execution.

Source SHA:
`f72f1e15ddffbbf7a40d0d49064a281adaf0ac1a11bd5bfc116fe4f8a3791d6c`.
Plan SHA:
`045ea6606921f9cf6a7a9af180617eae49e5e6197721f7c5d5d28932c9df37fc`.

Immutable source:
`/scratch/11617/sujato_ts/third_eye_research/releases/main_f72f1e15ddff`.
Campaign outputs:
`/scratch/11617/sujato_ts/third_eye_research/runs/forecasting_pilots_f72f1e15ddff`.

Outputs include `submissions.json`, `plan.json`, validation receipts,
`pilots/<model>/<task>/seed9142` and `seed9143`, and `pilot_inspection.json`.
The source package excludes credentials and model/data caches. Historical
source snapshots and results have not been replaced.

Eight development parent states ×three candidate updates ×five continuation
outcomes =24 candidate records and120 future outcomes. The two extra fresh pilot
seeds per stream are development seeds, not a confirmatory split. Training,
harvest attempts and terminal rules retain the frozen A2 semantics. Individual
verifier outcomes are captured during each original evaluation without rerunning
generation. No future outcome enters precommit features.

Local verification before submission:135 full-suite synthetic tests passed;
after adding the inspector contract, the three repeated-collection tests passed
again, and lint passed. The scheduled CPU suite tests the exact submitted source.
CUDA validation also loads both real pinned backbones, checks generation and a
discarded two-step adapter update/rollback, and exercises the configured code
sandbox. These smoke checks are not experimental performance evidence.

Status at22:47 IST: CPU validation was running; CUDA validation, pilots and
inspection were pending on their declared dependencies. No fresh scientific
result is available yet. Subsequent status and receipts should be appended here.

At22:48 IST,1065433 had completed successfully (0:0) in61 seconds:135 tests
passed in45.17 seconds and lint passed. Its completion receipt matches the
submitted source SHA.1065434 was pending for scheduler priority, with the eight
pilots and inspection still waiting on their declared dependencies. No failed
job was reported. Allocation delay is not a scientific result or a pipeline error.

October12 at10:50 IST: live scheduler and output metadata confirmed that both
validation jobs completed with0:0 exit status. CUDA validation took96 seconds:
the three repeat/item-capture tests passed in12.41 seconds, and both pinned
backbones passed generation, adapter-update and exact restoration preflight.
All eight pilot array tasks were RUNNING on allocated GPU compute nodes, with
elapsed runtime ranging from2h23m to4h19m. Inspection1065436 remained pending on
the array. No failure.json artifact was present in any pilot directory.

At that check, the first Llama/code seed9142 candidate had all five outcomes
and a completed record, and its second M1 checkpoint had been saved. Across
the eight tasks, nine additional continuation observation files were present;
the other tasks had each saved their first M1 and were continuing harvesting
or evaluation. No complete K=3 state had been published yet. These are execution
milestones, not a forecasting-quality result. Log timestamps ranged from10:15
to10:40 IST while long harvesting phases were active; a quiet log alone does
not establish that a task is hanging.

October12 at15:16 IST: Llama/code seed9142 (1065435_0) completed successfully
in8h32m14s with all three candidates, five outcomes per candidate and one
accepted M1. The other seven pilots were RUNNING with6h50m–7h11m elapsed;
none had a failure.json artifact. Ten of the24 planned candidate records were
complete (Llama/code4, Llama/math2, Qwen/code3, Qwen/math1), and50 fresh
continuation observation files were present in addition to the original draws.
Only one of eight complete K=3 states had been published. Inspection1065436
remained correctly pending on the array. There is no new forecasting-quality
or scientific-gate result at this checkpoint.

October12 at18:08 IST: both Llama/code pilots completed successfully. Seed9143
took9h43m56s; seed9142 remained the earlier8h32m14s completion. Six pilots were
still RUNNING, with no failure.json artifacts. Fifteen of24 candidates had
completed all five outcomes, and70 fresh observation files were saved in
addition to original draws. Inspection remained pending on the array.
The account reported2,195 SUs available, subject to outstanding job charges.
A provisional, explicitly assumption-based remaining-compute estimate is in
[the October12 projection](compute_projection_20261012.md); it does not freeze
sample counts or authorize automatic expansion.

October13 at10:08 IST: all eight pilots and inspection1065436 completed
successfully (0:0). The inspection'spassed=true is a structural result:
24 complete candidate records,120 continuation outcomes, distinct compositions,
matched budgets and checkpoint-bound per-item evaluation artifacts. All120
continuations were available; there were no terminal draws in this initial-state
cohort. The report explicitly remainsdevelopment_only=true and
automatic_expansion=false. No forecasting superiority or Gate2 decision was
made. The live queue was empty at this review hold.

Pilot GPU allocations totalled108.6611 node-hours. Including the minimum charges
for one CUDA validation and twoCPU jobs yields approximately109.08 SUs for
the stage, using the published1/0.33 queue rates. Model/task state times ranged
from8h32m to19h43m, with a balanced mean13.58h per state. These are measured
initial-state costs; later-generation costs remain unmeasured under this
five-draw collection. The account reported2,093 SUs remaining. Native labels and
the inspection report are being copied locally with remote SHA256 checksums.
