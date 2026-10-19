# Stage B execution record

Prepared October 18, 2026. Frozen declaration:
[fresh admission pilot](admission_stage_b_20261018.md).

Source SHA256:
`ef78aa1ed0565020b2fc727af62f8aaeac9cbf71ba7382f2aa839eb54da44156`.
Archive SHA256:
`4070f122c020640fa6740b6c94125f2106aefa523915bbab2011c39b14df91ca`.

Implemented the 16-parent/four-stream plan with three total native continuation
draws per candidate, fixed rendering from the first harvest, independent item
halves, H1/H2 admission fits, hold/replay baselines, actual update norms,
conditional parent/shared-item uncertainty and two power sensitivities.
Original eight-parent/five-draw collection remains supported. Exclude probe
features that use held-out scoring items. No pooling arm or automatic C launch.

Local verification: lint passes; 12 admission/planning tests pass; real tiny
adapter integration passes with three and five draws, with and without terminal
branches. Synthetic preparation, complete review, array submission/index offsets
and duplicate-submission blocking also pass. Fixtures are not research evidence.
The required full suite and real CUDA preflight remain cluster jobs.

Live metadata check at 13:27:53 UTC showed an empty user queue. Prior baseline
five-draw job1065435 completed all eight elements; their elapsed times ranged
from30,734 to70,986seconds. Allocation availability and new runtime limits are
not inferred from these values alone: allocated CPU preparation performs the
declared per-stream cost calculation before permitting new GPU collection.

Reconnected and uploaded the immutable release. Remote archive/release hashes
were verified before submitting the dependency chain. Submission succeeded on
October18 at approximately19:16IST. Latest scheduler/accounting verification at
19:20:39IST showed:

| Job | Purpose | Observed state | Required predecessor |
| --- | --- | --- | --- |
| 1078842 | Full CPU suite and lint | COMPLETED, exit0,81seconds | None |
| 1078843 | Seed/data audit and measured budget guard | COMPLETED, exit0,40seconds | afterok:1078842 |
| 1078844 | CUDA and four-stream real-model preflight | PENDING, Priority | afterok:1078843 |
| 1078845 | Bounded pilot submission controller | PENDING, Dependency | afterok:1078844 |

The controller submits the four GPU arrays and CPU review only after successful
validation/preparation/preflight. The16 GPU pilots are therefore **not yet
submitted** at this observation; their launch is automatic if the guards pass.
Any predecessor failure blocks descendants. No automatic confirmation launch.
Full cluster validation passed:176tests in66.68seconds, plus lint. Preparation
completed successfully, checking16 fresh parent configurations, manifest
integrity/disjointness and historical run metadata for the declared seeds.
No implementation failure was observed. CUDA preflight has not yet run.

Measured per-parent time ceilings, including the declared25% margin:

| Stream | Limit per parent | Parents |
| --- | ---: | ---: |
| Llama code | 8hours | 4 |
| Llama math | 13.5hours | 4 |
| Qwen code | 8.75hours | 4 |
| Qwen math | 16hours | 4 |

The four arrays reserve185GPU-node SUs in total. The full B chain, including
all CPU minima and CUDA preflight, reserves at most186.8525SUs, within192.
These are budget/time ceilings, not guaranteed durations or queue-time forecasts.
Downloaded plan and validation receipts are under
`runs/deployment/evidence/admission_stage_b_20261018/`.

The one-shot deployment continuation completed without retry. Transport request
70 records upload,71 submission, and72 live scheduler verification. Local receipt:
`runs/deployment/admission_b_ef78aa1ed056_submissions.json`. The remote campaign
receipt will record eventual pilot array/review IDs. Submitted Slurm jobs no
longer depend on the local SSH connection or workspace remaining open.

Remote campaign:
`/scratch/11617/sujato_ts/third_eye_research/runs/admission_stage_b_ef78aa1ed056`.

B's192-SU ceiling transfers72 from contingency before outcomes; total new
planned ceilings stay1,088 and the existing persistent ledger cap stays1,200.
The initial chain reserves1.5225SUs; ledger charged-plus-reserved is89.318611SUs.
The controller must reserve all16 GPU elements and review before their submission;
those additional reservations remain conditional on the measured192-SU B cap.
No computations are authorized on the login node.

## October 19: deployment failure and reviewed correction

Live review at 12:04:43 IST found an empty queue. CUDA job **1078844** had failed
after 57 seconds; controller **1078845** was dependency-cancelled without an
allocation. CUDA training/rollback/checkpoint smoke and all five repeated-draw
tests passed. The real code-stream verifier then raised:
`Code verification requires an isolated sandbox configuration`.

The deployment helper omitted exports for the existing sandbox, model-source
registry and offline cache. This was an implementation/deployment defect, not
a negative scientific result. **None of the 16 fresh pilots had been submitted
or collected.** Preserve the failed campaign and its outputs unchanged.

After reviewing the traceback and existing working deployment configuration,
corrected the helper to export all three paths explicitly. It checks the two
configuration files exist and records their hashes in `environment.json`:

- sandbox config SHA256:
  `c7b1c5bc6b051ac040e1b9dcdc340ef30eef529c51ec2159e7250b907f513bbd`;
- model-source registry SHA256:
  `6487d731b1954396bb5dd958f65540c6b679473cf4d9cc922ddacab4924093d5`.

Reuse the same immutable implementation/source SHA; no protocol, seeds,
candidates, optimizer budgets, features or analysis rules changed. The reviewed
replacement has new output directories so it cannot overwrite failure evidence.
This is one manually reviewed repair, not an automatic job retry.

New campaign:
`/scratch/11617/sujato_ts/third_eye_research/runs/admission_stage_b_ef78aa1ed056_env1`.

| Job | Purpose | State at 12:08 IST |
| --- | --- | --- |
| 1080365 | Full CPU suite/lint with corrected runtime environment | RUNNING |
| 1080366 | Repeat seed/data and cost preparation | PENDING, afterok:1080365 |
| 1080367 | CUDA and real four-stream preflight | PENDING, afterok:1080366 |
| 1080368 | Submit bounded pilots and review | PENDING, afterok:1080367 |

Reconciled every old job against complete final scheduler accounting before
reserving replacements. Original B attempt: two CPU billing minima (0.165 SU)
plus one GPU minimum (0.25 SU), cancelled controller zero: **0.415 SU total**.
Replacement chain plus all planned pilots/review remains at most 186.8525 SUs;
combined B maximum **187.2675 SUs**, below the 192 ceiling. Persistent ledger
charged plus reserved after replacement submission: **89.733611 SUs**; the large
GPU arrays remain unreserved/unsubmitted until the validated controller runs.
The overall 1,200-SU cap remains unchanged. No new scientific results yet.
