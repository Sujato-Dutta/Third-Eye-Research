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

Update at approximately 12:11 IST: replacement validation completed with exit 0
in 78 seconds (**176 tests in 63.56 seconds, plus lint**). Preparation completed
with exit 0 in 46 seconds, preserving the 16-parent grid and 186.8525-SU new-chain
ceiling. CUDA **1080367** is PENDING for Priority; controller **1080368** remains
dependency-blocked. No pilot array has been submitted yet.

Scheduler forecast was 04:00 CDT on October 19, equivalent to **14:30 IST**.
This is a mutable scheduling estimate. A read-only `sbatch --test-only` check
for the regular `gh` partition estimated October 20 instead, so no partition
change or extra GPU job was submitted. Actual launch still requires CUDA and
real-model/sandbox preflight to pass. Review receipts are downloaded to
`runs/deployment/evidence/admission_stage_b_20261019_env1/`.

Reconciled every old job against complete final scheduler accounting before
reserving replacements. Original B attempt: two CPU billing minima (0.165 SU)
plus one GPU minimum (0.25 SU), cancelled controller zero: **0.415 SU total**.
Replacement chain plus all planned pilots/review remains at most 186.8525 SUs;
combined B maximum **187.2675 SUs**, below the 192 ceiling. Persistent ledger
charged plus reserved after replacement submission: **89.733611 SUs**; the large
GPU arrays remain unreserved/unsubmitted until the validated controller runs.
The overall 1,200-SU cap remains unchanged. No new scientific results yet.

## October 19 evening: preflight passed and all pilots submitted

At 22:19:56 IST, CUDA job **1080367** was confirmed COMPLETED, exit 0, in
125 seconds. Tiny CUDA training/rollback/checkpoint tests and all five repeated-
draw tests passed. All four real-model/task stream preflights passed, with
adapter restoration and the pinned Llama/Qwen revisions verified. The sandbox,
offline model registry and rendering configuration were exercised successfully.

Controller **1080368** then failed in 10 seconds: Vista explicitly rejected
`sbatch` from its compute host and instructed submission through login resources.
No pilot job was accepted. The recorded first submission response had return
code 1 with that explicit host rejection; no later array attempt existed.

Added a standalone standard-library-only login dispatcher. It imports only the
existing serialization/accounting helpers, never models, datasets or numerical
analysis. It verifies the immutable release and successful source-bound CPU/CUDA
receipts, bound runtime configuration hashes, exact existing reservations and
absence of duplicate outputs/jobs before submission. All training, inference,
code execution, item audits, norms and statistical analysis remain compute jobs.
This follows the site's submission policy; no host spoofing or bypass was used.

Using those existing reservations, login dispatch succeeded. Live scheduler and
array accounting verification at **22:24:09 IST** confirmed all 16 elements:

| Array | Stream | Elements | State at observation |
| --- | --- | ---: | --- |
| 1081443 | Llama code | 0–3 | PENDING, Priority |
| 1081444 | Llama math | 0–3 | PENDING, Priority |
| 1081445 | Qwen code | 0–3 | PENDING, Priority |
| 1081446 | Qwen math | 0–3 | PENDING, Priority |
| 1081447 | CPU structural/statistical review | One job | PENDING, afterany of all four arrays |

The arrays permit four concurrent GPUs each, at most 16 total. The review runs
automatically after every array finishes, including failures; incomplete cohorts
stop before confirmation. Slurm supplied no reliable array start estimate at
this observation. No fresh pilot has completed and no new scientific result is
claimed. These accepted jobs persist independently of the SSH connection.

Reconciled all four replacement preparation/preflight/controller jobs before
dispatch. Total B preprocessing/preflight cost across both attempts is 0.9125 SU.
All pending pilot arrays plus review reserve 185.33 SUs, giving a conservative
**186.2425-SU B charged-plus-reserved total**, within 192. Persistent bounded-
additions ledger charged-plus-reserved is **274.038611 SUs**, within 1,200.
Reservations are ceilings, not actual consumed GPU hours.

Login dispatcher SHA256:
`ad72f75299f784804f2336915372c924dacc30d8f96784a931c99aca70b938e5`.
The compute jobs continue to use immutable source `ef78aa1ed056...`; no model or
scientific protocol change was required. Local main now makes the allocated
controller publish a dispatch-ready metadata handoff instead of calling sbatch,
preventing the same host error in future snapshots. Five scheduler/handoff
regression checks passed locally, including host rejection, ambiguous-result
handling, duplicate blocking and a no-subprocess compute handoff. Lint passes.
This local infrastructure update does not replace the frozen running snapshot.

Local dispatch receipt:
`runs/deployment/admission_b_ef78aa1ed056_env1_dispatch.json`.
The remote `login_dispatch.json` and updated `submissions.json` retain exact job
IDs, scheduler responses, prior accounting and dispatcher/source/plan hashes.

## October 21: cohort and review complete

Verified at 10:07:11 IST: all 16 pilot array elements completed with exit 0;
review **1081447** completed with exit 0 in 35 seconds. All 16 GPU allocations
overlapped. First pilot start: October 20, 13:40:59 IST. Final pilot end:
October 21, 02:33:37 IST. Review finished at 04:46:56 IST. The queue is empty.

Structural inspection passed for all 16 parents: K=3 labels and three native
continuation draws per candidate, with zero terminal observations. The report
and supporting artifacts were downloaded and hash-verified. Full interpretation:
[fresh development results](admission_results_20261021.md).

Fresh forced candidate-plus-continuation utility is -2.31pp versus hold, with
both conditional parent and shared-item intervals below zero. H1/H2 learned
gates are approximately +0.16/+0.13pp versus hold, with intervals crossing zero;
no restored forecasting or useful admission advantage over hold/replay has been
established. The privileged headroom proxy is +0.84pp but does not establish
learnability. The predeclared review stop remains in force: no confirmation,
scaling, recursive rollout or additional experiment was submitted.

Final accounting includes every array element, review and earlier failed
infrastructure attempt. B cost:144.033056SUs. Bounded-additions ledger:
231.829167SUs charged/reserved against1,200. About968.17SUs remain under that
project cap; this is not the account's current allocation balance.
