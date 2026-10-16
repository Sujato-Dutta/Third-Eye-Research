# Third Eye: fresh forecasting evidence for the December submission

Recorded October 11, 2026. The intended submission is now the next cycle in
late December, as instructed. The exact public submission date still needs
verification. Earlier October deadlines, protocols, results and failures remain
historical records; they are not retroactively rewritten.

The central question remains whether precommit information predicts the future
value of persistent parameter self-updates well enough to improve selection.
The completed eight-parent expected-value fit is development evidence only.
Neither a reliable forecasting advantage nor an oral-level result is established.

## Stage now authorized: bounded development and reliability pilots

Freeze this declaration before collecting new outcomes:

| Setting | Declaration |
|---|---|
| Backbones | Pinned Qwen3-4B and Llama-3.2-3B-Instruct |
| Streams | Each backbone on math and code |
| Fresh pilot seeds | 9142 and 9143 for each stream |
| Scope | Eight developmental trajectories, one initial state each |
| Candidates | K=3 distinct compositions; unique examples within each batch; overlap between batches permitted |
| Candidate batch size | Two verified corrections, unchanged |
| Candidate update | Frozen A2 configuration: LoRA rank16, LR1e-4, 50 optimizer steps, gradient accumulation8 |
| Continuations | Five outcomes per saved M1: original seed plus offsets50,000,000 /100,000,000 /150,000,000 /200,000,000 |
| Harvesting | Frozen A2: eight attempts, stride16; train-only verified corrections |
| Terminal continuation | Zero executed steps, M2=M1, continuation_available=0; no manufactured training examples |
| Evaluation | Existing immutable selection manifests: target64, OOD64, retention256; math train1024, code train306 |
| Capture | Save per-item verdicts and checkpoint hashes during the original evaluation; no extra scoring pass |
| Commitment | Only the selected M1 is committed; fresh M2 branches never advance the accepted trajectory |
| Parallelism | Eight allocated GH200 nodes maximum for this stage |
| Reservation ceiling | Eight tasks ×36h =288 GPU-node hours, plus short CPU and CUDA validation |
| Expansion | Disabled; inspect pilot_inspection.json and actual accounting before a larger cohort |

Expected outputs are24 candidate records and120 continuation outcomes. Terminal
outcomes count as observed outcomes, not executed updates. A scarcity stop or
failed state remains evidence and cannot silently be counted as complete.
The configuration retains T=5 to permit a separately authorized, consistent
resume; this invocation runs only one state. These pilots are not final held-out
confirmation and do not satisfy Gate2 by themselves.

Submission dependencies are CPU full-suite validation → real CUDA adapter and
five-draw/terminal verification → eight pilots → CPU inspection. All scientific
work runs in Slurm compute allocations. Login nodes handle only staging and
scheduler/metadata operations. Failed validation blocks pilot execution.

Inspection checks all K=3 labels, distinct candidate compositions, within-batch
uniqueness, matched budgets, five matched continuation seeds, checkpoint-bound
per-item outcomes, valid target/OOD/retention metrics and exact terminal identity.
It reports scarcity and within-versus-between candidate variation descriptively.
There is no new terminal-frequency threshold or automated science gate.

## What makes the subsequent study stronger

1. **Freeze a balanced collection and evaluation plan before collecting its
   outcomes.** Allocate whole trajectories to training, development and final
   evaluation, stratified across streams. Keep these pilots in development.
   Choose sample sizes from pilot variance and a declared practical selection
   improvement, with power/precision calculations on allocated CPU nodes.
2. **Train on expected continuation value and its uncertainty.** Compare the
   core expected-value model with matched H1, the earlier H2 model, linear/simple
   feature models, immediate-gain selection, random selection and no-update.
   Match information access, tuning effort and cost; historical failures remain
   documented. Choose checkpoints and decision thresholds on development only.
3. **Score selections independently.** Fix decisions before fresh continuation
   seeds and disjoint scoring items are exposed. Report paired selection utility,
   harmful updates, retention, calibration and intervals clustered by whole
   trajectory. Separate continuation variation from evaluation-item uncertainty.
   Any oracle-like comparison must select on one draw set and score on another.
4. **Test an explanation.** After the pilots, predeclare one controlled update
   magnitude or candidate-diversity intervention with matched comparisons. An
   association alone does not establish why rankings change.
5. **Demonstrate recursive usefulness and practical cost.** Compare fixed learned
   selectors through T=5 on untouched trajectories. Measure GPU-node hours,
   inference tokens, training steps, continuation availability and wall time.
   An uncertainty-based no-update action must be tested against the same no-update
   baseline, rather than credited for avoiding inherently harmful forced updates.
6. **Broaden only after the core comparison works.** A properly powered third
   backbone is more valuable than another four-state sanity check. Larger models
   remain ineligible before the existing Gate2 requirements are met.

Prior-work comparisons must cover [TuneAhead](https://arxiv.org/abs/2606.17660)
and [SEAL](https://arxiv.org/abs/2506.10943), identifying which information and
adaptation budgets can be matched. Repeated-label supervision and uncertainty
alone should not be described as unprecedented. The useful contribution must
be established by predictive performance, selection outcomes and explanation.

## Time and resource planning

The first stage has a36h per-task limit, not a promised36h elapsed completion:
queue delay, partial allocation and failed checks affect elapsed time. Once
pilots finish, use Slurm accounting and harvest/evaluation logs to project larger
cohorts. Do not extrapolate from GPU count alone.

The October11 resource check showed2,204 available SUs; `/work` used941.5GB of
1,024GB. New pilot checkpoints stay on scratch. A larger collection needs a
durable storage plan and a separately bounded SU budget. Do not delete historical
archives to create space without authorization.

A provisional internal target is to freeze experiments and claims well before
late December, leaving time for statistical review, reproducibility and writing.
Exact milestones follow measured pilot costs and the verified submission date.
The paper's strength will follow fresh evidence; acceptance or an oral slot
cannot be promised from the implementation or plan.
