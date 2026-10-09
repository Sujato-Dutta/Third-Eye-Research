# E1 structural replay failure: proposed checkpoint-based fallback

Proposal only, October8. No fallback trajectories or measurements are launched
by this document. Preserve E1's failed replay, native-evaluator comparison and
probe replay. Do not replace the sampled parent or relax its hash requirement.

The first original parent hash matches, but its saved-batch M1 replay does not.
Replaying the complete original feature/probe path produces the same new M1
hash as replay without it, and both differ from the original. Original branch
weights were pruned, so an exact original M1 cannot currently be recovered.
The collector and native evaluator agree, but their code-OOD parent total is
29/64 versus the original28/64. Package versions, source proof, revision and
GPU model match. The underlying cause has not yet been established.

The deadline-focused fallback is to **collect the already planned balanced
cohort and measure reliability from its actual saved checkpoints**. This changes
the supplement's execution order and reliability sample; it does not change
the A2 labeling/training protocol or the old results. It needs a recorded
scientific decision before releasing jobs, because E1 required a clean initial
noise review before replication.

Keep all eight planned T5 trajectories, seeds6142/6143 in each of four streams,
and all frozen methods. Predeclare eight independent reliability parents:
generation0 of seed6142 and generation4 of seed6143 per stream. This spans early
and late recursion with one parent per trajectory; it has eight independent
units instead of E1's twelve. Do not select by outcomes, swap failed seeds or
make success depend on a post-hoc terminal-frequency threshold.

For each parent, load its actual retained parent/M1/M2 weights and require their
stored hashes to match. Record aligned item predictions and verifier outcomes;
separate prediction repeatability from repeated verification of identical code.
Use the same planned fixed-batch optimizer repeat and two new standardized
continuation seeds, offsets50,000 and50,000,000/100,000,000 respectively. Keep K=3,
size2, fifty optimizer steps, eight correction attempts, training portions,
evaluation roles and M2=M1 terminal semantics unchanged. Do not overwrite single
continuation labels with averages. Disclose any original-versus-repeat metric
difference and its verification variability rather than claiming exact totals.

Early reliability jobs can overlap later cohort generations. Late reliability
jobs depend on their exact trajectory completing. Count the three original
recoveries, eight new trajectories and up to eight reliability jobs within the
twenty-GPU ceiling. Stop if retained weights, corrections, budgets or item
identities are inconsistent. Keep all failures and stop for scientific review
after reliability and balanced confirmation.

Planning cost: about174 GPU node-hours for the eight trajectories and roughly
53-93 for eight-parent repeats, plus validation/recovery. These are extrapolations,
not measured fallback runtimes. Approximately24-40 hours for the cohort and up
to8-18 hours for late repeats after allocation, with queue delays and longer
tails potentially exceeding October11. Fitting methods on fresh outcomes,
8B scaling and online rollout remain disabled.
