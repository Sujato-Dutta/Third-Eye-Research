# Data contracts

The evaluation component owns dataset preparation and official verifiers. This
training runner consumes immutable JSONL splits, not raw benchmark test sets.

Keep downloaded datasets in `data/raw/`, prepared splits in `data/processed/`,
and generated manifests in `data/manifests/`; all three are ignored by Git.
`data/examples/` contains synthetic arithmetic fixtures for interface checks.
They are not research data and must never support a paper claim.

Each JSONL record has `id`, `prompt`, `answer`, `split`, `task`, `difficulty`,
and optional `metadata`. IDs and normalized prompts must be disjoint across
all four splits. Roles are `train`, `target_dev`, `ood_dev`, `retention_dev`.
`task` is `math`, `code`, or `exact`. Use the same difficulty strata for matched
candidates. Final test roles are rejected by the meta-label runner.

OOD/retention proxy **development** sets are required for online features and
meta-labels. Keep MATH-500/HumanEval final evaluation and the final retention
test anchor outside this runner. Do not relabel official test data as dev.
The evaluation owner must additionally audit source IDs and semantic overlap;
the runner's exact normalized-prompt check cannot detect paraphrases.

The reference `answer` is supplied only to verification and development loss
evaluation. Candidate SFT targets must be the current model's verified
self-generated completions, never copied reference answers.
