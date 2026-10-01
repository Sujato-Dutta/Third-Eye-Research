# Data protocol

`experiments/prepare_data.py` downloads pinned Hub revisions and writes an
immutable benchmark version. Run it on a scheduled CPU node. The JSONL schema
is `id`, `prompt`, `answer`, `split`, `task`, `difficulty`, `metadata`.

Selection uses only `train`, `target_dev`, `ood_dev`, `retention_dev` through
`selection.json`. Final evaluation uses separate `target_test`, `ood_test`,
`retention_test` files through `final.json`. The correction and training APIs
reject every role other than `train`.

| Family | Update pool | Target development | Development proxy | Final target / OOD |
| --- | --- | --- | --- | --- |
| Math | GSM8K train subset | Disjoint GSM8K train holdout | MATH train | GSM8K test / all MATH-500 |
| Code | MBPP full train subset | Disjoint MBPP train holdout | MBPP validation | MBPP full test / all HumanEval |

The code proxy is a held-out MBPP proxy, not HumanEval OOD performance. MMLU
validation supplies the fixed 256-example retention development anchor; a
separate 256-example MMLU test sample supplies final retention. Defaults request
256 training prompts and 64 development examples. Exact audited counts,
excluded duplicates, source splits, seed and resolved commits are archived in
`provenance.json`. Do not silently treat a requested size as the actual count.

Audit uses source IDs, normalized raw prompts before instructions, and
five-token-shingle Jaccard >=0.85. Final sets have priority when overlapping
development examples are removed. This audits observable overlap; it cannot
establish absence from a backbone's pretraining corpus.

Manifest paths are relative to the manifest for portability. Hashes detect
changes after freezing. The example files are synthetic interface fixtures,
not benchmark data and not research evidence. Downloaded data stays untracked.

Sources: [GSM8K](https://huggingface.co/datasets/openai/gsm8k),
[MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp),
[MATH](https://huggingface.co/datasets/EleutherAI/hendrycks_math),
[MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500),
[HumanEval](https://huggingface.co/datasets/openai/openai_humaneval),
[MMLU](https://huggingface.co/datasets/cais/mmlu).
