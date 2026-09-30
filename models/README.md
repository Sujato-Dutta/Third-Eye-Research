# Model artifacts

Adapters are written under each run's `states/` and `accepted/` directories.
Place manually retained checkpoints under `models/` if desired. Model weights
are ignored by Git. Checkpoints include PEFT adapter weights, tokenizer files,
the exact protocol, adapter/weight hashes, resolved base revision, dependency
versions, and parent/candidate provenance. Load the same frozen base model and
apply the cumulative adapter; no full-model merges are required.

In-memory rollback stores only adapter tensors on CPU. Each update resets
AdamW momentum; accepted adapters carry all parameter changes across the chain.
Resume is from a completed **accepted** checkpoint, into a fresh run directory.
Recovery within a partially executed optimizer update is not implemented.
