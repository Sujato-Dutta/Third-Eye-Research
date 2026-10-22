# Independent scoring provenance and inference-cost check

Declared October 21 after the exploratory admission audit, before these jobs.
This is a bounded precision/cost development check, not fresh confirmation.
Original protocols, outcomes, gates and policy selection remain unchanged.
No final-suite outcomes will be generated, extracted into analysis or used to
choose a policy. Historical artifacts are read only to identify recorded item
use; existing prediction/correctness payloads are ignored.

## Fixed sequence

1. Validate an immutable source release with the full CPU test suite and lint.
2. In a CPU allocation, verify current selection/final manifest checksums and
   overlap, enumerate benchmark/source identities and prompt lengths, and scan
   historical JSON/JSONL artifacts under project `runs` and both amendment run
   directories. Match IDs, normalized rendered-prompt hashes, raw-source hashes
   and realized final-evaluation manifest hashes. A planned manifest reference
   alone is not evaluation evidence. Full correction examples also count as
   exposure. Archive searched artifact hashes and exact matched filenames;
   unreadable artifacts or unidentified final-evaluation references remain
   unresolved. Check all eight benchmark adapter files/configs before GPU work.
3. Run four independent inference benchmarks, at most four GPUs concurrently:
   Llama code, Llama math, Qwen code, Qwen math. Use the existing seed-12142
   initial parent and the first stored candidate's original continuation
   (row zero, draw zero). No outcome determines the checkpoint choice.
4. On each checkpoint, evaluate 16 development examples per role (target, OOD,
   retention), spaced across sorted prompt-character lengths including both
   extremes, breaking ties by ID. Use the exact saved configuration, max tokens,
   batching, fixed rendering, greedy evaluation and approved verifiers. Warm up
   with eight generated tokens on one development prompt and record its time.
   Measure loading/restoration, generation, verification and generated lengths.
   Verify adapters remain unchanged. Do not store completions or summarize
   correctness. This is 96 timed item-checkpoint evaluations per stream, plus
   separately recorded warmup, with zero optimizer steps.
5. A CPU review requires four successful benchmark receipts, reports timing and
   conditional suite-size extrapolations with a 25% margin, then stops. No
   confirmation, policy promotion, scaling or recursive job is launched.

## Interpretation boundaries

An exposure hit disqualifies an untouched-item claim. No recorded hit establishes
only absence in the searched surviving artifacts, not absence of deleted,
external, aggregate-only or unlogged use, nor absence of pretraining exposure.
Do not automatically designate a pool sealed based on the scan.

Timing on development examples estimates inference cost; it does not produce
independent accuracy evidence, new power, or a fully priced confirmation. Input
and completion lengths and verifier costs can shift. In particular, code final
OOD is HumanEval while code development OOD uses MBPP validation. Do not assume
the same distribution, precision or measured per-item cost. Price all candidate,
continuation and parent scores, model loads, checks and retries in a later design.

The next review must decide whether genuinely independent scoring items and an
adequately powered balanced design fit the cap, dropping optional recursion
before weakening confirmation. The exploratory backbone-H2 baseline remains a
development candidate; it has not replaced the predeclared primary H1 rule.

## Budget and operations

Reserve 0.11 SU for CPU validation, 0.33 for a one-hour CPU provenance audit,
8 SUs for four two-hour GH GPU benchmarks, and 0.165 for the half-hour CPU review:
**8.605 SUs maximum**, charged to the existing B-development envelope. B's
previous cost is 144.1981 of 192 SUs; total project ledger is 231.9942 of 1,200.
Reserve every job before submission and reconcile actual minimum billing and
failed/cancelled allocations afterward. Never retry automatically.

Submit only from login using scheduler/file/accounting operations. All suite
checks, artifact scans, testing, inference and numerical review run in allocated
Slurm compute jobs. Dependencies are full validation → provenance → four timing
jobs → CPU review. Invalid dependencies cancel downstream jobs. Authentication
ending does not affect submitted Slurm jobs.
