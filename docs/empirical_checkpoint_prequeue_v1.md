# E2 scheduler handoff: queue before execution is eligible

October 8 operational change only. The complete CPU/CUDA validation barrier,
E2 source, comparisons, sample and scientific budgets stay unchanged.

The scheduler currently estimates a substantial wait for CPU validation. Queue
all eight collection jobs with explicit `afterok` dependencies on both CPU and
CUDA validation, rather than entering the GPU queue only after a third CPU
launch-controller allocation. This does not permit experimental execution before
validation succeeds. Every collection process additionally checks both matching
passed receipts before constructing a backend. No CUDA dependency is bypassed.

A scheduler-only login service checks frozen small metadata, records each
submission immediately and retires only the unused pending E2 launch-controller
job. It imports no scientific package and performs no fitting or evaluation.
All pending/running GPU jobs, including validation and recoveries, count toward
the ceiling of twenty. The service cancels only its blocked collection entries
if validation fails and preserves that failure; no trajectory is replaced.

The CPU confirmation keeps its `afterany` dependency on all eight collections.
The existing E2 checkpoint callback submits controls as selected states become
available. Confirmation and noise review remain automatic, with incomplete
coverage reported explicitly and a scientific review stop. Handoff source hash,
validation job IDs, retired controller ID and all new job IDs are durable receipts.
