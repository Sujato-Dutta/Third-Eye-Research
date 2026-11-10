# Additional DGX worker — November 9, 2026

Account: `se23ucse205`; home: `/dgxa_home/se23ucse205`.
Authentication reads the locally saved `.env` without displaying credentials.
The previously inspected shared account is retired and must not be used.

## Verified limits

- Student association: one GPU allocation, 32 CPUs and 256 GB RAM in aggregate.
- Partitions: `cpu_student` and `gpu_student`; maximum duration seven days.
- Operating storage limit: **50,000,000,000 bytes across the whole account**.
- Warn at 45 GB; refuse new work at 48 GB, including its requested output reserve.
- A missing enforced filesystem quota does not relax this limit.

The initial compute-node audit `35058` found 50,231,947,264 allocated bytes,
49,823,059,497 apparent bytes, and no running or pending pre-existing jobs.
Accounting conservatively uses the larger byte count. The audit also confirmed
read access to the original account's verified Qwen weights and executable
environment; it did not alter them. Reuse requires a compute-node runtime check.

## Authorized reset

The user explicitly requested deletion of the account's existing data. CPU job
`35059` performs that reset on `dgxa`, with exact username, UID and resolved-home
checks. It refuses deletion if another account job exists or a target contains
a mounted directory. Recursive deletion does not follow symbolic links.

All existing projects, model downloads, caches and old environments are deletion
targets. Preserve SSH/shell configuration and the small audit directory. The
original account, its research artifacts and active experiments are outside the
deletion scope. The before/after receipt records each target's completion.

Job `35059` stopped at a read-only cached directory after removing initial cache
targets. The failed log and incomplete receipt were preserved. Follow-up CPU job
`35060` added owner permissions only within the approved, owned deletion
directories and completed the same target list. Usage fell from 50,231,975,936
to **118,784 bytes**; the post-reset whole-account scan passed the 50-GB guard.
The reset receipt and post-cleanup quota report are retained locally.

Storage watcher `35061` checks usage every twenty minutes for 167 hours in
a one-CPU Slurm allocation. It writes audit metadata only and must be renewed
after expiry if the account remains in use. Future research jobs must also run
the admission check before allocating output space; periodic monitoring alone
does not prevent an unrelated writer from filling the account.

Campaign directory: `/dgxa_home/se23ucse205/third_eye_account_audit_20261109`.
Local transport queue: `runs/deployment/dgx_replacement_requests_20261109/`.
Local receipts: `runs/deployment/dgx_replacement_20261109/`.

The subsequent [longer-horizon extension](long_horizon_execution_20261109.md)
uses this worker for four paired seeds per backbone and T=3/T=5 scoring, while
the primary confirmation continues independently. The original study's seeds,
endpoints and recipes remain unchanged by account preparation.
All storage scans, deletion, software checks and future computation require Slurm
compute allocations. Login-node work is limited to authentication, staging,
small metadata reads and scheduling. Storage monitoring must not be described
as a server-enforced quota or as controlling unrelated external writers.
