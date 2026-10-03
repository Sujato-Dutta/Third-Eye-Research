---
title: TACC Setup and Working Notes

---

# TACC Setup and Working Notes

## SSH Agent Forwarding

Use SSH agent forwarding so the GitHub private key stays on the laptop, and TACC only requests signatures through the forwarded agent. Do not copy private keys onto Vista, Lonestar6, Stampede3, or other shared systems. GitHub recommends this pattern for server access because the key remains on the local machine. ([GitHub Docs][1])

### Local Machine Setup

1. Create or use a passphrase-protected SSH key locally and add its public key to GitHub.
2. Load the key into the local agent.

macOS:

```bash
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
ssh-add -l
```

Linux:

```bash
ssh-add ~/.ssh/id_ed25519
ssh-add -l
```

3. Put host stanzas in `~/.ssh/config` on the laptop. This forwards your local agent to the TACC login nodes. If you use 1Password or another custom agent globally, explicitly selecting `IdentityAgent SSH_AUTH_SOCK` matters because `IdentityAgent` overrides `SSH_AUTH_SOCK`; OpenSSH supports `IdentityAgent SSH_AUTH_SOCK` to read the socket path from the environment. ([OpenBSD Manual Pages][2])

```sshconfig
Host vista vista.tacc.utexas.edu login*.vista.tacc.utexas.edu
  HostName vista.tacc.utexas.edu
  User YOUR_TACC_USERNAME
  ForwardAgent yes
  IdentityAgent SSH_AUTH_SOCK
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
  AddKeysToAgent yes
  UseKeychain yes

Host ls6 ls6.tacc.utexas.edu login*.ls6.tacc.utexas.edu
  HostName ls6.tacc.utexas.edu
  User YOUR_TACC_USERNAME
  ForwardAgent yes
  IdentityAgent SSH_AUTH_SOCK
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
  AddKeysToAgent yes
  UseKeychain yes

Host stampede3 stampede3.tacc.utexas.edu login*.stampede3.tacc.utexas.edu
  HostName stampede3.tacc.utexas.edu
  User YOUR_TACC_USERNAME
  ForwardAgent yes
  IdentityAgent SSH_AUTH_SOCK
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
  AddKeysToAgent yes
  UseKeychain yes
```

### Verification

On the laptop:

```bash
ssh-add -l
ssh vista    # or ls6 / stampede3
```

On the TACC login node:

```bash
echo "$SSH_AUTH_SOCK"
ssh-add -l
ssh -T git@github.com
```

If `SSH_AUTH_SOCK` is non-empty and `ssh-add -l` lists keys, forwarding is working.

## TACC Shell Helpers

Add these to `~/.bashrc`, `~/.zshrc`, or the shell startup file you actually use on TACC:

```bash
# Compact Slurm partition/node summary:
# PARTITION, availability, and nodes in allocated/idle/offline/total states.
alias nodes='sinfo -S+P -o "%18P %8a %20F"'

# User-local helper scripts.
export PATH="$HOME/.local/bin:$PATH"

# VISTA_IDEV_MONITOR_AUTOSTART
# Start only for interactive shells inside a Slurm allocation, and only if the
# optional monitor helper exists.
if [[ $- == *i* ]] && [[ -n "${SLURM_JOB_ID:-}" ]] && [[ -x "$HOME/.local/bin/vista-idev-monitor" ]]; then
  "$HOME/.local/bin/vista-idev-monitor" start "$$" >/dev/null 2>&1 || true
fi

# LLM/API secrets: keep these out of repos.
[ -f "$HOME/.config/llm/secrets.env" ] && . "$HOME/.config/llm/secrets.env"

# TACC account/quota summary.
if [ -x /usr/local/etc/taccinfo ]; then
  alias taccinfo='/usr/local/etc/taccinfo'
fi
```

Useful commands:

```bash
nodes                 # queue/node summary, using sinfo
taccinfo              # project balances, SU usage, and filesystem quotas
qlimits               # current queue limits; these can change
squeue -u "$USER"     # your pending/running Slurm jobs
showq -u              # TACC queue view for your jobs
```

TACC documents the `nodes` command's underlying `sinfo` format as a compact queue-status summary, and documents `/usr/local/etc/taccinfo` as the command-line source for project balances and disk quotas. ([TACC Job Management][4], [TACC Common User Issues][5])

## Compute Nodes and the Second Hop

If someone does `ssh login -> idev -> ssh compute`, agent forwarding must also be enabled on the login-to-compute SSH hop. Otherwise GitHub access can work on the login node but fail on the compute node.

Two good patterns:

### A. ProxyJump from the Laptop

Connect directly from the laptop to the allocated compute node through the login node:

```bash
ssh -J vista -A i614-054
```

Replace `i614-054` with the node you actually own. This keeps forwarding end-to-end.

### B. Forward Agent on TACC Internal Hops

Add a small `~/.ssh/config` on the TACC side:

```sshconfig
Host i* c* r*
  ForwardAgent yes
```

TACC supports SSH'ing to a compute node you own through an active batch job or interactive session, which is also useful for monitoring jobs. ([Vista User Guide][3])

Security note: agent forwarding is powerful. While you are logged into a host, that host can request signatures from your agent. Scope `ForwardAgent yes` to trusted TACC hosts and internal node patterns, not `Host *`. ([GitHub Docs][1])

## `idev` vs. Slurm Batch Jobs

`idev` is for interactive development, debugging, short tests, and manual inspection on compute nodes. It is not a bypass around Slurm: TACC's `idev` submits a Slurm job, waits for an allocation, then places you on the allocated compute node. The default is typically a short single-node development allocation, and options such as `-p`, `-N`, `-n`, and `-m` mirror common Slurm resource requests. ([TACC idev Guide][6])

Use `idev` when you need a live shell:

```bash
idev -p <queue> -N 1 -n 8 -m 60
```

Use `sbatch` for repeatable, long-running, or production work:

```bash
#!/bin/bash
#SBATCH -J myjob
#SBATCH -A <project>
#SBATCH -p <queue>
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 01:00:00
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
module list

ibrun ./run.sh
```

Do not run compute-heavy work on login nodes. Login nodes are for editing, compiling small code, submitting jobs, moving files, and checking status. TACC's Vista guide explicitly shows application launch examples inside a Slurm job script or `idev` session and warns against launching jobs on login nodes. ([Vista User Guide][3])

One important Slurm detail: shell variables such as `$WORK` and `$SCRATCH` do not expand inside `#SBATCH` directives. Use literal paths in directives, or set paths in the shell body of the script after the directives. ([Vista User Guide][3])

## Codex and Claude Code on TACC

Treat Codex, Claude Code, and similar coding agents as interactive development tools with the same login-node rules as a human user.

Recommended pattern:

1. Use the login node for lightweight work: editing, Git operations, reading docs, preparing scripts, and submitting jobs.
2. Start an `idev` session before asking an agent to run tests, compile heavily, process data, train models, or benchmark code.
3. For longer runs, have the agent generate or edit an `sbatch` script, submit it, and monitor it with `squeue`, `showq`, `sacct`, and job logs.
4. Keep API keys and tokens in `~/.config/llm/secrets.env` with restrictive permissions, and source that file from the shell startup file. Never commit secrets.
5. Put caches somewhere with enough quota. `$HOME` is small and easy to fill.

Example cache setup:

```bash
mkdir -p "$WORK/.cache"/{pip,huggingface,uv,npm}

export XDG_CACHE_HOME="$WORK/.cache"
export PIP_CACHE_DIR="$WORK/.cache/pip"
export HF_HOME="$WORK/.cache/huggingface"
export UV_CACHE_DIR="$WORK/.cache/uv"
export npm_config_cache="$WORK/.cache/npm"
```

Before letting an agent launch expensive work, make it answer three checks in the terminal:

```bash
hostname
echo "${SLURM_JOB_ID:-no-slurm-allocation}"
taccinfo
```

If `SLURM_JOB_ID` is empty, the agent is probably on a login node. It should not run the expensive command there.

## Resource Utilization and Monitoring

Be deliberate about resource requests. Requesting extra nodes, GPUs, wall time, or memory can make jobs wait longer and can waste allocation if the code does not use the resources. Queue limits can change, so check `qlimits` before assuming a partition's limits. ([Vista Running Jobs][7])

Practical workflow:

1. Start with a small `idev` run or a short batch job.
2. Measure CPU, memory, GPU, I/O, and wall time.
3. Scale one dimension at a time: tasks, threads, nodes, GPUs, or data size.
4. Use the smallest request that keeps the workload efficient and reliable.
5. Save logs so future runs are based on evidence instead of guesses.

Status and accounting:

```bash
nodes
squeue -u "$USER"
squeue --start -j <jobid>
scontrol show job=<jobid>
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,AllocNodes,AllocCPUS,MaxRSS,ExitCode
seff <jobid>              # if available on the system
```

Live utilization from a node you own:

```bash
ssh <allocated-node>
top                       # press 1 to see per-core CPU use
free -h
df -h
nvidia-smi                # GPU nodes, where available
```

TACC notes that SSH'ing to a node you own is useful for monitoring, including running `top` and pressing `1` to see CPU/thread use. ([Vista User Guide][3])

If utilization is poor, fix the job before scaling up. Common problems are too many idle MPI ranks, too many OpenMP threads, missing CPU/GPU binding, serial preprocessing inside a large allocation, slow shared-filesystem I/O, tiny-file storms, and over-requested wall time. For many independent small tasks, consider a launcher or job array instead of holding a large allocation mostly idle.

## `$HOME`, `$WORK`, `$SCRATCH`, and Temporary Storage

Use the right filesystem for the job:

| Location | Best use | Notes |
| --- | --- | --- |
| `$HOME` | Dotfiles, SSH config, small source files, small durable files | Small quota, backed up on many systems, not for heavy I/O. |
| `$WORK` | Persistent project repos, environments, shared datasets, important outputs | Larger quota and available across TACC systems through Stockyard, but not backed up. Monitor both space and file-count quotas. |
| `$SCRATCH` | Large temporary job input/output and high-volume run directories | Often much larger than `$HOME`/`$WORK`, but check `taccinfo` for the current system. Not backed up and subject to purge policies. Stage important outputs back to `$WORK` or another durable location. |
| `$TMPDIR` or `/tmp` | Node-local temporary files during a job | Local to one node and usually removed after the job. Copy anything important out before exit. |
| Corral | Longer-term shared/archive data when available | Better for durable project storage than scratch; still maintain important copies. |

TACC's system docs describe `$HOME`, `$WORK`, and `$SCRATCH` as available across HPC nodes, with `$HOME` intended for small permanent storage, `$WORK` as the global Stockyard filesystem, and `$SCRATCH` as temporary non-backed-up storage that may be purged. ([Stampede3 System Guide][8], [Vista System Guide][9], [TACC FAQ][10])

Good habits:

```bash
taccinfo
du -sh "$HOME" "$WORK" "$SCRATCH" 2>/dev/null
find "$WORK" -xdev -type f | wc -l
```

Prefer job-specific run directories:

```bash
RUN_DIR="$SCRATCH/runs/${SLURM_JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$RUN_DIR"

rsync -a "$WORK/project/input/" "$RUN_DIR/input/"
cd "$RUN_DIR"

ibrun "$WORK/project/bin/myprogram" input/config.yaml

mkdir -p "$WORK/project/results/${SLURM_JOB_ID:-manual}"
rsync -a "$RUN_DIR/output/" "$WORK/project/results/${SLURM_JOB_ID:-manual}/"
```

Avoid writing millions of tiny files to shared filesystems. Bundle tiny outputs where practical, write temporary intermediates to node-local storage or `$SCRATCH`, and copy only final artifacts to `$WORK`.

[1]: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/using-ssh-agent-forwarding "Using SSH agent forwarding"
[2]: https://man.openbsd.org/ssh_config "ssh_config(5) - OpenBSD manual pages"
[3]: https://docs.tacc.utexas.edu/hpc/vista/ "Vista User Guide"
[4]: https://docs.tacc.utexas.edu/hpc/frontera/jobmanagement/ "TACC Job Management"
[5]: https://docs.tacc.utexas.edu/basics/userissues/ "TACC Common User Issues"
[6]: https://docs.tacc.utexas.edu/software/idev/ "idev: Interactive Development User Guide"
[7]: https://docs.tacc.utexas.edu/hpc/vista/running/ "Vista Running Jobs"
[8]: https://docs.tacc.utexas.edu/hpc/3stampede/system/ "Stampede3 System Guide"
[9]: https://docs.tacc.utexas.edu/hpc/vista/system/ "Vista System Guide"
[10]: https://docs.tacc.utexas.edu/basics/faq/ "TACC FAQ"
