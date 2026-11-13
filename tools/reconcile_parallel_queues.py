"""Remove duplicate unstarted setup jobs; preserve canonical Slurm bindings."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOTS = {
    'positive': (1231, Path('/dgxa_home/se23uari167/third_eye_dgx_20261026/runs/positive_regime_20261109/temporary_worker_20261112')),
    'horizon': (1339, Path('/dgxa_home/se23ucse205/third_eye_horizon_20261109/temporary_worker_20261112')),
    'faculty': (1189, Path('/dgxb_home/muef2320/third_eye_account_audit_20261112/parallel')),
}


def command(args):
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=ROOTS)
    args = parser.parse_args()
    uid, root = ROOTS[args.kind]
    if os.getuid() != uid:
        raise RuntimeError('Owning account required')
    plan = json.loads((root / 'plan.json').read_text())
    rows = command(['squeue', '-u', str(uid), '-h', '-o', '%i|%T|%j'])
    jobs = {parts[0]: {'state': parts[1], 'name': parts[2]}
            for line in rows.splitlines() if (parts := line.split('|')) and len(parts) == 3}
    if args.kind == 'faculty':
        keep = {task['worker_job'] for task in plan['tasks']}
        prefix = 'te-worker-'
    else:
        selected = [task for task in plan['tasks'] if task['kind'] == args.kind]
        keep = {task['sentinel_job'] for task in selected}
        prefix = 'te-handoff-'
        for task in selected:
            if jobs.get(task['sentinel_job'], {}).get('state') != 'PENDING':
                raise RuntimeError('Canonical handoffs must remain unstarted during reconciliation')
            command(['scontrol', 'update', 'JobId=' + task['sentinel_job'],
                     'Dependency=afterany:' + task['worker_job']])
    cancelled = []
    for job, data in jobs.items():
        if data['name'].startswith(prefix) and job not in keep:
            if data['state'] != 'PENDING':
                raise RuntimeError('Do not cancel a running scientific task')
            command(['scancel', '--state=PENDING', job])
            cancelled.append(job)
        elif data['name'] == 'te-storage-' + args.kind:
            command(['scancel', job])
            cancelled.append(job)
    watcher = root.parent / 'watch-faculty.slurm' if args.kind == 'faculty' else root / 'watch.slurm'
    watch_job = command(['sbatch', '--parsable', str(watcher)]).split(';')[0]
    report = {'passed': True, 'kind': args.kind, 'cancelled_duplicate_or_obsolete_setup_jobs': cancelled,
              'retained_canonical_jobs': sorted(keep), 'storage_watch_job': watch_job,
              'running_scientific_jobs_touched': False}
    path = root / 'queue_reconciliation.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
