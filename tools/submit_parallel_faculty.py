"""Scheduler-only held submissions; preserve a durable receipt for every job."""
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path('/dgxb_home/muef2320/third_eye_account_audit_20261112/parallel')


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def main():
    if os.getuid() != 1189:
        raise RuntimeError('Named temporary account required')
    manifest = json.loads((ROOT / 'task_manifest.json').read_text())
    destination = ROOT / 'plan.json'
    plan = json.loads(destination.read_text()) if destination.exists() else {
        **manifest, 'armed': False, 'faculty_gpu_limit': 2,
        'scientific_settings_changed': False, 'owner_data_cleanup': False}
    if plan['armed']:
        raise RuntimeError('Do not resubmit an armed delegation')
    for task in plan['tasks']:
        if task.get('worker_job'):
            continue
        result = subprocess.run(['sbatch', '--parsable', '--hold', '--dependency=afterok:35441',
                                 str(ROOT / ('worker-' + task['key'] + '.slurm'))],
                                text=True, capture_output=True)
        task['submit_stdout'], task['submit_stderr'] = result.stdout, result.stderr
        if result.returncode or not re.fullmatch(r'\d+(?:;[A-Za-z0-9_.-]+)?\s*', result.stdout):
            write(destination, plan)
            raise RuntimeError('Submission failed; partial plan preserved')
        task['worker_job'] = result.stdout.strip().split(';')[0]
        write(destination, plan)
    print(json.dumps({'plan': str(destination), 'held_jobs':
                      {t['key']: t['worker_job'] for t in plan['tasks']}}), flush=True)


if __name__ == '__main__':
    main()
