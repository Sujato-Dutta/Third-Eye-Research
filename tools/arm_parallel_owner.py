"""Scheduler-only retirement of pending originals and registration of handoffs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOTS = {
    'positive': (1231, Path('/dgxa_home/se23uari167/third_eye_dgx_20261026/runs/positive_regime_20261109')),
    'horizon': (1339, Path('/dgxa_home/se23ucse205/third_eye_horizon_20261109')),
}


def command(args):
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def state(job):
    result = command(['sacct', '-S', '2026-11-09', '-j', job, '-X', '-n', '-P', '--format=JobID,State'])
    for line in result.splitlines():
        fields = line.split('|')
        if fields[0] == job:
            return fields[1].split()[0].rstrip('+')
    return 'UNKNOWN'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=ROOTS)
    args = parser.parse_args()
    uid, root = ROOTS[args.kind]
    if os.getuid() != uid:
        raise RuntimeError('Owning account required')
    control = root / 'temporary_worker_20261112'
    receipt = control / 'retirement_completed.json'
    if receipt.exists():
        raise FileExistsError('Inspect existing retirement receipt instead of repeating it')
    proof = json.loads((control / 'validation_completed.json').read_text())
    if not proof['passed']:
        raise ValueError('Completed compute-node validation required')
    plan = json.loads((control / 'plan.json').read_text())
    selected = [task for task in plan['tasks'] if task['kind'] == args.kind]
    held = []
    for task in selected:
        current = command(['squeue', '-h', '-j', task['original_job'], '-r', '-o', '%T'])
        if current != 'PENDING':
            for job in held:
                command(['scontrol', 'release', job])
            raise RuntimeError('Refuse to move a job that is no longer pending: ' + task['original_job'])
        command(['scontrol', 'hold', task['original_job']])
        held.append(task['original_job'])
    registry = None
    if args.kind == 'positive':
        originals = [f'35053_{i}' for i in range(2, 14) if state(f'35053_{i}') != 'COMPLETED']
        registry = {'original_tasks': originals,
                    'tasks': {task['key']: task['sentinel_job'] for task in selected}}
        write(control / 'review_dependencies.json', registry)
        command(['scontrol', 'update', 'JobId=35054',
                 'Dependency=afterok:' + ':'.join(originals + list(registry['tasks'].values()))])
    else:
        for task in selected:
            command(['scontrol', 'update', 'JobId=' + task['downstream_job'],
                     'Dependency=afterok:' + task['sentinel_job']])
    record = {'passed': False, 'kind': args.kind, 'held_original_jobs': held,
              'tasks': selected, 'review_dependencies': registry,
              'scientific_settings_changed': False, 'running_jobs_moved': 0}
    write(control / 'retirement_progress.json', record)
    for task in selected:
        command(['scancel', '--state=PENDING', task['original_job']])
    for attempt in range(30):
        if all(state(task['original_job']) == 'CANCELLED' for task in selected):
            break
        time.sleep(.25)
    else:
        raise RuntimeError('Retirement not confirmed; faculty workers must remain held')
    record['passed'] = True
    record['retired_before_start'] = True
    write(receipt, record)
    for task in selected:
        command(['scontrol', 'release', task['sentinel_job']])
    print(json.dumps({'passed': True, 'kind': args.kind,
          'retired_pending_jobs': held, 'handoffs_registered': len(selected)}), flush=True)


if __name__ == '__main__':
    main()
