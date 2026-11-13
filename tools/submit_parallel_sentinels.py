"""Owning-account held handoff jobs, with durable per-task submission receipts."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

ROOTS = {
    'positive': (1231, Path('/dgxa_home/se23uari167/third_eye_dgx_20261026/runs/positive_regime_20261109')),
    'horizon': (1339, Path('/dgxa_home/se23ucse205/third_eye_horizon_20261109')),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=ROOTS)
    args = parser.parse_args()
    uid, root = ROOTS[args.kind]
    if os.getuid() != uid:
        raise RuntimeError('Owning account required')
    control = root / 'temporary_worker_20261112'
    destination = control / 'plan.json'
    plan = json.loads(destination.read_text())
    if plan.get('armed'):
        raise RuntimeError('Do not resubmit an armed delegation')
    for task in plan['tasks']:
        if task['kind'] != args.kind or task.get('sentinel_job'):
            continue
        result = subprocess.run(['sbatch', '--parsable', '--hold',
                                 '--dependency=afterany:' + task['worker_job'],
                                 str(control / ('sentinel-' + task['key'] + '.slurm'))],
                                text=True, capture_output=True)
        task['sentinel_submit_stdout'], task['sentinel_submit_stderr'] = result.stdout, result.stderr
        if result.returncode or not re.fullmatch(r'\d+(?:;[A-Za-z0-9_.-]+)?\s*', result.stdout):
            raise RuntimeError('Handoff submission failed; inspect existing receipts')
        task['sentinel_job'] = result.stdout.strip().split(';')[0]
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(plan, indent=2, sort_keys=True) + '\n')
        temporary.replace(destination)
    print(json.dumps({'kind': args.kind, 'held_handoffs':
          {t['key']: t['sentinel_job'] for t in plan['tasks'] if t['kind'] == args.kind}}), flush=True)


if __name__ == '__main__':
    main()
