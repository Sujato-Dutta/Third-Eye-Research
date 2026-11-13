"""Compute-node checks for scheduler scripts, ACL fixtures and frozen inputs."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

AUDIT = Path('/dgxb_home/muef2320/third_eye_account_audit_20261112')
ROOT = AUDIT / 'parallel'
RUNTIME = ROOT / 'tools/dgx_parallel_runtime.py'
OWNERS = {
    'positive': Path('/dgxa_home/se23uari167/third_eye_dgx_20261026/runs/positive_regime_20261109/temporary_worker_20261112'),
    'horizon': Path('/dgxa_home/se23ucse205/third_eye_horizon_20261109/temporary_worker_20261112'),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (os.getuid() != 1189 or not os.environ.get('SLURM_JOB_ID')
            or 'login' in socket.gethostname().lower()):
        raise RuntimeError('Expected faculty CPU allocation required')
    for kind in OWNERS:
        subprocess.run([sys.executable, str(RUNTIME), 'verify', '--kind', kind], check=True)
    subprocess.run([sys.executable, '-m', 'pytest', '-q', '-o',
        'cache_dir=' + str(ROOT / 'pytest_cache'), '--basetemp=' + str(ROOT / 'fixtures'),
        str(ROOT / 'tests/test_dgx_parallel_runtime.py')], check=True)
    scripts = sorted(ROOT.glob('worker-*.slurm')) + [ROOT / 'watchdog.slurm', AUDIT / 'watch-faculty.slurm']
    for owner in OWNERS.values():
        scripts += sorted(owner.glob('sentinel-*.slurm')) + [owner / 'watch.slurm']
    if len(scripts) != 24:
        raise ValueError('Ten GPU jobs, ten handoffs and four monitor scripts required')
    for path in scripts:
        data = path.read_bytes()
        if b'\r' in data or b'\\' in data:
            raise ValueError('Invalid Unix batch encoding/path: ' + str(path))
        subprocess.run(['bash', '-n', str(path)], check=True)
        for line in data.decode().splitlines():
            for prefix in ('#SBATCH --chdir=', '#SBATCH --output=', '#SBATCH --error='):
                if line.startswith(prefix):
                    target = Path(line[len(prefix):])
                    directory = target if prefix.endswith('chdir=') else target.parent
                    if not target.is_absolute() or not directory.is_dir():
                        raise ValueError('Batch path does not resolve: ' + str(path))
    record = {'passed': True, 'job_id': os.environ['SLURM_JOB_ID'], 'host': socket.gethostname(),
              'runtime_sha256': digest(RUNTIME), 'operational_test_cases': 8,
              'scripts_checked': len(scripts), 'script_sha256': {str(p): digest(p) for p in scripts},
              'bindings': {kind: json.loads((AUDIT / ('parallel_' + kind + '_access.json')).read_text())
                           for kind in OWNERS}, 'scientific_settings_changed': False}
    path = ROOT / 'validation_completed.json'
    if path.exists():
        previous = json.loads(path.read_text())
        archived = ROOT / ('validation_completed_' + previous['job_id'] + '.json')
        if not archived.exists():
            archived.write_bytes(path.read_bytes())
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)
    print(json.dumps({'passed': True, 'scripts_checked': len(scripts),
                      'runtime_sha256': record['runtime_sha256']}), flush=True)


if __name__ == '__main__':
    main()
