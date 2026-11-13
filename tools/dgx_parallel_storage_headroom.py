"""Retire one verified duplicate transport bundle, preserving extracted results."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

ROOT = Path('/dgxa_home/se23uari167/third_eye_dgx_20261026')
EXPECTED = 'a50d822016c319163512cea1eb578767e918ed239d458cc9ae63ed5e93b848e9'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def usage():
    result = subprocess.run(['du', '-x', '-B1', '-s', str(Path.home())],
                            text=True, capture_output=True, check=True)
    return int(result.stdout.split()[0])


def main():
    if (os.getuid() != 1231 or not os.environ.get('SLURM_JOB_ID')
            or 'login' in socket.gethostname().lower()):
        raise RuntimeError('Owning account and Slurm allocation required')
    target = ROOT / 'transfers/payload.tar.gz'
    receipt = ROOT / 'runs/positive_regime_20261109/temporary_worker_20261112/transport_retirement.json'
    if receipt.exists():
        raise FileExistsError('Preserve earlier maintenance receipt')
    if (target.is_symlink() or target.resolve(strict=True) != target
            or target.stat().st_uid != os.getuid() or target.stat().st_size != 818572008
            or digest(target) != EXPECTED):
        raise ValueError('Duplicate transport bundle failed exact source checks')
    prepared = json.loads((ROOT / 'prepared.json').read_text())
    if (not prepared['passed'] or digest(ROOT / 'imported/registry.json') != prepared['registry_sha256']
            or not (ROOT / 'imported/payload').is_dir()):
        raise ValueError('Verified extracted study data must remain present')
    record = {'passed': False, 'job_id': os.environ['SLURM_JOB_ID'], 'target': str(target),
              'sha256': EXPECTED, 'before_home_bytes': usage(),
              'verified_local_backup': 'F:/Third-Eye-Research/runs/deployment/dgx_transfer_20261026/payload.tar.gz',
              'extracted_results_preserved': True, 'scientific_content_changed': False}
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    target.unlink()
    record.update({'passed': True, 'after_home_bytes': usage()})
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
