"""Read-only inventory of an authorized temporary account, inside Slurm only."""
import datetime
import json
import os
from pathlib import Path
import pwd
import shutil
import socket
import subprocess


def command(args, timeout=180):
    result = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
    return {'stdout': result.stdout, 'stderr': result.stderr, 'exit_code': result.returncode}


def main():
    home = Path.home().resolve()
    if (not os.environ.get('SLURM_JOB_ID') or 'login' in socket.gethostname().lower()
            or os.getuid() != 1189 or home != Path('/dgxb_home/muef2320')):
        raise RuntimeError('Expected account and Slurm compute allocation required')
    output = home / 'third_eye_account_audit_20261112'
    output.mkdir(exist_ok=True)
    roots = []
    for candidate in (home, Path('/dgxa_home/muef2320'), Path('/dgxb_home/muef2320')):
        if not candidate.exists() or any(os.path.samefile(candidate, old) for old in roots):
            continue
        if candidate.stat().st_uid != os.getuid():
            raise RuntimeError('Storage root is not owned by this account')
        roots.append(candidate)
    storage = []
    total = 0
    large = []
    for root in roots:
        allocated = command(['du', '-x', '-B1', '-s', str(root)])
        apparent = command(['du', '-x', '-B1', '--apparent-size', '-s', str(root)])
        breakdown = command(['du', '-x', '-B1', '--max-depth=2', str(root)])
        if any(item['exit_code'] for item in (allocated, apparent, breakdown)):
            raise RuntimeError('Incomplete storage scan; no new model staging permitted')
        count = max(int(allocated['stdout'].split()[0]), int(apparent['stdout'].split()[0]))
        total += count
        storage.append({'root': str(root), 'conservative_bytes': count,
                        'allocated': allocated, 'apparent': apparent, 'breakdown': breakdown})
        for directory, folders, files in os.walk(root, followlinks=False):
            folders[:] = [name for name in folders if not (Path(directory) / name).is_symlink()]
            for name in files:
                path = Path(directory) / name
                if path.is_symlink():
                    continue
                info = path.stat()
                if info.st_size >= 512 * 1024**2:
                    large.append({'path': str(path), 'bytes': info.st_size,
                                  'owner_uid': info.st_uid, 'hardlinks': info.st_nlink,
                                  'suffix': path.suffix})
    primary = Path('/dgxa_home/se23uari167/third_eye_dgx_20261026')
    probes = {
        'shared_python': primary / '.venv/bin/python',
        'shared_qwen4_index': primary / 'models/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/'
            '1cfa9a7208912126459214e8b04321603b3df60c/model.safetensors.index.json',
        'primary_request': primary / 'runs/positive_regime_20261109/request.json',
    }
    report = {
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'job_id': os.environ['SLURM_JOB_ID'], 'host': socket.gethostname(),
        'account': pwd.getpwuid(os.getuid()).pw_name, 'uid': os.getuid(),
        'partition': os.environ.get('SLURM_JOB_PARTITION'), 'storage': storage,
        'total_conservative_bytes': total, 'provisional_owner_limit_bytes': 200_000_000_000,
        'under_provisional_limit': total < 200_000_000_000,
        'quota': command(['quota', '-s']) if shutil.which('quota') else {'available': False},
        'filesystem_capacity_not_account_quota': command(['df', '-B1', '-T', str(home)]),
        'cpu_count_visible_not_allocation': os.cpu_count(),
        'allocated_cpus': os.environ.get('SLURM_CPUS_PER_TASK'),
        'gpu_types_on_node_not_allocated': command(['nvidia-smi', '--query-gpu=name,memory.total',
                                                   '--format=csv,noheader']),
        'shared_read_access': {key: {'path': str(path), 'readable': os.access(path, os.R_OK),
                                     'executable': os.access(path, os.X_OK)}
                               for key, path in probes.items()},
        'large_files_metadata_only': sorted(large, key=lambda row: row['bytes'], reverse=True)[:100],
        'automatic_cleanup': False, 'owner_projects_modified': False,
    }
    temporary = output / 'inventory.tmp'
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    temporary.replace(output / 'inventory.json')
    print(json.dumps({key: report[key] for key in ('job_id', 'host', 'account',
          'total_conservative_bytes', 'under_provisional_limit', 'owner_projects_modified')}), flush=True)


if __name__ == '__main__':
    main()
