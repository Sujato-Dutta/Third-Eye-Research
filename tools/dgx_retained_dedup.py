"""Lossless inode sharing for identical, inactive historical research artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

ROOT = Path('/dgxa_home/se23uari167/third_eye_dgx_20261026')
CAMPAIGN = ROOT / 'runs/positive_regime_20261109'
SCOPES = [ROOT / 'runs' / name for name in (
    'scale_8777e39cac29', 'review_controls_9d722d4af1b3_sandboxfix',
    'next_round_dgx_20261108',
)]
SCOPES.append(ROOT / 'imported/payload')
NAMES = {'adapter_model.safetensors', 'adapter_model.safetensors.gz', 'tokenizer.json',
         'tokenizer_config.json', 'special_tokens_map.json', 'merges.txt', 'vocab.json'}
PLAN = CAMPAIGN / 'historical_dedup_plan_v3_20261111.json'
RECEIPT = CAMPAIGN / 'historical_dedup_receipt_v3_20261111.json'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def checked(path):
    path = Path(path)
    if path.is_symlink() or not any(path.is_relative_to(scope) for scope in SCOPES):
        raise ValueError('Artifact outside explicit historical allowlist')
    if path.resolve(strict=True) != path or not path.is_file():
        raise ValueError('Refuse indirect or missing artifact')
    info = path.stat()
    if info.st_uid != os.getuid() or path.name not in NAMES:
        raise ValueError('Wrong artifact owner or filename')
    return info


def usage():
    result = subprocess.run(['du', '-x', '-B1', '-s', str(Path.home())],
                            capture_output=True, text=True, check=True)
    return int(result.stdout.split()[0])


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def plan():
    if PLAN.exists():
        raise FileExistsError('Preserve previous inventory')
    sizes = {}
    for scope in SCOPES:
        if scope.resolve(strict=True) != scope or scope.is_symlink():
            raise ValueError('Unsafe historical scope')
        for directory, folders, files in os.walk(scope, followlinks=False):
            folders[:] = [name for name in folders if not (Path(directory) / name).is_symlink()]
            for name in sorted(set(files) & NAMES):
                path = Path(directory) / name
                info = checked(path)
                key = (info.st_size, info.st_mode & 0o777, name)
                sizes.setdefault(key, []).append(path)
    groups = []
    for (size, mode, name), paths in sizes.items():
        if len(paths) < 2:
            continue
        hashes = {}
        inodes = {}
        for path in sorted(paths):
            info = checked(path)
            inode = (info.st_dev, info.st_ino)
            inodes.setdefault(inode, []).append(path)
        for aliases in inodes.values():
            hashes.setdefault(digest(aliases[0]), []).append(aliases)
        for sha, same in hashes.items():
            if len(same) > 1:
                groups.append({'sha256': sha, 'bytes': size, 'mode': mode,
                               'canonical': str(same[0][0]),
                               'reclaimable_inodes': len(same) - 1,
                               'duplicates': [str(p) for aliases in same[1:] for p in aliases]})
    record = {'job_id': os.environ['SLURM_JOB_ID'], 'home_bytes': usage(),
              'groups': groups, 'potential_bytes': sum(g['bytes'] * g['reclaimable_inodes'] for g in groups),
              'scientific_content_changed': False, 'files_removed': False}
    write(PLAN, record)
    print(json.dumps({k: v for k, v in record.items() if k != 'groups'}), flush=True)


def apply():
    if RECEIPT.exists():
        raise FileExistsError('Preserve previous application receipt')
    gpu = subprocess.run(['squeue', '-h', '-u', str(os.getuid()), '-t', 'R', '-o', '%P'],
                         capture_output=True, text=True, check=True).stdout
    if 'gpu' in gpu.lower():
        raise ValueError('No primary-account GPU may run during storage maintenance')
    record = {'job_id': os.environ['SLURM_JOB_ID'], 'plan_sha256': digest(PLAN),
              'before_home_bytes': usage(), 'records': [], 'passed': False,
              'scientific_content_changed': False, 'paths_removed': False}
    write(RECEIPT, record)
    for group in json.loads(PLAN.read_text())['groups']:
        source = Path(group['canonical'])
        checked(source)
        if digest(source) != group['sha256']:
            raise ValueError('Canonical changed since inventory')
        for name in group['duplicates']:
            target = Path(name)
            source_info, target_info = checked(source), checked(target)
            if (digest(target) != group['sha256'] or source_info.st_dev != target_info.st_dev
                    or target_info.st_size != group['bytes']
                    or source_info.st_mode & 0o777 != target_info.st_mode & 0o777):
                raise ValueError('Duplicate changed since inventory')
            temporary = target.with_name(target.name + '.dedup-link')
            if temporary.exists() or temporary.is_symlink():
                raise FileExistsError('Preserve partial maintenance artifact')
            os.link(source, temporary)
            os.replace(temporary, target)
            if not os.path.samefile(source, target) or digest(target) != group['sha256']:
                raise ValueError('Replacement failed exact-byte verification')
            record['records'].append({'path': str(target), 'canonical': str(source),
                                      'sha256': group['sha256'], 'bytes': group['bytes']})
            write(RECEIPT, record)
    record.update({'passed': True, 'after_home_bytes': usage()})
    record['reclaimed_bytes'] = record['before_home_bytes'] - record['after_home_bytes']
    write(RECEIPT, record)
    print(json.dumps({k: v for k, v in record.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('plan', 'apply'))
    args = parser.parse_args()
    if (not os.environ.get('SLURM_JOB_ID') or 'login' in socket.gethostname()
            or Path.home().resolve() != ROOT.parent or os.getuid() != 1231):
        raise RuntimeError('Named account and Slurm compute allocation required')
    (plan if args.mode == 'plan' else apply)()
