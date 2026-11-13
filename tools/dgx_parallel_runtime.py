"""Dispatch frozen experiments across accounts; retain canonical paths and inputs."""
import argparse
import datetime as dt
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

PRIMARY = Path('/dgxa_home/se23uari167/third_eye_dgx_20261026')
HORIZON = Path('/dgxa_home/se23ucse205/third_eye_horizon_20261109')
AUDIT = Path('/dgxb_home/muef2320/third_eye_account_audit_20261112')
DEADLINE = dt.datetime(2026, 11, 17, 18, 29, 59, tzinfo=dt.timezone.utc)
PROFILES = {
    'positive': {'uid': 1231, 'root': PRIMARY / 'runs/positive_regime_20261109',
        'source': PRIMARY / 'releases/positive_regime_d5c3bc9fb65a',
        'sha': 'd5c3bc9fb65a2e416736ff42be7f7ebadf5a44943afc7277c52ef846fed0d1e0',
        'module': 'positive_regime_confirmation', 'admission': 50_000_000_000},
    'horizon': {'uid': 1339, 'root': HORIZON, 'source': HORIZON / 'release_adb48501b796',
        'sha': 'adb48501b796b2c20538a0e0142379b13f9f3f659338f6bff18c4200ebd72d2a',
        'module': 'long_horizon_trajectories', 'admission': 48_000_000_000},
}


def tasks():
    rows = []
    for index in (2, 3):
        for model, original, archive in (
            ('qwen3-8b', {2: '35082', 3: '35084'}, {2: '35083', 3: '35085'}),
            ('llama3.1-8b', {2: '35091', 3: '35093'}, {2: '35092', 3: '35094'}),
        ):
            rows.append({'key': f'horizon-{model}-{index}', 'kind': 'horizon',
                'model': model, 'index': index, 'hours': 32,
                'original_job': original[index], 'downstream_job': archive[index]})
    rows += [{'key': f'positive-{i}', 'kind': 'positive', 'model': 'qwen3-4b',
              'index': i, 'hours': 24, 'original_job': f'35053_{i}',
              'downstream_job': '35054'} for i in range(14, 20)]
    return rows


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def remaining_hours(now=None):
    return (DEADLINE - (now or dt.datetime.now(dt.timezone.utc))).total_seconds() / 3600


def command(args):
    result = subprocess.run(args, text=True, capture_output=True)
    if (result.returncode and args[0] == 'sacct'
            and 'localhost:7031' in result.stderr):
        result = subprocess.run(args, text=True, capture_output=True,
                                env=accounting_environment())
    result.check_returncode()
    return result.stdout.strip()


def accounting_environment():
    # A private client configuration fixes the site's loopback database address.
    # System configuration and all training settings remain untouched.
    original = Path('/etc/slurm/slurm.conf').read_text()
    corrected, count = re.subn(r'(?m)^(\s*AccountingStorageHost\s*=\s*)localhost\s*$',
                              r'\g<1>dgx-login01', original)
    if count != 1:
        raise RuntimeError('Expected exactly one loopback accounting address')
    profile = next((p for p in PROFILES.values() if p['uid'] == os.getuid()), None)
    root = (profile['root'] / 'temporary_worker_20261112') if profile else AUDIT / 'parallel'
    path = root / ('accounting-client-' + str(os.getuid()) + '.conf')
    # Each Slurm job gets its own file to avoid concurrent publication races.
    path = path.with_name(path.stem + '-' + os.environ['SLURM_JOB_ID'] + '.conf')
    path.write_text(corrected)
    path.chmod(0o600)
    return dict(os.environ, SLURM_CONF=str(path))


def state(job):
    text = command(['sacct', '-S', '2026-11-09', '-j', job, '-X', '-n', '-P',
                    '--format=JobID,State'])
    members = []
    for row in text.splitlines():
        fields = row.split('|')
        if fields[0] == job:
            return fields[1].split()[0].rstrip('+')
        if fields[0].startswith(job + '_'):
            members.append(fields[1].split()[0].rstrip('+'))
    if members:
        if all(value == 'COMPLETED' for value in members):
            return 'COMPLETED'
        for value in ('RUNNING', 'COMPLETING', 'PENDING', 'FAILED', 'TIMEOUT', 'CANCELLED'):
            if value in members:
                return value
    return 'UNKNOWN'


def runner_for(kind):
    profile = PROFILES[kind]
    sys.path[:0] = [str(profile['source'] / 'src'), str(profile['source'] / 'experiments')]
    runner = importlib.import_module(profile['module'])
    request = runner.checked(profile['root'] / 'request.json')
    if request['source_tree_sha256'] != profile['sha'] or Path(request['output']) != profile['root']:
        raise ValueError('Frozen source or canonical output differs')
    os.environ['THIRD_EYE_MODEL_SOURCES'] = request['model_sources']
    return runner, request


def folder_for(task, runner):
    root = PROFILES[task['kind']]['root'] / 'parents'
    if task['kind'] == 'positive':
        return root / str(runner.DECLARATION['seeds'][task['index']])
    return root / task['model'] / str(runner.MODELS[task['model']]['seeds'][task['index']])


def inspect(task, runner, request):
    if task['kind'] == 'positive':
        from third_eye.evaluation.final import load_final_manifest
        _, final = load_final_manifest(request['final_manifest'], request['manifest'])
        runner.inspect_parent(request, task['index'], final)
    else:
        runner.inspect(request, task['model'], task['index'])


def verify_archives(folder):
    retained = read(folder / 'lossless_retention.json')
    if not retained['passed'] or not retained.get('scientific_outcomes_unchanged'):
        raise ValueError('Complete verified lossless retention required')
    for record in retained['records']:
        archive = folder / record['archive']
        if (archive.is_symlink() or not archive.resolve().is_relative_to(folder.resolve())
                or sha(archive) != record['archive_sha256']):
            raise ValueError('Retained adapter bytes differ')


def find_task(plan, key):
    selected = next(t for t in plan['tasks'] if t['key'] == key)
    reference = next(t for t in tasks() if t['key'] == key)
    if any(selected.get(field) != value for field, value in reference.items()):
        raise ValueError('Fixed task identity or budget differs')
    return selected


def verify_site(kind):
    if os.getuid() != 1189:
        raise RuntimeError('Temporary account required')
    runner, request = runner_for(kind)
    profile = PROFILES[kind]
    receipt = read(profile['root'] / 'temporary_worker_20261112/access_completed.json')
    if not receipt['passed'] or receipt['owner_uid'] != profile['uid']:
        raise ValueError('Owning-account access validation missing')
    for task in (t for t in tasks() if t['kind'] == kind):
        folder = folder_for(task, runner)
        if folder.exists() or not os.access(folder.parent, os.W_OK):
            raise ValueError('Only writable, unstarted task destinations may be delegated')
    probe = profile['root'] / 'temporary_worker_20261112/acl_probe'
    child = probe / 'delegate_created'
    child.mkdir(exist_ok=True)
    if child.is_symlink() or child.stat().st_uid != 1189:
        raise ValueError('Compatibility fixture must belong to this delegate')
    fixture = child / 'fixture.json'
    if fixture.exists():
        if read(fixture) != {'fixture': True}:
            raise ValueError('Preserve an unexpected compatibility fixture')
    else:
        fixture.write_text('{"fixture": true}\n')
    try:
        (probe / 'owner_protected.txt').unlink()
    except PermissionError:
        pass
    else:
        raise ValueError('Sticky directory failed to protect owner fixture')
    write(AUDIT / f'parallel_{kind}_access.json', {'passed': True,
          'source_tree_sha256': request['source_tree_sha256'],
          'request_sha256': sha(profile['root'] / 'request.json'),
          'job_id': os.environ['SLURM_JOB_ID'], 'new_task_folders_only': True})


def worker(plan, key):
    task = find_task(plan, key)
    if not plan.get('armed'):
        raise RuntimeError('Delegation must be armed after retiring original pending jobs')
    validation = read(AUDIT / 'parallel/validation_completed.json')
    if not validation['passed'] or validation['runtime_sha256'] != sha(__file__):
        raise RuntimeError('Matching compute-node operational validation required')
    if os.getuid() != 1189 or str(os.environ['SLURM_JOB_ID']) != task['worker_job']:
        raise RuntimeError('Named faculty task allocation required')
    if remaining_hours() < task['hours'] + .5:
        raise RuntimeError('Protective runtime no longer fits the authorized lease')
    if state(task['original_job']) != 'CANCELLED':
        raise RuntimeError('Original unstarted job must be retired before delegation')
    if not read(AUDIT / 'gpu_preflight.json')['passed']:
        raise ValueError('Real temporary-account GPU validation required')
    runner, request = runner_for(task['kind'])
    profile = PROFILES[task['kind']]
    storage = read(profile['root'] / 'temporary_worker_20261112/status/storage.json')
    age = time.time() - storage['unix_time']
    if age < -60 or age > 900 or storage['bytes'] + 2_000_000_000 >= profile['admission']:
        raise RuntimeError('Fresh safe owning-account storage snapshot required')
    home_storage = read(AUDIT / 'storage_latest.json')
    if home_storage['total_conservative_bytes'] + 15_000_000_000 >= 190_000_000_000:
        raise RuntimeError('Insufficient faculty whole-account reserve')
    folder = folder_for(task, runner)
    if folder.exists():
        raise FileExistsError('Preserve an existing experiment, including partial evidence')
    if task['kind'] == 'positive':
        runner.run_parent(request, task['index'])
    else:
        runner.run(request, task['model'], task['index'])
    inspect(task, runner, request)
    if task['kind'] == 'positive':
        from third_eye.experiments.scale import compact_checkpoints
        compact_checkpoints(folder)
        verify_archives(folder)
    write(folder / 'temporary_handoff.json', {'passed': True, 'key': key,
          'worker_job': task['worker_job'], 'completed_sha256': sha(folder / 'completed.json'),
          'request_sha256': sha(profile['root'] / 'request.json'),
          'source_tree_sha256': request['source_tree_sha256'],
          'original_job_retired_before_start': task['original_job'], 'scientific_settings_changed': False})


def pending_dependencies(jobs):
    complete = {'COMPLETED'}
    return [job for job in jobs if state(job) not in complete]


def reroute_after_fallback(task, fallback):
    import fcntl
    profile = PROFILES[task['kind']]
    control = profile['root'] / 'temporary_worker_20261112'
    with (control / 'dependencies.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        if task['kind'] == 'positive':
            registry_path = control / 'review_dependencies.json'
            registry = read(registry_path)
            registry['tasks'][task['key']] = fallback
            jobs = registry['original_tasks'] + list(registry['tasks'].values())
            active = pending_dependencies(jobs)
            if fallback not in active:
                active.append(fallback)
            command(['scontrol', 'update', 'JobId=' + task['downstream_job'],
                     'Dependency=afterok:' + ':'.join(active)])
            write(registry_path, registry)
        else:
            command(['scontrol', 'update', 'JobId=' + task['downstream_job'],
                     'Dependency=afterok:' + fallback])


def sentinel(plan, key):
    task = find_task(plan, key)
    profile = PROFILES[task['kind']]
    if os.getuid() != profile['uid']:
        raise RuntimeError('Owning-account CPU allocation required')
    target = profile['root'] / 'temporary_worker_20261112' / (key + '_completion.json')
    if target.exists():
        previous = read(target)
        if previous.get('fallback_submitted'):
            reroute_after_fallback(task, previous['fallback_submitted'])
            previous['fallback_dependency_registered'] = True
            write(target, previous)
            return
    terminal = state(task['worker_job'])
    if terminal in {'PENDING', 'RUNNING', 'COMPLETING', 'CONFIGURING', 'UNKNOWN'}:
        raise RuntimeError('Faculty allocation must be terminal before handoff or fallback')
    runner, request = runner_for(task['kind'])
    folder = folder_for(task, runner)
    if (folder / 'completed.json').exists():
        # A complete result is inspected regardless of scheduler exit; never replace
        # a scored outcome merely because its effect is weak or its archival step failed.
        inspect(task, runner, request)
        if task['kind'] == 'positive':
            from third_eye.experiments.scale import compact_checkpoints
            if not (folder / 'lossless_retention.json').exists():
                compact_checkpoints(folder)
            verify_archives(folder)
        write(target, {'passed': True, 'worker_job_state': terminal,
              'completed_sha256': sha(folder / 'completed.json'), 'retrained': False})
        return
    quarantine = None
    if folder.exists():
        quarantine = profile['root'] / 'failure_evidence' / (key + '-' + task['worker_job'])
        quarantine.parent.mkdir(exist_ok=True)
        if quarantine.exists():
            raise FileExistsError('Preserve earlier failure evidence')
        folder.rename(quarantine)
    args = ['sbatch', '--parsable']
    if task['kind'] == 'positive':
        args += ['--array=' + str(task['index']) + '%1', str(profile['root'] / 'remaining.slurm')]
    else:
        args += [str(profile['root'] / (task['model'] + '-' + str(task['index']) + '.slurm'))]
    output = command(args)
    if not re.fullmatch(r'\d+(?:;[A-Za-z0-9_.-]+)?', output):
        raise ValueError('Fallback submission did not return a unique job ID')
    fallback = output.split(';')[0]
    # Record the submission before changing downstream dependencies, for recovery.
    record = {'passed': False, 'fallback_submitted': fallback, 'worker_job_state': terminal,
              'quarantined_partial': str(quarantine) if quarantine else None,
              'same_frozen_seed': True, 'scientific_outcomes_used': False}
    write(target, record)
    reroute_after_fallback(task, fallback)
    record['fallback_dependency_registered'] = True
    write(target, record)


def watch(kind):
    profile = PROFILES[kind]
    if os.getuid() != profile['uid']:
        raise RuntimeError('Owning account required for complete private-home scans')
    destination = profile['root'] / 'temporary_worker_20261112/status/storage.json'
    while remaining_hours() > 0:
        allocated = int(command(['du', '-x', '-B1', '-s', str(Path.home())]).split()[0])
        apparent = int(command(['du', '-x', '-B1', '--apparent-size', '-s', str(Path.home())]).split()[0])
        total = max(allocated, apparent)
        write(destination, {'bytes': total, 'allocated_bytes': allocated, 'apparent_bytes': apparent,
              'unix_time': time.time(),
              'job_id': os.environ['SLURM_JOB_ID'], 'owner_uid': os.getuid(),
              'hard_limit_bytes': 50_000_000_000, 'safe': total + 2_000_000_000 < profile['admission']})
        time.sleep(min(120, max(0, remaining_hours() * 3600)))


def watchdog(plan_path):
    if os.getuid() != 1189:
        raise RuntimeError('Faculty account required for its own lease enforcement')
    while True:
        plan = read(plan_path)
        if plan.get('armed'):
            rows = command(['sacct', '-S', '2026-11-12', '-j',
                            ','.join(['35441'] + [t['worker_job'] for t in plan['tasks']]),
                            '-X', '-n', '-P', '--format=JobID,State'])
            states = {row.split('|')[0]: row.split('|')[1].split()[0].rstrip('+')
                      for row in rows.splitlines() if '|' in row}
            failed_check = states.get('35441') in {'FAILED', 'CANCELLED', 'TIMEOUT', 'DEADLINE', 'NODE_FAIL'}
            active = False
            for task in plan['tasks']:
                current = states.get(task['worker_job'], 'UNKNOWN')
                if current in {'PENDING', 'RUNNING', 'COMPLETING', 'CONFIGURING', 'UNKNOWN'}:
                    active = True
                if current == 'PENDING' and (failed_check or remaining_hours() < task['hours'] + .5):
                    # Filter atomically at Slurm: do not interrupt a task that just started.
                    command(['scancel', '--state=PENDING', task['worker_job']])
                elif current == 'RUNNING' and remaining_hours() <= 0:
                    command(['scancel', task['worker_job']])
            write(AUDIT / 'parallel/watchdog_status.json', {'unix_time': time.time(),
                  'remaining_hours': remaining_hours(), 'gpu_preflight_state': states.get('35441'),
                  'task_states': states, 'lease_end_utc': DEADLINE.isoformat()})
            if not active:
                return
        if remaining_hours() <= 0:
            return
        time.sleep(min(60, max(0, remaining_hours() * 3600)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('verify', 'worker', 'sentinel', 'watch', 'watchdog'))
    parser.add_argument('--kind', choices=PROFILES)
    parser.add_argument('--plan')
    parser.add_argument('--key')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or 'login' in socket.gethostname().lower():
        raise RuntimeError('All execution requires Slurm compute nodes')
    if args.mode == 'verify':
        verify_site(args.kind)
    elif args.mode == 'watch':
        watch(args.kind)
    elif args.mode == 'watchdog':
        watchdog(args.plan)
    else:
        plan = read(args.plan)
        if plan['deadline_utc'] != DEADLINE.isoformat():
            raise ValueError('Temporary account authorization window differs')
        (worker if args.mode == 'worker' else sentinel)(plan, args.key)


if __name__ == '__main__':
    main()
