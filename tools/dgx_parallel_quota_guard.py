"""Stop only delegated writes when whole-account reserves are consumed."""
import argparse
import json
import os
import socket
import subprocess
import time

from dgx_parallel_runtime import AUDIT, PROFILES, remaining_hours, tasks, write


def unsafe(snapshot, limit, now):
    return (now - snapshot['unix_time'] > 900 or snapshot['unix_time'] > now + 60
            or snapshot['bytes'] + 2_000_000_000 >= limit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if os.getuid() != 1189 or not os.environ.get('SLURM_JOB_ID') or 'login' in socket.gethostname().lower():
        raise RuntimeError('Faculty CPU allocation required')
    if args.self_test:
        assert not unsafe({'unix_time': 1000, 'bytes': 47_999_999_999}, 50_000_000_000, 1001)
        assert unsafe({'unix_time': 1000, 'bytes': 48_000_000_000}, 50_000_000_000, 1001)
        assert unsafe({'unix_time': 1000, 'bytes': 1}, 50_000_000_000, 1901)
        assert unsafe({'unix_time': 1061, 'bytes': 1}, 50_000_000_000, 1000)
        print(json.dumps({'passed': True, 'checks': 4}), flush=True)
        return
    plan = json.loads((AUDIT / 'parallel/plan.json').read_text())
    expected = {task['key']: task for task in tasks()}
    for task in plan['tasks']:
        if any(task.get(k) != v for k, v in expected[task['key']].items()):
            raise ValueError('Only the frozen delegated task whitelist may be managed')
    while remaining_hours() > 0:
        now = time.time()
        blocked, snapshots = [], {}
        for kind, profile in PROFILES.items():
            try:
                snapshot = json.loads((profile['root'] / 'temporary_worker_20261112/status/storage.json').read_text())
                snapshots[kind] = snapshot
                if unsafe(snapshot, profile['admission'], now):
                    blocked.append(kind)
            except (OSError, ValueError, KeyError):
                blocked.append(kind)
        try:
            faculty = json.loads((AUDIT / 'storage_latest.json').read_text())
            if faculty['total_conservative_bytes'] + 15_000_000_000 >= 190_000_000_000:
                blocked = list(PROFILES)
        except (OSError, ValueError, KeyError):
            blocked = list(PROFILES)
        output = subprocess.run(['squeue', '-u', '1189', '-h', '-o', '%i|%T|%j'],
                                capture_output=True, text=True, check=True).stdout
        jobs = {parts[0]: parts[1:] for line in output.splitlines()
                if len(parts := line.split('|')) == 3}
        active, cancelled = False, []
        for task in plan['tasks']:
            row = jobs.get(task['worker_job'])
            if not row:
                continue
            if row[1] != 'te-worker-' + task['key']:
                raise ValueError('Unexpected scheduler job identity')
            active = True
            if task['kind'] in blocked and row[0] in ('PENDING', 'RUNNING', 'CONFIGURING'):
                subprocess.run(['scancel', task['worker_job']], check=True)
                cancelled.append(task['worker_job'])
        write(AUDIT / 'parallel/quota_guard_status.json', {
            'unix_time': now, 'blocked_kinds': blocked, 'storage': snapshots,
            'cancelled_delegated_jobs': cancelled, 'owner_data_deleted': False,
            'original_running_jobs_touched': False})
        if not active:
            return
        time.sleep(min(60, max(0, remaining_hours() * 3600)))


if __name__ == '__main__':
    main()
