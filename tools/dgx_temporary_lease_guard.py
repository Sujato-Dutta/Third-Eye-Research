"""Enforce the existing temporary lease without depending on Slurm accounting."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from dgx_parallel_runtime import AUDIT, DEADLINE, find_task, read, remaining_hours, resource_profile, write


def cancel_pending(hours_left, required_hours, preflight_active, preflight_passed):
    return (hours_left < required_hours + .5
            or not (preflight_active or preflight_passed))


def run():
    if (os.getuid() != 1189 or not os.environ.get('SLURM_JOB_ID')
            or 'login' in socket.gethostname().lower()):
        raise RuntimeError('Temporary-account Slurm CPU allocation required')
    if '--self-test' in sys.argv:
        assert not cancel_pending(33, 32, True, False)
        assert not cancel_pending(33, 32, False, True)
        assert cancel_pending(32.49, 32, True, False)
        assert cancel_pending(33, 32, False, False)
        print('Lease admission boundary checks: 4 passed', flush=True)
        return
    plan = read(AUDIT / 'parallel/plan.json')
    if not plan.get('armed') or plan['deadline_utc'] != DEADLINE.isoformat():
        raise ValueError('The existing armed lease is required')
    tasks = [find_task(plan, row['key']) for row in plan['tasks']]
    failures = 0
    while True:
        left = remaining_hours()
        # At lease expiry, cancellation does not depend on a successful query.
        if left <= 0:
            ids = [task['worker_job'] for task in tasks] + ['35441']
            result = subprocess.run(['scancel', *ids], capture_output=True, text=True, timeout=30)
            write(AUDIT / 'parallel/lease_guard_status.json', {
                'unix_time': time.time(), 'lease_expired': True,
                'cancellation_requested': ids, 'returncode': result.returncode,
                'stderr': result.stderr, 'original_jobs_touched': False})
            return
        try:
            result = subprocess.run(['squeue', '-u', '1189', '-h', '-o', '%i|%T|%j'],
                                    capture_output=True, text=True, timeout=20, check=True)
            jobs = {}
            for line in result.stdout.splitlines():
                job, state, name = line.split('|')
                jobs[job] = (state, name)
            cancelled, active = [], []
            for task in tasks:
                job = task['worker_job']
                if job not in jobs:
                    continue
                state, name = jobs[job]
                if name != 'te-worker-' + task['key']:
                    raise ValueError('Unexpected scheduler identity for delegated job')
                active.append(job)
                resource = resource_profile(task)
                proof = Path(resource['proof'])
                passed = proof.exists() and read(proof).get('passed') is True
                if state == 'PENDING' and cancel_pending(
                        left, resource['hours'], resource['preflight_job'] in jobs, passed):
                    subprocess.run(['scancel', '--state=PENDING', job],
                                   capture_output=True, text=True, timeout=20, check=True)
                    cancelled.append(job)
            write(AUDIT / 'parallel/lease_guard_status.json', {
                'unix_time': time.time(), 'remaining_hours': left,
                'lease_end_utc': DEADLINE.isoformat(), 'active_delegated_jobs': active,
                'cancelled_pending_jobs': cancelled, 'query_failures': failures,
                'scheduler_source': 'squeue', 'original_jobs_touched': False})
            if not active:
                return
        except (subprocess.SubprocessError, OSError) as error:
            failures += 1
            write(AUDIT / 'parallel/lease_guard_query_error.json', {
                'unix_time': time.time(), 'query_failures': failures,
                'error': str(error), 'retrying': True})
        time.sleep(min(60, max(0, remaining_hours() * 3600)))


if __name__ == '__main__':
    run()
