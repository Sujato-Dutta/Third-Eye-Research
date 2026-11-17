"""Atomically select validated hardware for pending jobs; preserve fixed job IDs."""
import argparse
import json
import os
import socket

from dgx_parallel_runtime import (AUDIT, command, read, remaining_hours,
                                 resource_profile, sha, validate_resource_proof, write)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight-job', required=True)
    parser.add_argument('--validation-job', required=True)
    args = parser.parse_args()
    if os.getuid() != 1189 or not os.environ.get('SLURM_JOB_ID') or 'login' in socket.gethostname().lower():
        raise RuntimeError('Faculty Slurm CPU allocation required')
    root = AUDIT / 'parallel'
    completed = root / 'mig_activation_completed.json'
    if completed.exists():
        raise FileExistsError('Inspect the earlier activation instead of dispatching again')
    validation = read(root / 'validation_completed.json')
    if (not validation['passed'] or validation['job_id'] != args.validation_job
            or validation['runtime_sha256'] != sha(root / 'tools/dgx_parallel_runtime.py')):
        raise RuntimeError('Matching complete CPU validation required')
    plan = read(root / 'plan.json')
    if not plan['armed']:
        raise RuntimeError('Original jobs must already be retired with owning-account handoffs')
    proof_path = AUDIT / ('mig_preflight_' + args.preflight_job + '.json')
    proof = read(proof_path) if proof_path.exists() else {'passed': False}
    use_mig = bool(proof.get('passed')) and remaining_hours() > 40.5
    selected = [t for t in plan['tasks'] if t['kind'] == 'positive']
    resources = {'jobs': {t['worker_job']: {'type': 'a100_3g.20gb', 'hours': 40,
                  'preflight_job': args.preflight_job} for t in selected} if use_mig else {},
                 'preflight_job': args.preflight_job, 'scientific_settings_changed': False,
                 'purpose': 'hardware-only dispatch within the November 17 lease'}
    if use_mig:
        for task in selected:
            profile = resource_profile(task, resources)
            validate_resource_proof(task, profile, proof)
    for task in plan['tasks']:
        state = command(['squeue', '-h', '-j', task['worker_job'], '-o', '%T'])
        if state != 'PENDING':
            raise RuntimeError('Only pending tasks can change hardware: ' + task['worker_job'])
    for task in plan['tasks']:
        command(['scontrol', 'hold', task['worker_job']])
    record = {'passed': False, 'job_id': os.environ['SLURM_JOB_ID'], 'use_mig': use_mig,
              'resources': resources, 'updated': [], 'scientific_settings_changed': False,
              'canonical_job_ids_preserved': True, 'preflight_passed': proof.get('passed', False)}
    write(root / 'mig_activation_progress.json', record)
    try:
        for task in plan['tasks']:
            profile = resource_profile(task, resources)
            command(['scontrol', 'update', 'JobId=' + task['worker_job'],
                     'Gres=gpu:' + profile['type'] + ':1',
                     'TimeLimit=' + str(profile['hours']) + ':00:00',
                     # Successful completed dependencies can already be purged
                     # from Slurm; their matching durable receipts were checked above.
                     'Dependency=' + ('0' if profile['type'] != 'a100' else 'afterok:35441')])
            record['updated'].append(task['worker_job'])
            write(root / 'mig_activation_progress.json', record)
        write(root / 'resource_plan.json', resources)
    except Exception:
        # Jobs remain held until their original safe bindings have been restored.
        for task in plan['tasks']:
            command(['scontrol', 'update', 'JobId=' + task['worker_job'], 'Gres=gpu:a100:1',
                     'TimeLimit=' + str(task['hours']) + ':00:00',
                     'Dependency=afterok:35441'])
        write(root / 'resource_plan.json', {'jobs': {}})
        for task in plan['tasks']:
            command(['scontrol', 'release', task['worker_job']])
        raise
    # Replace the old monitor so its loop understands per-resource preflights and runtimes.
    active = command(['squeue', '-u', '1189', '-h', '-o', '%i']).splitlines()
    if '35507' in active:
        command(['scancel', '35507'])
    # The original five-day monitor limit no longer fits the remaining lease.
    # Override it explicitly; its loop and the research-job guards share the cutoff.
    minutes = max(1, int(remaining_hours() * 60) - 1)
    record['watchdog_job'] = command(['sbatch', '--parsable', '--time=' + str(minutes),
        '--deadline=2026-11-17T23:59:59', str(root / 'watchdog.slurm')]).split(';')[0]
    for task in plan['tasks']:
        command(['scontrol', 'release', task['worker_job']])
    record['passed'] = True
    record['research_jobs_on_mig'] = [t['worker_job'] for t in selected] if use_mig else []
    record['preflight_sha256'] = sha(proof_path) if proof_path.exists() else None
    write(completed, record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
