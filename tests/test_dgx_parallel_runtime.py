"""Operational guarantees for delegation; scientific runners remain frozen."""
import datetime as dt
import importlib.util
import os
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / 'tools/dgx_parallel_runtime.py'
spec = importlib.util.spec_from_file_location('parallel_runtime', MODULE)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def test_delegation_keeps_fixed_seeds_and_excludes_running_prefix():
    rows = runtime.tasks()
    assert len(rows) == 10
    assert len({row['original_job'] for row in rows}) == 10
    assert [row['index'] for row in rows if row['kind'] == 'positive'] == list(range(14, 20))
    assert {(row['model'], row['index']) for row in rows if row['kind'] == 'horizon'} == {
        (model, index) for model in ('qwen3-8b', 'llama3.1-8b') for index in (2, 3)}


def test_mutated_task_identity_is_rejected():
    row = dict(runtime.tasks()[0])
    row['index'] = 0
    with pytest.raises(ValueError, match='identity'):
        runtime.find_task({'tasks': [row]}, row['key'])


def test_default_resources_keep_existing_full_gpu_budget():
    task = {**runtime.tasks()[-1], 'worker_job': '40000'}
    profile = runtime.resource_profile(task, {'jobs': {}})
    assert profile['type'] == 'a100' and profile['hours'] == task['hours']
    assert profile['preflight_job'] == '35441'


def test_mig_resources_are_limited_to_qwen4_confirmation():
    entry = {'type': 'a100_3g.20gb', 'hours': 40, 'preflight_job': '35960'}
    resources = {'jobs': {'40000': entry}}
    task = {**runtime.tasks()[-1], 'worker_job': '40000'}
    assert runtime.resource_profile(task, resources)['hours'] == 40
    with pytest.raises(ValueError, match='Only unchanged'):
        runtime.resource_profile({**runtime.tasks()[0], 'worker_job': '40000'}, resources)


def test_mig_proof_requires_original_batches_and_full_training():
    task = {**runtime.tasks()[-1], 'worker_job': '40000'}
    profile = runtime.resource_profile(task, {'jobs': {'40000': {
        'type': 'a100_3g.20gb', 'hours': 40, 'preflight_job': '35960'}}})
    proof = {'passed': True, 'resource_type': 'a100_3g.20gb', 'model': 'qwen3-4b',
             'job_id': '35960', 'exact_rollback': True, 'quantization': 'none', 'dtype': 'bfloat16',
             'generation_batch_size': 32, 'sampled_batch_size': 16, 'steps': 25,
             'final_items_evaluated': 0, 'source_tree_sha256': runtime.PROFILES['positive']['sha'],
             'stages': {name: {} for name in ('load', 'train_25_steps',
                 'greedy_batch32_tokens512', 'sampled_batch16_tokens512', 'mcq_batch32')}}
    runtime.validate_resource_proof(task, profile, proof)
    with pytest.raises(ValueError, match='Matched model'):
        runtime.validate_resource_proof(task, profile, {**proof, 'generation_batch_size': 1})


def test_lease_ends_at_november_17_eod_ist():
    end = runtime.DEADLINE.astimezone(dt.timezone(dt.timedelta(hours=5, minutes=30)))
    assert (end.day, end.hour, end.minute, end.second) == (17, 23, 59, 59)
    assert runtime.remaining_hours(runtime.DEADLINE + dt.timedelta(seconds=1)) < 0


def test_archive_validation_rejects_modified_bytes(tmp_path):
    path = tmp_path / 'adapter.gz'
    path.write_bytes(b'original-archive')
    runtime.write(tmp_path / 'lossless_retention.json', {
        'passed': True, 'scientific_outcomes_unchanged': True,
        'records': [{'archive': path.name, 'archive_sha256': runtime.sha(path)}]})
    runtime.verify_archives(tmp_path)
    path.write_bytes(b'corrupt-archive')
    with pytest.raises(ValueError, match='bytes'):
        runtime.verify_archives(tmp_path)


def test_fallback_routing_keeps_uncompleted_prerequisites(monkeypatch):
    monkeypatch.setattr(runtime, 'state', lambda job: {'1': 'COMPLETED', '2': 'RUNNING',
                                                     '3': 'PENDING'}[job])
    assert runtime.pending_dependencies(['1', '2', '3']) == ['2', '3']


def test_completed_fallback_array_is_recognized(monkeypatch):
    monkeypatch.setattr(runtime, 'command', lambda args: '35550_14|COMPLETED|\n')
    assert runtime.state('35550') == 'COMPLETED'


def test_loopback_accounting_retry_is_limited_to_accounting(monkeypatch):
    calls = []
    def run(args, **kwargs):
        calls.append(kwargs)
        return runtime.subprocess.CompletedProcess(args, 0 if 'env' in kwargs else 1,
            '35441|PENDING|' if 'env' in kwargs else '',
            '' if 'env' in kwargs else 'Connection refused localhost:7031')
    monkeypatch.setattr(runtime.subprocess, 'run', run)
    monkeypatch.setattr(runtime, 'accounting_environment', lambda: {'SLURM_CONF': '/private/client.conf'})
    assert runtime.command(['sacct']) == '35441|PENDING|'
    assert len(calls) == 2 and calls[-1]['env']['SLURM_CONF'] == '/private/client.conf'
    with pytest.raises(runtime.subprocess.CalledProcessError):
        runtime.command(['sbatch'])
    assert len(calls) == 3


def test_repeated_sentinel_never_submits_a_second_fallback(tmp_path, monkeypatch):
    task = dict(next(t for t in runtime.tasks() if t['kind'] == 'positive'))
    task['worker_job'] = '40000'
    monkeypatch.setitem(runtime.PROFILES, 'positive', {'root': tmp_path, 'uid': os.getuid()})
    runtime.write(tmp_path / 'temporary_worker_20261112' / (task['key'] + '_completion.json'),
                  {'passed': False, 'fallback_submitted': '40001'})
    calls = []
    monkeypatch.setattr(runtime, 'reroute_after_fallback', lambda task, job: calls.append(job))
    monkeypatch.setattr(runtime, 'runner_for', lambda kind: pytest.fail('Must reuse existing fallback'))
    runtime.sentinel({'tasks': [task]}, task['key'])
    assert calls == ['40001']
