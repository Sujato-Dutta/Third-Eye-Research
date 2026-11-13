"""Local source/staging preparation only; never executes scientific computation."""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('parallel_runtime', Path(__file__).with_name('dgx_parallel_runtime.py'))
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
OUTPUT = Path('runs/deployment/parallel_20261112')
PYTHON = runtime.PRIMARY / '.venv/bin/python'
REMOTE = runtime.AUDIT / 'parallel'


def script(path, name, account, partition, cpus, memory, duration, cwd, logdir, commands, gpu=False):
    lines = ['#!/bin/bash', '#SBATCH --job-name=' + name, '#SBATCH --account=' + account,
             '#SBATCH --partition=' + partition, '#SBATCH --nodes=1', '#SBATCH --ntasks=1',
             '#SBATCH --cpus-per-task=' + str(cpus), '#SBATCH --mem=' + memory,
             '#SBATCH --time=' + duration, '#SBATCH --chdir=' + cwd.as_posix(),
             '#SBATCH --output=' + (logdir / (name + '-%j.out')).as_posix(),
             '#SBATCH --error=' + (logdir / (name + '-%j.err')).as_posix()]
    if account == 'faculty':
        lines += ['#SBATCH --deadline=2026-11-17T23:00:00']
    if gpu:
        lines += ['#SBATCH --gres=gpu:a100:1', '#SBATCH --no-requeue']
    cache = runtime.AUDIT if account == 'faculty' else logdir.parent
    lines += ['set -euo pipefail',
              '[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* ]] || exit 1',
              'export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false',
              'export OMP_NUM_THREADS=' + ('4' if gpu else '1') + ' OPENBLAS_NUM_THREADS=1',
              'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1',
              'export HF_HOME=' + (cache / 'hf_cache').as_posix(),
              'export XDG_CACHE_HOME=' + (cache / 'cache').as_posix(),
              'export CUDA_CACHE_PATH=' + (cache / 'cuda_cache').as_posix(),
              'export TRITON_CACHE_DIR=' + (cache / 'triton_cache').as_posix(),
              'export TORCH_HOME=' + (cache / 'torch_cache').as_posix(),
              'export THIRD_EYE_SYMBOLIC_WORKERS=1 THIRD_EYE_VERIFIER_WORKERS=' + ('16' if gpu else '1')]
    if gpu:
        lines += ['[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || exit 1',
                  'export CUBLAS_WORKSPACE_CONFIG=:4096:8']
    else:
        lines += ['export CUDA_VISIBLE_DEVICES=""']
    lines += commands
    if any('\\' in line or '\r' in line for line in lines):
        raise ValueError('Remote batch scripts must contain literal Unix paths and line endings')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')


def main():
    OUTPUT.mkdir(exist_ok=True)
    for task in runtime.tasks():
        kind, key = task['kind'], task['key']
        owner = runtime.PROFILES[kind]
        control = owner['root'] / 'temporary_worker_20261112'
        script(OUTPUT / ('worker-' + key + '.slurm'), 'te-worker-' + key,
               'faculty', 'gpu_faculty', 16, '64G', str(task['hours']) + ':00:00',
               runtime.AUDIT.parent, runtime.AUDIT / 'logs',
               [PYTHON.as_posix() + ' ' + (REMOTE / 'tools/dgx_parallel_runtime.py').as_posix() +
                ' worker --plan ' + (REMOTE / 'plan.json').as_posix() + ' --key ' + key], gpu=True)
        script(OUTPUT / ('sentinel-' + key + '.slurm'), 'te-handoff-' + key,
               'student', 'cpu_student', 1, '4G', '02:00:00',
               runtime.PRIMARY.parent if kind == 'positive' else runtime.HORIZON.parent,
               control / 'logs', [PYTHON.as_posix() + ' ' + (control / 'runtime.py').as_posix() +
               ' sentinel --plan ' + (control / 'plan.json').as_posix() + ' --key ' + key])
    for kind, owner in runtime.PROFILES.items():
        control = owner['root'] / 'temporary_worker_20261112'
        script(OUTPUT / ('watch-' + kind + '.slurm'), 'te-storage-' + kind,
               'student', 'cpu_student', 1, '2G', '6-00:00:00',
               runtime.PRIMARY.parent if kind == 'positive' else runtime.HORIZON.parent,
               control / 'logs', [PYTHON.as_posix() + ' ' + (control / 'runtime.py').as_posix() + ' watch --kind ' + kind])
    script(OUTPUT / 'watch-faculty.slurm', 'te-storage-faculty', 'faculty', 'cpu_faculty',
           1, '2G', '4-23:00:00', runtime.AUDIT.parent, runtime.AUDIT / 'logs',
           [PYTHON.as_posix() + ' ' + (runtime.AUDIT / 'storage.py').as_posix() + ' watch --output ' +
            runtime.AUDIT.as_posix() + ' --duration-hours 115 --interval-seconds 1200' +
            ' --limit-bytes 200000000000 --warning-bytes 180000000000 --admission-limit-bytes 190000000000 --measure-timeout 900'])
    script(OUTPUT / 'watchdog.slurm', 'te-temporary-lease', 'faculty', 'cpu_faculty',
           1, '2G', '4-23:00:00', runtime.AUDIT.parent, runtime.AUDIT / 'logs',
           [PYTHON.as_posix() + ' ' + (REMOTE / 'tools/dgx_parallel_runtime.py').as_posix() +
            ' watchdog --plan ' + (REMOTE / 'plan.json').as_posix()])
    (OUTPUT / 'task_manifest.json').write_text(json.dumps({'deadline_utc': runtime.DEADLINE.isoformat(),
          'tasks': runtime.tasks()}, indent=2) + '\n')
    print(json.dumps({'prepared_tasks': len(runtime.tasks()), 'output': str(OUTPUT)}))


if __name__ == '__main__':
    main()
