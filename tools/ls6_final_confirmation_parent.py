"""Run fixed Qwen3-4B confirmation seed 2026110913 after DGX refused admission."""
import os
from pathlib import Path
import subprocess
import sys

import ls6_runtime as runtime
from ls6_finish_scoring_exports import retained_weights


def main():
    runtime.require_compute()
    runtime.environment()
    setup = runtime.read(runtime.CONTROL / 'setup_completed.json')
    preflight = runtime.read(runtime.CONTROL / 'preflight_completed.json')
    if (not setup['passed'] or setup['runtime_sha256'] != runtime.sha(runtime.__file__)
            or not preflight['passed'] or preflight['job_id'] != '3592178'
            or preflight['binding_sha256'] != setup['binding_sha256']
            or setup['binding_sha256'] != runtime.sha(runtime.CONTROL / 'binding.json')):
        raise ValueError('Current compute validation required')

    if len(sys.argv) == 1:
        lane = runtime.gpu_lanes()[0]
        subprocess.run([sys.executable, __file__, 'run'],
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=lane), check=True)
        return

    runner, request = runtime.imports('positive')
    index = 13
    seed = runner.DECLARATION['seeds'][index]
    if seed != 2026110913:
        raise ValueError('Fixed confirmation seed differs')
    folder = Path(request['output']) / 'parents' / str(seed)
    runner.run_parent(request, index)
    _, final = runner.load_final_manifest(request['final_manifest'], request['manifest'])
    runner.inspect_parent(request, index, final)
    records = retained_weights(folder, runtime.read(folder / 'commitment.json'))
    runtime.write(folder / 'lossless_retention.json', {
        'passed': True, 'job_id': os.environ['SLURM_JOB_ID'], 'records': records,
        'policy': 'Verify each retained tensor against its training seal',
        'scientific_outcomes_unchanged': True,
    })
    runtime.write(runtime.CONTROL / 'positive_13_completed.json', {
        'passed': True, 'seed': seed, 'job_id': os.environ['SLURM_JOB_ID'],
        'dgx_admission_failure_job': '38040_13',
        'completed_sha256': runtime.sha(folder / 'completed.json'),
    })


if __name__ == '__main__':
    main()
