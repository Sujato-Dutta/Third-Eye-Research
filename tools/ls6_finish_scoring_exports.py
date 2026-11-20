"""Verify sealed scores and export them without relying on hard links."""
import gzip
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import ls6_runtime as runtime


def retained_weights(folder, commitment):
    records = []
    for name in sorted(commitment['states']):
        state_path = folder / 'checkpoints' / name / 'state.json'
        for filename, expected in runtime.read(state_path)['weights'].items():
            if Path(filename).name != filename:
                raise ValueError('Unexpected checkpoint filename')
            raw = state_path.parent / filename
            packed = raw.with_suffix(raw.suffix + '.gz')
            for path in (raw, packed):
                if path.is_symlink():
                    raise ValueError('Checkpoint files must be regular files')
            if raw.is_file():
                if runtime.sha(raw) != expected:
                    raise ValueError(f'Checkpoint checksum differs: {raw}')
                records.append({'original': str(raw.relative_to(folder)), 'original_sha256': expected})
                continue
            digest = hashlib.sha256()
            with gzip.open(packed, 'rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024**2), b''):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError(f'Compressed checkpoint checksum differs: {packed}')
            records.append({'original': str(raw.relative_to(folder)), 'original_sha256': expected,
                            'archive': str(packed.relative_to(folder)), 'archive_sha256': runtime.sha(packed)})
    if len(records) != 11:
        raise ValueError('Expected all eleven checkpoint tensors')
    return records


def finish(index):
    runtime.require_compute()
    setup = runtime.read(runtime.CONTROL / 'setup_completed.json')
    if setup['runtime_sha256'] != runtime.sha(runtime.__file__):
        raise ValueError('Original validated runtime differs')
    runner, request = runtime.imports('positive')
    key = f'positive-{index}'
    folder = Path(request['output']) / 'parents' / str(runner.DECLARATION['seeds'][index])
    done = runtime.read(folder / 'completed.json')
    if done['recovery']['training_repeated'] is not False:
        raise ValueError('Expected scoring-only recovery provenance')
    _, final = runner.load_final_manifest(request['final_manifest'], request['manifest'])
    runner.inspect_parent(request, index, final)
    records = retained_weights(folder, runtime.read(folder / 'commitment.json'))
    runtime.write(folder / 'lossless_retention.json', {
        'passed': True, 'job_id': os.environ['SLURM_JOB_ID'], 'records': records,
        'policy': 'Verify each raw or losslessly compressed tensor against its training seal',
        'scientific_outcomes_unchanged': True,
    })
    receipt = runtime.ROOT / 'results' / (key + '.json')
    if not receipt.exists():
        runtime.export_result(folder, key)
    elif runtime.read(receipt)['completed_sha256'] != runtime.sha(folder / 'completed.json'):
        raise ValueError('Existing export refers to different outcomes')
    print(f'Verified and exported {key}', flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        finish(int(sys.argv[1]))
    else:
        runtime.require_compute()
        for parent in runtime.FAILED:
            subprocess.run([sys.executable, __file__, str(parent)], check=True)
