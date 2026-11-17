"""Read-only, compute-node verification of the six preserved MIG attempts."""
import json
import os
import socket
import time

from dgx_parallel_runtime import AUDIT, PRIMARY, read, sha, write


def main():
    if (os.getuid() != 1189 or not os.environ.get('SLURM_JOB_ID')
            or 'login' in socket.gethostname().lower()):
        raise RuntimeError('Temporary-account Slurm CPU allocation required')
    root = PRIMARY / 'runs/positive_regime_20261109/failure_evidence'
    expected = {'m0', 'm1'} | {f'd{d}_{arm}' for d in range(3)
                                            for arm in ('native', 'direct', 'mitigated')}
    rows = []
    for index, job in ((14,35465),(15,35466),(16,35467),(17,35468),(18,35469),(19,35471)):
        folder = root / f'positive-{index}-{job}'
        row = {'index': index, 'failed_job': str(job), 'folder': str(folder)}
        try:
            commitment = read(folder / 'commitment.json')
            if set(commitment['states']) != expected:
                raise ValueError('Incomplete sealed checkpoint set')
            if sha(folder / 'design.json') != commitment['design_sha256']:
                raise ValueError('Design commitment differs')
            files = []
            for name, sealed in commitment['states'].items():
                state_path = folder / 'checkpoints' / name / 'state.json'
                if sha(state_path) != sealed['state_sha256']:
                    raise ValueError('Checkpoint commitment differs: ' + name)
                state = read(state_path)
                if state['adapter_hash'] != sealed['adapter_hash']:
                    raise ValueError('Adapter identity differs: ' + name)
                for filename, digest in state['weights'].items():
                    path = state_path.parent / filename
                    if (path.is_symlink() or not path.resolve().is_relative_to(folder.resolve())
                            or sha(path) != digest):
                        raise ValueError('Stored checkpoint bytes differ: ' + name)
                    files.append({'checkpoint': name, 'filename': filename,
                                  'bytes': path.stat().st_size, 'sha256': digest})
            row.update(passed=True, checkpoints=len(expected), weights=files,
                       commitment_sha256=sha(folder/'commitment.json'),
                       completed_present=(folder/'completed.json').exists(),
                       scoring_directories=sorted(p.name for p in (folder/'scoring').iterdir()),
                       checkpoint_bytes=sum(record['bytes'] for record in files))
        except (OSError, ValueError, KeyError) as error:
            row.update(passed=False, error=str(error))
        rows.append(row)
    write(AUDIT/'parallel/failed_checkpoint_inventory_20261116.json', {
        'unix_time':time.time(), 'job_id':os.environ['SLURM_JOB_ID'],
        'read_only':True, 'scientific_outcomes_inspected':False,
        'results':rows, 'all_sealed_checkpoint_bytes_verified':all(r['passed'] for r in rows)})
    print(json.dumps({'verified':sum(r['passed'] for r in rows), 'attempts':len(rows)}), flush=True)


if __name__ == '__main__':
    main()
