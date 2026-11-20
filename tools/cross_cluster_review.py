"""Apply the frozen reviews to verified per-item outcomes across both clusters."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np


DGX = {
    'positive': Path('/dgxa_home/se23uari167/third_eye_dgx_20261026/runs/positive_regime_20261109'),
    'horizon': Path('/dgxa_home/se23ucse205/third_eye_horizon_20261109'),
}
DGX_INDICES = {'positive': tuple(range(13)), 'horizon': (0, 1)}
QWEN = 'qwen3-8b'
OTHER_MODELS = ('llama-3.2-3b-instruct', 'qwen3-4b', 'llama3.1-8b')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def collect(runner, request, kind, indices, verified_archives=()):
    root = Path(request['output'])
    if kind == 'positive':
        _, final = runner.load_final_manifest(request['final_manifest'], request['manifest'])
    data = {role: [] for role in runner.ROLES}
    parents = []
    for index in indices:
        seed = (runner.DECLARATION['seeds'][index] if kind == 'positive'
                else runner.MODELS[QWEN]['seeds'][index])
        folder = root / 'parents'
        if kind == 'horizon':
            folder /= QWEN
        folder /= str(seed)
        if index not in verified_archives and not read(folder / 'lossless_retention.json')['passed']:
            raise ValueError('A verified retained checkpoint is required')
        contrasts = (runner.inspect_parent(request, index, final) if kind == 'positive'
                     else runner.inspect(request, QWEN, index))
        done = read(folder / 'completed.json')
        parent = {'index': index, 'seed': seed, 'elapsed_seconds': done['elapsed_seconds'],
                  'completed_sha256': sha(folder / 'completed.json')}
        if kind == 'positive':
            parent.update(initial_available=done['initial_available'],
                          continuation_available=done['continuation_available'])
        parents.append(parent)
        for role in runner.ROLES:
            data[role].append(contrasts[role])
    return {role: np.stack(xs) for role, xs in data.items()}, parents


def export(kind):
    import runtime as dgx

    root = DGX[kind]
    assert Path(dgx.CAMPAIGN if kind == 'positive' else dgx.ROOT) == root
    runner = dgx.imports()
    request = runner.checked(root / 'request.json')
    data, parents = collect(runner, request, kind, DGX_INDICES[kind])
    out = root / 'review_transfer_20261120'
    out.mkdir(exist_ok=True)
    target = out / 'contrasts.npz'
    with (out / 'contrasts.partial').open('wb') as stream:
        np.savez_compressed(stream, **data)
    (out / 'contrasts.partial').replace(target)
    record = {'passed': True, 'kind': kind, 'parents': parents, 'contrasts_sha256': sha(target),
              'request_sha256': sha(root / 'request.json'), 'job_id': os.environ['SLURM_JOB_ID'],
              'source_tree_sha256': request['source_tree_sha256'], 'other_reviews': {}}
    if kind == 'horizon':
        for model in OTHER_MODELS:
            review = read(root / f'{model}_review.json')
            if not review['passed'] or review['request_sha256'] != record['request_sha256']:
                raise ValueError(f'Unverified DGX model review: {model}')
            record['other_reviews'][model] = review
    write(out / 'receipt.json', record)
    print(f'Exported {len(parents)} inspected {kind} parents', flush=True)


def combine(kind):
    import ls6_runtime as ls6

    runner, request = ls6.imports(kind)
    root = Path(request['output'])
    staged = ls6.CONTROL / 'dgx_review' / kind
    record = read(staged / 'receipt.json')
    request_hash = sha(root / 'request.json')
    if (not record['passed'] or record['kind'] != kind or record['request_sha256'] != request_hash
            or record['source_tree_sha256'] != request['source_tree_sha256']
            or tuple(parent['index'] for parent in record['parents']) != DGX_INDICES[kind]
            or record['contrasts_sha256'] != sha(staged / 'contrasts.npz')):
        raise ValueError('Inspected DGX outcomes or original request identity differ')
    verified = ls6.read(ls6.CONTROL / 'results_verified.json')
    if not verified['passed']:
        raise ValueError('Passed Lonestar6 result-archive verification required')
    keys = {row['key'] for row in verified['results'] if row['status'] == 'verified_for_import'}
    expected = ({f'positive-{i}' for i in range(14, 20)} if kind == 'positive'
                else {f'horizon-{QWEN}-{i}' for i in (2, 3)})
    if not expected.issubset(keys):
        raise ValueError('A required Lonestar6 result archive is missing')
    indices = tuple(range(13, 20)) if kind == 'positive' else (2, 3)
    local, parents = collect(runner, request, kind, indices,
                             verified_archives=indices if kind == 'horizon' else ())
    parents = record['parents'] + parents
    with np.load(staged / 'contrasts.npz', allow_pickle=False) as dgx:
        matrices = {r: np.concatenate((dgx[r], local[r]), axis=0) for r in runner.ROLES}
    result = {'passed': True, 'request_sha256': request_hash, 'declaration': runner.DECLARATION,
              'automatic_expansion': False, 'forecaster_fits': 0, 'job_id': os.environ['SLURM_JOB_ID'],
              'elapsed_parent_seconds': [p['elapsed_seconds'] for p in parents],
              'dgx_export_receipt_sha256': sha(staged / 'receipt.json'),
              'parent_completed_sha256': {str(p['seed']): p['completed_sha256'] for p in parents}}
    if kind == 'positive':
        result.update(statistics=runner.summarize(matrices, runner.DECLARATION['bootstrap_replicates'],
                                                 runner.DECLARATION['bootstrap_seed']),
                      initial_scarce_parents=sum(not p['initial_available'] for p in parents),
                      terminal_draws=sum(3 - sum(p['continuation_available']) for p in parents))
        write(root / 'confirmation_review.json', result)
    else:
        result['statistics'] = runner.summarize(matrices, QWEN)
        write(root / f'{QWEN}_review.json', result)
        all_reviews = {**record['other_reviews'], QWEN: result}
        if set(all_reviews) != set(runner.MODELS):
            raise ValueError('Complete four-model review required')
        write(root / 'integrated_cross_cluster_review.json', {
            'passed': True, 'models': all_reviews, 'declaration': runner.DECLARATION,
            'request_sha256': request_hash, 'forecaster_fits': 0, 'automatic_expansion': False,
        })
    print(f'Completed {kind} review', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('export', 'combine'))
    parser.add_argument('kind', choices=tuple(DGX))
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or 'login' in socket.gethostname().lower():
        raise RuntimeError('A Slurm compute allocation is required')
    (export if args.mode == 'export' else combine)(args.kind)
