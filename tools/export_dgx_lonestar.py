"""Read-only, compute-node export of the fixed outstanding work, without base weights."""
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import tarfile
import time

P = Path('/dgxa_home/se23uari167/third_eye_dgx_20261026')
H = Path('/dgxa_home/se23ucse205/third_eye_horizon_20261109')
F = Path('/dgxb_home/muef2320/third_eye_account_audit_20261112/parallel')
C = P/'runs/positive_regime_20261109'
FAILED = ((14,35465),(15,35466),(16,35467),(17,35468),(18,35469),(19,35471))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    tmp.replace(path)


def pack(output, paths, metadata):
    if output.exists():
        raise FileExistsError('Preserve the previous export')
    index = {}
    tmp = output.with_suffix('.partial')
    with tarfile.open(tmp, 'w:gz', compresslevel=1, dereference=True) as archive:
        for path in sorted(set(paths)):
            resolved = path.resolve()
            if not any(resolved.is_relative_to(root.resolve()) for root in (P,H)):
                raise ValueError('Outside the explicitly authorized scientific roots')
            name = 'mirror/'+str(path).lstrip('/')
            index[name] = {'original':str(path), 'bytes':path.stat().st_size, 'sha256':sha(path)}
            archive.add(path, arcname=name, recursive=False)
        payload = (json.dumps({'files':index, **metadata}, sort_keys=True, indent=2)+'\n').encode()
        member = tarfile.TarInfo('export_index.json')
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    tmp.replace(output)
    write(output.with_suffix('.receipt.json'), {'passed':True, 'archive':str(output),
        'archive_sha256':sha(output), 'bytes':output.stat().st_size,
        'files':len(index), 'job_id':os.environ['SLURM_JOB_ID'],
        'read_only_sources':True, 'base_weights_exported':False, 'unix_time':time.time()})


def main():
    if (os.getuid()!=1189 or not os.environ.get('SLURM_JOB_ID')
            or 'login' in socket.gethostname().lower()):
        raise RuntimeError('Named temporary-account Slurm CPU allocation required')
    paths = [C/'request.json', H/'request.json', H/'model_plan.json']
    requests = [read(p) for p in (C/'request.json',H/'request.json')]
    for request in requests:
        for key in ('config','template','manifest','final_manifest','development_review','primary_request','model_sources'):
            if key in request:
                path = Path(request[key])
                if sha(path) != request[key+'_sha256']:
                    raise ValueError('Frozen input binding differs: '+key)
                paths.append(path)
        for key in ('manifest','final_manifest'):
            path = Path(request[key])
            for record in read(path)['splits'].values():
                data = (path.parent/record['path']).resolve()
                if sha(data) != record['sha256']:
                    raise ValueError('Frozen example bytes differ')
                paths.append(data)
        mapping = read(request['model_sources'])
        paths += [Path(p) for p in mapping.values()]
        source = Path(request['source'])
        for directory in ('src','experiments','cluster','tests'):
            paths += [p for p in (source/directory).rglob('*')
                      if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')]
        paths += list(source.glob('requirements*.txt'))
        paths += [source/n for n in ('pytest.ini','ruff.toml') if (source/n).exists()]
    meta = {'version':'ls6_migration_20261116', 'positive_request':str(C/'request.json'),
            'horizon_request':str(H/'request.json'), 'model_plan':str(H/'model_plan.json'),
            'scientific_sources_unchanged':True, 'final_outcomes_used_for_transfer_decisions':False}
    pack(F/'lonestar_control_20261116.tar.gz', paths, meta)
    for index, job in FAILED:
        folder = C/'failure_evidence'/f'positive-{index}-{job}'
        seal = read(folder/'commitment.json')
        if len(seal['states']) != 11:
            raise ValueError('All sealed checkpoints required')
        for name, state in seal['states'].items():
            state_path = folder/'checkpoints'/name/'state.json'
            if sha(state_path) != state['state_sha256']:
                raise ValueError('Sealed checkpoint metadata changed')
            for filename, digest in read(state_path)['weights'].items():
                if sha(state_path.parent/filename) != digest:
                    raise ValueError('Sealed checkpoint weights changed')
        pack(F/f'lonestar_parent_{index}_20261116.tar.gz',
             [p for p in folder.rglob('*') if p.is_file()],
             {'parent_index':index,'failed_job':str(job),'original_folder':str(folder),
              'commitment_sha256':sha(folder/'commitment.json'),
              'scientific_outcomes_used_for_transfer_decisions':False})
    print(json.dumps({'passed':True, 'exports':7, 'sources_unchanged':True}), flush=True)


if __name__=='__main__':
    main()
