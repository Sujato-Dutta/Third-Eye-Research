"""Compute-node verification of transferred outputs; no outcome-based selection."""
import hashlib
import json
import os
from pathlib import Path
import socket
import tarfile
import time

ROOT = Path('/scratch/11617/sujato_ts/third_eye_ls6_20261116')


def sha(stream):
    h = hashlib.sha256()
    for block in iter(lambda:stream.read(8*1024**2),b''):
        h.update(block)
    return h.hexdigest()


def main():
    if not os.environ.get('SLURM_JOB_ID') or 'login' in socket.gethostname().lower():
        raise RuntimeError('A Slurm CPU allocation is required')
    owners = json.loads((ROOT/'control/ownership.json').read_text())
    rows = []
    for key,owner in owners['tasks'].items():
        row = {'key':key,'dgx_job':owner['dgx_job'],'migrated':owner['run_on_lonestar']}
        if not owner['run_on_lonestar']:
            row['status'] = 'retained_on_dgx'
            rows.append(row)
            continue
        try:
            receipt = json.loads((ROOT/'results'/(key+'.json')).read_text())
            archive = ROOT/'results'/(key+'.tar.gz')
            with archive.open('rb') as stream:
                if sha(stream)!=receipt['archive_sha256'] or not receipt['passed']:
                    raise ValueError('Result archive identity differs')
            with tarfile.open(archive,'r:gz') as packed:
                entries = packed.getmembers()
                if {p.name for p in entries}!=set(receipt['files']) or len(entries)!=len(receipt['files']):
                    raise ValueError('Result archive membership differs')
                for member in entries:
                    if (not member.isfile() or Path(member.name).is_absolute()
                            or '..' in Path(member.name).parts):
                        raise ValueError('Unsafe result archive member')
                    with packed.extractfile(member) as stream:
                        if sha(stream)!=receipt['files'][member.name]['sha256']:
                            raise ValueError('Transferred result bytes differ')
            row.update(status='verified_for_import',archive_sha256=receipt['archive_sha256'])
        except (OSError,ValueError,KeyError) as error:
            row.update(status='incomplete_or_invalid',error=str(error))
        rows.append(row)
    result = {'passed':all(r['status']!='incomplete_or_invalid' for r in rows),
        'results':rows,'job_id':os.environ['SLURM_JOB_ID'],'unix_time':time.time(),
        'scientific_outcomes_used_for_decisions':False,
        'complete_cohort_science_reviews':'Existing DGX reviews follow verified result import'}
    path = ROOT/'control/results_verified.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    temporary.replace(path)
    print(json.dumps({'passed':result['passed'],'counts':{
        value:sum(r['status']==value for r in rows)
        for value in ('verified_for_import','retained_on_dgx','incomplete_or_invalid')}}),flush=True)


if __name__=='__main__':
    main()
