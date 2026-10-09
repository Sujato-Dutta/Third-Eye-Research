"""Standard-library scheduler service for R1 and completed E2 review dependencies."""

from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scheduler(args):
    r=subprocess.run(args,text=True,capture_output=True,timeout=60)
    if r.returncode: raise RuntimeError(r.stdout+r.stderr)
    return r.stdout


def accounting(ids):
    raw=scheduler(['/usr/bin/sacct','-j',','.join(ids),'-X','--format=JobIDRaw,State,ExitCode','-n','-P'])
    result={}
    for line in raw.splitlines():
        if not line.strip(): continue
        job,state,exit_code=line.strip().split('|')[:3]
        if job in ids: result[job]=dict(state=state.split()[0].rstrip('+'),exit_code=exit_code)
    if set(result)!=set(ids): raise RuntimeError('Incomplete scheduler accounting')
    return result


def trim_completed(arguments, statuses):
    result=[]
    for value in arguments:
        if not value.startswith('--dependency='): result.append(value);continue
        if not value.startswith('--dependency=afterany:'): raise ValueError('Only afterany reviews can be reconciled')
        ids=value.split(':')[1:]
        if set(ids)-statuses.keys(): raise ValueError('Every dependency needs accounting')
        remaining=[job for job in ids if statuses[job]['state'] not in TERMINAL]
        if remaining: result.append('--dependency=afterany:'+':'.join(remaining))
    return result


def write(path,record):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record,indent=2));temporary.replace(path)


def checked(root):
    out=root/'runs/a2/empirical_v1/checkpoint_fallback_v3/independent_selection_r1'
    r=json.loads((out/'release.json').read_text())
    if any(sha(root/p)!=h for p,h in r['files'].items()): raise RuntimeError('R1 source changed')
    core=out.parent
    if sha(core/'release.json')!=r['core_release_sha256']: raise RuntimeError('Core release changed')
    receipt=json.loads((out/'cpu_passed.json').read_text())
    if receipt['status']!='passed' or receipt['release_sha256']!=sha(out/'release.json'): raise RuntimeError('R1 CPU validation required')
    return out,r


def relay(root):
    import empirical_checkpoint_relay_v3 as original
    import empirical_relay_v1 as broker
    out,release=checked(root)
    directory=out/'scheduler_reconciliation';directory.mkdir(exist_ok=True)
    def validate(request, deployed):
        original.validate(request,deployed)
        arguments=request['arguments']
        if arguments[-1:]!=['review'] or str(root/'tools/empirical_checkpoint_cpu_v3.slurm') not in arguments: return
        deps=[a for a in arguments if a.startswith('--dependency=')]
        if not deps: return
        ids=deps[0].split(':')[1:]
        known={json.loads(p.read_text())['job_id'] for p in (out.parent/'noise_submissions').glob('task_*.json') if json.loads(p.read_text()).get('status')=='queued'}
        if set(ids)-known: raise RuntimeError('Review dependencies must be retained E2 controls')
        statuses=accounting(ids)
        queue=scheduler(['/usr/bin/squeue','-h','-u','sujato_ts','-o','%i|%T'])
        active={line.split('|')[0]:line.split('|')[1] for line in queue.splitlines()}
        for job in ids:
            if job in active: statuses[job]['state']=active[job]
        normalized=trim_completed(arguments,statuses)
        record=dict(original_arguments=arguments,normalized_arguments=normalized,accounting=statuses,created_utc=datetime.now(timezone.utc).isoformat(),rule='Remove only already-terminal afterany dependencies; failures still produce incomplete review')
        name=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
        write(directory/f'{name}.json',record)
        request['arguments']=normalized
        original.validate(request,deployed)
    broker.validate=validate
    deadline=datetime.fromisoformat(release['scope']['deadline_utc'])+timedelta(hours=2)
    broker.serve(root,out.parent/'submission_queue',deadline)


def watch(root):
    out,release=checked(root)
    target=out/'submission.json'
    if target.exists(): raise RuntimeError('Prior R1 submission attempt; reconcile before retry')
    core=out.parent
    mains=[j['job_id'] for j in json.loads((core/'path_repair_v1/submissions.json').read_text())['jobs']]
    deadline=datetime.fromisoformat(release['scope']['deadline_utc'])+timedelta(hours=2)
    while True:
        noise=[json.loads(p.read_text())['job_id'] for p in (core/'noise_submissions').glob('task_*.json') if json.loads(p.read_text()).get('status')=='queued']
        states=accounting(mains+noise)
        live=scheduler(['/usr/bin/squeue','-h','-u','sujato_ts','-o','%i|%T'])
        active={line.split('|')[0] for line in live.splitlines()}
        ready=all(job not in active and states[job]['state'] in TERMINAL for job in mains+noise)
        write(out/'watch_status.json',dict(status='waiting_for_core_and_controls',updated_utc=datetime.now(timezone.utc).isoformat(),main_jobs=mains,known_noise_jobs=noise,accounting=states))
        if ready or datetime.now(timezone.utc)>=deadline: break
        time.sleep(60)
    # All completed jobs are verified through accounting; no purged dependency.
    attempt=dict(status='submitting_cpu_review',main_jobs=mains,known_noise_jobs=noise,accounting=states,created_utc=datetime.now(timezone.utc).isoformat())
    write(target,attempt)
    raw=scheduler(['/usr/bin/sbatch','--parsable','--account=IRI23021','--partition=gg','--nodes=1','--ntasks=1','--cpus-per-task=72','--time=01:00:00',str(root/'tools/empirical_independent_selection_r1.slurm'),'review'])
    ids=re.findall(r'^\s*(\d+)(?:;[A-Za-z0-9_.-]+)?\s*$',raw,re.M)
    if len(ids)!=1: raise RuntimeError('Uncertain R1 submission; do not retry')
    attempt.update(status='queued_cpu_review',job_id=ids[0]);write(target,attempt)


def main():
    if 'login' not in socket.gethostname().lower() or os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Scheduler services require login host')
    root=Path.cwd()
    if len(sys.argv)!=2 or sys.argv[1] not in ('watch','relay'): raise ValueError('Specify watch or relay')
    {'watch':watch,'relay':relay}[sys.argv[1]](root)


if __name__=='__main__': main()
