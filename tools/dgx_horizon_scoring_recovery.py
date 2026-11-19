"""Recover sealed trajectory scoring without repeating training or harvesting."""
import argparse
import importlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path('/dgxa_home/se23ucse205/third_eye_horizon_20261109')
SOURCE = ROOT/'release_adb48501b796'
PYTHON = Path('/dgxa_home/se23uari167/third_eye_dgx_20261026/.venv/bin/python')
MODEL = 'llama3.1-8b'

def read(path):
    return json.loads(Path(path).read_text())

def write(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)

def context(index):
    if os.getuid()!=1339 or not os.environ.get('SLURM_JOB_ID') or 'login' in socket.gethostname().lower():
        raise RuntimeError('Owning-account Slurm compute allocation required')
    sys.path[:0] = [str(SOURCE/'src'),str(SOURCE/'experiments')]
    runner = importlib.import_module('long_horizon_trajectories')
    request = runner.checked(ROOT/'request.json')
    os.environ['THIRD_EYE_MODEL_SOURCES'] = request['model_sources']
    cfg = runner.cohort_config(request,MODEL,index)
    seed = runner.MODELS[MODEL]['seeds'][index]
    folder = ROOT/'parents'/MODEL/str(seed)
    return runner,request,cfg,seed,folder

def audit(index):
    runner,request,cfg,seed,folder = context(index)
    seal = read(folder/'commitment.json')
    expected = {'m0','m1'}|{p+'_t'+str(t) for p in runner.POLICIES for t in range(2,6)}
    if (set(seal['states'])!=expected or seal['inherited_scores'] or seal['primary_binding'] is not None
            or seal['source_tree_sha256']!=request['source_tree_sha256']
            or seal['design_sha256']!=runner.file_digest(folder/'design.json')):
        raise ValueError('Complete unchanged ten-state training seal required')
    for name,state in seal['states'].items():
        checkpoint = folder/'checkpoints'/name
        saved = read(checkpoint/'state.json')
        if (runner.file_digest(checkpoint/'state.json')!=state['state_sha256']
                or saved['adapter_hash']!=state['adapter_hash']):
            raise ValueError('Checkpoint identity changed')
        restored_cfg = runner.saved_config(saved['config'])
        if restored_cfg.fingerprint!=saved['config_hash']:
            raise ValueError('Stored configuration fingerprint differs')
        for key in ('model','protocol','data','evaluation'):
            if saved['config'].get(key)!=cfg.to_dict().get(key):
                raise ValueError('Frozen configuration differs: '+key)
        for filename,digest in saved['weights'].items():
            if Path(filename).name!=filename or runner.file_digest(checkpoint/filename)!=digest:
                raise ValueError('Sealed weight bytes differ')
        if 'donor' in state:
            donor_path = folder/state['donor']
            donor = read(donor_path)
            if (runner.file_digest(donor_path)!=state['donor_sha256']
                    or runner.digest(donor['batch'])!=donor['batch_sha256']):
                raise ValueError('Donor commitment differs')
    record = {'passed':True,'seed':seed,'states':10,'training_repeated':False,
        'commitment_sha256':runner.file_digest(folder/'commitment.json'),
        'operational_script_sha256':runner.file_digest(__file__),
        'request_sha256':runner.file_digest(ROOT/'request.json'),
        'source_tree_sha256':request['source_tree_sha256'],'job_id':os.environ['SLURM_JOB_ID']}
    write(folder/'scoring_recovery_audit.json',record)
    return record

def score(index,name):
    runner,request,cfg,seed,folder = context(index)
    if name not in runner.SCORED:
        raise ValueError('Only frozen scoring checkpoints may be evaluated')
    from third_eye.cluster import require_gpu_allocation
    from third_eye.training.hf_backend import HFBackend
    from third_eye.evaluation.final import load_final_manifest,evaluate_final
    from third_eye.evaluation.verifiers import VerifierRegistry
    import torch
    require_gpu_allocation()
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_properties(0).total_memory<38_000_000_000:
        raise ValueError('One full A100 required')
    if torch.cuda.mem_get_info()[0]<35_000_000_000:
        raise RuntimeError('Insufficient free allocated-GPU memory before model loading')
    seal = read(folder/'commitment.json')
    seal_hash = runner.file_digest(folder/'commitment.json')
    proof = read(folder/'scoring_recovery_audit.json')
    if (not proof['passed'] or proof['commitment_sha256']!=seal_hash
            or proof['operational_script_sha256']!=runner.file_digest(__file__)):
        raise ValueError('Passed same-seal CPU audit required')
    _,final = load_final_manifest(request['final_manifest'],request['manifest'])
    state = seal['states'][name]
    checkpoint = folder/'checkpoints'/name
    if runner.file_digest(checkpoint/'state.json')!=state['state_sha256']:
        raise ValueError('Checkpoint seal changed after audit')
    dest = folder/'scoring'/name
    if (dest/'scores.json').exists():
        runner.read_scored_items(dest,final,state['adapter_hash'],seal_hash)
        return
    if dest.exists():
        evidence = folder/'failure_evidence'
        evidence.mkdir(exist_ok=True)
        dest.rename(evidence/(name+'-partial-before-recovery-'+os.environ['SLURM_JOB_ID']))
    backend = HFBackend(runner.saved_config(read(checkpoint/'state.json')['config']))
    backend.load_checkpoint(checkpoint)
    backend.config = cfg
    verifier = VerifierRegistry.from_spec('third_eye.evaluation.benchmarks:make_verifier')
    started = time.perf_counter()
    evaluate_final(backend,final,verifier,dest,512,0,{
        'commitment_sha256':seal_hash,'checkpoint':name,'parent_seed':seed,
        'model':MODEL,'request_sha256':runner.file_digest(ROOT/'request.json')})
    runner.read_scored_items(dest,final,state['adapter_hash'],seal_hash)
    write(folder/'recovery_timings'/(name+'.json'),{'seconds':time.perf_counter()-started,
        'job_id':os.environ['SLURM_JOB_ID'],'cold_backend':True,'training_repeated':False})

def recover(index):
    runner,request,cfg,seed,folder = context(index)
    audit(index)
    if (folder/'completed.json').exists():
        runner.inspect(request,MODEL,index)
        return
    started = time.perf_counter()
    for name in runner.SCORED:
        # A fresh process for each state prevents allocator history and unused
        # training objects from accumulating across checkpoint evaluations.
        subprocess.run([str(PYTHON),__file__,'score','--index',str(index),'--checkpoint',name],check=True)
    scores = {name:{'scores_sha256':runner.file_digest(folder/'scoring'/name/'scores.json'),
                    'items_sha256':runner.file_digest(folder/'scoring'/name/'items.jsonl'),
                    'seconds':read(folder/'recovery_timings'/(name+'.json'))['seconds']
                        if (folder/'recovery_timings'/(name+'.json')).exists() else 0}
              for name in runner.SCORED}
    previous = read(folder/'progress.json')['elapsed_seconds']
    write(folder/'completed.json',{'passed':True,'model':MODEL,'seed':seed,
        'request_sha256':runner.file_digest(ROOT/'request.json'),
        'commitment_sha256':runner.file_digest(folder/'commitment.json'),
        'source_tree_sha256':request['source_tree_sha256'],'scores':scores,
        'elapsed_seconds':previous+time.perf_counter()-started,'forecaster_fits':0,
        'recovery':{'training_repeated':False,'cold_backend_per_checkpoint':True,
                    'failed_scoring_waste_excluded_from_elapsed':True}})
    runner.inspect(request,MODEL,index)

def controller(index):
    runner,request,cfg,seed,folder = context(index)
    if (folder/'completed.json').exists():
        runner.inspect(request,MODEL,index)
        return
    audit(index)
    submission = folder/'scoring_recovery_submission.json'
    if submission.exists():
        job = str(read(submission)['gpu_job'])
    else:
        job = subprocess.check_output(['sbatch','--parsable',
            '--job-name=te-horizon-'+MODEL+'-'+str(index),
            '--export=ALL,RECOVERY_INDEX='+str(index),
            str(ROOT/'scoring_recovery_gpu_20261118.slurm')],text=True).strip()
        if not job.isdigit():
            raise RuntimeError('Inspect submission before retrying: '+job)
        write(submission,{'passed':True,'audit_job':os.environ['SLURM_JOB_ID'],
            'gpu_job':job,'training_repeated':False,'scientific_completion':False})
    registry = ROOT/'temporary_worker_20261112'/('horizon-'+MODEL+'-'+str(index)+'_completion.json')
    receipt = read(registry)
    if str(receipt['fallback_submitted'])!=job:
        receipt.setdefault('fallback_history',[]).append({
            'job':str(receipt['fallback_submitted']),'state':'FAILED',
            'reason':'sealed trajectory requires scoring-only recovery'})
        receipt.update(fallback_submitted=job,recovery_kind='scoring_only_gpu',
            recovery_proof_path=str(folder/'scoring_recovery_audit.json'),training_repeated=False)
        write(registry,receipt)
    archive_job = '35092' if index==2 else '35094'
    subprocess.run(['scontrol','update','JobId='+archive_job,'Dependency=afterok:'+job],check=True)
    print(json.dumps({'passed':True,'gpu_job':job,'scientific_completion':False,
                      'training_repeated':False}),flush=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('audit','score','recover','controller'))
    parser.add_argument('--index',type=int,choices=(2,3),required=True)
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    if args.mode=='score': score(args.index,args.checkpoint)
    else: globals()[args.mode](args.index)
