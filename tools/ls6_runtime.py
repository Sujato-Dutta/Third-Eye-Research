"""Operational relocation of unchanged frozen experiments onto Lonestar6.

Original requests/configs/manifests retain their bytes and identities. Separate
bindings relocate filesystem paths; the scientific source is never patched.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import datetime as dt
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tarfile
import time

ROOT = Path('/scratch/11617/sujato_ts/third_eye_ls6_20261116')
CONTROL = ROOT/'control'
P = '/dgxa_home/se23uari167/third_eye_dgx_20261026'
H = '/dgxa_home/se23ucse205/third_eye_horizon_20261109'
POS = P+'/runs/positive_regime_20261109'
SOURCES = {'positive':'d5c3bc9fb65a2e416736ff42be7f7ebadf5a44943afc7277c52ef846fed0d1e0',
           'horizon':'adb48501b796b2c20538a0e0142379b13f9f3f659338f6bff18c4200ebd72d2a'}
FAILED = {14:35465,15:35466,16:35467,17:35468,18:35469,19:35471}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def mirror(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts or not any(path.is_relative_to(Path(p)) for p in (P,H)):
        raise ValueError('Unexpected original scientific path')
    return ROOT/'imports/mirror'/str(path).lstrip('/')


def tasks():
    return ([{'key':f'horizon-{model}-{index}', 'kind':'horizon','model':model,'index':index}
             for index in (2,3) for model in ('qwen3-8b','llama3.1-8b')]
            +[{'key':f'positive-{i}','kind':'positive','model':'qwen3-4b','index':i} for i in FAILED])


def task(key):
    return next(t for t in tasks() if t['key']==key)


def require_compute():
    if not os.environ.get('SLURM_JOB_ID') or 'login' in socket.gethostname().lower():
        raise RuntimeError('All setup, hashing, tests, download and research require Slurm compute nodes')
    if os.environ.get('THIRD_EYE_LS6_ROOT',str(ROOT)) != str(ROOT):
        raise ValueError('Unexpected staging root')


def unpack(name):
    archive = ROOT/(name+'.tar.gz')
    receipt = read(ROOT/(name+'.tar.receipt.json'))
    if not receipt['passed'] or sha(archive)!=receipt['archive_sha256']:
        raise ValueError('Export archive checksum differs')
    with tarfile.open(archive,'r:gz') as packed:
        meta = read_index = json.load(packed.extractfile('export_index.json'))
        expected = set(read_index['files']) | {'export_index.json'}
        members = packed.getmembers()
        if {p.name for p in members} != expected or len(members)!=len(expected):
            raise ValueError('Export membership differs')
        for member in members:
            namepath = Path(member.name)
            if (not member.isfile() or namepath.is_absolute() or '..' in namepath.parts
                    or not (ROOT/'imports'/namepath).resolve().is_relative_to((ROOT/'imports').resolve())):
                raise ValueError('Unsafe export member')
            if member.name=='export_index.json':
                continue
            entry = read_index['files'][member.name]
            dest = ROOT/'imports'/member.name
            if dest.exists():
                if dest.is_symlink() or sha(dest)!=entry['sha256']:
                    raise ValueError('Existing imported bytes differ')
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix+'.importing')
            with packed.extractfile(member) as src, tmp.open('xb') as dst:
                shutil.copyfileobj(src,dst,8*1024**2)
            if tmp.stat().st_size!=entry['bytes'] or sha(tmp)!=entry['sha256']:
                raise ValueError('Imported scientific bytes differ')
            tmp.replace(dest)
    return meta


def environment():
    os.environ.update(HF_HOME=str(ROOT/'hf_cache'),HF_HUB_CACHE=str(ROOT/'hf_cache/hub'),
                      HF_HUB_DISABLE_XET='1',TOKENIZERS_PARALLELISM='false')


def install_models(meta):
    from huggingface_hub import hf_hub_download
    plans = read(mirror(meta['model_plan']))['models']
    old_h = read(mirror(meta['horizon_request']))
    mapping, records = {}, {}
    for plan in plans:
        if plan['id'] not in ('qwen3-8b','llama3.1-8b'):
            continue
        folder = ROOT/'models'/plan['id']
        folder.mkdir(parents=True,exist_ok=True)
        def download(item):
            name, expected = item
            path = Path(hf_hub_download(plan['source_repository'],name,
                revision=plan['source_revision'],local_dir=folder,token=False))
            if path.stat().st_size!=expected['bytes']:
                raise ValueError('Pinned model file size differs')
            digest = sha(path)
            if expected.get('sha256'):
                valid = digest==expected['sha256']
            else:
                blob = path.read_bytes()
                valid = hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()==expected['git_blob']
            if not valid:
                raise ValueError('Pinned model file bytes differ: '+name)
            return name,digest
        with ThreadPoolExecutor(max_workers=4) as pool:
            verified = dict(pool.map(download,sorted(plan['files'].items())))
        old_mapping = read(mirror(old_h['model_sources']))
        proof = read(mirror(old_mapping[plan['official']]))
        proof['runtime_directory'] = str(folder)
        if any(verified[n]!=v for n,v in proof['verified_weight_sha256'].items()):
            raise ValueError('Relocated model weight proof differs')
        path = ROOT/'models'/(plan['id']+'_source.json')
        write(path,proof)
        mapping[plan['official']] = str(path)
        records[plan['id']] = {'verified_files':verified,'proof_sha256':sha(path)}
    write(ROOT/'models/model_sources.json',mapping)
    # Keep Qwen4B on its original official repository identity. Introducing a
    # replacement-source proof would invalidate the saved checkpoint identity.
    registry = read(CONTROL/'qwen4_registry.json')
    if registry['revision']!='1cfa9a7208912126459214e8b04321603b3df60c':
        raise ValueError('Pinned Qwen4B revision differs')
    def qwen_file(item):
        name, expected = item
        if not (name.endswith('.safetensors') or name in {
                'config.json','generation_config.json','model.safetensors.index.json',
                'tokenizer.json','tokenizer_config.json','special_tokens_map.json','merges.txt','vocab.json'}
                or 'chat_template' in name):
            return None
        path = Path(hf_hub_download('Qwen/Qwen3-4B',name,revision=registry['revision'],
                                  cache_dir=ROOT/'hf_cache/hub',token=False))
        digest = sha(path)
        if expected.get('sha256'):
            valid = digest==expected['sha256']
        else:
            blob = path.read_bytes()
            valid = hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()==expected['git_blob']
        if not valid:
            raise ValueError('Pinned Qwen4B model bytes differ')
        return name,digest
    with ThreadPoolExecutor(max_workers=4) as pool:
        records['qwen3-4b'] = dict(row for row in pool.map(qwen_file,sorted(registry['files'].items())) if row)
    write(CONTROL/'models_completed.json',{'passed':True,'models':records,'job_id':os.environ['SLURM_JOB_ID']})


def bind(meta):
    for kind, original_path in (('positive',meta['positive_request']),('horizon',meta['horizon_request'])):
        old = read(mirror(original_path))
        if old['source_tree_sha256']!=SOURCES[kind]:
            raise ValueError('Unexpected frozen source identity')
        out = ROOT/kind
        out.mkdir(exist_ok=True)
        request_path = out/'request.json'
        if request_path.exists() and sha(request_path)!=sha(mirror(original_path)):
            raise ValueError('Preserve the original request identity')
        shutil.copyfile(mirror(original_path),request_path)
        bound = dict(old,source=str(mirror(old['source'])),output=str(out),
                     validation=str(out/'validation/completed.json'))
        for field in ('config','template','manifest','final_manifest','development_review','model_sources','primary_request'):
            if field in old:
                bound[field] = str(mirror(old[field]))
                if sha(bound[field])!=old[field+'_sha256']:
                    raise ValueError('Original input bytes differ: '+field)
        if kind=='horizon':
            bound['primary_request'] = str(ROOT/'positive/request.json')
            bound['model_sources'] = str(ROOT/'models/model_sources.json')
            bound['model_sources_sha256'] = sha(bound['model_sources'])
        write(CONTROL/('bound_'+kind+'.json'),bound)
    write(CONTROL/'binding.json',{'version':'ls6_migration_20261116','scientific_sources':SOURCES,
        'original_request_sha256':{k:sha(ROOT/k/'request.json') for k in SOURCES},
        'bound_request_sha256':{k:sha(CONTROL/('bound_'+k+'.json')) for k in SOURCES},
        'changes':'filesystem locations only; scientific configs, manifests, source and seed grid unchanged',
        'historical_zero_vista_su_fields':'unchanged historical declaration; new Lonestar6 charges accounted separately',
        'qwen4_registry_sha256':sha(CONTROL/'qwen4_registry.json'),
        'model_preparation_sha256':sha(CONTROL/'models_completed.json')})


def imports(kind, validated=True):
    binding = read(CONTROL/'binding.json')
    path = CONTROL/('bound_'+kind+'.json')
    if sha(path)!=binding['bound_request_sha256'][kind]:
        raise ValueError('Relocation binding changed')
    request = read(path)
    sys.path[:0] = [str(Path(request['source'])/'src'),str(Path(request['source'])/'experiments')]
    runner = importlib.import_module('positive_regime_confirmation' if kind=='positive' else 'long_horizon_trajectories')
    # checked(False) retains every source/data/recipe check. CPU validation is
    # bound separately to both original request bytes and the relocation map.
    runner.checked(path,validation=False)
    if sha(ROOT/kind/'request.json')!=binding['original_request_sha256'][kind]:
        raise ValueError('Original request identity changed')
    if validated:
        valid = read(ROOT/kind/'validation/completed.json')
        if not valid['passed'] or valid['binding_sha256']!=sha(CONTROL/'binding.json'):
            raise ValueError('Complete bound CPU validation required')
    os.environ['THIRD_EYE_MODEL_SOURCES'] = request['model_sources']
    return runner,request


def validate(kind):
    runner,request = imports(kind,False)
    from third_eye.data.splits import load_manifest
    from third_eye.evaluation.final import load_final_manifest
    _,splits = load_manifest(request['manifest'])
    _,final = load_final_manifest(request['final_manifest'],request['manifest'])
    if len(splits['train'])!=1024 or {k:len(v) for k,v in final.items()}!={
            'target_test':1319,'ood_test':500,'retention_test':1024}:
        raise ValueError('Frozen example counts differ')
    if kind=='positive':
        for index in FAILED:
            runner.configured(request,2026110900+index)
    else:
        for model in ('qwen3-8b','llama3.1-8b'):
            for index in (2,3):
                cfg = runner.cohort_config(request,model,index)
                if cfg.model.dtype!='bfloat16' or cfg.model.quantization!='none':
                    raise ValueError('Precision differs')
    write(ROOT/kind/'validation/completed.json',{'passed':True,
        'binding_sha256':sha(CONTROL/'binding.json'),
        'request_sha256':sha(ROOT/kind/'request.json'),
        'source_tree_sha256':SOURCES[kind],'job_id':os.environ['SLURM_JOB_ID']})


def prepare():
    environment()
    runtime_sha256 = sha(__file__)
    meta = unpack('lonestar_control_20261116')
    source = mirror(read(mirror(meta['positive_request']))['source'])
    uv = ROOT/'bootstrap/bin/uv'
    subprocess.run([str(uv),'pip','install','--python',sys.executable,
                    '-r',str(source/'requirements-evaluation.txt')],check=True)
    import importlib.metadata as metadata
    for package,version in {'torch':'2.8.0','transformers':'4.56.2','peft':'0.17.1',
                            'accelerate':'1.10.1','numpy':'2.2.6','math-verify':'0.8.0'}.items():
        if metadata.version(package).split('+')[0]!=version:
            raise ValueError('Pinned package version differs: '+package)
    (ROOT/'validation').mkdir(parents=True, exist_ok=True)
    for kind,field,names in (
            ('positive','positive_request',['tests/test_positive_regime.py']),
            ('horizon','horizon_request',['tests/test_positive_regime.py','tests/test_long_horizon.py'])):
        frozen = mirror(read(mirror(meta[field]))['source'])
        subprocess.run([sys.executable,'-m','pytest','-q',*names,
                        '--basetemp',str(ROOT/'validation'/kind)],cwd=frozen,check=True)
    install_models(meta)
    bind(meta)
    for kind in SOURCES:
        subprocess.run([sys.executable,__file__,'validate','--kind',kind],check=True)
    # Operational tests exercise relocation, the unique task grid and archive
    # safety. The scientific sources themselves remain byte-identical.
    if len(tasks())!=10 or len({t['key'] for t in tasks()})!=10:
        raise AssertionError('Unique frozen outstanding task grid required')
    for path in ('/etc/passwd',P+'/../secret',H+'/../secret'):
        try:
            mirror(path)
        except ValueError:
            pass
        else:
            raise AssertionError('Path boundary check failed')
    if sha(__file__)!=runtime_sha256:
        raise RuntimeError('Operational runtime changed during setup; revalidate it')
    write(CONTROL/'setup_completed.json',{'passed':True,'binding_sha256':sha(CONTROL/'binding.json'),
        'runtime_sha256':runtime_sha256,
        'job_id':os.environ['SLURM_JOB_ID'],'tasks':tasks(),'new_su_ceiling':197,
        'versions':{p:metadata.version(p) for p in ('torch','transformers','peft','accelerate','numpy','math-verify')}})


def preflight_model(model):
    import torch
    kind = 'positive' if model=='qwen3-4b' else 'horizon'
    runner,request = imports(kind)
    from third_eye.data.schema import Correction
    from third_eye.data.splits import load_manifest
    from third_eye.evaluation.final import load_final_manifest
    from third_eye.evaluation.verifiers import VerifierRegistry
    from third_eye.evaluation.runner import verify_completions
    from third_eye.training.hf_backend import HFBackend
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_properties(0).total_memory<38_000_000_000:
        raise ValueError('One full A100 of at least 40GB required per lane')
    cfg = runner.configured(request,2026110914) if kind=='positive' else runner.cohort_config(request,model,2)
    backend = HFBackend(cfg)
    _,splits = load_manifest(request['manifest'])
    fixtures = [Correction(e,e.answer,1) for e in splits['train'][:8]]
    verifier = VerifierRegistry.from_spec('third_eye.evaluation.benchmarks:make_verifier')
    if not all(verify_completions(verifier,[c.example for c in fixtures],[c.completion for c in fixtures])):
        raise ValueError('Training fixtures fail verification')
    before,snapshot = backend.state_hash(),backend.snapshot()
    fit = backend.train(fixtures,2026111601,max_steps=25)
    backend.restore(snapshot)
    if fit['optimizer_steps']!=25 or backend.state_hash()!=before:
        raise ValueError('Full optimizer budget/rollback failed')
    examples = sorted(splits['train']+splits['target_dev']+splits['ood_dev'],
                      key=lambda e:len(backend.tokenizer.encode(e.prompt)),reverse=True)
    generated = backend.generate_many([e.prompt for e in examples[:32]],512,[0]*32)
    sampled = backend.generate_sampled_many([e.prompt for e in examples[:16]],512,
        list(range(16)),cfg.protocol.correction_temperature)
    if len(generated)!=32 or len(sampled)!=16:
        raise ValueError('Frozen generation batches incomplete')
    _,final = load_final_manifest(request['final_manifest'],request['manifest'])
    # Exercise the entire frozen retention role, including its true longest
    # prompts. Outcomes are discarded and cannot affect any scientific choice.
    choices = backend.predict_choices([e.prompt for e in final['retention_test']])
    if len(choices)!=1024 or backend.state_hash()!=before:
        raise ValueError('Full retention scoring/identity failed')
    proof = {'passed':True,'model':model,'source_tree_sha256':SOURCES[kind],
        'request_sha256':sha(ROOT/kind/'request.json'),'binding_sha256':sha(CONTROL/'binding.json'),
        'training':fit,'exact_rollback':True,'research_update':False,
        'generation_batch':32,'sampled_batch':16,'max_new_tokens':512,
        'retention_items_exercised':1024,'retention_outcomes_retained':False,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'gpu':torch.cuda.get_device_properties(0).name,'gpu_bytes':torch.cuda.get_device_properties(0).total_memory,
        'config':cfg.to_dict(),'job_id':os.environ['SLURM_JOB_ID']}
    write(CONTROL/('preflight_'+model+'.json'),proof)
    if kind=='horizon':
        write(ROOT/'horizon/compatibility'/model/'completed.json',proof)


def gpu_lanes():
    import torch
    count = torch.cuda.device_count()
    if count!=3 or any(torch.cuda.get_device_properties(i).total_memory<38_000_000_000 for i in range(count)):
        raise ValueError('A standard Lonestar6 node with three full 40GB A100s is required')
    mask = os.environ.get('CUDA_VISIBLE_DEVICES')
    lanes = mask.split(',') if mask else [str(i) for i in range(count)]
    if len(lanes)!=3:
        raise ValueError('Unexpected scheduler GPU mask')
    return lanes


def preflight():
    setup = read(CONTROL/'setup_completed.json')
    if not setup['passed'] or setup['runtime_sha256']!=sha(__file__):
        raise ValueError('CPU setup required')
    lanes = gpu_lanes()
    processes = [subprocess.Popen([sys.executable,__file__,'preflight-model','--model',model],
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=lane))
        for model,lane in zip(('qwen3-4b','qwen3-8b','llama3.1-8b'),lanes)]
    exits = [p.wait() for p in processes]
    if any(exits):
        raise RuntimeError('Real full-workload model validation failed: '+str(exits))
    write(CONTROL/'preflight_completed.json',{'passed':True,'binding_sha256':setup['binding_sha256'],
        'models':['qwen3-4b','qwen3-8b','llama3.1-8b'],'job_id':os.environ['SLURM_JOB_ID']})


def ownership():
    if read(CONTROL/'setup_completed.json')['runtime_sha256']!=sha(__file__):
        raise ValueError('Operational runtime differs from CPU validation')
    record = read(CONTROL/'ownership.json')
    if not record['passed'] or set(record['tasks'])!={t['key'] for t in tasks()}:
        raise ValueError('Complete cross-cluster ownership receipt required')
    for key,entry in record['tasks'].items():
        if entry['run_on_lonestar'] and entry['dgx_state']!='PENDING_HELD':
            raise ValueError('Only an explicitly held DGX seed can be migrated')
        if not entry['run_on_lonestar'] and entry['dgx_state'] not in ('RUNNING','COMPLETING','COMPLETED'):
            raise ValueError('Original execution must remain attributable')
    for index in FAILED:
        if record['tasks'][f'positive-{index}']['run_on_lonestar']:
            exported = unpack(f'lonestar_parent_{index}_20261116')
            runner,request = imports('positive')
            cfg = runner.configured(request,2026110900+index)
            folder = mirror(exported['original_folder'])
            seal = read(folder/'commitment.json')
            if sha(folder/'commitment.json')!=exported['commitment_sha256']:
                raise ValueError('Original training seal differs')
            for name,entry in seal['states'].items():
                saved = read(folder/'checkpoints'/name/'state.json')
                expected = replace(cfg,training=replace(cfg.training,
                    learning_rate=entry.get('learning_rate',cfg.training.learning_rate)))
                if expected.fingerprint!=saved['config_hash'] or saved['model_source'] is not None:
                    raise ValueError('Saved checkpoint cannot be loaded with the frozen configuration')
    write(CONTROL/'ownership_completed.json',{'passed':True,'ownership_sha256':sha(CONTROL/'ownership.json'),
        'binding_sha256':sha(CONTROL/'binding.json'),'job_id':os.environ['SLURM_JOB_ID']})


def recover_positive(request,runner,index):
    from third_eye.training.hf_backend import HFBackend
    from third_eye.evaluation.final import load_final_manifest,evaluate_final
    from third_eye.evaluation.verifiers import VerifierRegistry
    from third_eye.experiments.confirmation import read_scored_items
    seed = 2026110900+index
    folder = Path(request['output'])/'parents'/str(seed)
    folder.parent.mkdir(exist_ok=True)
    origin = mirror(POS+f'/failure_evidence/positive-{index}-{FAILED[index]}')
    if not folder.exists():
        origin.rename(folder)
    seal = read(folder/'commitment.json')
    seal_hash = sha(folder/'commitment.json')
    if sha(folder/'design.json')!=seal['design_sha256']:
        raise ValueError('Original training commitment changed')
    cfg = runner.configured(request,seed)
    backend = HFBackend(cfg)
    verifier = VerifierRegistry.from_spec('third_eye.evaluation.benchmarks:make_verifier')
    _,final = load_final_manifest(request['final_manifest'],request['manifest'])
    started = time.perf_counter()
    scores,seen = {},{}
    names = ['m0','m1']+[f'd{d}_{arm}' for d in range(3) for arm in ('native','direct','mitigated')]
    for name in names:
        state = seal['states'][name]
        checkpoint = folder/'checkpoints'/name
        if sha(checkpoint/'state.json')!=state['state_sha256']:
            raise ValueError('Sealed state changed')
        # Half-LR checkpoint configs were sealed under their actual training LR.
        backend.config = replace(cfg,training=replace(cfg.training,
            learning_rate=state.get('learning_rate',cfg.training.learning_rate)))
        backend.load_checkpoint(checkpoint)
        backend.config = cfg
        dest = folder/'scoring'/name
        metadata = {'commitment_sha256':seal_hash,'checkpoint':name,'seed':seed,
                    'request_sha256':sha(ROOT/'positive/request.json')}
        stamp = time.perf_counter()
        if dest.exists() and not (dest/'scores.json').exists():
            evidence = folder/'failure_evidence'
            evidence.mkdir(exist_ok=True)
            dest.rename(evidence/(name+'-partial-'+os.environ['SLURM_JOB_ID']))
        if (dest/'scores.json').exists():
            read_scored_items(dest,final,state['adapter_hash'],seal_hash)
        elif state['adapter_hash'] in seen:
            prior = folder/'scoring'/seen[state['adapter_hash']]
            dest.mkdir()
            shutil.copyfile(prior/'items.jsonl',dest/'items.jsonl')
            write(dest/'scores.json',{**read(prior/'scores.json'),**metadata,
                'identity_reuse':str(prior),'identity_reuse_items_sha256':sha(prior/'items.jsonl')})
        else:
            evaluate_final(backend,final,verifier,dest,512,0,metadata)
        seen[state['adapter_hash']] = name
        scores[name] = {'scores_sha256':sha(dest/'scores.json'),
                        'items_sha256':sha(dest/'items.jsonl'),'seconds':time.perf_counter()-stamp}
        write(folder/'progress_recovery.json',{'phase':'final_scoring','scores':scores,
            'recovery_job':os.environ['SLURM_JOB_ID'],'training_repeated':False})
    initial = read(folder/'initial_pool.json')
    previous_elapsed = read(folder/'progress.json')['elapsed_seconds']
    write(folder/'completed.json',{'passed':True,'seed':seed,'initial_available':len(initial['batch'])==8,
        'continuation_available':[read(folder/'donors'/f'{d}.json')['continuation_available'] for d in range(3)],
        'elapsed_seconds':previous_elapsed+time.perf_counter()-started,'commitment_sha256':seal_hash,
        'request_sha256':sha(ROOT/'positive/request.json'),'source_tree_sha256':SOURCES['positive'],
        'scores':scores,'forecaster_fits':0,'recovery':{'training_repeated':False,
            'original_failed_job':str(FAILED[index]),'original_partial_evidence_preserved':True,
            'original_failed_scoring_walltime_not_in_elapsed_seconds':True,
            'binding_sha256':sha(CONTROL/'binding.json')}})
    runner.inspect_parent(request,index,final)
    from third_eye.experiments.scale import compact_checkpoints
    compact_checkpoints(folder)
    return folder


def export_result(folder,key):
    destination = ROOT/'results'
    destination.mkdir(exist_ok=True)
    archive = destination/(key+'.tar.gz')
    if archive.exists():
        raise FileExistsError('Never replace an exported scientific result')
    inventory = {}
    with tarfile.open(archive.with_suffix('.partial'),'w:gz',compresslevel=1,dereference=True) as packed:
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                relative = path.relative_to(folder).as_posix()
                inventory[relative] = {'sha256':sha(path),'bytes':path.stat().st_size}
                packed.add(path,arcname=relative,recursive=False)
    archive.with_suffix('.partial').replace(archive)
    write(destination/(key+'.json'),{'passed':True,'task':task(key),'archive_sha256':sha(archive),
        'files':inventory,'completed_sha256':sha(folder/'completed.json'),
        'original_request_sha256':sha(ROOT/task(key)['kind']/'request.json'),
        'binding_sha256':sha(CONTROL/'binding.json'),'job_id':os.environ['SLURM_JOB_ID']})


def execute(key):
    row = task(key)
    runner,request = imports(row['kind'])
    if row['kind']=='positive':
        folder = recover_positive(request,runner,row['index'])
    else:
        runner.run(request,row['model'],row['index'])
        runner.inspect(request,row['model'],row['index'])
        folder = Path(request['output'])/'parents'/row['model']/str(runner.MODELS[row['model']]['seeds'][row['index']])
        # Original DGX archive dependencies compact these raw checkpoints after
        # verified import; preserve their existing archival/review pipeline.
    export_result(folder,key)


def lane(gpu,bundle):
    owners = read(CONTROL/'ownership.json')['tasks']
    claims = CONTROL/'claims'
    claims.mkdir(exist_ok=True)
    failures = []
    for row in tasks():
        if not owners[row['key']]['run_on_lonestar']:
            continue
        claim = claims/row['key']
        try:
            claim.mkdir()
        except FileExistsError:
            continue
        write(claim/'owner.json',{'job_id':os.environ['SLURM_JOB_ID'],'bundle':bundle,'gpu_mask':gpu})
        log = (ROOT/'logs'/(row['key']+'-'+os.environ['SLURM_JOB_ID']+'.log')).open('x')
        with log:
            code = subprocess.call([sys.executable,__file__,'task','--key',row['key']],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT)
        write(claim/'exit.json',{'exit_code':code,'result_available':(ROOT/'results'/(row['key']+'.json')).exists()})
        if code:
            failures.append(row['key'])
    return failures


def bundle(name):
    if read(CONTROL/'setup_completed.json')['runtime_sha256']!=sha(__file__):
        raise ValueError('Operational runtime differs from CPU validation')
    deadline = dt.datetime.fromisoformat(read(CONTROL/'bound_horizon.json')['deadline_utc'].replace('Z','+00:00'))
    if (deadline-dt.datetime.now(dt.timezone.utc)).total_seconds()<32.5*3600:
        raise RuntimeError('Protective runtime no longer fits the original execution window')
    for stage in ('setup','preflight','ownership'):
        record = read(CONTROL/(stage+'_completed.json'))
        if not record['passed'] or record['binding_sha256']!=sha(CONTROL/'binding.json'):
            raise ValueError('Complete matching prerequisite required: '+stage)
    if read(CONTROL/'ownership_completed.json')['ownership_sha256']!=sha(CONTROL/'ownership.json'):
        raise ValueError('Ownership changed')
    with ThreadPoolExecutor(max_workers=3) as pool:
        failures = list(pool.map(lambda gpu:lane(gpu,name),gpu_lanes()))
    write(CONTROL/(name+'_completed.json'),{'passed':not any(failures),'failures':failures,
        'job_id':os.environ['SLURM_JOB_ID'],'charged_node_hour_rate':3,'node_walltime_ceiling_hours':32})
    if any(failures):
        raise RuntimeError('Preserved task failures require inspection; no silent retries')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('prepare','validate','preflight','preflight-model','ownership','bundle0','bundle1','task'))
    parser.add_argument('--kind',choices=SOURCES)
    parser.add_argument('--model')
    parser.add_argument('--key')
    args = parser.parse_args()
    require_compute()
    environment()
    if args.mode=='prepare': prepare()
    elif args.mode=='validate': validate(args.kind)
    elif args.mode=='preflight': preflight()
    elif args.mode=='preflight-model': preflight_model(args.model)
    elif args.mode=='ownership': ownership()
    elif args.mode.startswith('bundle'): bundle(args.mode)
    else: execute(args.key)


if __name__=='__main__':
    main()
