"""Validate unchanged Qwen4B training/inference on an allocated 20-GB MIG slice."""
import gc
import json
import os
from pathlib import Path
import socket
import sys
import time

PRIMARY = Path('/dgxa_home/se23uari167/third_eye_dgx_20261026')
AUDIT = Path('/dgxb_home/muef2320/third_eye_account_audit_20261112')
SOURCE = PRIMARY / 'releases/positive_regime_d5c3bc9fb65a'
sys.path[:0] = [str(SOURCE / 'src'), str(SOURCE / 'experiments')]


def main():
    if (os.getuid() != 1189 or not os.environ.get('SLURM_JOB_ID')
            or not os.environ.get('CUDA_VISIBLE_DEVICES') or 'login' in socket.gethostname().lower()):
        raise RuntimeError('Faculty Slurm GPU allocation required')
    os.environ['HF_HUB_CACHE'] = str(PRIMARY / 'models/hf_cache/hub')
    import torch
    import positive_regime_confirmation as runner
    from third_eye.data.schema import Correction
    from third_eye.data.splits import load_manifest
    from third_eye.io import file_digest
    from third_eye.training.hf_backend import HFBackend
    from third_eye.evaluation.runner import verify_completions
    from third_eye.evaluation.verifiers import VerifierRegistry

    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one Slurm-visible MIG device required')
    gpu = torch.cuda.get_device_properties(0)
    if 'MIG' not in gpu.name or not 19_000_000_000 <= gpu.total_memory <= 23_000_000_000:
        raise RuntimeError('Expected one 20-GB A100 MIG slice')
    destination = AUDIT / ('mig_preflight_' + os.environ['SLURM_JOB_ID'] + '.json')
    report = {'passed': False, 'job_id': os.environ['SLURM_JOB_ID'],
              'host': socket.gethostname(), 'gpu': gpu.name, 'gpu_memory_bytes': gpu.total_memory,
              'slurm_cuda_mask': os.environ['CUDA_VISIBLE_DEVICES'],
              'resource_type': 'a100_3g.20gb', 'model': 'qwen3-4b',
              'final_items_evaluated': 0, 'scientific_settings_changed': False,
              'research_update_retained': False, 'stages': {}}
    started = time.perf_counter()
    def publish():
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        temporary.replace(destination)
    def stage(name, operation):
        torch.cuda.reset_peak_memory_stats()
        before = time.perf_counter()
        result = operation()
        torch.cuda.synchronize()
        report['stages'][name] = {'seconds': time.perf_counter() - before,
                                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                  'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
        publish()
        print(json.dumps({'stage': name, **report['stages'][name]}), flush=True)
        return result
    try:
        request_path = PRIMARY / 'runs/positive_regime_20261109/request.json'
        request = runner.checked(request_path)
        os.environ['THIRD_EYE_MODEL_SOURCES'] = request['model_sources']
        cfg = runner.configured(request, runner.DECLARATION['seeds'][0])
        report.update({'request_sha256': file_digest(request_path),
                       'source_tree_sha256': request['source_tree_sha256'],
                       'quantization': cfg.model.quantization, 'dtype': cfg.model.dtype,
                       'generation_batch_size': cfg.protocol.generation_batch_size,
                       'sampled_batch_size': cfg.protocol.sampled_batch_size,
                       'steps': cfg.training.max_steps})
        backend = stage('load', lambda: HFBackend(cfg))
        _, selection = load_manifest(request['manifest'])
        fixtures = sorted(selection['train'], key=lambda e: len(e.prompt + e.answer), reverse=True)[:32]
        corrections = [Correction(e, e.answer, 1) for e in fixtures[:8]]
        verifier = VerifierRegistry.from_spec('third_eye.evaluation.benchmarks:make_verifier')
        if not all(verify_completions(verifier, [c.example for c in corrections], [c.completion for c in corrections])):
            raise RuntimeError('Training fixtures must pass the original verifier')
        original = backend.snapshot()
        original_hash = backend.state_hash()
        training = stage('train_25_steps', lambda: backend.train(corrections, 2026111501, max_steps=25))
        backend.restore(original)
        report['exact_rollback'] = backend.state_hash() == original_hash
        if not report['exact_rollback'] or training['optimizer_steps'] != 25:
            raise RuntimeError('Exact rollback and matched training budget required')
        del original
        gc.collect()
        torch.cuda.empty_cache()
        prompts = [e.prompt for e in fixtures]
        stage('greedy_batch32_tokens512', lambda: backend.generate_many(prompts, 512, list(range(32))))
        gc.collect()
        torch.cuda.empty_cache()
        stage('sampled_batch16_tokens512', lambda: backend.generate_sampled_many(
            prompts[:16], 512, list(range(16)), cfg.protocol.correction_temperature))
        gc.collect()
        torch.cuda.empty_cache()
        # Development items only: no sealed final questions or outcomes are loaded.
        retention = selection['retention_dev']
        mc_prompts = [e.prompt for e in sorted(retention, key=lambda e: len(e.prompt), reverse=True)[:32]]
        if len(mc_prompts) != 32:
            raise RuntimeError('Full-size development MCQ batch required')
        stage('mcq_batch32', lambda: backend.predict_choices(mc_prompts))
        report['passed'] = True
    except Exception as error:
        report['failure'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter() - started
        publish()
        print(json.dumps({'passed': report['passed'], 'report': str(destination)}), flush=True)


if __name__ == '__main__':
    main()
