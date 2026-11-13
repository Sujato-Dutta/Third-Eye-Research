"""One allocated GPU step and exact rollback; no retained research update."""
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
    if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES')
            or 'login' in socket.gethostname().lower() or os.getuid() != 1189):
        raise RuntimeError('Named temporary account and allocated GPU required')
    import torch
    import positive_regime_confirmation as runner
    from third_eye.data.schema import Correction
    from third_eye.data.splits import load_manifest
    from third_eye.evaluation.runner import verify_completions
    from third_eye.evaluation.verifiers import VerifierRegistry
    from third_eye.io import file_digest
    from third_eye.training.hf_backend import HFBackend

    inventory_path = AUDIT / 'inventory.json'
    inventory = json.loads(inventory_path.read_text())
    if inventory['total_conservative_bytes'] + 2_000_000_000 >= 190_000_000_000:
        raise RuntimeError('Insufficient safe temporary storage headroom')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one Slurm-visible GPU')
    gpu = torch.cuda.get_device_properties(0)
    if 'A100' not in gpu.name or gpu.total_memory < 38_000_000_000:
        raise RuntimeError('A full A100 allocation is required')
    request_path = PRIMARY / 'runs/positive_regime_20261109/request.json'
    request = runner.checked(request_path)
    os.environ['THIRD_EYE_MODEL_SOURCES'] = request['model_sources']
    cfg = runner.configured(request, runner.DECLARATION['seeds'][0])
    started = time.perf_counter()
    backend = HFBackend(cfg)
    _, selection = load_manifest(request['manifest'])
    fixtures = [Correction(example, example.answer, 1) for example in selection['train'][:8]]
    verifier = VerifierRegistry.from_spec('third_eye.evaluation.benchmarks:make_verifier')
    if not all(verify_completions(verifier, [c.example for c in fixtures], [c.completion for c in fixtures])):
        raise RuntimeError('Training-only compatibility fixtures must pass their verifier')
    before = backend.snapshot()
    before_hash = backend.state_hash()
    training = backend.train(fixtures, 2026111201, max_steps=1)
    backend.restore(before)
    exact = backend.state_hash() == before_hash
    if not exact or training['optimizer_steps'] != 1:
        raise RuntimeError('One-step training and exact rollback check failed')
    record = {'passed': True, 'job_id': os.environ['SLURM_JOB_ID'],
              'host': socket.gethostname(), 'gpu': gpu.name, 'gpu_memory_bytes': gpu.total_memory,
              'slurm_cuda_mask_preserved': True, 'training': training, 'exact_rollback': exact,
              'source_tree_sha256': request['source_tree_sha256'],
              'primary_request_sha256': file_digest(request_path),
              'inventory_sha256': file_digest(inventory_path),
              'elapsed_seconds': time.perf_counter() - started,
              'research_update': False, 'final_items_evaluated': 0,
              'owner_projects_modified': False}
    destination = AUDIT / 'gpu_preflight.json'
    if destination.exists():
        raise FileExistsError('Preserve a previous compatibility report')
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    temporary.replace(destination)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
