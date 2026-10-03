"""Benchmark allocated GPU batching without claiming research accuracy."""

from dataclasses import replace
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from third_eye.cluster import require_gpu_allocation
from third_eye.config import Config
from third_eye.data.splits import load_manifest
from third_eye.data.schema import Correction, Example
from third_eye.io import digest, write_json
from third_eye.provenance import execution_environment
from third_eye.training.hf_backend import HFBackend
from verify_backend import make_tiny_backend
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantization", choices=("none", "nf4"), default="nf4")
    parser.add_argument("--output", default="runs/gpu_resource_profile.json")
    parser.add_argument(
        "--config", default="experiments/configs/qwen3_4b_batch8_mcq_b2_pilot.json"
    )
    parser.add_argument("--manifest", default="data/processed/v1/math/selection.json")
    parser.add_argument(
        "--correction-pool",
        help="Optional saved training-only correction pool for update/rollback smoke",
    )
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Profiling artifacts are immutable; choose a new output")
    require_gpu_allocation()
    tiny = make_tiny_backend("cuda", args.quantization)
    prompts, seeds = ["one", "one plus two", "two plus two"], [30, 31, 32]
    original = [
        tiny.generate(p, 8, seed=s, temperature=0.7) for p, s in zip(prompts, seeds)
    ]
    tiny.config = replace(
        tiny.config, protocol=replace(tiny.config.protocol, sampled_batch_size=3)
    )
    before = tiny.state_hash()
    assert tiny.generate_sampled_many(prompts, 8, seeds, 0.7) == original
    assert (
        tiny.generate_sampled_many(prompts[::-1], 8, seeds[::-1], 0.7) == original[::-1]
    )
    assert tiny.state_hash() == before
    del tiny
    torch.cuda.empty_cache()
    config = Config.load(args.config)
    config = replace(
        config, model=replace(config.model, quantization=args.quantization)
    )
    backend = HFBackend(config)
    _, splits = load_manifest(args.manifest)
    prompts = [ex.prompt for ex in splits["train"][:32]]
    before = backend.state_hash()
    report = {
        "status": "passed",
        "execution": execution_environment(),
        "model": config.model.name,
        "revision": backend.resolved_revision,
        "quantization": args.quantization,
        "cases": [],
    }
    for size in [8, 16, 32]:
        backend.config = replace(
            config, protocol=replace(config.protocol, generation_batch_size=size)
        )
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        outputs = backend.generate_many(prompts, 128, [42] * len(prompts))
        elapsed = time.perf_counter() - started
        case = {
            "mode": "greedy",
            "batch_size": size,
            "prompts": len(prompts),
            "seconds": elapsed,
            "tokens": sum(
                len(backend.tokenizer.encode(s, add_special_tokens=False))
                for s in outputs
            ),
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "completion_hash": digest(outputs),
        }
        report["cases"].append(case)
        print(json.dumps(case), flush=True)
    prompts, seeds = prompts[:8], list(range(100042, 100050))
    for size in [1, 8, 16]:
        backend.config = replace(
            config, protocol=replace(config.protocol, sampled_batch_size=size)
        )
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        outputs = backend.generate_sampled_many(prompts, 64, seeds, 0.7)
        elapsed = time.perf_counter() - started
        assert backend.generate_sampled_many(prompts, 64, seeds, 0.7) == outputs
        case = {
            "mode": "sampled",
            "batch_size": size,
            "prompts": len(prompts),
            "seconds": elapsed,
            "tokens": sum(
                len(backend.tokenizer.encode(s, add_special_tokens=False))
                for s in outputs
            ),
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "completion_hash": digest(outputs),
        }
        report["cases"].append(case)
        print(json.dumps(case), flush=True)
    assert backend.state_hash() == before
    # Also exercise intended 16-row retries and a larger retention batch.
    backend.config = replace(
        config,
        protocol=replace(
            config.protocol, sampled_batch_size=16, generation_batch_size=32
        ),
    )
    prompts = [ex.prompt for ex in splits["train"][:16]]
    first = backend.generate_sampled_many(prompts, 64, list(range(16)), 0.7)
    assert backend.generate_sampled_many(prompts, 64, list(range(16)), 0.7) == first
    assert (
        len(backend.predict_choices([ex.prompt for ex in splits["retention_dev"][:32]]))
        == 32
    )
    assert backend.state_hash() == before
    if args.correction_pool:
        saved = json.loads(Path(args.correction_pool).read_text())
        corrections = [
            Correction(Example(**item["example"]), item["completion"], item["attempt"])
            for item in saved["items"][:2]
        ]
        parent = backend.snapshot()
        report["training_smoke"] = backend.train(corrections, 42, max_steps=1)
        assert backend.state_hash() != before
        backend.restore(parent)
        assert backend.state_hash() == before
    write_json(args.output, report)


if __name__ == "__main__":
    main()
