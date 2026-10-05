"""Profile unchanged correction generation in an isolated GPU allocation.

This diagnostic never trains, publishes research labels, changes a protocol,
or resumes a campaign. Its tools directory is outside the scientific inventory.
"""

import argparse
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.trace and args.trace.exists()):
        parser.error("Diagnostic artifacts are immutable; choose new output paths")
    if args.max_new_tokens < 1:
        parser.error("Diagnostic token count must be positive")
    sys.path.insert(0, str(args.repository.resolve() / "src"))
    from third_eye.cluster import require_gpu_allocation

    require_gpu_allocation()
    import torch
    from third_eye.config import Config
    from third_eye.data.splits import load_manifest
    from third_eye.io import digest, write_json
    from third_eye.provenance import execution_environment
    from third_eye.training.hf_backend import HFBackend
    from third_eye.training.sampling import SeededSampling

    config = Config.load(args.config)
    if not str(config.model.device).startswith("cuda"):
        parser.error("This diagnostic requires a CUDA configuration")
    _, splits = load_manifest(args.manifest)
    count = config.protocol.sampled_batch_size
    examples = splits["train"][:count]
    if len(examples) != count or any(ex.split != "train" for ex in examples):
        raise ValueError("Require one full frozen sampled batch of training prompts")
    prompts = [ex.prompt for ex in examples]
    seeds = [
        config.protocol.seed + 100_000 + i * config.protocol.correction_attempts
        for i in range(count)
    ]
    backend = HFBackend(config)
    adapter_before = backend.state_hash()

    def generate():
        return backend.generate_sampled_many(
            prompts,
            args.max_new_tokens,
            seeds,
            config.protocol.correction_temperature,
        )

    # Warm caches, then time the uninstrumented path. Profiling time is not a
    # throughput estimate because collection itself adds overhead.
    generate()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    baseline = generate()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()

    original = SeededSampling.__call__

    def measured(self, input_ids, scores):
        with torch.profiler.record_function("third_eye.seeded_sampling"):
            return original(self, input_ids, scores)

    SeededSampling.__call__ = measured
    try:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as profile:
            instrumented = generate()
            torch.cuda.synchronize()
    finally:
        SeededSampling.__call__ = original
    if baseline != instrumented:
        raise RuntimeError("Instrumented generation changed completions")
    if backend.state_hash() != adapter_before:
        raise RuntimeError("Read-only profiling changed the adapter")
    events = []
    for event in profile.key_averages():
        events.append(
            {
                "name": event.key,
                "calls": event.count,
                "cpu_total_us": event.cpu_time_total,
                "cpu_self_us": event.self_cpu_time_total,
                "device_total_us": getattr(event, "device_time_total", 0),
                "device_self_us": getattr(event, "self_device_time_total", 0),
            }
        )
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        profile.export_chrome_trace(str(args.trace))
    tokens = sum(
        len(backend.tokenizer.encode(x, add_special_tokens=False)) for x in baseline
    )
    report = {
        "status": "passed",
        "purpose": "performance diagnostic; not research labels or a protocol change",
        "execution": execution_environment(),
        "model": config.model.name,
        "revision": backend.resolved_revision,
        "attention_implementation": getattr(
            backend.model.config, "_attn_implementation", None
        ),
        "sampled_batch_size": count,
        "diagnostic_max_new_tokens": args.max_new_tokens,
        "unprofiled_seconds": seconds,
        "decoded_tokens": tokens,
        "decoded_tokens_per_second": tokens / seconds,
        "peak_cuda_bytes": peak,
        "completion_hash": digest(baseline),
        "adapter_unchanged": True,
        "instrumentation_completion_parity": True,
        "sampling_events": [
            x for x in events if x["name"] == "third_eye.seeded_sampling"
        ],
        "top_device_self_events": sorted(
            events, key=lambda x: x["device_self_us"], reverse=True
        )[:25],
        "top_cpu_self_events": sorted(
            events, key=lambda x: x["cpu_self_us"], reverse=True
        )[:25],
        "limitations": "Short training-prompt generation only; excludes full harvesting, "
        "verifier work, retry-prompt lengths and training. Inclusive "
        "profiler times overlap and must not be summed as runtime shares.",
    }
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output),
                "unprofiled_seconds": seconds,
                "decoded_tokens_per_second": tokens / seconds,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
