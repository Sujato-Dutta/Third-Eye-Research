"""Compare inference batching on training prompts without changing research runs."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument(
        "--model-key", choices=("qwen3_4b", "llama3_2_3b"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=64)
    args = parser.parse_args()
    cases_path = args.output.with_suffix(".cases.jsonl")
    if args.output.exists() or cases_path.exists():
        parser.error("Benchmark artifacts are immutable")
    if args.rows < 64:
        parser.error("Use at least 64 training prompts to exercise all batch sizes")
    sys.path.insert(0, str(args.repository.resolve() / "src"))
    from third_eye.cluster import require_gpu_allocation

    require_gpu_allocation()
    import torch
    from third_eye.config import Config
    from third_eye.data.splits import load_manifest
    from third_eye.io import digest, write_json, append_jsonl
    from third_eye.provenance import execution_environment
    from third_eye.training.hf_backend import HFBackend

    config = Config.load(args.config)
    if config.model.device != "cuda" or config.protocol.sampled_batch_size != 16:
        parser.error(
            "Require the unchanged CUDA A1 configuration with sampled batch 16"
        )
    backend = HFBackend(config)
    root = backend.snapshot()
    cases = []
    for task in ("math", "code"):
        _, splits = load_manifest(args.data_root / task / "selection.json")
        examples = splits["train"][: args.rows]
        if len(examples) != args.rows or any(x.split != "train" for x in examples):
            raise ValueError("Require enough unique training prompts")
        seeds = [
            config.protocol.seed + 100_000 + i * config.protocol.correction_attempts
            for i in range(len(examples))
        ]
        prompts = [x.prompt for x in examples]
        pilot = args.pilot_root / f"{args.model_key}_{task}_1042"
        completed = json.loads((pilot / "completed.json").read_text())
        if completed["status"] != "complete" or len(completed["accepted"]) != 1:
            raise ValueError("Require a completed original one-state A1 pilot")
        for state in ("root", "trained"):
            backend.config = config
            backend.restore(root)
            if state == "trained":
                backend.load_checkpoint(pilot / "accepted/generation_1")
            adapter_hash = backend.state_hash()
            baseline = None
            baseline_seconds = None
            for size in (16, 32, 64):
                # This temporary process-local config never enters a campaign.
                backend.config = replace(
                    config, protocol=replace(config.protocol, sampled_batch_size=size)
                )
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                outputs = backend.generate_sampled_many(
                    prompts,
                    config.protocol.max_new_tokens,
                    seeds,
                    config.protocol.correction_temperature,
                )
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                if baseline is None:
                    baseline, baseline_seconds = outputs, seconds
                if backend.state_hash() != adapter_hash:
                    raise RuntimeError("Inference benchmark changed adapter weights")
                case = {
                    "task": task,
                    "adapter_state": state,
                    "batch_size": size,
                    "prompts": len(examples),
                    "max_new_tokens": config.protocol.max_new_tokens,
                    "seconds": seconds,
                    "speedup_over_16": baseline_seconds / seconds,
                    "completion_hash": digest(outputs),
                    "exact_completion_parity": outputs == baseline,
                    "different_completions": sum(
                        a != b for a, b in zip(outputs, baseline)
                    ),
                    "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                    "adapter_unchanged": True,
                }
                cases.append(case)
                append_jsonl(cases_path, case)
                print(json.dumps(case), flush=True)
    backend.config = config
    backend.restore(root)
    matching = [
        size
        for size in (32, 64)
        if all(x["exact_completion_parity"] for x in cases if x["batch_size"] == size)
    ]
    write_json(
        args.output,
        {
            "status": "complete",
            "purpose": "performance/equivalence diagnostic only",
            "model": config.model.name,
            "revision": backend.resolved_revision,
            "execution": execution_environment(),
            "cases": cases,
            "matching_batches_on_tested_prompts": matching,
            "scientific_runs_changed": False,
            "limitations": "Direct training prompts, not complete retry harvesting. "
            "Matching finite cases does not establish whole-pool or label "
            "equivalence. No inference batch change is automatically deployed.",
        },
    )


if __name__ == "__main__":
    main()
