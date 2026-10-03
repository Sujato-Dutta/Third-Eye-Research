"""Scheduled real-model check of conditional multiple-choice scoring."""

from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from third_eye.cluster import require_gpu_allocation
from third_eye.config import Config
from third_eye.data.splits import load_manifest
from third_eye.io import write_json
from third_eye.training.hf_backend import HFBackend

import torch


def main():
    require_gpu_allocation()
    config = Config.load("experiments/configs/qwen3_4b_batch8_mcq_pilot.json")
    backend = HFBackend(config)
    _, splits = load_manifest("data/processed/v1/math/selection.json")
    prompts = [example.prompt for example in splits["retention_dev"][:8]]
    before, padding = backend.state_hash(), backend.tokenizer.padding_side
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    predictions = backend.predict_choices(prompts)
    seconds = time.perf_counter() - started
    assert len(predictions) == 8 and all(p in "ABCD" for p in predictions)
    assert backend.predict_choices(prompts) == predictions
    backend.config = replace(
        config, protocol=replace(config.protocol, generation_batch_size=1)
    )
    assert backend.predict_choices(prompts) == predictions
    assert backend.state_hash() == before
    assert backend.tokenizer.padding_side == padding
    report = {
        "status": "passed",
        "protocol": "conditional_likelihood",
        "batch_size": config.protocol.generation_batch_size,
        "prompts": len(prompts),
        "seconds": seconds,
        "peak_cuda_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
        "model": config.model.name,
        "revision": backend.resolved_revision,
        "job_id": os.environ["SLURM_JOB_ID"],
        "checks": [
            "repeatability",
            "batched/unbatched agreement",
            "adapter isolation",
            "padding restoration",
        ],
    }
    write_json("runs/choice_verification.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
