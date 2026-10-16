"""Validation-only investigation of nonmatching historical update replay."""

import argparse
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crossed_corrections import donor_input, prepared, read, rows_for
from third_eye.cluster import require_compute_allocation, require_gpu_allocation
from third_eye.config import Config
from third_eye.data.splits import load_manifest
from third_eye.io import digest, file_digest, write_json
from third_eye.provenance import execution_environment, source_inventory, versions


def tensor_digest(tensor):
    import torch

    value = tensor.detach().cpu().contiguous()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(
            value.reshape(-1).view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
    }


def model_buffers(backend):
    return {name: tensor_digest(value) for name, value in backend.model.named_buffers()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--index", required=True, type=int, choices=(0, 1))
    args = parser.parse_args()
    require_compute_allocation()
    require_gpu_allocation("cuda")
    source = source_inventory()
    validation = read(args.validation)
    if (
        validation.get("passed") is not True
        or validation["source_tree_sha256"] != source["source_tree_sha256"]
    ):
        raise ValueError("Full source-bound CPU validation required")
    campaign = Path(args.campaign)
    plan = read(campaign / "plan.json")
    # Reuse the complete original audit, without changing its source identity.
    parent_index = (0, 4)[args.index]
    prepared(plan, parent_index)
    pilot = read(plan["pilot_plan"])
    task = pilot["tasks"][parent_index]
    cfg = Config.load(task["config"])
    rows = rows_for(task, pilot)
    row = rows[0]
    _, splits = load_manifest(task["manifest"])
    draw, batch, _ = donor_input(row, 50_000_000, splits, cfg)
    if not batch:
        raise ValueError("Declared diagnostic native donor is unexpectedly terminal")
    native_directory = (
        Path(row["checkpoints"]["t1"]).parent
        / "continuation_replicates"
        / str(draw["seed"])
    )
    first = json.loads(
        (native_directory / "training.jsonl").read_text().splitlines()[0]
    )
    destination = Path(args.output) / str(args.index)
    destination.mkdir(parents=True, exist_ok=False)
    from third_eye.training.hf_backend import HFBackend
    import torch

    records = []
    # The original template uses process-local wall time. Reconstruct its date
    # from the untouched original training-log mtime, rejecting midnight windows.
    ended = (native_directory / "training.jsonl").stat().st_mtime
    started = ended - draw["training"]["seconds"] - 60
    original_date = datetime.fromtimestamp(ended).strftime("%d %b %Y")
    if datetime.fromtimestamp(started).strftime("%d %b %Y") != original_date:
        raise ValueError("Original training rendering date is ambiguous at midnight")
    for condition in ("fresh", "fresh_repeat", "historical_date_full_update"):
        backend = HFBackend(
            cfg, prompt_context={"date_string": datetime.now().strftime("%d %b %Y")}
        )
        backend.load_checkpoint(row["checkpoints"]["t1"])
        start, saved = backend.state_hash(), backend.snapshot()
        before = model_buffers(backend)
        if condition == "historical_date_full_update":
            backend.prompt_context = {"date_string": original_date}
        encoded_sha = digest(
            [backend._encode(c.example.prompt, c.completion) for c in batch]
        )
        warmed = model_buffers(backend)
        if backend.state_hash() != start:
            raise RuntimeError("Diagnostic forward changed recipient adapter")
        try:
            log = backend.train(
                batch,
                draw["seed"],
                max_steps=cfg.training.max_steps
                if condition == "historical_date_full_update"
                else 1,
                log_path=destination / (condition + ".jsonl"),
            )
            result = {
                "condition": condition,
                "encoded_batch_sha256": encoded_sha,
                "m1_hash": start,
                "one_step_hash": backend.state_hash(),
                "executed_steps": log["optimizer_steps"],
                "historical_date": original_date
                if condition == "historical_date_full_update"
                else None,
                "native_full_adapter_exact_match": (
                    backend.state_hash() == draw["continuation_adapter_hash"]
                    if condition == "historical_date_full_update"
                    else None
                ),
                "training": log,
                "native_first_step": first,
                "native_loss_exact_match": log["loss_first"] == first["loss"],
                "buffers_before": before,
                "buffers_after_context": warmed,
                "buffers_after_training": model_buffers(backend),
                "model_attention_implementation": getattr(
                    backend.model.config, "_attn_implementation", None
                ),
                "tokenizer_padding_side": backend.tokenizer.padding_side,
            }
            write_json(destination / (condition + ".json"), result)
            records.append(result)
        finally:
            backend.config = cfg
            backend.restore(saved)
            if backend.state_hash() != start:
                raise RuntimeError("Diagnostic rollback failed")
        del backend, saved
        gc.collect()
        torch.cuda.empty_cache()
    # Validate actual frozen base files as well as the earlier metadata proof.
    proof = read(Path(row["checkpoints"]["t1"]) / "state.json")["model_source"]
    weights = {}
    if proof and "runtime_directory" in proof:
        for name, expected in proof["verified_weight_sha256"].items():
            weights[name] = {
                "expected": expected,
                "actual": file_digest(Path(proof["runtime_directory"]) / name),
            }
    write_json(
        destination / "completed.json",
        {
            "status": "diagnostic_complete_not_scientific_pass",
            "source_tree_sha256": source["source_tree_sha256"],
            "stream": task["stream"],
            "original_plan_hash": digest(plan),
            "records": records,
            "weight_checks": weights,
            "execution_environment": execution_environment(),
            "versions": versions(),
            "date_reconstruction": {
                "log_mtime": ended,
                "conservative_start": started,
                "date_string": original_date,
                "process_tzname": list(time.tzname),
                "tz_env": __import__("os").environ.get("TZ"),
            },
            "automatic_expansion": False,
            "forbidden_for_forecaster_training": True,
            "limitations": "One-step validation probes only; no new H2 labels, threshold changes or authorization to mix historical and crossed executions.",
        },
    )


if __name__ == "__main__":
    main()
