"""Development-only paired training probes and cluster admission checks."""

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np

from third_eye.cluster import require_compute_allocation, require_gpu_allocation
from third_eye.data.schema import Correction, Example
from third_eye.data.splits import load_manifest
from third_eye.evaluation.runner import generate_greedy, verify_completions
from third_eye.evaluation.verifiers import VerifierRegistry
from third_eye.io import digest, file_digest, write_json

ROLES = ("target_dev", "ood_dev", "retention_dev")
PROBE_STEPS = 2
# A smoke configuration. Method hyperparameters are chosen on development seeds.
SMOKE_LAMBDA = 0.5
SMOKE_PROBE_ITEMS = 12


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def score(gain, effects_before, fixed_mask, coefficient):
    effects_before, fixed_mask = np.asarray(effects_before), np.asarray(fixed_mask)
    if (effects_before.shape != fixed_mask.shape or fixed_mask.dtype != np.bool_ or
            not np.isfinite(effects_before).all() or not np.isfinite(gain) or
            not np.isfinite(coefficient) or coefficient < 0):
        raise ValueError("Need finite matched item effects, a fixed-item mask and lambda >= 0")
    penalty = float(np.maximum(effects_before[fixed_mask], 0).mean()) if fixed_mask.any() else 0.0
    return float(gain - coefficient * penalty), penalty


def probe(backend, start, expected_hash, batch, optimizer_seed, evaluate, steps=PROBE_STEPS):
    """Measure effects on a temporary adapter and verify rollback even after errors."""
    backend.restore(start)
    if backend.state_hash() != expected_hash:
        raise ValueError("Probe did not start at its committed adapter")
    try:
        training = backend.train(batch, optimizer_seed, max_steps=steps)
        if training["optimizer_steps"] != steps:
            raise ValueError("Probe optimizer budget changed")
        values = evaluate()
        return values, training
    finally:
        backend.restore(start)
        if backend.state_hash() != expected_hash:
            raise RuntimeError("Probe did not restore its starting adapter")


def checked_inputs(args):
    request = read(args.request)
    if file_digest(request["manifest"]) != request["manifest_sha256"]:
        raise ValueError("Development manifest checksum differs")
    source = read(args.registry)
    if source["model"] != "Qwen/Qwen3-4B" or source["revision"] != source["source_revision"]:
        raise ValueError("Expected the pinned official Qwen3-4B source")
    if file_digest(args.parent / "completed.json") != args.parent_sha256:
        raise ValueError("The archived parent differs from the integrated review")
    completion = read(args.parent / "completed.json")
    seal_path = args.parent / "commitment.json"
    seal = read(seal_path)
    if not completion["passed"] or file_digest(seal_path) != completion["commitment_sha256"]:
        raise ValueError("Archived parent commitment differs")
    for state in ("m0", "m1"):
        path = args.parent / "checkpoints" / state / "state.json"
        if file_digest(path) != seal["states"][state]["state_sha256"]:
            raise ValueError("Archived state checksum differs")
        saved = read(path)
        if (saved["adapter_hash"] != seal["states"][state]["adapter_hash"] or
                saved["resolved_revision"] != source["revision"]):
            raise ValueError("Archived adapter source differs")
        for name in saved["weights"]:
            weight = path.parent / name
            if not os.access(weight, os.R_OK) and not os.access(str(weight)+".gz", os.R_OK):
                raise FileNotFoundError(weight)
    _, splits = load_manifest(request["manifest"])
    train = {example.id: example for example in splits["train"]}
    batches = []
    for draw, expected in enumerate(seal["donor_sha256"]):
        path = args.parent / "donors" / f"{draw}.json"
        donor = read(path)
        if file_digest(path) != expected or digest(donor["batch"]) != donor["batch_sha256"]:
            raise ValueError("Donor correction checksum differs")
        corrections = [Correction(Example(**row["example"]), row["completion"], row["attempt"])
                       for row in donor["batch"]]
        if (len(corrections) != 8 or len({c.example.id for c in corrections}) != 8 or
                any(train.get(c.example.id) != c.example for c in corrections)):
            raise ValueError("Expected eight distinct corrections from the bound training split")
        batches.append(corrections)
    if len(batches) != 3:
        raise ValueError("Three archived candidate batches are required")
    return splits, batches


def admission(args):
    from huggingface_hub import snapshot_download

    checked_inputs(args)
    source = read(args.registry)
    location = Path(snapshot_download(
        source["model"], revision=source["revision"], local_files_only=True,
    ))
    for name, expected in source["files"].items():
        path = location / name
        if path.stat().st_size != expected["bytes"] or file_digest(path) != expected["sha256"]:
            raise ValueError(f"Cache file differs: {name}")
    disk = shutil.disk_usage(tempfile.gettempdir())
    if disk.free < 2_000_000_000:
        raise ValueError("Need 2 GB of node-local space for temporary adapters")
    usage = None
    if args.home_cap is not None:
        usage = int(subprocess.check_output(
            ["du", "-sx", "-B1", str(Path.home())], text=True,
        ).split()[0])
        if usage + 2_000_000_000 > args.home_cap:
            raise ValueError(f"Home usage {usage} lacks the 2 GB operating reserve")
    return {
        "passed": True, "node": os.uname().nodename, "job_id": os.environ["SLURM_JOB_ID"],
        "home_bytes": usage, "node_local_free_bytes": disk.free,
        "script_sha256": file_digest(__file__), "registry_sha256": file_digest(args.registry),
        "request_sha256": file_digest(args.request), "parent_sha256": args.parent_sha256,
    }


def run_smoke(args, splits, batches):
    from long_horizon_trajectories import copy_checkpoint, saved_config
    import torch
    from third_eye.training.hf_backend import HFBackend

    require_gpu_allocation()
    if torch.cuda.get_device_properties(0).total_memory < 38_000_000_000:
        raise ValueError("Pilot requires a full GPU with at least 38 GB available capacity")
    report = read(args.validated)
    if (not report["passed"] or report["script_sha256"] != file_digest(__file__) or
            report["request_sha256"] != file_digest(args.request) or
            report["registry_sha256"] != file_digest(args.registry) or
            report["parent_sha256"] != args.parent_sha256):
        raise ValueError("Matching CPU validation receipt required")
    if args.home_cap is not None:
        usage = int(subprocess.check_output(
            ["du", "-sx", "-B1", str(Path.home())], text=True,
        ).split()[0])
        if usage + 2_000_000_000 > args.home_cap:
            raise ValueError("Home headroom changed after CPU validation")
    if shutil.disk_usage(tempfile.gettempdir()).free < 2_000_000_000:
        raise ValueError("GPU node has insufficient temporary space")

    started = time.perf_counter()
    snapshots, hashes = {}, {}
    state = read(args.parent / "checkpoints" / "m0" / "state.json")
    if state["model_source"] is not None:
        raise ValueError("Smoke expects the original official-cache model source")
    cfg = saved_config(state["config"])
    backend = HFBackend(cfg)
    for name in ("m0", "m1"):
        with tempfile.TemporaryDirectory(prefix="third-eye-probe-") as temp:
            path = Path(temp) / "adapter"
            recovered = copy_checkpoint(args.parent / "checkpoints" / name, path)
            backend.config = saved_config(recovered["config"])
            backend.load_checkpoint(path)
        snapshots[name], hashes[name] = backend.snapshot(), backend.state_hash()
    backend.config = cfg
    verifier = VerifierRegistry.from_spec("third_eye.evaluation.benchmarks:make_verifier")
    dev = {}
    for role in ROLES:
        items = []
        for example in splits[role]:
            try:
                backend._encode(example.prompt, example.answer)
            except ValueError as error:
                if "SFT example exceeds" not in str(error):
                    raise
                continue
            items.append(example)
            if len(items) == SMOKE_PROBE_ITEMS:
                break
        if len(items) != SMOKE_PROBE_ITEMS:
            raise ValueError(f"Insufficient complete development targets in {role}")
        dev[role] = items

    def losses():
        return {k: np.asarray([backend.measure_loss([item]) for item in v]) for k, v in dev.items()}

    bases, passed = {}, {}
    for name in ("m0", "m1"):
        backend.restore(snapshots[name])
        bases[name] = losses()
        predictions = generate_greedy(backend, dev["target_dev"], 512, [7]*SMOKE_PROBE_ITEMS)
        passed[name] = np.asarray(verify_completions(verifier, dev["target_dev"], predictions), dtype=bool)
        if backend.state_hash() != hashes[name]:
            raise ValueError("Development measurements changed the checkpoint")
    fixed = ~passed["m0"] & passed["m1"]
    candidates = []
    for draw, batch in enumerate(batches):
        effects, timing = {}, {}
        for name in ("m0", "m1"):
            after, train = probe(backend, snapshots[name], hashes[name], batch, 2026112200, losses)
            effects[name] = {role: bases[name][role] - after[role] for role in ROLES}
            timing[name] = train
        gain = float(np.mean([x.mean() for x in effects["m1"].values()]))
        value, penalty = score(gain, effects["m0"]["target_dev"], fixed, SMOKE_LAMBDA)
        candidates.append({"batch": draw, "estimated_gain": gain, "redundancy_penalty": penalty,
                           "score": value, "training": timing})
        gc.collect()
    return {
        "passed": True, "job_id": os.environ["SLURM_JOB_ID"],
        "gpu": torch.cuda.get_device_name(0), "exact_rollback": True,
        "scientific_outcomes": False, "final_eval_read": False,
        "fixed_development_items": int(fixed.sum()), "steps_per_probe": PROBE_STEPS,
        "candidate_diagnostics": candidates, "seconds": time.perf_counter()-started,
    }


def main():
    require_compute_allocation()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("admit", "smoke"))
    for key in ("request", "registry", "parent", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--home-cap", type=int)
    parser.add_argument("--validated", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.mode == "admit":
        result = admission(args)
    else:
        if args.validated is None:
            parser.error("smoke requires --validated")
        result = run_smoke(args, *checked_inputs(args))
    write_json(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
