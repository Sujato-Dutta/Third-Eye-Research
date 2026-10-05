"""Control retry-cap effects within one initialized Llama process."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from third_eye.cluster import require_gpu_allocation

    require_gpu_allocation()
    from third_eye.config import Config
    from third_eye.data.splits import load_manifest
    from third_eye.evaluation.benchmarks import make_verifier
    from third_eye.evaluation.runner import evaluate
    from third_eye.io import digest, file_digest, write_json
    from third_eye.provenance import execution_environment
    from third_eye.training.hf_backend import HFBackend
    from third_eye.updates.corrections import collect_corrections
    from retry_prefix_backend import RetryPrefixBackend

    if args.output.exists():
        parser.error("Control artifacts are immutable")
    cfg = Config.load(args.config)
    p = cfg.protocol
    if (
        p.amendment != "A1"
        or p.correction_attempts != 16
        or p.sampled_batch_size != 16
        or p.depth != 1
    ):
        parser.error("Require the original one-state A1 protocol")
    manifest, splits = load_manifest(args.manifest)
    labels_path = args.reference_run / "meta_labels.jsonl"
    rows = [json.loads(s) for s in labels_path.read_text().splitlines() if s.strip()]
    if len(rows) != 3 or any(
        r["config_hash"] != cfg.fingerprint or r["manifest_hash"] != digest(manifest)
        for r in rows
    ):
        raise ValueError("Original reference provenance mismatch")
    seed = p.seed + rows[0]["generation"] * 1_000_000
    original_pool_path = (
        args.reference_run / "states" / rows[0]["state_id"] / "correction_pool.json"
    )
    original = json.loads(original_pool_path.read_text())
    hashes = {
        str(path): file_digest(path) for path in [labels_path, original_pool_path]
    }
    backend = HFBackend(cfg)
    parent_hash = backend.state_hash()
    if parent_hash != rows[0]["parent_adapter_hash"]:
        raise ValueError("Initialized parent adapter differs from reference")
    verifier = make_verifier()
    # Mirror the original label pipeline's development evaluation before harvest.
    baseline = evaluate(backend, splits, verifier, p.max_new_tokens, seed)
    if backend.state_hash() != parent_hash:
        raise RuntimeError("Evaluation changed adapter")
    began = time.perf_counter()
    shorter, short_stats = collect_corrections(
        RetryPrefixBackend(backend, seed),
        splits["train"],
        verifier,
        replace(p, correction_attempts=8),
        seed,
    )
    short_seconds = time.perf_counter() - began
    if backend.state_hash() != parent_hash:
        raise RuntimeError("Short harvest changed adapter")
    # Persist the first half before beginning the longer control.
    write_json(
        args.output.with_suffix(".short.json"),
        {
            "status": "short_harvest_complete",
            "stats": short_stats,
            "items": [x.to_dict() for x in shorter],
            "seconds": short_seconds,
        },
    )
    began = time.perf_counter()
    full, full_stats = collect_corrections(backend, splits["train"], verifier, p, seed)
    full_seconds = time.perf_counter() - began
    if backend.state_hash() != parent_hash or any(
        file_digest(path) != h for path, h in hashes.items()
    ):
        raise RuntimeError("Control changed adapter or original evidence")
    prefix = [x.to_dict() for x in full if x.attempt <= 8]
    short_items = [x.to_dict() for x in shorter]
    expected_attempts = full_stats["unresolved_failures"] * 8 + sum(
        min(x.attempt, 8) for x in full
    )
    exact = short_items == prefix
    write_json(
        args.output,
        {
            "status": "complete",
            "purpose": "paired retry-prefix control; zero optimizer updates",
            "original_config_hash": cfg.fingerprint,
            "manifest_hash": digest(manifest),
            "seed": seed,
            "seed_stride": 16,
            "parent_adapter_hash": parent_hash,
            "baseline": baseline.to_dict(),
            "baseline_matches_historical": baseline.to_dict()
            == rows[0]["evaluation"]["parent"],
            "short_stats": short_stats,
            "full_stats": full_stats,
            "short_seconds": short_seconds,
            "full_seconds": full_seconds,
            "within_session_prefix_exact": exact,
            "within_session_accounting_exact": short_stats["initial_failures"]
            == full_stats["initial_failures"]
            and short_stats["revision_attempts"] == expected_attempts,
            "full_pool_matches_historical": [x.to_dict() for x in full]
            == original["items"],
            "short_items": short_items,
            "full_items": [x.to_dict() for x in full],
            "adapter_unchanged": True,
            "research_runs_changed": False,
            "execution": execution_environment(),
            "model_source_proof": backend.source_proof,
            "attention_implementation": backend.model.config._attn_implementation,
            "limitations": [
                "A fresh root harvest is not a recursive eight-attempt science trial.",
                "The shorter harvest runs first; warmup/order can influence timing.",
                "Historical differences and within-session cap effects are reported separately.",
            ],
        },
    )
    print(
        json.dumps({"output": str(args.output), "within_session_prefix_exact": exact}),
        flush=True,
    )


if __name__ == "__main__":
    main()
