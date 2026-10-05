"""Replay only the first eight A1 correction attempts on a GPU allocation."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from retry_prefix_backend import RetryPrefixBackend


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
    from third_eye.io import digest, file_digest, write_json
    from third_eye.provenance import execution_environment
    from third_eye.training.hf_backend import HFBackend
    from third_eye.updates.corrections import collect_corrections
    from audit_retry_budget import audit_pool

    if args.output.exists():
        parser.error("Diagnostic evidence is immutable")
    config = Config.load(args.config)
    p = config.protocol
    if p.amendment != "A1" or p.correction_attempts != 16 or p.sampled_batch_size != 16:
        parser.error("Require unchanged frozen A1 configuration")
    if config.model.device != "cuda" or p.depth != 1:
        parser.error("Require original one-state CUDA pilot")
    manifest, splits = load_manifest(args.manifest)
    rows = [
        json.loads(x)
        for x in (args.reference_run / "meta_labels.jsonl").read_text().splitlines()
        if x.strip()
    ]
    if len(rows) != 3 or len({r["state_id"] for r in rows}) != 1:
        raise ValueError("Require a complete reference K=3 state")
    if any(
        r["config_hash"] != config.fingerprint or r["manifest_hash"] != digest(manifest)
        for r in rows
    ):
        raise ValueError("Reference configuration/manifest provenance mismatch")
    reference_path = (
        args.reference_run / "states" / rows[0]["state_id"] / "correction_pool.json"
    )
    reference = json.loads(reference_path.read_text())
    before = file_digest(reference_path)
    accounting, expected = audit_pool(reference, 8)
    seed = p.seed + rows[0]["generation"] * 1_000_000
    backend = HFBackend(config)
    adapter_hash = backend.state_hash()
    if adapter_hash != rows[0]["parent_adapter_hash"]:
        raise ValueError("Initialized adapter differs from the original pilot parent")
    started = time.perf_counter()
    pool, stats = collect_corrections(
        RetryPrefixBackend(backend, seed, cap=8, seed_stride=16),
        splits["train"],
        make_verifier(),
        replace(p, correction_attempts=8),
        seed,
    )
    seconds = time.perf_counter() - started
    if backend.state_hash() != adapter_hash or file_digest(reference_path) != before:
        raise RuntimeError("Diagnostic changed adapter or historical evidence")
    actual = [c.to_dict() for c in pool]
    expected_by_id = {x["example"]["id"]: x for x in expected}
    actual_by_id = {x["example"]["id"]: x for x in actual}
    write_json(
        args.output,
        {
            "status": "complete",
            "purpose": "eight-attempt prefix replay; no training or research labels",
            "cap": 8,
            "seed_stride": 16,
            "seed": seed,
            "model": config.model.name,
            "manifest_hash": digest(manifest),
            "original_config_hash": config.fingerprint,
            "reference_pool_sha256": before,
            "reference_prefix_accounting": accounting,
            "pool_prefix_exact": actual == expected,
            "accounting_matches_prefix": stats["initial_failures"]
            == reference["stats"]["initial_failures"]
            and stats["revision_attempts"] == accounting["prefix_revision_attempts"],
            "different_correction_ids": sorted(
                k
                for k in expected_by_id.keys() | actual_by_id.keys()
                if expected_by_id.get(k) != actual_by_id.get(k)
            ),
            "seconds": seconds,
            "stats": stats,
            "items": actual,
            "historical_full_harvest_seconds": reference["stats"]["total_seconds"],
            "adapter_unchanged": True,
            "scientific_runs_changed": False,
            "execution": execution_environment(),
            "limitations": [
                "Only original root checkpoint; not recursive H=2 equivalence.",
                "Historical timings are not a same-session paired speedup benchmark.",
                "A shorter budget is an experimental protocol change, not an automatic A1 replacement.",
            ],
        },
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "pool_prefix_exact": actual == expected,
                "verified_corrections": len(pool),
                "seconds": seconds,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
