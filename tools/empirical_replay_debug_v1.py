# ruff: noqa: E402
"""Preserve an aggregate mismatch and compare the native evaluation path."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools")]
from empirical_noise_v1 import (
    BASE,
    frozen,
    initialize,
    collect_items,
    original_m1,
    original_m2,
)
from empirical_common_v1 import prepared
from third_eye.cluster import require_gpu_allocation
from third_eye.evaluation.runner import evaluate
from third_eye.evaluation.verifiers import VerifierRegistry
from third_eye.io import write_json
from third_eye.provenance import execution_environment, versions


def main():
    frozen()
    require_gpu_allocation()
    task = prepared()["noise_tasks"][0]
    out = BASE / "replay_debug_v1"
    out.mkdir(exist_ok=False)
    cfg, splits, verifier, backend = initialize(task, out)
    row = task["selected_records"][0]
    seed = cfg.protocol.seed
    original = json.loads((Path(task["source"]) / "run.json").read_text())
    write_json(
        out / "environment.json",
        dict(
            original_software=original["software_versions"],
            current_software=versions(),
            original_execution=original["execution"],
            current_execution=execution_environment(),
            original_model_source=original["model_source"],
            current_model_source=backend.source_proof,
            original_revision=original["resolved_revision"],
            current_revision=backend.resolved_revision,
        ),
    )
    measured = collect_items(backend, splits, verifier, cfg.protocol, seed)
    write_json(out / "parent_items.json", measured)
    print(
        json.dumps(
            dict(
                original_parent=row["evaluation"]["parent"],
                measured_parent=measured["metrics"],
            )
        ),
        flush=True,
    )
    native = evaluate(
        backend, splits, VerifierRegistry(verifier), cfg.protocol.max_new_tokens, seed
    )
    write_json(
        out / "parent_comparison.json",
        dict(
            original=row["evaluation"]["parent"],
            collector=measured["metrics"],
            native=native.to_dict(),
            exact_collector_native=measured["metrics"] == native.to_dict(),
            exact_original_native=row["evaluation"]["parent"] == native.to_dict(),
        ),
    )
    parent = backend.snapshot()
    original_m1(backend, task, row, parent, out)
    original_m2(backend, task, row, out)
    write_json(
        out / "adapter_replay_passed.json",
        dict(
            parent_hash=row["parent_adapter_hash"],
            m1_hash=row["candidate_adapter_hash"],
            m2_hash=backend.state_hash(),
        ),
    )


if __name__ == "__main__":
    main()
