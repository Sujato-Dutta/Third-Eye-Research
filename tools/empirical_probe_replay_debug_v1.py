# ruff: noqa: E402
"""Test whether the original reversible feature probe is needed for exact replay."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools")]
from empirical_noise_v1 import BASE, frozen, batch, same_hash
from empirical_common_v1 import prepared
from third_eye.cluster import require_gpu_allocation
from third_eye.config import Config
from third_eye.data.splits import load_manifest
from third_eye.experiments.features import extract_features
from third_eye.io import write_json


def main():
    frozen()
    require_gpu_allocation()
    from third_eye.training.hf_backend import HFBackend

    task = prepared()["noise_tasks"][0]
    row = task["selected_records"][0]
    out = BASE / "probe_replay_debug_v1"
    out.mkdir(exist_ok=False)
    cfg = Config.load(task["config"])
    _, splits = load_manifest(task["manifest"])
    backend = HFBackend(cfg)
    backend.load_checkpoint(BASE / "replay_debug_v1/parent")
    same_hash(backend, row["parent_adapter_hash"], "debug parent")
    seed = row["runtime"]["candidate"]["seed"]
    items = batch(task, row)
    features = extract_features(
        backend,
        items,
        splits["retention_dev"],
        cfg.protocol,
        seed,
        out / "original_probe.jsonl",
    )
    same_hash(backend, row["parent_adapter_hash"], "native probe rollback")
    write_json(
        out / "feature_comparison.json",
        dict(original=row["precommit"]["features"], replayed=features),
    )
    log = backend.train(items, seed, log_path=out / "training.jsonl")
    actual = backend.state_hash()
    write_json(
        out / "result.json",
        dict(
            original_hash=row["candidate_adapter_hash"],
            replayed_hash=actual,
            matched=actual == row["candidate_adapter_hash"],
            training=log,
        ),
    )
    backend.save_checkpoint(
        out / "m1",
        dict(
            stage="native_probe_replay_diagnostic",
            matched_original=actual == row["candidate_adapter_hash"],
        ),
    )
    same_hash(backend, row["candidate_adapter_hash"], "M1 after native feature path")


if __name__ == "__main__":
    main()
