"""Frozen paired-horizon scale checks and lossless checkpoint retention."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from third_eye.experiments.mechanism import choose
from third_eye.experiments.smollm_confirmation import NAMES, ROLES, matrix
from third_eye.io import file_digest, write_json

MODELS = {
    "qwen3-8b": {
        "official": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "repository": "Qwen/Qwen3-8B",
        "source_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "seeds": {"code": list(range(2026110200, 2026110204)),
                  "math": list(range(2026110100, 2026110104))},
    },
    "llama3.1-8b": {
        "official": "meta-llama/Llama-3.1-8B-Instruct",
        "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
        "repository": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "source_revision": "d10aef7999a2b5ba950ab3974312feeedbfe0b77",
        "seeds": {"code": list(range(2026110400, 2026110404)),
                  "math": list(range(2026110300, 2026110304))},
    },
}
DECLARATION = {
    "version": "scale_20261028", "models": MODELS,
    "parents_per_model": 8, "parents_per_task": 4,
    "continuation_draws": 3,
    "scoring_items": {"target_test": 128, "ood_test": 128, "retention_test": 256},
    "primary": "selected_future_minus_selected_immediate_utility",
    "update_steps": 50, "candidate_size": 2, "candidates": 3,
    "correction_attempts": 8, "learning_rate": 1e-4, "rank": 16,
    "selection": "development_H1_equal_head_utility_uniform_ties",
    "terminal": "M2_equals_M1_no_fabricated_training",
    "frozen_date": "11 Oct 2026", "forecaster_fits": 0,
    "retention": "hash_verified_lossless_gzip_adapters_after_complete_scoring",
    "purpose": "balanced_scale_sensitivity_not_equal_precision_replication",
    "Vista_SUs": 0, "concurrent_GPUs": 1,
}


def validate_authorization(path, config_path, manifest_path, *, mode, policy,
                           generations, offsets, capture):
    from third_eye.config import Config
    from third_eye.io import digest
    from third_eye.provenance import source_inventory

    auth = json.loads(Path(path).read_text())
    cfg = Config.load(config_path)
    if (auth.get("purpose") != "frozen_empirical_scale_confirmation"
            or auth.get("declaration_sha256") != digest(DECLARATION)
            or auth.get("source_tree_sha256") != source_inventory()["source_tree_sha256"]
            or mode != "labels" or policy != "random" or generations != 1
            or offsets != [50_000_000, 100_000_000] or capture is not True):
        raise ValueError("Only frozen empirical scale label collection is authorized")
    ready = json.loads(Path(auth["preflight"]).read_text())
    if (not ready["passed"] or file_digest(auth["preflight"]) != auth["preflight_sha256"]
            or ready["source_tree_sha256"] != auth["source_tree_sha256"]):
        raise ValueError("Matching real GPU compatibility required")
    allowed = [t for t in auth["tasks"] if t["seed"] == cfg.protocol.seed
               and t["config_hash"] == cfg.fingerprint
               and t["manifest_sha256"] == file_digest(manifest_path)]
    if len(allowed) != 1 or cfg.model.name != MODELS[auth["model"]]["official"]:
        raise ValueError("Unregistered empirical scale task")
    return auth


def actions(stream, immediate):
    x = np.asarray(immediate, float)
    if stream not in {m + "/" + t for m in MODELS for t in ("code", "math")}:
        raise ValueError("Unregistered scale stream")
    if x.shape != (3, 3) or not np.isfinite(x).all():
        raise ValueError("Three finite development candidates required")
    mass = choose(x.mean(1))
    return {"hold": [0.0] * 3, "forced": mass.tolist(),
            "replay": (mass * ((x[:, 0] > 0) & (x[:, 1] >= 0) & (x[:, 2] >= 0))).tolist()}


def compact_checkpoints(parent):
    """Preserve every tensor byte; score/label paths remain immediately readable."""
    parent = Path(parent).resolve()
    if not json.loads((parent / "completed.json").read_text())["passed"]:
        raise ValueError("Only complete scored parents may be compacted")
    receipt = parent / "lossless_retention.json"
    if receipt.exists():
        raise FileExistsError("Retention is immutable; inspect existing receipt")
    records = []
    # Include adapter tokenizer copies: retain their paths through identical-byte
    # hardlinks, avoiding repeated 9-12MB tokenizer files without deleting them.
    shared = {}
    for state_path in sorted(parent.rglob("state.json")):
        if state_path.is_symlink():
            raise ValueError("Refuse symlink checkpoint")
        saved = json.loads(state_path.read_text())
        if "weights" not in saved:
            continue
        for filename, expected in saved["weights"].items():
            original = state_path.parent / filename
            original.resolve().relative_to(parent)
            if original.is_symlink() or file_digest(original) != expected:
                raise ValueError("Checkpoint weight changed before archival")
            packed = original.with_suffix(original.suffix + ".gz")
            if packed.exists():
                raise FileExistsError("Preserve an earlier partial archival attempt")
            temporary = packed.with_suffix(packed.suffix + ".tmp")
            with original.open("rb") as src, temporary.open("xb") as dst:
                with gzip.GzipFile(filename="", mode="wb", fileobj=dst,
                                   compresslevel=6, mtime=0) as zipped:
                    shutil.copyfileobj(src, zipped, 8 * 1024**2)
            h = hashlib.sha256()
            with gzip.open(temporary, "rb") as src:
                for chunk in iter(lambda: src.read(8 * 1024**2), b""):
                    h.update(chunk)
            if h.hexdigest() != expected:
                raise ValueError("Lossless archival verification failed")
            temporary.replace(packed)
            record = {"original": str(original.relative_to(parent)),
                      "original_sha256": expected,
                      "archive": str(packed.relative_to(parent)),
                      "archive_sha256": file_digest(packed),
                      "original_bytes": original.stat().st_size,
                      "archive_bytes": packed.stat().st_size}
            records.append(record)
            # Persist recoverability before removing the redundant raw copy.
            write_json(parent / "lossless_retention_progress.json", records)
            original.unlink()
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            item = state_path.parent / name
            if not item.is_file() or item.is_symlink():
                continue
            sha = file_digest(item)
            if sha in shared:
                import os

                temporary = item.with_suffix(item.suffix + ".link")
                os.link(shared[sha], temporary)
                temporary.replace(item)
            else:
                shared[sha] = item
    write_json(receipt, {"passed": True, "policy": DECLARATION["retention"],
                         "records": records, "scientific_outcomes_unchanged": True})
    return records


def restore_archived_checkpoint(directory):
    directory = Path(directory)
    saved = json.loads((directory / "state.json").read_text())
    for name, expected in saved["weights"].items():
        destination = directory / name
        if destination.exists():
            if file_digest(destination) != expected:
                raise ValueError("Existing restored checkpoint changed")
            continue
        temporary = destination.with_suffix(destination.suffix + ".restore")
        with gzip.open(destination.with_suffix(destination.suffix + ".gz"), "rb") as src:
            with temporary.open("xb") as dst:
                shutil.copyfileobj(src, dst, 8 * 1024**2)
        if file_digest(temporary) != expected:
            raise ValueError("Restored tensor checksum differs")
        temporary.replace(destination)


def summarize(states, model, output, metadata, draws=5000):
    expected = {model + "/code", model + "/math"}
    if model not in MODELS or len(states) != 8 or {s["stream"] for s in states} != expected:
        raise ValueError("Complete eight-parent scale check required")
    if any(sum(s["stream"] == g for s in states) != 4 for g in expected):
        raise ValueError("Four parents per task required")
    rng = np.random.default_rng(20261028)
    weights = np.zeros((draws, 8))
    for group in sorted(expected):
        ids = np.array([i for i, s in enumerate(states) if s["stream"] == group])
        chosen = rng.choice(ids, (draws, 4))
        for i in ids:
            weights[:, i] = (chosen == i).sum(1) / 8
    point, boot = np.zeros((3, len(NAMES))), np.zeros((draws, 3, len(NAMES)))
    shared, seen = {}, {}
    for i, state in enumerate(states):
        for r, role in enumerate(ROLES):
            identity = tuple(tuple(v) for v in state["ids"][role])
            if any(set(previous) & set(identity) and previous != identity
                   for previous in seen.get(role, [])):
                raise ValueError("Partially shared item pools require explicit resampling")
            seen.setdefault(role, []).append(identity)
            key = role, identity
            if key not in shared:
                shared[key] = rng.integers(len(identity), size=(draws, len(identity)))
            values = matrix(state, role)
            if values.shape != (len(NAMES), len(identity)) or not np.isfinite(values).all():
                raise ValueError("Complete finite scoring items required")
            point[r] += values.mean(1) / 8
            boot[:, r] += values[:, shared[key]].mean(2).T * weights[:, i, None]

    def intervals(p, b):
        return {name: {"mean_pp": float(100 * p[j]),
                       "ci95_pp": (100 * np.quantile(b[:, j], [.025, .975])).tolist(),
                       "ci975_pp": (100 * np.quantile(b[:, j], [.0125, .9875])).tolist()}
                for j, name in enumerate(NAMES)}

    report = {**metadata, "declaration": DECLARATION, "model": model, "parents": 8,
              "aggregate": intervals(point.mean(0), boot.mean(1)),
              "heads": {r: intervals(point[i], boot[:, i]) for i, r in enumerate(ROLES)},
              "scored_checkpoints": 104, "continuation_draws": 72,
              "multiplicity": "Two model-specific selected_change primaries use 97.5% intervals; other endpoints exploratory",
              "limitations": "Four seeds per task; frozen small-batch LoRA and H=2; scale sensitivity, not a causal size effect or universal replication"}
    output = Path(output)
    output.mkdir(exist_ok=False)
    write_json(output / "report.json", report)
    return report
