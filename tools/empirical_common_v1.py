"""Immutable supplemental study contracts; no edits to the A2 implementation."""

from collections import defaultdict
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools")]
from third_eye.config import Config, ModelConfig, TrainingConfig, ProtocolConfig
from third_eye.io import digest, file_digest, write_json
from third_eye.provenance import source_inventory
from third_eye.forecasting.dataset import read_records, split_records

BASE = ROOT / "runs/a2/empirical_v1"
SNAPSHOT = ROOT / "runs/a2/provisional_forecasting_fast_20261008"


def compute_only():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Empirical research requires Slurm compute allocation")


def checked():
    compute_only()
    release = json.loads((BASE / "release.json").read_text())
    if source_inventory(ROOT)["source_tree_sha256"] != release["base_source_sha256"]:
        raise RuntimeError("A2 source changed")
    for name, sha in release["files"].items():
        if file_digest(ROOT / name) != sha:
            raise RuntimeError(f"Empirical release changed: {name}")
    if file_digest(BASE / "scope.json") != release["scope_sha256"]:
        raise RuntimeError("Empirical scope changed")
    if file_digest(SNAPSHOT / "snapshot_manifest.json") != release["snapshot_sha256"]:
        raise RuntimeError("Input snapshot changed")
    plan = json.loads((ROOT / "runs/a2/execution_plan.json").read_text())
    if any(
        file_digest(ROOT / p) != h
        for p, h in plan["resource_policy"]["operational_files_sha256"].items()
    ):
        raise RuntimeError("A2 operational source changed")
    return json.loads((BASE / "scope.json").read_text())


def config(raw):
    return Config(
        ModelConfig(**raw["model"]),
        TrainingConfig(**raw["training"]),
        ProtocolConfig(**raw["protocol"]),
    )


def snapshot_parts():
    manifest = json.loads((SNAPSHOT / "snapshot_manifest.json").read_text())
    paths = []
    for item in manifest["trajectories"]:
        p = SNAPSHOT / "snapshot" / f"task_{item['index']}" / "meta_labels.jsonl"
        if file_digest(p) != item["snapshot_labels_sha256"]:
            raise RuntimeError("Snapshot label checksum differs")
        paths.append(p)
    parts = split_records(read_records(paths), 42)
    assert {
        p: sorted({r["trajectory_id"] for r in rs}) for p, rs in parts.items()
    } == manifest["trajectory_assignments"]
    return manifest, parts


def select_noise(manifest, parts):
    """Metadata-only selection: three generations and distinct trajectories/stream."""
    allowed = {r["trajectory_id"] for p in ("train", "validation") for r in parts[p]}
    items = {t["trajectory_id"]: t for t in manifest["trajectories"]}
    states = defaultdict(list)
    for p in ("train", "validation"):
        for r in parts[p]:
            states[r["state_id"]].append(r)
    streams = defaultdict(list)
    for rows in states.values():
        t = items[rows[0]["trajectory_id"]]
        path = Path(t["source"]).parts
        i = path.index("labels")
        streams["/".join(path[i + 1 : i + 3])].append(rows)
    result, used = [], set()
    for stream in sorted(streams):
        for generation in (0, 2, 4):
            eligible = [
                rs
                for rs in streams[stream]
                if rs[0]["generation"] == generation
                and rs[0]["trajectory_id"] in allowed
                and rs[0]["trajectory_id"] not in used
            ]
            if not eligible:
                raise RuntimeError(
                    f"No eligible independent {stream} generation {generation}"
                )
            rows = min(
                eligible,
                key=lambda rs: digest(
                    [
                        "empirical-noise-v1",
                        42,
                        stream,
                        generation,
                        rs[0]["trajectory_id"],
                    ]
                ),
            )
            tid = rows[0]["trajectory_id"]
            used.add(tid)
            result.append(
                {
                    "index": len(result),
                    "stream": stream,
                    "generation": generation,
                    "trajectory_id": tid,
                    "state_id": rows[0]["state_id"],
                    "source": items[tid]["source"],
                    "snapshot_index": items[tid]["index"],
                    "snapshot_record_hashes": [digest(r) for r in rows],
                }
            )
    if len(result) != 12 or len(used) != 12 or len(streams) != 4:
        raise RuntimeError(
            "Need twelve independent parent trajectories across four streams"
        )
    return result


def preparation():
    scope = checked()
    manifest, parts = snapshot_parts()
    tasks = select_noise(manifest, parts)
    target = BASE / "prepared.json"
    if target.exists():
        saved = json.loads(target.read_text())
        if [(t["state_id"], t["trajectory_id"]) for t in saved["noise_tasks"]] != [
            (t["state_id"], t["trajectory_id"]) for t in tasks
        ]:
            raise RuntimeError("Prepared sample changed")
        return saved
    plan = json.loads((ROOT / "runs/a2/execution_plan.json").read_text())
    for item in tasks:
        command = plan["stages"]["labels"][item["snapshot_index"]]["commands"][0]
        item["config"] = command[command.index("--config") + 1]
        item["manifest"] = command[command.index("--manifest") + 1]
        item["config_sha256"] = file_digest(item["config"])
        item["manifest_sha256"] = file_digest(item["manifest"])
        source = Path(item["source"])
        inputs = [
            source / "run.json",
            source / "meta_labels.jsonl",
            source / "ledger.jsonl",
        ]
        # Replay ancestors from preserved batches, never re-harvest old pools.
        original_rows = read_records([source / "meta_labels.jsonl"])
        ledger = [
            json.loads(line)
            for line in (source / "ledger.jsonl").read_text().splitlines()
        ]
        ancestors = []
        for generation in range(item["generation"]):
            acceptance = next(
                r
                for r in ledger
                if r.get("status") == "accepted" and r["generation"] == generation + 1
            )
            row = next(
                r
                for r in original_rows
                if r["state_id"] == acceptance["state_id"]
                and r["candidate_id"] == acceptance["candidate_id"]
            )
            ancestors.append(row)
        selected = [r for r in original_rows if r["state_id"] == item["state_id"]]
        if [digest(r) for r in selected] != item["snapshot_record_hashes"]:
            raise RuntimeError("Selected original records differ from snapshot")
        for row in ancestors + selected:
            branch = source / "states" / row["state_id"] / row["candidate_id"]
            inputs.extend(
                [
                    branch / "batch.json",
                    branch / "record.json",
                    branch / "continuation_batch.json",
                ]
            )
        item["ancestor_records"] = ancestors
        item["selected_records"] = selected
        item["input_sha256"] = {str(p): file_digest(p) for p in inputs}
    replication = []
    for stream in sorted({t["stream"] for t in tasks}):
        example = next(t for t in tasks if t["stream"] == stream)
        original = Config.load(example["config"])
        for seed in scope["replication_seeds"]:
            cfg = replace(original, protocol=replace(original.protocol, seed=seed))
            path = BASE / "configs" / f"{stream.replace('/', '_')}_{seed}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if Config.load(path).fingerprint != cfg.fingerprint:
                    raise RuntimeError("Replication config already differs")
            else:
                write_json(path, cfg.to_dict())
            replication.append(
                {
                    "index": len(replication),
                    "stream": stream,
                    "seed": seed,
                    "config": str(path),
                    "config_sha256": file_digest(path),
                    "manifest": example["manifest"],
                    "manifest_sha256": example["manifest_sha256"],
                    "output": str(BASE / "replication" / stream / f"seed{seed}"),
                    "generations": 5,
                }
            )
    record = {
        "noise_tasks": tasks,
        "replication_tasks": replication,
        "replication_released": False,
        "scope_sha256": file_digest(BASE / "scope.json"),
    }
    write_json(target, record)
    return record


def prepared():
    checked()
    record = json.loads((BASE / "prepared.json").read_text())
    for task in record["noise_tasks"] + record["replication_tasks"]:
        if (
            file_digest(task["config"]) != task["config_sha256"]
            or file_digest(task["manifest"]) != task["manifest_sha256"]
        ):
            raise RuntimeError("Prepared task configuration/manifest changed")
        if any(file_digest(p) != h for p, h in task.get("input_sha256", {}).items()):
            raise RuntimeError("Replay input changed")
    return record
