"""Compute-only fixed checkpoint x saved verified correction batch experiment."""

import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecasting_pilots import inspect_task, item_artifact
from third_eye.cluster import require_compute_allocation, require_gpu_allocation
from third_eye.config import Config
from third_eye.data.schema import Correction, Example
from third_eye.data.splits import load_manifest
from third_eye.evaluation.runner import evaluate
from third_eye.evaluation.verifiers import VerifierRegistry
from third_eye.experiments.crossed import DECLARATION, aggregate, state_review
from third_eye.experiments.rendering import FIXED_CONTEXT, historical_training_date
from third_eye.io import digest, file_digest, write_json
from third_eye.provenance import source_inventory
from third_eye.updates.candidates import batch_hash


def read(path):
    return json.loads(Path(path).read_text())


def checked(plan):
    if (
        plan["declaration"] != DECLARATION
        or plan["source_tree_sha256"] != source_inventory()["source_tree_sha256"]
    ):
        raise ValueError("Crossed experiment declaration/source changed")
    for key in ("pilot_plan", "inspection"):
        if file_digest(plan[key]) != plan[key + "_sha256"]:
            raise ValueError("Historical input changed")
    validation = read(plan["validation"])
    if (
        validation.get("passed") is not True
        or validation["source_tree_sha256"] != plan["source_tree_sha256"]
    ):
        raise ValueError("Full source-bound validation required")
    if read(plan["inspection"]).get("passed") is not True:
        raise ValueError("Pilot inspection missing")
    pilot = read(plan["pilot_plan"])
    if len(pilot["tasks"]) != 8:
        raise ValueError("Eight historical controls required")
    return pilot


def rows_for(task, pilot):
    for key in ("config", "manifest"):
        if file_digest(task[key]) != task[key + "_sha256"]:
            raise ValueError("Original config/manifest changed")
    return inspect_task(
        task, pilot["continuation_seed_offsets"], pilot["source_tree_sha256"]
    )[0]


def validate_batch(items, pool, splits, size, available, expected_hash):
    """No relabeling, regeneration, substitutions or within-batch duplicates."""
    batch = tuple(
        Correction(Example(**c["example"]), c["completion"], c["attempt"])
        for c in items
    )
    if type(available) is not int or available not in (0, 1):
        raise ValueError("Explicit continuation availability required")
    if len(batch) != (size if available else 0) or len(
        {c.example.id for c in batch}
    ) != len(batch):
        raise ValueError("Fixed-size unique correction batch required")
    verified = {digest(c) for c in pool["items"]}
    training = {e.id: e for e in splits["train"]}
    if any(
        digest(c.to_dict()) not in verified or c.example != training.get(c.example.id)
        for c in batch
    ):
        raise ValueError("Saved correction is not in original verified train pool")
    if (batch_hash(batch) if available else None) != expected_hash:
        raise ValueError("Saved continuation batch hash changed")
    return batch


def checkpoint_files(path, adapter_hash, cfg):
    path = Path(path)
    state = read(path / "state.json")
    if state["adapter_hash"] != adapter_hash or state["config_hash"] != cfg.fingerprint:
        raise ValueError("Saved checkpoint identity/config differs")
    if not state["weights"]:
        raise ValueError("Saved adapter has no weight files")
    paths = [path / "state.json"]
    for name, sha in state["weights"].items():
        weight = path / name
        weight.resolve().relative_to(path.resolve())
        if file_digest(weight) != sha:
            raise ValueError("Saved adapter weights changed")
        paths.append(weight)
    return paths


def donor_input(row, offset, splits, cfg):
    seed = row["continuation"]["seed"] + offset
    draw = next(d for d in row["continuation_replicates"] if d["seed"] == seed)
    directory = (
        Path(row["checkpoints"]["t1"]).parent / "continuation_replicates" / str(seed)
    )
    paths = [
        directory / "pool.json",
        directory / "batch.json",
        directory / "observation.json",
        Path(draw["items_path"]),
    ]
    if read(directory / "observation.json") != draw:
        raise ValueError("Saved draw differs from original label")
    batch = validate_batch(
        read(paths[1]),
        read(paths[0]),
        splits,
        cfg.protocol.candidate_size,
        draw["continuation_available"],
        draw["batch_hash"],
    )
    if draw["optimizer_steps"] != (cfg.training.max_steps if batch else 0):
        raise ValueError("Historical optimizer budget differs")
    if draw["candidate_adapter_hash"] != row["candidate_adapter_hash"]:
        raise ValueError("Donor checkpoint differs")
    if not batch and draw["continuation_adapter_hash"] != row["candidate_adapter_hash"]:
        raise ValueError("Historical terminal changed M1")
    paths += checkpoint_files(directory / "t2", draw["continuation_adapter_hash"], cfg)
    return draw, batch, paths


def inventory_task(task, pilot):
    cfg = Config.load(task["config"])
    if (
        cfg.protocol.candidates != 3
        or cfg.protocol.candidate_size != 2
        or cfg.training.rank != 16
        or cfg.training.max_steps != 50
        or cfg.training.learning_rate != 1e-4
        or cfg.training.gradient_accumulation_steps != 8
    ):
        raise ValueError("Frozen original update budget differs")
    _, splits = load_manifest(task["manifest"])
    rows = rows_for(task, pilot)
    if [r["candidate_id"].split("-", 1)[0] for r in rows] != ["k0", "k1", "k2"]:
        raise ValueError("Frozen candidate order changed")
    if (
        len({r["continuation"]["seed"] for r in rows}) != 1
        or len({r["parent_adapter_hash"] for r in rows}) != 1
    ):
        raise ValueError("Common parent and matched continuation seeds required")
    paths = [
        Path(task["config"]),
        Path(task["manifest"]),
        Path(task["output"]) / "meta_labels.jsonl",
    ]
    terminals, seconds, legacy_dates = 0, [], []
    for row in rows:
        paths += list(Path(row["checkpoints"]["t1"]).glob("*.jinja"))
        paths += list(Path(row["checkpoints"]["t1"]).glob("tokenizer*.json"))
        paths += checkpoint_files(
            row["checkpoints"]["t1"], row["candidate_adapter_hash"], cfg
        )
        paths += [Path(row["evaluation_artifacts"][k]) for k in ("parent", "t1")]
        for offset in DECLARATION["offsets"]:
            draw, batch, saved = donor_input(row, offset, splits, cfg)
            paths += saved
            terminals += int(not batch)
            seconds.append(draw.get("training", {}).get("seconds", 0))
            if batch:
                log = (
                    Path(row["checkpoints"]["t1"]).parent
                    / "continuation_replicates"
                    / str(draw["seed"])
                    / "training.jsonl"
                )
                paths.append(log)
                if volatile_template(row):
                    legacy_dates.append(
                        {
                            "path": str(log),
                            **historical_training_date(
                                log, draw["training"]["seconds"]
                            ),
                        }
                    )
    return {
        "stream": task["stream"],
        "trajectory_id": rows[0]["trajectory_id"],
        "terminal_donors": terminals,
        "native_training_seconds": seconds,
        "legacy_dates": legacy_dates,
        "files": {str(p.resolve()): file_digest(p) for p in sorted(set(paths))},
    }


def prepare(plan, pilot):
    root = Path(plan["output"])
    root.mkdir(exist_ok=False)
    proof_root = Path(plan["date_replay_diagnostic"])
    for index in (0, 1):
        proof = read(proof_root / str(index) / "completed.json")
        native = [
            r
            for r in proof["records"]
            if r["condition"] == "historical_date_full_update"
        ]
        if (
            proof["source_tree_sha256"] != plan["date_replay_source_sha256"]
            or len(native) != 1
            or native[0]["native_full_adapter_exact_match"] is not True
        ):
            raise ValueError("Date-restored exact historical replay has not passed")
        if any(w["actual"] != w["expected"] for w in proof["weight_checks"].values()):
            raise ValueError("Original frozen base weights changed")
    tasks = [inventory_task(t, pilot) for t in pilot["tasks"]]
    estimated_checkpoint_bytes = 0
    for task in pilot["tasks"]:
        rows = rows_for(task, pilot)
        size = max(
            sum(
                p.stat().st_size
                for p in Path(r["checkpoints"]["t1"]).rglob("*")
                if p.is_file()
            )
            for r in rows
        )
        estimated_checkpoint_bytes += 27 * size
    # Include smoke checkpoints, logs and a conservative serialization margin.
    required_bytes = int(estimated_checkpoint_bytes * 1.5) + 2 * 1024**3
    free_bytes = shutil.disk_usage(root).free
    if free_bytes < required_bytes:
        raise ValueError("Insufficient scratch free space for crossed checkpoints")
    if (
        len({t["trajectory_id"] for t in tasks}) != 8
        or any(
            sum(t["stream"] == g for t in tasks) != 2
            for g in {t["stream"] for t in tasks}
        )
        or len({t["stream"] for t in tasks}) != 4
    ):
        raise ValueError("Balanced distinct parents required")
    write_json(
        root / "prepared.json",
        {
            "passed": True,
            "plan_hash": digest(plan),
            "tasks": tasks,
            "additional_harvesting": 0,
            "off_diagonal_cells": 144,
            "native_identity_replays": 72,
            "fresh_native_cells": 72,
            "fixed_prompt_context": FIXED_CONTEXT,
            "array_tasks": 24,
            "scratch_free_bytes": free_bytes,
            "estimated_required_bytes": required_bytes,
        },
    )


def prepared(plan, index=None):
    value = read(Path(plan["output"]) / "prepared.json")
    if value.get("passed") is not True or value["plan_hash"] != digest(plan):
        raise ValueError("Compute-node preparation prerequisite missing")
    tasks = value["tasks"] if index is None else [value["tasks"][index]]
    for task in tasks:
        if any(file_digest(p) != sha for p, sha in task["files"].items()):
            raise ValueError("Prepared input changed")
        if any(
            Path(d["path"]).stat().st_mtime != d["log_mtime"]
            for d in task.get("legacy_dates", [])
        ):
            raise ValueError("Historical date evidence timestamp changed")
    return value


def execute_cell(
    backend, cfg, splits, verifier, recipient, donor, offset, directory, diagonal=False
):
    """Always reset to recipient M1, including after a failed update/evaluation."""
    draw, batch, _ = donor_input(donor, offset, splits, cfg)
    seed = draw["seed"]
    backend.load_checkpoint(recipient["checkpoints"]["t1"])
    m1_hash = backend.state_hash()
    if m1_hash != recipient["candidate_adapter_hash"]:
        raise ValueError("Recipient checkpoint hash differs")
    saved = backend.snapshot()
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        training = (
            backend.train(batch, seed, log_path=directory / "training.jsonl")
            if batch
            else {"optimizer_steps": 0, "seed": seed, "seconds": 0.0}
        )
        if training["optimizer_steps"] != (cfg.training.max_steps if batch else 0):
            raise ValueError("Crossed update budget differs")
        after = backend.state_hash()
        if not batch and after != m1_hash:
            raise RuntimeError("Terminal donor did not preserve recipient")
        if diagonal:
            if after != draw["continuation_adapter_hash"]:
                raise RuntimeError(
                    "Native diagonal replay differs; do not mix incompatible executions"
                )
            items_path = draw["items_path"]
            kind = "native_diagonal_exact_replay"
        elif not batch:
            items_path = recipient["evaluation_artifacts"]["t1"]
            kind = "crossed_terminal_identity"
        else:
            backend.save_checkpoint(
                directory / "t2",
                {
                    "artifact_kind": "counterfactual_crossed_correction",
                    "recipient": recipient["candidate_id"],
                    "donor": donor["candidate_id"],
                    "seed": seed,
                    "forbidden_for_forecaster_training": True,
                },
            )
            items_path = str((directory / "evaluation.json").resolve())
            evaluate(
                backend,
                splits,
                verifier,
                cfg.protocol.max_new_tokens,
                cfg.protocol.seed + recipient["generation"] * 1_000_000,
                item_output=items_path,
            )
            kind = (
                "matched_context_native_continuation"
                if recipient["candidate_id"] == donor["candidate_id"]
                else "counterfactual_crossed_correction"
            )
        item_artifact(items_path, after, splits)
        result = {
            "artifact_kind": kind,
            "recipient": recipient["candidate_id"],
            "donor": donor["candidate_id"],
            "recipient_adapter_hash": m1_hash,
            "continuation_adapter_hash": after,
            "donor_batch_hash": draw["batch_hash"],
            "seed": seed,
            "offset": offset,
            "continuation_available": int(bool(batch)),
            "training": training,
            "items_path": items_path,
            "items_sha256": file_digest(items_path),
            "seconds": time.perf_counter() - started,
            "forbidden_for_forecaster_training": True,
            "input_rendering": getattr(backend, "input_rendering", None),
        }
        write_json(directory / "cell.json", result)
        return result
    finally:
        backend.restore(saved)
        if backend.state_hash() != m1_hash:
            raise RuntimeError("Crossed experiment rollback failed")


def replay_native(backend, cfg, splits, verifier, row, offset, directory):
    draw, batch, _ = donor_input(row, offset, splits, cfg)
    context = FIXED_CONTEXT
    if batch and volatile_template(row):
        log = (
            Path(row["checkpoints"]["t1"]).parent
            / "continuation_replicates"
            / str(draw["seed"])
            / "training.jsonl"
        )
        recovered = historical_training_date(log, draw["training"]["seconds"])
        context = {"date_string": recovered["date_string"]}
    before = backend.prompt_context
    backend.prompt_context = context
    try:
        return execute_cell(
            backend, cfg, splits, verifier, row, row, offset, directory, diagonal=True
        )
    finally:
        backend.prompt_context = before


def volatile_template(row):
    root = Path(row["checkpoints"]["t1"])
    templates = list(root.glob("*.jinja"))
    if not templates:
        raise ValueError("Saved tokenizer template required for temporal-context audit")
    return any("strftime_now" in p.read_text() for p in templates)


def fixed_items(path, adapter_hash, splits, cfg):
    artifact = item_artifact(path, adapter_hash, splits)
    expected = {**cfg.model.chat_kwargs, **FIXED_CONTEXT}
    rendering = artifact.get("input_rendering", {})
    if (
        rendering.get("chat_kwargs") != expected
        or len(rendering.get("template_sha256", "")) != 64
    ):
        raise ValueError(
            "Historical or mismatched rendering context cannot enter the fresh matrix"
        )
    return artifact


def preflight(plan, pilot):
    require_gpu_allocation("cuda")
    prepared(plan)
    from third_eye.training.hf_backend import HFBackend
    import torch

    verifier = VerifierRegistry.from_spec(
        "third_eye.evaluation.benchmarks:make_verifier"
    )
    reports, streams = [], set()
    for index, task in enumerate(pilot["tasks"]):
        if task["stream"] in streams:
            continue
        cfg = Config.load(task["config"])
        _, splits = load_manifest(task["manifest"])
        row = rows_for(task, pilot)[0]
        backend = HFBackend(cfg, prompt_context=FIXED_CONTEXT)
        directory = Path(plan["output"]) / "cuda" / str(index)
        cell = replay_native(
            backend,
            cfg,
            splits,
            verifier,
            row,
            DECLARATION["offsets"][0],
            directory,
        )
        # Historical scores may contain other dates. Do not use them as cells.
        # Check two fresh full evaluations of the same checkpoint/fixed context.
        backend.load_checkpoint(
            Path(row["checkpoints"]["t1"]).parent
            / "continuation_replicates"
            / str(cell["seed"])
            / "t2"
        )
        path = directory / "replayed_evaluation.json"
        evaluate(
            backend,
            splits,
            verifier,
            cfg.protocol.max_new_tokens,
            cfg.protocol.seed + row["generation"] * 1_000_000,
            item_output=path,
        )
        repeat_path = directory / "repeated_fixed_evaluation.json"
        evaluate(
            backend,
            splits,
            verifier,
            cfg.protocol.max_new_tokens,
            cfg.protocol.seed + row["generation"] * 1_000_000,
            item_output=repeat_path,
        )
        if read(path) != read(repeat_path):
            raise RuntimeError("Fixed-context evaluation is not exactly repeatable")
        # Also exercise a real off-diagonal transfer before the array is eligible.
        donor = rows_for(task, pilot)[1]
        crossed = execute_cell(
            backend,
            cfg,
            splits,
            verifier,
            row,
            donor,
            DECLARATION["offsets"][0],
            directory / "crossed_smoke",
        )
        reports.append(
            {
                "stream": task["stream"],
                "native_cell": cell,
                "discarded_crossed_smoke": crossed,
            }
        )
        streams.add(task["stream"])
        del backend
        gc.collect()
        torch.cuda.empty_cache()
    if len(streams) != 4:
        raise ValueError("Four-stream real CUDA verification required")
    write_json(
        Path(plan["output"]) / "cuda_verified.json",
        {"passed": True, "plan_hash": digest(plan), "streams": reports},
    )


def run(plan, pilot, index):
    require_gpu_allocation("cuda")
    parent_index, recipient_index = divmod(index, 3)
    prepared(plan, parent_index)
    cuda = read(Path(plan["output"]) / "cuda_verified.json")
    if cuda.get("passed") is not True or cuda["plan_hash"] != digest(plan):
        raise ValueError("Real CUDA replay prerequisite missing")
    task = pilot["tasks"][parent_index]
    rows = rows_for(task, pilot)
    cfg = Config.load(task["config"])
    _, splits = load_manifest(task["manifest"])
    from third_eye.training.hf_backend import HFBackend

    root = Path(plan["output"]) / f"task_{index}"
    root.mkdir(exist_ok=False)
    backend = HFBackend(cfg, prompt_context=FIXED_CONTEXT)
    verifier = VerifierRegistry.from_spec(
        "third_eye.evaluation.benchmarks:make_verifier"
    )
    results = []
    try:
        recipient = rows[recipient_index]
        backend.load_checkpoint(
            Path(recipient["checkpoints"]["t1"]).parent.parent / "parent"
        )
        parent_path = root / "evaluation_parent.json"
        evaluate(
            backend,
            splits,
            verifier,
            cfg.protocol.max_new_tokens,
            cfg.protocol.seed + recipient["generation"] * 1_000_000,
            item_output=parent_path,
        )
        backend.load_checkpoint(recipient["checkpoints"]["t1"])
        immediate_path = root / "evaluation_m1.json"
        evaluate(
            backend,
            splits,
            verifier,
            cfg.protocol.max_new_tokens,
            cfg.protocol.seed + recipient["generation"] * 1_000_000,
            item_output=immediate_path,
        )
        recipient = {
            **recipient,
            "evaluation_artifacts": {
                **recipient["evaluation_artifacts"],
                "t1": str(immediate_path.resolve()),
            },
        }
        replays = [
            replay_native(
                backend,
                cfg,
                splits,
                verifier,
                rows[recipient_index],
                offset,
                root / f"legacy_replay_{s}",
            )
            for s, offset in enumerate(DECLARATION["offsets"])
        ]
        for j, donor in enumerate(rows):
            for s, offset in enumerate(DECLARATION["offsets"]):
                result = execute_cell(
                    backend,
                    cfg,
                    splits,
                    verifier,
                    recipient,
                    donor,
                    offset,
                    root / f"donor_{j}_draw_{s}",
                )
                result.update(
                    recipient_index=recipient_index, donor_index=j, draw_index=s
                )
                results.append(result)
        write_json(
            root / "completed.json",
            {
                "passed": True,
                "plan_hash": digest(plan),
                "index": index,
                "cells": results,
                "native_exact_replays": replays,
                "parent_items": str(parent_path.resolve()),
                "parent_items_sha256": file_digest(parent_path),
                "immediate_items": str(immediate_path.resolve()),
                "immediate_items_sha256": file_digest(immediate_path),
            },
        )
    except Exception as exc:
        write_json(
            root / "failure.json",
            {
                "plan_hash": digest(plan),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed_cells": len(results),
            },
        )
        raise


def inspect(plan, pilot):
    root = Path(plan["output"])
    prepared(plan)
    missing = [i for i in range(24) if not (root / f"task_{i}/completed.json").exists()]
    report = {
        "status": "incomplete_review_stop" if missing else "science_review_required",
        "missing_tasks": missing,
        "automatic_expansion": False,
        "development_only": True,
    }
    states = []
    if not missing:
        for p, task in enumerate(pilot["tasks"]):
            cfg = Config.load(task["config"])
            _, splits = load_manifest(task["manifest"])
            rows = rows_for(task, pilot)
            parent, m1 = None, []
            cube = [[[None] * 3 for _ in range(3)] for _ in range(3)]
            for i in range(3):
                receipt = read(root / f"task_{3 * p + i}/completed.json")
                if (
                    receipt.get("passed") is not True
                    or receipt["plan_hash"] != digest(plan)
                    or receipt["index"] != 3 * p + i
                    or len(receipt["cells"]) != 9
                ):
                    raise ValueError("Crossed completion receipt changed")
                for name in ("parent", "immediate"):
                    if (
                        file_digest(receipt[name + "_items"])
                        != receipt[name + "_items_sha256"]
                    ):
                        raise ValueError("Fresh baseline evaluation changed")
                current_parent = fixed_items(
                    receipt["parent_items"], rows[i]["parent_adapter_hash"], splits, cfg
                )
                if parent is not None and current_parent != parent:
                    raise ValueError(
                        "Repeated fixed-context parent evaluation differs across recipients"
                    )
                parent = current_parent
                m1.append(
                    fixed_items(
                        receipt["immediate_items"],
                        rows[i]["candidate_adapter_hash"],
                        splits,
                        cfg,
                    )
                )
                if m1[-1]["input_rendering"] != parent["input_rendering"]:
                    raise ValueError("Parent and M1 rendering templates differ")
                replays = receipt.get("native_exact_replays", [])
                if len(replays) != 3 or len({r["seed"] for r in replays}) != 3:
                    raise ValueError(
                        "All three original exact identity replays required"
                    )
                for s, replay in enumerate(replays):
                    native, _, _ = donor_input(
                        rows[i], DECLARATION["offsets"][s], splits, cfg
                    )
                    if (
                        replay["continuation_adapter_hash"]
                        != native["continuation_adapter_hash"]
                        or replay["seed"] != native["seed"]
                        or replay["recipient_adapter_hash"]
                        != rows[i]["candidate_adapter_hash"]
                    ):
                        raise ValueError("Historical native replay changed")
                for c in receipt["cells"]:
                    j, s = c["donor_index"], c["draw_index"]
                    if (
                        c["recipient_index"] != i
                        or j not in range(3)
                        or s not in range(3)
                        or cube[i][j][s] is not None
                    ):
                        raise ValueError("Duplicated/misassigned crossed cell")
                    draw, batch, _ = donor_input(
                        rows[j],
                        DECLARATION["offsets"][s],
                        splits,
                        Config.load(task["config"]),
                    )
                    if (
                        c["recipient"] != rows[i]["candidate_id"]
                        or c["recipient_adapter_hash"]
                        != rows[i]["candidate_adapter_hash"]
                        or c["donor"] != rows[j]["candidate_id"]
                        or c["donor_batch_hash"] != draw["batch_hash"]
                        or c["seed"] != draw["seed"]
                        or c["continuation_available"] != int(bool(batch))
                    ):
                        raise ValueError(
                            "Cell checkpoint/batch/seed provenance changed"
                        )
                    expected_steps = (
                        Config.load(task["config"]).training.max_steps if batch else 0
                    )
                    if (
                        c["training"]["optimizer_steps"] != expected_steps
                        or c["training"]["seed"] != draw["seed"]
                    ):
                        raise ValueError("Cell optimizer budget/seed differs")
                    if (
                        not batch
                        and c["continuation_adapter_hash"]
                        != rows[i]["candidate_adapter_hash"]
                    ):
                        raise ValueError("Terminal crossed cell changed recipient")
                    if file_digest(c["items_path"]) != c["items_sha256"]:
                        raise ValueError("Cell item outcomes changed")
                    cube[i][j][s] = fixed_items(
                        c["items_path"], c["continuation_adapter_hash"], splits, cfg
                    )
                    if cube[i][j][s]["input_rendering"] != parent["input_rendering"]:
                        raise ValueError("Cell and baseline rendering templates differ")
            state = state_review(parent, m1, cube)
            states.append(
                {
                    "stream": task["stream"],
                    "trajectory_id": rows[0]["trajectory_id"],
                    **state,
                }
            )
        report.update(aggregate(states), states=states)
    path = root / "report.json"
    if path.exists():
        raise ValueError("Never overwrite crossed scientific review")
    write_json(path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "preflight", "run", "inspect"))
    parser.add_argument("--plan", required=True)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    require_compute_allocation()
    plan = read(args.plan)
    pilot = checked(plan)
    if args.mode == "prepare":
        prepare(plan, pilot)
    elif args.mode == "preflight":
        preflight(plan, pilot)
    elif args.mode == "run":
        if args.index is None or not 0 <= args.index < 24:
            parser.error("Require one frozen recipient task index 0..23")
        run(plan, pilot, args.index)
    else:
        inspect(plan, pilot)


if __name__ == "__main__":
    main()
