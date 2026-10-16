"""Compute-only, bounded evidence strengthening in the main Third Eye project."""

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecasting_pilots import inspect_task, item_artifact
from third_eye.cluster import require_compute_allocation, require_gpu_allocation
from third_eye.config import Config
from third_eye.data.schema import Correction, Example
from third_eye.data.splits import load_manifest
from third_eye.evaluation.runner import evaluate
from third_eye.evaluation.verifiers import VerifierRegistry
from third_eye.experiments.continuations import collect_scoring_replicates
from third_eye.experiments.mechanism import (
    OFFSETS,
    aggregate,
    state_comparison,
    intervention_export,
)
from third_eye.forecasting.continuation import repeated_targets
from third_eye.io import digest, file_digest, write_json
from third_eye.provenance import source_inventory
from third_eye.updates.candidates import batch_hash


def read(path):
    return json.loads(Path(path).read_text())


def checked(plan):
    if plan["source_tree_sha256"] != source_inventory()["source_tree_sha256"]:
        raise ValueError("Main source differs from frozen mechanism plan")
    if plan["candidate_lr_factor"] != 0.5 or plan["continuation_offsets"] != list(
        OFFSETS
    ):
        raise ValueError("Declared intervention changed")
    for key in ("pilot_plan", "inspection"):
        if file_digest(plan[key]) != plan[key + "_sha256"]:
            raise ValueError("Historical input changed")
    validation = read(plan["validation"])
    if (
        not validation["passed"]
        or validation["source_tree_sha256"] != plan["source_tree_sha256"]
    ):
        raise ValueError("Full suite validation prerequisite missing")
    if not read(plan["inspection"])["passed"]:
        raise ValueError("Original pilot inspection did not pass")
    pilot = read(plan["pilot_plan"])
    if len(pilot["tasks"]) != 8:
        raise ValueError("Eight balanced pilot controls required")
    return pilot


def inputs(task, pilot):
    return inspect_task(
        task, pilot["continuation_seed_offsets"], pilot["source_tree_sha256"]
    )[0]


def artifacts(rows, splits):
    parent = item_artifact(
        rows[0]["evaluation_artifacts"]["parent"],
        rows[0]["parent_adapter_hash"],
        splits,
    )
    m1 = [
        item_artifact(
            r["evaluation_artifacts"]["t1"], r["candidate_adapter_hash"], splits
        )
        for r in rows
    ]
    futures = []
    for offset in OFFSETS:
        draw_items = []
        for row in rows:
            base_seed = row.get("reference_continuation_seed")
            if base_seed is None:
                base_seed = row["continuation"]["seed"]
            seed = base_seed + offset
            draw = next(d for d in row["continuation_replicates"] if d["seed"] == seed)
            draw_items.append(
                item_artifact(
                    draw["items_path"], draw["continuation_adapter_hash"], splits
                )
            )
        futures.append(draw_items)
    return parent, m1, futures


def prepare(plan, pilot):
    root = Path(plan["output"])
    root.mkdir(exist_ok=False)
    states = []
    for task in pilot["tasks"]:
        rows = inputs(task, pilot)
        _, splits = load_manifest(task["manifest"])
        parent, m1, futures = artifacts(rows, splits)
        states.append(
            dict(
                stream=task["stream"],
                trajectory_id=rows[0]["trajectory_id"],
                comparison=state_comparison(parent, m1, futures),
            )
        )
    write_json(
        root / "existing_independent_review.json",
        {**aggregate(states), "states": states},
    )
    write_json(
        root / "prepared.json",
        dict(
            passed=True,
            plan_hash=digest(plan),
            source_tree_sha256=plan["source_tree_sha256"],
            label_hashes=[
                file_digest(Path(t["output"]) / "meta_labels.jsonl")
                for t in pilot["tasks"]
            ],
        ),
    )


def run(plan, pilot, index):
    require_gpu_allocation("cuda")
    root = Path(plan["output"])
    prerequisite = read(root / "prepared.json")
    if not prerequisite["passed"] or prerequisite["plan_hash"] != digest(plan):
        raise ValueError("Preparation prerequisite changed")
    task = pilot["tasks"][index]
    if (
        file_digest(Path(task["output"]) / "meta_labels.jsonl")
        != prerequisite["label_hashes"][index]
    ):
        raise ValueError("Historical labels changed after preparation")
    rows = inputs(task, pilot)
    config = Config.load(task["config"])
    _, splits = load_manifest(task["manifest"])
    destination = root / f"task_{index}"
    destination.mkdir(exist_ok=False)
    from third_eye.training.hf_backend import HFBackend

    backend = HFBackend(config)
    verifier = VerifierRegistry.from_spec(
        "third_eye.evaluation.benchmarks:make_verifier"
    )
    small_config = replace(
        config,
        training=replace(
            config.training, learning_rate=config.training.learning_rate * 0.5
        ),
    )
    measured = []
    saved, before = backend.snapshot(), backend.state_hash()
    try:
        for row in rows:
            directory = destination / row["candidate_id"]
            directory.mkdir()
            original = Path(row["checkpoints"]["t1"])
            batch = [
                Correction(Example(**c["example"]), c["completion"], c["attempt"])
                for c in read(original.parent / "batch.json")
            ]
            if batch_hash(batch) != row["batch_hash"]:
                raise ValueError("Fixed intervention batch changed")
            backend.load_checkpoint(original)
            if backend.state_hash() != row["candidate_adapter_hash"]:
                raise ValueError("Control candidate checkpoint hash mismatch")
            control_snapshot = backend.snapshot()
            backend.load_checkpoint(original.parent.parent / "parent")
            if backend.state_hash() != row["parent_adapter_hash"]:
                raise ValueError("Intervention parent checkpoint hash mismatch")
            parent_snapshot = backend.snapshot()
            control_norm = (
                sum(
                    float(
                        (control_snapshot[k].float() - parent_snapshot[k].float())
                        .square()
                        .sum()
                    )
                    for k in parent_snapshot
                )
                ** 0.5
            )
            backend.config = small_config
            try:
                log = backend.train(
                    batch,
                    row["runtime"]["candidate"]["seed"],
                    log_path=directory / "training_t1.jsonl",
                )
            finally:
                backend.config = config
            if log["optimizer_steps"] != config.training.max_steps:
                raise ValueError("Intervention optimizer budget changed")
            after_snapshot = backend.snapshot()
            intervention_norm = (
                sum(
                    float(
                        (after_snapshot[k].float() - parent_snapshot[k].float())
                        .square()
                        .sum()
                    )
                    for k in parent_snapshot
                )
                ** 0.5
            )
            candidate_hash = backend.state_hash()
            backend.save_checkpoint(
                directory / "t1",
                dict(
                    intervention="candidate_half_lr",
                    parent_adapter_hash=row["parent_adapter_hash"],
                    baseline_candidate_adapter_hash=row["candidate_adapter_hash"],
                ),
            )
            m1 = evaluate(
                backend,
                splits,
                verifier,
                config.protocol.max_new_tokens,
                config.protocol.seed + row["generation"] * 1_000_000,
                item_output=directory / "evaluation_t1.json",
            ).to_dict()
            result = copy.deepcopy(row)
            result.pop("continuation_reference", None)
            result.update(
                candidate_adapter_hash=candidate_hash,
                checkpoints={"t1": str((directory / "t1").resolve())},
                intervention=dict(
                    candidate_lr_factor=0.5,
                    candidate_config_hash=small_config.fingerprint,
                    continuation_config_hash=config.fingerprint,
                    batch_hash=row["batch_hash"],
                    control_adapter_delta_norm=control_norm,
                    intervention_adapter_delta_norm=intervention_norm,
                    forbidden_for_forecaster_training=True,
                ),
            )
            result["runtime"]["candidate"] = log
            result["evaluation"]["t1"] = m1
            result["labels"]["h1"] = {
                h: m1[h] - row["evaluation"]["parent"][h]
                for h in ("target", "ood", "retention")
            }
            result["evaluation_artifacts"]["t1"] = str(
                (directory / "evaluation_t1.json").resolve()
            )
            labeler = SimpleNamespace(
                backend=backend,
                config=config,
                splits=splits,
                verifier=verifier,
                capture_item_outcomes=True,
            )
            result["continuation_replicates"] = collect_scoring_replicates(
                labeler,
                result,
                directory,
                [row["continuation"]["seed"] + offset for offset in OFFSETS],
            )
            result = intervention_export(result)
            write_json(directory / "intervention_record.json", result)
            measured.append(result)
        repeated_targets(measured)
        parent, control_m1, control_m2 = artifacts(rows, splits)
        _, small_m1, small_m2 = artifacts(measured, splits)
        comparison = state_comparison(
            parent, control_m1, control_m2, small_m1, small_m2
        )
        write_json(
            destination / "completed.json",
            dict(
                passed=True,
                plan_hash=digest(plan),
                stream=task["stream"],
                trajectory_id=rows[0]["trajectory_id"],
                candidates=3,
                continuation_draws=9,
                comparison=comparison,
                terminal_draws=sum(
                    1 - d["continuation_available"]
                    for r in measured
                    for d in r["continuation_replicates"]
                ),
                manipulation=[r["intervention"] for r in measured],
            ),
        )
    except Exception as exc:
        write_json(
            destination / "failure.json",
            dict(
                error_type=type(exc).__name__,
                error=str(exc),
                complete_candidates=len(measured),
                plan_hash=digest(plan),
            ),
        )
        raise
    finally:
        backend.config = config
        backend.restore(saved)
        if backend.state_hash() != before:
            raise RuntimeError("Intervention rollback failed")


def inspect(plan):
    root = Path(plan["output"])
    states, missing = [], []
    for index in range(8):
        path = root / f"task_{index}/completed.json"
        if not path.exists():
            missing.append(index)
            continue
        state = read(path)
        if not state["passed"] or state["plan_hash"] != digest(plan):
            raise ValueError("Completion receipt differs from plan")
        states.append(state)
    report = dict(
        status="incomplete_review_stop" if missing else "science_review_required",
        missing=missing,
        completed_states=len(states),
        states=states,
        automatic_expansion=False,
        development_only=True,
    )
    if not missing:
        report.update(aggregate(states))
    if (root / "report.json").exists():
        raise ValueError("Never overwrite scientific review")
    write_json(root / "report.json", report)


def preflight(plan, pilot):
    require_gpu_allocation("cuda")
    from third_eye.training.hf_backend import HFBackend
    import gc
    import torch

    results = []
    checked_models = set()
    for task in pilot["tasks"]:
        config = Config.load(task["config"])
        if config.model.name in checked_models:
            continue
        rows = inputs(task, pilot)
        original = Path(rows[0]["checkpoints"]["t1"])
        batch = [
            Correction(Example(**c["example"]), c["completion"], c["attempt"])
            for c in read(original.parent / "batch.json")
        ]
        backend = HFBackend(config)
        backend.load_checkpoint(original.parent.parent / "parent")
        parent, before = backend.snapshot(), backend.state_hash()
        baseline = backend.train(
            batch, rows[0]["runtime"]["candidate"]["seed"], max_steps=2
        )
        control_hash = backend.state_hash()
        backend.restore(parent)
        backend.config = replace(
            config,
            training=replace(
                config.training, learning_rate=config.training.learning_rate * 0.5
            ),
        )
        try:
            small = backend.train(
                batch, rows[0]["runtime"]["candidate"]["seed"], max_steps=2
            )
            small_hash = backend.state_hash()
        finally:
            backend.config = config
            backend.restore(parent)
        if (
            backend.state_hash() != before
            or small_hash == control_hash
            or small["optimizer_steps"] != baseline["optimizer_steps"]
        ):
            raise ValueError("Real-model half-LR/rollback verification failed")
        results.append(
            dict(
                model=config.model.name,
                passed=True,
                parent_hash=before,
                control_hash=control_hash,
                half_lr_hash=small_hash,
            )
        )
        checked_models.add(config.model.name)
        del backend
        gc.collect()
        torch.cuda.empty_cache()
    write_json(
        Path(plan["output"]) / "cuda_verified.json",
        dict(passed=True, plan_hash=digest(plan), models=results),
    )


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
        cuda = read(Path(plan["output"]) / "cuda_verified.json")
        if not cuda["passed"] or cuda["plan_hash"] != digest(plan):
            raise ValueError("CUDA prerequisite missing or changed")
        if args.index is None or not 0 <= args.index < 8:
            parser.error("Require one declared task index from 0 to7")
        run(plan, pilot, args.index)
    else:
        inspect(plan)


if __name__ == "__main__":
    main()
