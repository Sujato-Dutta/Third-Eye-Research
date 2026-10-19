"""Bounded fresh-data pilots for the core Third Eye expected-value forecaster."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from third_eye.cluster import require_compute_allocation, require_gpu_allocation
from third_eye.config import Config
from third_eye.data.splits import load_manifest
from third_eye.forecasting.continuation import reliability_audit, repeated_targets
from third_eye.io import digest, file_digest, write_json
from third_eye.provenance import source_inventory
from third_eye.updates.candidates import batch_hash
from third_eye.data.schema import Correction, Example


def checked_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan["source_tree_sha256"] != source_inventory()["source_tree_sha256"]:
        raise ValueError("Pilot plan/source snapshot differs")
    expected = {"development_reliability_pilots": 8, "admission_development_pilots": 16}
    if (
        plan["purpose"] not in expected
        or len(plan["tasks"]) != expected[plan["purpose"]]
    ):
        raise ValueError("Expected declared balanced developmental pilots")
    for task in plan["tasks"]:
        if (
            file_digest(task["config"]) != task["config_sha256"]
            or file_digest(task["manifest"]) != task["manifest_sha256"]
        ):
            raise ValueError("Pilot config/manifest changed")
    return plan


def item_artifact(path, adapter_hash, splits):
    value = json.loads(Path(path).read_text())
    if value["schema_version"] != 1 or value["adapter_hash"] != adapter_hash:
        raise ValueError("Per-item evaluation checkpoint differs")
    for role, head in zip(
        ("target_dev", "ood_dev", "retention_dev"), ("target", "ood", "retention")
    ):
        items = value["items"][role]
        expected = [(ex.id, ex.prompt_hash) for ex in splits[role]]
        if [(item["id"], item["prompt_sha256"]) for item in items] != expected:
            raise ValueError("Per-item evaluation examples/order differ")
        if len(set(expected)) != len(expected) or any(
            type(item["correct"]) is not int or item["correct"] not in (0, 1)
            for item in items
        ):
            raise ValueError("Duplicated or invalid per-item outcomes")
        metric = sum(item["correct"] for item in items) / len(items)
        if not math.isclose(metric, value["metrics"][head], abs_tol=1e-12):
            raise ValueError("Aggregate metric differs from saved item outcomes")
    return value


def inspect_task(task, offsets, source_sha):
    output = Path(task["output"])
    run = json.loads((output / "run.json").read_text())
    cfg = Config.load(task["config"])
    manifest, splits = load_manifest(task["manifest"])
    if (
        run["config_hash"] != cfg.fingerprint
        or run["manifest_hash"] != digest(manifest)
        or run["source_tree_sha256"] != source_sha
        or run["continuation_seed_offsets"] != offsets
        or run["capture_item_outcomes"] is not True
    ):
        raise ValueError("Run provenance/collection settings differ")
    completed = json.loads((output / "completed.json").read_text())
    if (
        completed["status"] != "complete"
        or completed["completed_states"] != 1
        or completed["requested_generations"] != 1
    ):
        raise ValueError("Pilot did not complete its one planned state")
    rows = [
        json.loads(line)
        for line in (output / "meta_labels.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != 3 or len({r["state_id"] for r in rows}) != 1:
        raise ValueError("Pilot must contain exactly one complete K=3 state")
    draw_count = 1 + len(offsets)
    repeated_targets(rows, minimum_replicates=draw_count)
    hashes = set()
    terminal = 0
    pool = json.loads(next(output.glob("states/*/correction_pool.json")).read_text())
    verified_pool = {digest(c) for c in pool["items"]}
    for row in rows:
        if row["config_hash"] != cfg.fingerprint or row["manifest_hash"] != digest(
            manifest
        ):
            raise ValueError("Label provenance differs")
        directory = Path(row["checkpoints"]["t1"]).parent
        batch = [
            Correction(Example(**c["example"]), c["completion"], c["attempt"])
            for c in json.loads((directory / "batch.json").read_text())
        ]
        if len(batch) != cfg.protocol.candidate_size or len(
            {c.example.id for c in batch}
        ) != len(batch):
            raise ValueError("Candidate batch size/uniqueness differs")
        if batch_hash(batch) != row["batch_hash"]:
            raise ValueError("Candidate batch hash differs")
        if any(digest(c.to_dict()) not in verified_pool for c in batch):
            raise ValueError("Candidate example is absent from the saved verified pool")
        hashes.add(row["batch_hash"])
        if row["runtime"]["candidate"]["optimizer_steps"] != cfg.training.max_steps:
            raise ValueError("Candidate optimizer budget differs")
        parent = item_artifact(
            row["evaluation_artifacts"]["parent"], row["parent_adapter_hash"], splits
        )
        immediate = item_artifact(
            row["evaluation_artifacts"]["t1"], row["candidate_adapter_hash"], splits
        )
        for head in ("target", "ood", "retention"):
            if (
                parent["metrics"][head] != row["evaluation"]["parent"][head]
                or immediate["metrics"][head] != row["evaluation"]["t1"][head]
                or not math.isclose(
                    immediate["metrics"][head] - parent["metrics"][head],
                    row["labels"]["h1"][head],
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("Immediate label differs from item outcomes")
        draws = row["continuation_replicates"]
        expected_seeds = [row["continuation"]["seed"] + o for o in [0, *offsets]]
        if len(draws) != draw_count or [d["seed"] for d in draws] != expected_seeds:
            raise ValueError("Continuation seed grid differs")
        for draw in draws:
            item = item_artifact(
                draw["items_path"], draw["continuation_adapter_hash"], splits
            )
            for head in ("target", "ood", "retention"):
                if not math.isclose(
                    item["metrics"][head] - parent["metrics"][head],
                    draw["h2"][head],
                    abs_tol=1e-12,
                ):
                    raise ValueError("Future label differs from item outcomes")
            terminal += 1 - draw["continuation_available"]
    if len(hashes) != 3:
        raise ValueError("Candidate compositions are not distinct")
    return rows, {
        "stream": task["stream"],
        "seed": task["seed"],
        "complete": True,
        "candidates": 3,
        "continuation_draws": 3 * draw_count,
        "terminal_draws": terminal,
        "evaluation_sizes": {r: len(splits[r]) for r in splits},
        "state_seconds": json.loads(
            next(output.glob("states/*/summary.json")).read_text()
        )["seconds"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "run", "inspect"))
    parser.add_argument("--plan", required=True)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    require_compute_allocation()
    plan = checked_plan(args.plan)
    if args.mode == "preflight":
        require_gpu_allocation("cuda")
        from third_eye.training.hf_backend import HFBackend
        from third_eye.evaluation.verifiers import VerifierRegistry
        import gc
        import torch

        checked = set()
        reports = []
        verifier = VerifierRegistry.from_spec(
            "third_eye.evaluation.benchmarks:make_verifier"
        )
        for task in plan["tasks"]:
            cfg = Config.load(task["config"])
            _, splits = load_manifest(task["manifest"])
            # Exercise the real sandbox/benchmark verifier for each task family.
            ex = splits["train"][0]
            verifier.verify(ex, ex.answer)
            validation_key = (
                task["stream"]
                if plan["purpose"] == "admission_development_pilots"
                else cfg.model.name
            )
            if validation_key in checked:
                continue
            backend = HFBackend(cfg)
            parent, before = backend.snapshot(), backend.state_hash()
            try:
                backend.measure_loss(splits["train"][:2])
                backend.generate(splits["train"][0].prompt, 8, seed=cfg.protocol.seed)
                batch = [Correction(e, e.answer, 1) for e in splits["train"][:2]]
                backend.train(batch, cfg.protocol.seed, max_steps=2)
                if backend.state_hash() == before:
                    raise RuntimeError(
                        "Real-backbone preflight did not update the adapter"
                    )
            finally:
                backend.restore(parent)
            if backend.state_hash() != before:
                raise RuntimeError("Real-backbone preflight did not restore the parent")
            reports.append(
                {
                    "model": cfg.model.name,
                    "revision": backend.resolved_revision,
                    "adapter_restored": True,
                    "purpose": "discarded implementation smoke; not experiment evidence",
                }
            )
            checked.add(validation_key)
            del backend, parent
            gc.collect()
            torch.cuda.empty_cache()
        write_json(
            Path(plan["cuda_validation"]).parent / "real_backbones.json",
            {"passed": True, "models": reports},
        )
        return
    if args.mode == "run":
        require_gpu_allocation("cuda")
        if args.index is None or not 0 <= args.index < len(plan["tasks"]):
            parser.error("Run requires an index inside the declared task grid")
        task = plan["tasks"][args.index]
        for key in ("cpu_validation", "cuda_validation"):
            receipt = json.loads(Path(plan[key]).read_text())
            if (
                receipt.get("passed") is not True
                or receipt.get("source_tree_sha256") != plan["source_tree_sha256"]
            ):
                raise ValueError(
                    "Pilot requires matching successful CPU and CUDA validation"
                )
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("run.py")),
                "--config",
                task["config"],
                "--manifest",
                task["manifest"],
                "--output",
                task["output"],
                "--generations",
                "1",
                "--policy",
                "random",
                "--capture-item-outcomes",
                "--continuation-seed-offsets",
                *map(str, plan["continuation_seed_offsets"]),
            ],
            check=True,
        )
        return
    reports, records = [], []
    for task in plan["tasks"]:
        try:
            rows, report = inspect_task(
                task, plan["continuation_seed_offsets"], plan["source_tree_sha256"]
            )
            records.extend(rows)
            reports.append(report)
        except (OSError, ValueError, KeyError, StopIteration) as exc:
            reports.append(
                {
                    "stream": task["stream"],
                    "seed": task["seed"],
                    "complete": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    complete = all(r["complete"] for r in reports)
    write_json(
        Path(args.plan).parent / "pilot_inspection.json",
        {
            "schema_version": 1,
            "passed": complete,
            "development_only": True,
            "automatic_expansion": False,
            "tasks": reports,
            "reliability": reliability_audit(records) if records else None,
            "interpretation": "Structural/reliability review only. No terminal-rate pass threshold, forecasting advantage or Gate 2 decision.",
        },
    )
    if not complete:
        raise SystemExit(
            "Incomplete/invalid pilots: inspect preserved artifacts; do not expand"
        )


if __name__ == "__main__":
    main()
