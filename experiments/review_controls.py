"""Paired review controls on imported, immutable initial parent checkpoints."""

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from third_eye.cluster import require_compute_allocation, require_gpu_allocation
from third_eye.config import Config
from third_eye.data.schema import Correction, Example
from third_eye.data.splits import load_manifest
from third_eye.evaluation.final import evaluate_final, load_final_manifest
from third_eye.evaluation.runner import verify_completions
from third_eye.evaluation.verifiers import VerifierRegistry
from third_eye.experiments.revision import (
    pooled_corrections,
    reference_corrections,
    restore_portable,
)
from third_eye.io import file_digest, write_json
from third_eye.provenance import source_inventory
from third_eye.updates.candidates import batch_hash, sample_batches
from third_eye.updates.corrections import collect_corrections


def read(path):
    return json.loads(Path(path).read_text())


def checked(request_path):
    require_compute_allocation()
    request = read(request_path)
    if request["source_tree_sha256"] != source_inventory()["source_tree_sha256"]:
        raise ValueError("Current controls source differs")
    validation = read(request["validation"])
    if (
        not validation["passed"]
        or validation["source_tree_sha256"] != request["source_tree_sha256"]
    ):
        raise ValueError("Matching full CPU validation required")
    if file_digest(request["registry"]) != request["registry_sha256"]:
        raise ValueError("Transferred original inputs differ")
    registry = read(request["registry"])
    prepared = read(request["prepared"])
    if (
        not prepared["passed"]
        or prepared["registry_sha256"] != request["registry_sha256"]
    ):
        raise ValueError("Allocated import/model verification required")
    return request, registry


def mapped(request, registry, relative):
    path = (Path(request["payload"]) / relative).resolve()
    path.relative_to(Path(request["payload"]).resolve())
    if (
        relative in registry["files"]
        and file_digest(path) != registry["files"][relative]["sha256"]
    ):
        raise ValueError("Transferred file checksum differs")
    return path


def inputs(request, registry, task):
    cfg = Config.load(mapped(request, registry, task["config"]))
    manifest = mapped(request, registry, task["manifest"])
    _, dev = load_manifest(manifest)
    _, final = load_final_manifest(
        mapped(request, registry, task["final_manifest"]), manifest
    )
    if (
        cfg.training.max_steps != 50
        or cfg.protocol.candidate_size != 2
        or cfg.protocol.correction_attempts != 8
    ):
        raise ValueError("Original fixed budgets required")
    batches = []
    train = {ex.id: ex for ex in dev["train"]}
    for draw in task["draws"]:
        batch = tuple(
            Correction(Example(**c["example"]), c["completion"], c["attempt"])
            for c in draw["batch"]
        )
        if len({c.example.id for c in batch}) != len(batch) or len(batch) not in (0, 2):
            raise ValueError("Fixed unique verified donor batches required")
        if any(train.get(c.example.id) != c.example for c in batch):
            raise ValueError("Donor is not a training example")
        batches.append(batch)
    return cfg, dev, final, batches


def quota():
    used = int(
        subprocess.run(
            ["du", "-x", "-B1", "-s", str(Path.home())],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()[0]
    )
    if used + 500_000_000 > 50_000_000_000:
        raise ValueError("50GB policy requires more disk headroom before next cell")
    return used


def verifier_for():
    return VerifierRegistry.from_spec("third_eye.evaluation.benchmarks:make_verifier")


def preflight(request, registry):
    require_gpu_allocation()
    from third_eye.training.hf_backend import HFBackend
    import torch

    output = Path(request["output"]) / "preflight"
    output.mkdir(parents=True, exist_ok=False)
    timings = []
    for stream in sorted({t["stream"] for t in registry["tasks"]}):
        task = next(t for t in registry["tasks"] if t["stream"] == stream)
        cfg, dev, final, batches = inputs(request, registry, task)
        backend, verifier = HFBackend(cfg), verifier_for()
        restore_portable(backend, mapped(request, registry, task["m1"]))
        before, before_hash = backend.snapshot(), backend.state_hash()
        batch = next((b for b in batches if b), ())
        if not batch or not all(
            verify_completions(
                verifier, [c.example for c in batch], [c.completion for c in batch]
            )
        ):
            raise ValueError("Archived training corrections fail native verifier")
        import time

        try:
            train = backend.train(batch, cfg.protocol.seed)
            if train["optimizer_steps"] != 50:
                raise ValueError("Real recipe compatibility budget differs")
            for state in ("trained", "receiver"):
                if state == "receiver":
                    backend.restore(before)
                for role in ("target_test", "ood_test", "retention_test"):
                    from third_eye.evaluation.runner import generate_greedy

                    examples = final[role][:8]
                    start = time.perf_counter()
                    completions = generate_greedy(
                        backend,
                        examples,
                        cfg.protocol.max_new_tokens,
                        [cfg.protocol.seed] * 8,
                    )
                    verdicts = verify_completions(verifier, examples, completions)
                    if len(verdicts) != 8:
                        raise ValueError("Incomplete verifier timing")
                    torch.cuda.synchronize()
                    timings.append(
                        {
                            "stream": stream,
                            "role": role,
                            "state": state,
                            "seconds": time.perf_counter() - start,
                            "count": 8,
                            "accuracy_recorded": False,
                            "train_seconds": train["seconds"],
                        }
                    )
        finally:
            backend.restore(before)
        if backend.state_hash() != before_hash:
            raise ValueError("Compatibility update persisted")
        del backend
        gc.collect()
        torch.cuda.empty_cache()
    coverage = {}
    for stream in sorted({t["stream"] for t in registry["tasks"]}):
        profile = [v for v in timings if v["stream"] == stream]
        scoring = 17 * sum(
            max(v["seconds"] / v["count"] for v in profile if v["role"] == role) * count
            for role, count in [
                ("target_test", 128),
                ("ood_test", 128),
                ("retention_test", 256),
            ]
        )
        old_harvest = max(
            d["observation"].get("pool_statistics", {}).get("total_seconds", 0)
            for t in registry["tasks"]
            if t["stream"] == stream
            for d in t["draws"]
        )
        required = 1.25 * (
            scoring
            + 15 * max(v["train_seconds"] for v in profile)
            + 2 * old_harvest * 4
            + 600
        )
        coverage[stream] = {
            "required_seconds": required,
            "reserved_seconds": request["per_parent_seconds"],
            "includes_all17_scored_cells": True,
            "historical_harvest_multiplier": 4,
        }
    write_json(output / "runtime_coverage.json", coverage)
    if any(v["required_seconds"] > v["reserved_seconds"] for v in coverage.values()):
        raise ValueError(
            "Complete paired controls exceed reserved walltime; no collection"
        )
    write_json(
        output / "completed.json",
        {
            "passed": True,
            "source_tree_sha256": request["source_tree_sha256"],
            "registry_sha256": request["registry_sha256"],
            "timings": timings,
            "runtime_coverage": coverage,
            "discarded_compatibility_updates": True,
            "cross_hardware_native_baselines_required": True,
        },
    )
    print(json.dumps({"preflight_passed": True, "streams": 4}))


def run(request, registry, index):
    require_gpu_allocation()
    from third_eye.training.hf_backend import HFBackend

    ready = read(Path(request["output"]) / "preflight/completed.json")
    if (
        not ready["passed"]
        or ready["source_tree_sha256"] != request["source_tree_sha256"]
        or ready["registry_sha256"] != request["registry_sha256"]
    ):
        raise ValueError("Matching GPU compatibility check required")
    task = registry["tasks"][index]
    cfg, dev, final, batches = inputs(request, registry, task)
    root = Path(request["output"]) / "parents" / task["stream"] / str(task["seed"])
    root.mkdir(parents=True, exist_ok=False)
    backend, verifier = HFBackend(cfg), verifier_for()
    # Validate every archived and reference solution before any experimental cell.
    for batch in batches:
        for candidate in (batch, reference_corrections(batch)):
            if not all(
                verify_completions(
                    verifier,
                    [c.example for c in candidate],
                    [c.completion for c in candidate],
                )
            ):
                raise ValueError(
                    "Saved/reference training solution is structurally invalid"
                )
    receivers = {}
    original_hashes = {"m0": task["parent_hash"], "m1": task["candidate_hash"]}
    for receiver in ("m0", "m1"):
        restore_portable(backend, mapped(request, registry, task[receiver]))
        if backend.state_hash() != original_hashes[receiver]:
            raise ValueError("Wrong transferred receiver")
        receivers[receiver] = backend.snapshot()

    records = []

    def score(name, receiver, batch, steps, seed, draw=None, train=True):
        used = quota()
        backend.restore(receivers[receiver])
        before = backend.state_hash()
        directory = root / name
        directory.mkdir(exist_ok=False)
        write_json(directory / "batch.json", [c.to_dict() for c in batch])
        try:
            log = (
                backend.train(
                    batch, seed, max_steps=steps, log_path=directory / "training.jsonl"
                )
                if train and batch
                else {"optimizer_steps": 0, "seconds": 0}
            )
            if log["optimizer_steps"] != (steps if train and batch else 0):
                raise ValueError("Cell optimizer budget differs")
            after = backend.state_hash()
            if (not train or not batch) and after != before:
                raise ValueError("Identity/terminal cell changed receiver")
            metadata = {
                "experiment": "paired_DGX_review_controls",
                "source_tree_sha256": request["source_tree_sha256"],
                "registry_sha256": request["registry_sha256"],
                "receiver": receiver,
                "draw": draw,
                "training_seed": seed,
                "steps": steps,
                "cross_hardware_baseline_recomputed": True,
            }
            if train:
                backend.save_checkpoint(directory / "adapter", metadata)
            result = evaluate_final(
                backend,
                final,
                verifier,
                directory / "scoring",
                cfg.protocol.max_new_tokens,
                cfg.protocol.seed,
                metadata,
            )
            record = {
                **metadata,
                "name": name,
                "receiver_adapter_hash": before,
                "result_adapter_hash": after,
                "training": log,
                "batch_hash": batch_hash(batch) if batch else None,
                "batch_size": len(batch),
                "continuation_available": int(bool(batch)),
                "scores": result["scores"],
                "scores_sha256": file_digest(directory / "scoring/scores.json"),
                "items_sha256": file_digest(directory / "scoring/items.jsonl"),
                "home_used_bytes_before": used,
            }
            write_json(directory / "completed.json", record)
            records.append(record)
            write_json(
                root / "progress.json",
                {
                    "completed": [r["name"] for r in records],
                    "expected_scored_cells": 17,
                },
            )
        finally:
            backend.restore(receivers[receiver])
        return record

    score("m0_before", "m0", (), 0, cfg.protocol.seed, train=False)
    score("m1_before", "m1", (), 0, cfg.protocol.seed, train=False)
    for d in task["selected_draws"]:
        seed = task["draws"][d]["observation"]["seed"]
        d1 = batches[d]
        backend.restore(receivers["m0"])
        pool, stats = collect_corrections(
            backend,
            dev["train"],
            verifier,
            cfg.protocol,
            task["new_harvest_seeds"][str(d)],
        )
        d0 = (
            sample_batches(
                pool,
                2,
                1,
                task["new_harvest_seeds"][str(d)],
                cfg.protocol.candidate_sampling,
            )[0]
            if len(pool) >= 2
            else ()
        )
        write_json(
            root / f"draw{d}_m0_harvest.json",
            {
                "stats": stats,
                "pool": [c.to_dict() for c in pool],
                "batch": [c.to_dict() for c in d0],
                "seed": task["new_harvest_seeds"][str(d)],
            },
        )
        for receiver, donor, batch in [
            ("m0", "d0", d0),
            ("m0", "d1", d1),
            ("m1", "d0", d0),
            ("m1", "d1", d1),
        ]:
            score(f"draw{d}_{receiver}_{donor}_50", receiver, batch, 50, seed, d)
        for steps in (10, 25):
            score(f"draw{d}_m1_d1_{steps}", "m1", d1, steps, seed, d)
        score(f"draw{d}_m1_gold_50", "m1", reference_corrections(d1), 50, seed, d)
    pooled = pooled_corrections(batches)
    pooled_seed = task["draws"][min(task["selected_draws"])]["observation"]["seed"]
    score("m1_pooled_50", "m1", pooled, 50, pooled_seed)
    if len(records) != 17:
        raise ValueError("Complete paired factorial/recipe set required")
    write_json(
        root / "completed.json",
        {
            "passed": True,
            "stream": task["stream"],
            "seed": task["seed"],
            "source_tree_sha256": request["source_tree_sha256"],
            "registry_sha256": request["registry_sha256"],
            "records": records,
            "original_Vista_results_preserved": True,
        },
    )
    print(
        json.dumps(
            {"paired_parent_completed": True, "index": index, "scored_cells": 17}
        )
    )


def analyze_saved(request, registry):
    import numpy as np
    from third_eye.experiments.smollm_confirmation import matrix, NAMES
    from third_eye.experiments.review_statistics import ROLES, summarize

    states = registry["states"]
    for state in states:
        state["artifacts"] = {
            k: {r: np.asarray(x) for r, x in heads.items()}
            for k, heads in state["artifacts"].items()
        }
    matrices = [{role: matrix(state, role) for role in ROLES} for state in states]
    dest = Path(request["output"]) / "saved_data_analysis"
    report = summarize(
        states,
        matrices,
        NAMES,
        dest,
        {
            "source_tree_sha256": request["source_tree_sha256"],
            "registry_sha256": request["registry_sha256"],
            "status": "review_driven_exploratory_original32_parents",
        },
    )
    changes = {}
    for endpoint, h1, h2 in [("selected", 0, 1), ("uniform", 2, 3)]:
        points = np.asarray(
            [np.mean([m[r].mean(1) for r in ROLES], axis=0) for m in matrices]
        )
        changes[endpoint] = {
            "parents": len(states),
            "observed_positive_to_negative": int(
                np.sum((points[:, h1] > 0) & (points[:, h2] < 0))
            ),
            "observed_negative_to_positive": int(
                np.sum((points[:, h1] < 0) & (points[:, h2] > 0))
            ),
            "qualification": "signs of observed parent means; not latent individual causal effects",
        }
    write_json(dest / "sign_transitions.json", changes)
    write_json(
        dest / "completed.json",
        {
            "passed": True,
            "report_sha256": file_digest(dest / "report.json"),
            "parents": report["parents"],
        },
    )
    print(json.dumps({"saved_data_analysis_complete": True, "parents": 32}))


def review(request, registry):
    import numpy as np
    from third_eye.experiments.review_statistics import (
        CONTROL_NAMES,
        ROLES,
        control_matrix,
        summarize,
    )

    states, matrices, hashes = [], [], {}
    for task in registry["tasks"]:
        root = Path(request["output"]) / "parents" / task["stream"] / str(task["seed"])
        receipt = read(root / "completed.json")
        if (
            not receipt["passed"]
            or receipt["source_tree_sha256"] != request["source_tree_sha256"]
            or receipt["registry_sha256"] != request["registry_sha256"]
            or len(receipt["records"]) != 17
        ):
            raise ValueError(
                "Complete source-bound sixteen-parent control cohort required"
            )
        _, _, final, _ = inputs(request, registry, task)
        cells = {}
        for record in receipt["records"]:
            name = record["name"]
            if name in cells:
                raise ValueError("Repeated cell identifier")
            dest = root / name / "scoring"
            for filename, key in [
                ("scores.json", "scores_sha256"),
                ("items.jsonl", "items_sha256"),
            ]:
                if file_digest(dest / filename) != record[key]:
                    raise ValueError("Immutable control output changed")
                hashes[str(dest / filename)] = record[key]
            rows = [
                json.loads(line)
                for line in (dest / "items.jsonl").read_text().splitlines()
                if line.strip()
            ]
            heads = {}
            for role in ROLES:
                observed = {r["id"]: r["passed"] for r in rows if r["role"] == role}
                expected = [ex.id for ex in final[role]]
                if set(observed) != set(expected) or sum(
                    r["role"] == role for r in rows
                ) != len(expected):
                    raise ValueError("Control scoring identities/counts differ")
                heads[role] = np.asarray([observed[i] for i in expected], float)
            cells[name] = heads
        state = {
            "stream": task["stream"],
            "seed": task["seed"],
            "ids": {
                r: [(ex.id, ex.prompt_hash) for ex in examples]
                for r, examples in final.items()
            },
        }
        states.append(state)
        matrices.append(
            {r: control_matrix(cells, task["selected_draws"], r) for r in ROLES}
        )
    if len(states) != 16:
        raise ValueError("Complete sixteen-parent follow-up required")
    dest = Path(request["output"]) / "paired_controls_analysis"
    summarize(
        states,
        matrices,
        CONTROL_NAMES,
        dest,
        {
            "source_tree_sha256": request["source_tree_sha256"],
            "registry_sha256": request["registry_sha256"],
            "output_hashes": hashes,
            "cross_hardware_native_baselines_recomputed": True,
            "status": "paired_followup_on_inspected_original_parents_not_fresh_replication",
        },
        adjusted=True,
    )
    write_json(
        dest / "completed.json",
        {
            "passed": True,
            "report_sha256": file_digest(dest / "report.json"),
            "parents": 16,
            "automatic_expansion": False,
        },
    )
    print(json.dumps({"all_paired_controls_complete": True, "parents": 16}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["preflight", "run", "analyze-saved", "review"]
    )
    parser.add_argument("--request", required=True)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    request, registry = checked(args.request)
    if args.command == "preflight":
        preflight(request, registry)
    elif args.command == "run":
        run(request, registry, args.index)
    elif args.command == "analyze-saved":
        analyze_saved(request, registry)
    else:
        review(request, registry)


if __name__ == "__main__":
    main()
