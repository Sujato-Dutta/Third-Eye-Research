"""Collect the approved16-run third-family extension; no forecasting or scaling."""

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np

from finish_confirmation import run_task
from forecasting_pilots import inspect_task
from third_eye.cluster import require_compute_allocation, require_gpu_allocation
from third_eye.config import Config
from third_eye.data.schema import Correction
from third_eye.data.splits import load_manifest
from third_eye.evaluation.final import load_final_manifest
from third_eye.experiments.confirmation import checkpoints, read_scored_items
from third_eye.experiments.smollm_confirmation import DECLARATION, actions, summarize
from third_eye.io import file_digest, write_json
from third_eye.provenance import source_inventory


def read(path):
    return json.loads(Path(path).read_text())


def checked(path):
    require_compute_allocation()
    request = read(path)
    if request["source_tree_sha256"] != source_inventory()["source_tree_sha256"]:
        raise ValueError("Frozen extension source changed")
    validation = read(request["validation"])
    if (
        not validation.get("passed")
        or validation["source_tree_sha256"] != request["source_tree_sha256"]
    ):
        raise ValueError("Full matching CPU validation required")
    for value in request["inputs"].values():
        if file_digest(value["path"]) != value["sha256"]:
            raise ValueError("Frozen extension input changed")
    return request


def plan_for(request):
    output = Path(request["output"])
    plan = read(output / "plan.json")
    receipt = read(output / "prepared.json")
    if (
        plan["declaration"] != DECLARATION
        or plan["source_tree_sha256"] != request["source_tree_sha256"]
    ):
        raise ValueError("Extension declaration/source differs")
    if (
        receipt["plan_sha256"] != file_digest(output / "plan.json")
        or not receipt["passed"]
    ):
        raise ValueError("Source-bound preparation required")
    for value in plan["frozen_files"].values():
        if file_digest(value["path"]) != value["sha256"]:
            raise ValueError("Frozen configuration or scoring suite differs")
    plan["plan_sha256"] = file_digest(output / "plan.json")
    return plan


def prepare(request):
    output = Path(request["output"])
    if (output / "plan.json").exists():
        raise FileExistsError("Extension preparation is immutable")
    old = read(request["inputs"]["core_plan"]["path"])
    identity = read(request["inputs"]["model_identity"]["path"])
    if (
        identity["model"] != DECLARATION["model"]
        or identity["revision"] != DECLARATION["revision"]
    ):
        raise ValueError("Original official SmolLM revision required")
    for p, h in identity["file_sha256"].items():
        if file_digest(p) != h:
            raise ValueError("Official cached model snapshot changed")
    seeds = {seed for values in DECLARATION["seeds"].values() for seed in values}
    if len(seeds) != 16:
        raise ValueError("Extension seeds must be unique")
    collision_count = 0
    for root in request["history_roots"]:
        for p in Path(root).rglob("run.json"):
            if output in p.parents:
                continue
            row = read(p)
            seed = row.get("config", {}).get("protocol", {}).get("seed")
            if seed in seeds:
                raise ValueError("New seed already used: " + str(p))
            collision_count += 1
    frozen = {}
    tasks = []
    references = {}

    def bind(path):
        frozen[str(path)] = {"path": str(path), "sha256": file_digest(path)}

    for task_name in ("code", "math"):
        original = next(
            t for t in old["tasks"] if t["stream"] == "qwen3-4b/" + task_name
        )
        for field in ("config", "manifest", "final_manifest"):
            if file_digest(original[field]) != original[field + "_sha256"]:
                raise ValueError("Core suites/configuration changed")
        manifest, dev = load_manifest(original["manifest"])
        _, final = load_final_manifest(original["final_manifest"], original["manifest"])
        if {k: len(v) for k, v in final.items()} != DECLARATION["scoring_items"]:
            raise ValueError("Fixed independent scoring sizes required")
        for path in [original["manifest"], original["final_manifest"]]:
            bind(Path(path))
        for entry in manifest["splits"].values():
            bind(Path(original["manifest"]).parent / entry["path"])
        for entry in read(original["final_manifest"])["splits"].values():
            bind(Path(original["final_manifest"]).parent / entry["path"])
        references[task_name] = original
        for seed in DECLARATION["seeds"][task_name]:
            cfg = deepcopy(read(original["config"]))
            cfg["model"].update(
                name=DECLARATION["model"],
                revision=DECLARATION["revision"],
                chat_kwargs={"enable_thinking": False, "date_string": "11 Oct 2026"},
            )
            cfg["protocol"]["seed"] = seed
            path = output / "configs" / f"{task_name}_{seed}.json"
            write_json(path, cfg)
            loaded = Config.load(path)
            if (
                loaded.protocol.candidates != 3
                or loaded.protocol.candidate_size != 2
                or loaded.protocol.correction_attempts != 8
                or loaded.protocol.correction_seed_stride != 16
                or loaded.training.max_steps != 50
                or loaded.training.rank != 16
                or loaded.training.learning_rate != 1e-4
            ):
                raise ValueError("Original fixed update and harvest budgets required")
            bind(path)
            tasks.append(
                {
                    **original,
                    "stream": "smollm3-3b/" + task_name,
                    "index": len(tasks),
                    "seed": seed,
                    "config": str(path),
                    "config_sha256": file_digest(path),
                    "output": str(output / "parents" / task_name / f"seed{seed}"),
                }
            )
    plan = {
        "source_tree_sha256": request["source_tree_sha256"],
        "declaration": DECLARATION,
        "tasks": tasks,
        "frozen_files": frozen,
        "reference_tasks": references,
        "history_run_records_checked": collision_count,
        "automatic_expansion": False,
    }
    write_json(output / "plan.json", plan)
    write_json(
        output / "prepared.json",
        {
            "passed": True,
            "plan_sha256": file_digest(output / "plan.json"),
            "source_tree_sha256": request["source_tree_sha256"],
        },
    )
    print(
        json.dumps(
            {
                "prepared": True,
                "parents": 16,
                "third_model": DECLARATION["model"],
                "forecaster_fits": 0,
            }
        )
    )


def preflight(request, task_name):
    require_gpu_allocation()
    import torch
    from third_eye.training.hf_backend import HFBackend
    from third_eye.evaluation.verifiers import VerifierRegistry
    from third_eye.evaluation.runner import generate_greedy, verify_completions

    plan = plan_for(request)
    task = next(t for t in plan["tasks"] if t["stream"].endswith("/" + task_name))
    cfg = Config.load(task["config"])
    _, dev = load_manifest(task["manifest"])
    backend = HFBackend(cfg)
    verifier = VerifierRegistry.from_spec(
        "third_eye.evaluation.benchmarks:make_verifier"
    )
    output = Path(request["output"]) / "preflight" / task_name
    output.mkdir(parents=True, exist_ok=False)
    parent = backend.snapshot()
    parent_hash = backend.state_hash()
    fixtures = [Correction(ex, ex.answer, 1) for ex in dev["train"][:2]]
    if not all(
        verify_completions(
            verifier,
            [item.example for item in fixtures],
            [item.completion for item in fixtures],
        )
    ):
        raise ValueError(
            "Gold train compatibility fixtures must pass the native verifier"
        )
    train = backend.train(
        fixtures, cfg.protocol.seed, log_path=output / "training.jsonl"
    )
    if train["optimizer_steps"] != 50:
        raise ValueError("Compatibility optimizer budget differs")
    trained_hash = backend.state_hash()
    backend.save_checkpoint(
        output / "adapter", {"purpose": "discarded_compatibility_fixture"}
    )
    backend.restore(parent)
    if backend.state_hash() != parent_hash:
        raise ValueError("Compatibility rollback failed")
    saved = backend.load_checkpoint(output / "adapter")
    if saved["adapter_hash"] != trained_hash:
        raise ValueError("Compatibility checkpoint restore failed")
    observations = []
    for state in ("trained", "parent"):
        if state == "parent":
            backend.restore(parent)
        for role in ("target_dev", "ood_dev", "retention_dev"):
            examples = dev[role][:8]
            start = time.perf_counter()
            texts = generate_greedy(
                backend,
                examples,
                cfg.protocol.max_new_tokens,
                [cfg.protocol.seed] * len(examples),
            )
            verdicts = verify_completions(verifier, examples, texts)
            torch.cuda.synchronize()
            if len(verdicts) != 8:
                raise ValueError("Incomplete timing/verifier case")
            observations.append(
                {
                    "role": role,
                    "state": state,
                    "count": 8,
                    "seconds": time.perf_counter() - start,
                    "accuracy_recorded": False,
                }
            )
        if task_name == "code":
            _, all_final = load_final_manifest(
                request["inputs"]["calibration_final"]["path"], task["manifest"]
            )
            wanted = {
                x["id"]
                for x in read(request["inputs"]["calibration_items"]["path"])["items"]
            }
            examples = [ex for ex in all_final["ood_test"] if ex.id in wanted]
            if len(examples) != 8:
                raise ValueError("Exactly eight excluded code timing items required")
            start = time.perf_counter()
            texts = generate_greedy(
                backend, examples, cfg.protocol.max_new_tokens, [cfg.protocol.seed] * 8
            )
            verdicts = verify_completions(verifier, examples, texts)
            torch.cuda.synchronize()
            if len(verdicts) != 8:
                raise ValueError("Direct code timing incomplete")
            observations.append(
                {
                    "role": "ood_test",
                    "state": state,
                    "count": 8,
                    "seconds": time.perf_counter() - start,
                    "accuracy_recorded": False,
                }
            )
    backend.restore(parent)
    if backend.state_hash() != parent_hash:
        raise ValueError("Compatibility fixture persisted")
    del backend
    gc.collect()
    torch.cuda.empty_cache()
    report = {
        "passed": True,
        "source_tree_sha256": request["source_tree_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "observations": observations,
        "train_seconds": train["seconds"],
        "discarded_fixture": True,
        "forecaster_fits": 0,
    }
    write_json(output / "completed.json", report)
    print(json.dumps({"preflight_passed": task_name, "fixture_discarded": True}))


def price(request):
    plan = plan_for(request)
    results = {}
    old_tasks = read(request["inputs"]["old_smol_tasks"]["path"])["tasks"]
    for task_name in ("code", "math"):
        pre = read(Path(request["output"]) / "preflight" / task_name / "completed.json")
        if (
            not pre["passed"]
            or pre["source_tree_sha256"] != request["source_tree_sha256"]
            or pre["plan_sha256"] != plan["plan_sha256"]
        ):
            raise ValueError("Matching both-task CUDA verification required")
        rates = {
            r: max(
                v["seconds"] / v["count"] for v in pre["observations"] if v["role"] == r
            )
            for r in ("target_dev", "ood_dev", "retention_dev")
        }
        final_rates = {**rates}
        if task_name == "code":
            final_rates["ood_dev"] = max(
                v["seconds"] / v["count"]
                for v in pre["observations"]
                if v["role"] == "ood_test"
            )
        old = []
        for t in old_tasks:
            if not t["stream"].endswith("/" + task_name):
                continue
            folder = Path(t["output"])
            run = read(folder / "run.json")
            rows = [
                json.loads(x)
                for x in (folder / "meta_labels.jsonl").read_text().splitlines()
                if x.strip()
            ]
            ledger = [
                json.loads(x)
                for x in (folder / "ledger.jsonl").read_text().splitlines()
                if x.strip()
            ]
            base = max(
                row["seconds"] for row in ledger if row.get("status") == "complete"
            )
            extras = max(0, 2 - len(run.get("continuation_seed_offsets") or []))
            per_draw = max(
                row["continuation"]["pool_statistics"]["total_seconds"]
                + row["runtime"]["continuation"]["seconds"]
                for row in rows
            ) + sum(
                rates[r] * n
                for r, n in [
                    ("target_dev", 64),
                    ("ood_dev", 64),
                    ("retention_dev", 256),
                ]
            )
            old.append(base + 3 * extras * per_draw)
        scoring = 13 * sum(
            final_rates[r] * n
            for r, n in [("target_dev", 128), ("ood_dev", 128), ("retention_dev", 256)]
        )
        needed = 1.25 * (max(old) + scoring + 120)
        results[task_name] = {
            "required_seconds": needed,
            "reserved_seconds": DECLARATION["per_parent_seconds"],
            "historical_collection_adjusted_seconds": max(old),
            "final_scoring_seconds": scoring,
            "includes_three_draws": True,
        }
    output = Path(request["output"]) / "preflight"
    write_json(output / "cost_report.json", results)
    if any(x["required_seconds"] > x["reserved_seconds"] for x in results.values()):
        raise ValueError(
            "Measured complete runtime exceeds reservation; preserve plan before fresh collection"
        )
    write_json(
        output / "completed.json",
        {
            "passed": True,
            "source_tree_sha256": request["source_tree_sha256"],
            "plan_sha256": plan["plan_sha256"],
            "cost_report_sha256": file_digest(output / "cost_report.json"),
        },
    )
    print(json.dumps({"full_runtime_coverage_passed": True, "cost": results}))


def collect(request, index):
    plan = plan_for(request)
    ready = read(Path(request["output"]) / "preflight/completed.json")
    if (
        not ready["passed"]
        or ready["source_tree_sha256"] != request["source_tree_sha256"]
        or ready["plan_sha256"] != plan["plan_sha256"]
    ):
        raise ValueError("Successful priced CUDA verification required")
    run_task(request, plan, plan["tasks"][index], decision_rule=actions)


def review(request):
    plan = plan_for(request)
    states = []
    hashes = {}
    for task in plan["tasks"]:
        folder = Path(task["output"])
        done = read(folder / "completed.json")
        decision = read(folder / "commitment.json")
        if (
            not done.get("passed")
            or done["source_tree_sha256"] != request["source_tree_sha256"]
            or done["plan_sha256"] != plan["plan_sha256"]
            or done["commitment_sha256"] != file_digest(folder / "commitment.json")
        ):
            raise ValueError("All sixteen source-bound complete runs required")
        if decision["actions"] != actions(
            task["stream"], decision["development_h1_heads"]
        ):
            raise ValueError("Frozen development action differs")
        rows, _ = inspect_task(
            {**task, "output": str(folder / "collection")},
            [50_000_000, 100_000_000],
            request["source_tree_sha256"],
        )
        _, expected = load_final_manifest(task["final_manifest"], task["manifest"])
        artifacts = {}
        for name, _, adapter_hash in checkpoints(rows):
            dest = folder / "scoring" / name
            for filename, key in [
                ("scores.json", "scores_sha256"),
                ("items.jsonl", "items_sha256"),
            ]:
                h = file_digest(dest / filename)
                if h != done["scores"][name][key]:
                    raise ValueError("Immutable scored output differs")
                hashes[str(dest / filename)] = h
            metadata = read(dest / "scores.json")
            if (
                metadata["source_tree_sha256"] != request["source_tree_sha256"]
                or metadata["plan_sha256"] != plan["plan_sha256"]
            ):
                raise ValueError("Scored provenance differs")
            artifacts[name] = read_scored_items(
                dest, expected, adapter_hash, done["commitment_sha256"]
            )
        for k, row in enumerate(rows):
            for d, obs in enumerate(row["continuation_replicates"]):
                if not obs["continuation_available"] and any(
                    not np.array_equal(
                        artifacts[f"m2_{k}_{d}"][r], artifacts[f"m1_{k}"][r]
                    )
                    for r in expected
                ):
                    raise ValueError("Terminal outcomes changed M1")
        states.append(
            {
                "stream": task["stream"],
                "seed": task["seed"],
                "actions": decision["actions"],
                "artifacts": artifacts,
                "ids": {
                    r: [(ex.id, ex.prompt_hash) for ex in examples]
                    for r, examples in expected.items()
                },
                "terminal_draws": sum(
                    1 - obs["continuation_available"]
                    for row in rows
                    for obs in row["continuation_replicates"]
                ),
            }
        )
    dest = Path(request["output"]) / "evidence"
    dest.mkdir(exist_ok=False)
    summarize(
        states,
        dest,
        {
            "source_tree_sha256": request["source_tree_sha256"],
            "plan_sha256": plan["plan_sha256"],
            "input_hashes": hashes,
        },
    )
    write_json(
        dest / "completed.json",
        {
            "passed": True,
            "parents": 16,
            "report_sha256": file_digest(dest / "report.json"),
            "automatic_expansion": False,
        },
    )
    print(json.dumps({"extension_complete": True, "parents": 16, "forecaster_fits": 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["prepare", "preflight", "price", "run", "review"]
    )
    parser.add_argument("--request", required=True)
    parser.add_argument("--task", choices=["math", "code"])
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    request = checked(args.request)
    if args.command == "preflight":
        preflight(request, args.task)
    elif args.command == "run":
        collect(request, args.index)
    else:
        {"prepare": prepare, "price": price, "review": review}[args.command](request)


if __name__ == "__main__":
    main()
