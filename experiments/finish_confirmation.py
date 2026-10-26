"""One fixed empirical confirmation cohort; no forecaster fitting or expansion."""

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from admission_pilots import load_pilot_states
from forecasting_pilots import inspect_task, item_artifact
from third_eye.cluster import require_compute_allocation, require_gpu_allocation
from third_eye.config import Config
from third_eye.data.splits import load_manifest
from third_eye.evaluation.final import evaluate_final, load_final_manifest
from third_eye.experiments.admission import views
from third_eye.experiments.confirmation import (
    DECLARED,
    actions,
    checkpoints,
    primary_power,
    read_scored_items,
)
from third_eye.experiments.confirmation_design import bootstrap
from third_eye.experiments.mechanism import ROLES, choose, folds
from third_eye.io import digest, file_digest, publish_jsonl_batch, write_json
from third_eye.provenance import source_inventory


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked_request(path):
    require_compute_allocation()
    request = read(path)
    source = source_inventory()["source_tree_sha256"]
    validation = read(request["validation"])
    if (
        request["source_tree_sha256"] != source
        or request["declaration"] != DECLARED
        or validation.get("passed") is not True
        or validation["source_tree_sha256"] != source
    ):
        raise ValueError(
            "Matching frozen source/declaration and CPU validation required"
        )
    for key, value in request["inputs"].items():
        if file_digest(value["path"]) != value["sha256"]:
            raise ValueError("Bound evidence changed: " + key)
    return request


def checked_plan(request):
    path = Path(request["output"]) / "plan.json"
    plan = read(path)
    if (
        plan["source_tree_sha256"] != request["source_tree_sha256"]
        or plan["declaration"] != DECLARED
        or plan["request_sha256"] != request["request_sha256"]
    ):
        raise ValueError("Frozen cohort plan differs")
    for value in plan["frozen_files"].values():
        if file_digest(value["path"]) != value["sha256"]:
            raise ValueError("Frozen cohort file changed")
    completed = read(Path(request["output"]) / "prepared.json")
    if completed.get("passed") is not True or completed["plan_sha256"] != file_digest(
        path
    ):
        raise ValueError("Complete matching preparation required")
    plan["plan_sha256"] = file_digest(path)
    return plan


def prepare(request):
    output = Path(request["output"])
    if (output / "plan.json").exists() or (output / "configs").exists():
        raise FileExistsError("Preparation is immutable; no silent rerun")
    inputs = request["inputs"]
    design = read(inputs["design_report"]["path"])
    for value in design["inputs"].values():
        if file_digest(value["path"]) != value["sha256"]:
            raise ValueError("Previously checked development input changed")
    if (
        read(inputs["design_completion"]["path"])["report_sha256"]
        != inputs["design_report"]["sha256"]
    ):
        raise ValueError("Design receipt differs")
    pilot = read(design["inputs"]["pilot_plan"]["path"])
    if (
        file_digest(design["inputs"]["pilot_plan"]["path"])
        != design["inputs"]["pilot_plan"]["sha256"]
    ):
        raise ValueError("Development plan changed")
    classification = read(design["inputs"]["classification_report"]["path"])
    if (
        classification["remaining_flagged_artifacts"]
        or classification["original_unreadable_artifacts"]
    ):
        raise ValueError("Recorded exposure unresolved")
    states, coverage, _ = load_pilot_states(pilot)
    for path, sha in read(design["inputs"]["pilot_report"]["path"])[
        "input_hashes"
    ].items():
        if file_digest(path) != sha:
            raise ValueError("Previously reviewed development artifact changed")
    if len(states) != 16 or any(not row["complete"] for row in coverage):
        raise ValueError("Complete source-bound development cohort required")
    counts = {
        task + "/" + role: DECLARED["scoring_items"][role.replace("_dev", "_test")]
        for task in ("math", "code")
        for role in ROLES
    }
    mass = np.zeros((16, 6, 3))
    for i, state in enumerate(states):
        immediate, _, _ = views(state["panel"], folds(state["parent"]), 0)
        mass[i, :] = choose(immediate.mean(1))
    point, boot, _, _ = bootstrap(states, mass, 32, counts, draws=2000)
    power = primary_power(point, boot)
    write_json(output / "primary_precision.json", power)
    if power["conditional_planned_power"] < 0.8:
        raise ValueError(
            "Fixed single-primary confirmation lacks declared planning precision; no label collection"
        )
    proposed = next(
        row
        for row in design["rows"]
        if row["parents"] == 32 and row["requested_target_ood_size"] == 128
    )
    if (
        proposed["item_counts"] != counts
        or proposed["cost"]["base_plus_scoring_reservation_sus"] + 2 > 560
    ):
        raise ValueError("Frozen N/items or complete cost exceeds confirmation ceiling")
    if (
        proposed["cost"]["walltime_seconds_by_stream"]
        != request["walltime_seconds_by_stream"]
    ):
        raise ValueError("Reserved stream runtimes differ from computed cost")
    seeds = {seed for values in DECLARED["seed_grid"].values() for seed in values}
    if len(seeds) != 32:
        raise ValueError("Independent, non-overlapping fresh seed grid required")
    audited = []
    for root in request["history_roots"]:
        for path in Path(root).rglob("run.json"):
            if output in path.parents:
                continue
            run = read(path)
            seed = run.get("config", {}).get("protocol", {}).get("seed")
            if seed in seeds:
                raise ValueError("Fresh seed collision: " + str(path))
            audited.append({"path": str(path), "sha256": file_digest(path)})
    write_json(output / "seed_audit.json", audited)
    calibration = design["calibration_reservation"]
    if (
        calibration["purpose"] != "cost_only_excluded_from_confirmation_scoring"
        or len(calibration["items"]) != 8
    ):
        raise ValueError("Fixed cost-only code identities required")
    write_json(output / "code_calibration.json", calibration)
    frozen = {}

    def bind(key, path):
        frozen[key] = {"path": str(path), "sha256": file_digest(path)}

    bind("code_calibration", output / "code_calibration.json")
    suites = {}
    shared_pools = {}
    for selection, audit in pilot["manifest_audits"].items():
        if (
            file_digest(selection) != audit["sha256"]
            or file_digest(audit["reserved_final_manifest"])
            != audit["reserved_final_manifest_sha256"]
        ):
            raise ValueError("Original item suites changed")
        manifest, dev = load_manifest(selection)
        _, final = load_final_manifest(audit["reserved_final_manifest"], selection)
        task = dev["train"][0].task
        folder = output / "scoring" / task
        folder.mkdir(parents=True, exist_ok=False)
        entries = {}
        for role, examples in final.items():
            excluded = (
                {row["id"] for row in calibration["items"]}
                if task == "code" and role == "ood_test"
                else set()
            )
            ordered = sorted(
                (ex for ex in examples if ex.id not in excluded),
                key=lambda ex: digest(
                    [
                        "third-eye-final-confirmation-20261022",
                        role,
                        ex.id,
                        ex.prompt_hash,
                    ]
                ),
            )
            size = DECLARED["scoring_items"][role]
            if len(ordered) < size:
                raise ValueError("Insufficient independent scoring items")
            identity = tuple((ex.id, ex.prompt_hash) for ex in ordered[:size])
            for previous in shared_pools.get(role, []):
                if set(previous) & set(identity) and previous != identity:
                    raise ValueError(
                        "Final suites partially overlap; explicit shared-identity inference required before collection"
                    )
            shared_pools.setdefault(role, []).append(identity)
            path = folder / (role + ".jsonl")
            publish_jsonl_batch(path, [asdict(ex) for ex in ordered[:size]])
            entries[role] = {
                "path": path.name,
                "sha256": file_digest(path),
                "count": size,
            }
            bind(task + "/" + role, path)
        path = folder / "final.json"
        write_json(
            path,
            {
                "schema_version": 1,
                "selection_hash": digest(manifest),
                "splits": entries,
            },
        )
        load_final_manifest(path, selection)
        suites[task] = {"path": str(path), "sha256": file_digest(path)}
        bind(task + "/final_manifest", path)
    tasks, code_sources, bounds = [], [], {}
    benchmarks = [read(design["inputs"][f"benchmark_{i}"]["path"]) for i in range(4)]
    for stream in sorted(DECLARED["seed_grid"]):
        template = next(task for task in pilot["tasks"] if task["stream"] == stream)
        if (
            file_digest(template["config"]) != template["config_sha256"]
            or file_digest(template["manifest"]) != template["manifest_sha256"]
        ):
            raise ValueError("Original configuration or selection suite changed")
        bind("selection/" + stream, template["manifest"])
        if stream.endswith("/code"):
            code_sources.append(template)
            old = next(
                value for value in benchmarks if value["task"]["stream"] == stream
            )
            bounds[stream] = 4 * max(
                (row["generation_seconds"] + row["verification_seconds"]) / row["count"]
                for row in old["observations"]
                if row["role"] == "ood_dev"
            )
        for seed in DECLARED["seed_grid"][stream]:
            cfg = read(template["config"])
            cfg["protocol"]["seed"] = seed
            path = output / "configs" / f"{stream.replace('/', '_')}_{seed}.json"
            write_json(path, cfg)
            checked = Config.load(path)
            if (
                checked.protocol.correction_attempts != 8
                or checked.training.max_steps != 50
                or checked.protocol.candidate_size != 2
                or checked.protocol.candidates != 3
            ):
                raise ValueError("Frozen training/correction protocol changed")
            task = {
                "index": len(tasks),
                "stream": stream,
                "seed": seed,
                "config": str(path),
                "config_sha256": file_digest(path),
                "manifest": template["manifest"],
                "manifest_sha256": template["manifest_sha256"],
                "final_manifest": suites[stream.split("/")[1]]["path"],
                "final_manifest_sha256": suites[stream.split("/")[1]]["sha256"],
                "output": str(output / "parents" / stream / f"seed{seed}"),
            }
            tasks.append(task)
            bind("config/" + str(task["index"]), path)
    plan = {
        "source_tree_sha256": request["source_tree_sha256"],
        "request_sha256": request["request_sha256"],
        "declaration": DECLARED,
        "purpose": "final_empirical_confirmation_no_forecaster",
        "tasks": tasks,
        "code_preflight_sources": code_sources,
        "code_rate_bounds": bounds,
        "frozen_files": frozen,
        "power": power,
        "cost": proposed["cost"],
        "automatic_expansion": False,
    }
    write_json(output / "plan.json", plan)
    write_json(
        output / "prepared.json",
        {
            "passed": True,
            "plan_sha256": file_digest(output / "plan.json"),
            "source_tree_sha256": request["source_tree_sha256"],
            "automatic_expansion": False,
        },
    )
    print(
        json.dumps(
            {
                "prepared": True,
                "parents": 32,
                "scoring_items": DECLARED["scoring_items"],
                "planned_primary_power": power["conditional_planned_power"],
                "forecaster_fits": 0,
            }
        )
    )


def code_preflight(request):
    from third_eye.evaluation.runner import generate_greedy, verify_completions
    from third_eye.evaluation.verifiers import VerifierRegistry
    from third_eye.training.hf_backend import HFBackend
    import torch

    require_gpu_allocation()
    plan = checked_plan(request)
    output = Path(request["output"]) / "code_preflight"
    output.mkdir(parents=True, exist_ok=False)
    calibration = read(Path(request["output"]) / "code_calibration.json")
    if (
        file_digest(calibration["original_final_manifest"])
        != calibration["original_final_sha256"]
    ):
        raise ValueError("Code calibration source changed")
    _, final = load_final_manifest(
        calibration["original_final_manifest"],
        plan["code_preflight_sources"][0]["manifest"],
    )
    wanted = calibration["items"]
    examples = [
        next(
            ex
            for ex in final["ood_test"]
            if ex.id == row["id"] and ex.prompt_hash == row["prompt_sha256"]
        )
        for row in wanted
    ]
    verifier = VerifierRegistry.from_spec(
        "third_eye.evaluation.benchmarks:make_verifier"
    )
    observations = []
    development = read(request["inputs"]["design_report"]["path"])
    pilot = read(development["inputs"]["pilot_plan"]["path"])
    for task in plan["code_preflight_sources"]:
        if (
            file_digest(task["config"]) != task["config_sha256"]
            or file_digest(task["manifest"]) != task["manifest_sha256"]
        ):
            raise ValueError("Bound code preflight input changed")
        cfg = Config.load(task["config"])
        rows, _ = inspect_task(
            task, [50_000_000, 100_000_000], pilot["source_tree_sha256"]
        )
        selected = [
            value for value in checkpoints(rows) if value[0] in {"parent", "m2_0_0"}
        ]
        backend = HFBackend(cfg)
        for name, checkpoint, expected in selected:
            saved = backend.load_checkpoint(checkpoint)
            if saved["adapter_hash"] != expected:
                raise ValueError("Code timing adapter identity differs")
            start = time.perf_counter()
            texts = generate_greedy(
                backend,
                examples,
                cfg.protocol.max_new_tokens,
                [cfg.protocol.seed] * len(examples),
            )
            torch.cuda.synchronize()
            verdicts = verify_completions(verifier, examples, texts)
            seconds = time.perf_counter() - start
            if len(verdicts) != 8 or backend.state_hash() != expected:
                raise ValueError("Code preflight incomplete or mutated adapter")
            observations.append(
                {
                    "stream": task["stream"],
                    "checkpoint": name,
                    "seconds": seconds,
                    "count": 8,
                    "seconds_per_item": seconds / 8,
                    "accuracy_recorded": False,
                }
            )
            if seconds / 8 > plan["code_rate_bounds"][task["stream"]]:
                write_json(
                    output / "failed_cost_bound.json",
                    {
                        "observations": observations,
                        "bound": plan["code_rate_bounds"][task["stream"]],
                    },
                )
                raise ValueError(
                    "Direct HumanEval timing exceeds reserved bound; no fresh labels"
                )
        del backend
        gc.collect()
        torch.cuda.empty_cache()
    write_json(
        output / "report.json",
        {
            "source_tree_sha256": request["source_tree_sha256"],
            "plan_sha256": plan["plan_sha256"],
            "observations": observations,
            "excluded_from_confirmation_items": wanted,
            "forecaster_fits": 0,
        },
    )
    write_json(
        output / "completed.json",
        {
            "passed": True,
            "report_sha256": file_digest(output / "report.json"),
            "source_tree_sha256": request["source_tree_sha256"],
            "plan_sha256": plan["plan_sha256"],
        },
    )
    print(
        json.dumps(
            {
                "direct_code_timing_passed": True,
                "future_confirmation_items_scored": False,
            }
        )
    )


def run(request, index):
    require_gpu_allocation()
    plan = checked_plan(request)
    preflight = read(Path(request["output"]) / "code_preflight/completed.json")
    if (
        preflight.get("passed") is not True
        or preflight["source_tree_sha256"] != request["source_tree_sha256"]
        or preflight["plan_sha256"] != plan["plan_sha256"]
    ):
        raise ValueError("Successful source-bound direct code verification required")
    task = plan["tasks"][index]
    run_task(request, plan, task)


def run_task(request, plan, task, decision_rule=actions):
    """Collect one immutable parent using a validated caller's frozen policy."""
    from third_eye.evaluation.verifiers import VerifierRegistry
    from third_eye.training.hf_backend import HFBackend

    require_gpu_allocation()
    if (
        file_digest(task["config"]) != task["config_sha256"]
        or file_digest(task["manifest"]) != task["manifest_sha256"]
    ):
        raise ValueError("Fresh configuration or selection suite changed")
    output = Path(task["output"])
    output.mkdir(parents=True, exist_ok=False)
    collection = output / "collection"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("run.py")),
            "--config",
            task["config"],
            "--manifest",
            task["manifest"],
            "--output",
            str(collection),
            "--generations",
            "1",
            "--policy",
            "random",
            "--capture-item-outcomes",
            "--continuation-seed-offsets",
            "50000000",
            "100000000",
        ],
        check=True,
    )
    source_task = {**task, "output": str(collection)}
    rows, inspection = inspect_task(
        source_task, [50_000_000, 100_000_000], request["source_tree_sha256"]
    )
    cfg = Config.load(task["config"])
    _, dev = load_manifest(task["manifest"])
    parent = item_artifact(
        rows[0]["evaluation_artifacts"]["parent"], rows[0]["parent_adapter_hash"], dev
    )
    assignment = folds(parent)
    immediate = []
    for row in rows:
        item = item_artifact(
            row["evaluation_artifacts"]["t1"], row["candidate_adapter_hash"], dev
        )
        immediate.append(
            [
                np.mean(
                    [
                        item["items"][role][i]["correct"]
                        - parent["items"][role][i]["correct"]
                        for i in assignment[role][0]
                    ]
                )
                for role in ROLES
            ]
        )
        for observation in row["continuation_replicates"]:
            if observation["optimizer_steps"] != (
                50 if observation["continuation_available"] else 0
            ):
                raise ValueError("Standardized continuation budget differs")
            if (
                not observation["continuation_available"]
                and observation["continuation_adapter_hash"]
                != row["candidate_adapter_hash"]
            ):
                raise ValueError("Terminal continuation changed M1")
    decision = {
        "source_tree_sha256": request["source_tree_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "stream": task["stream"],
        "seed": task["seed"],
        "development_h1_heads": immediate,
        "actions": decision_rule(task["stream"], immediate),
        "selection_orientation": 0,
        "no_scoring_outcomes_used": True,
        "forecaster_fits": 0,
    }
    write_json(output / "commitment.json", decision)
    commitment_hash = file_digest(output / "commitment.json")
    # Final labels are first accessed only after all policy actions are durable.
    final_manifest, splits = load_final_manifest(
        task["final_manifest"], task["manifest"]
    )
    backend = HFBackend(cfg)
    verifier = VerifierRegistry.from_spec(
        "third_eye.evaluation.benchmarks:make_verifier"
    )
    scores = {}
    for name, checkpoint, expected in checkpoints(rows):
        saved = backend.load_checkpoint(checkpoint)
        if saved["adapter_hash"] != expected:
            raise ValueError("Scored checkpoint identity differs")
        destination = output / "scoring" / name
        evaluate_final(
            backend,
            splits,
            verifier,
            destination,
            cfg.protocol.max_new_tokens,
            cfg.protocol.seed,
            {
                "config_hash": cfg.fingerprint,
                "final_manifest_hash": digest(final_manifest),
                "commitment_sha256": commitment_hash,
                "model": cfg.model.name,
                "checkpoint": str(checkpoint),
                "policy": "fixed_offline_empirical_confirmation",
                "plan_sha256": plan["plan_sha256"],
            },
        )
        scores[name] = {
            "scores_sha256": file_digest(destination / "scores.json"),
            "items_sha256": file_digest(destination / "items.jsonl"),
        }
        write_json(
            output / "progress.json",
            {"completed_checkpoints": list(scores), "total": 13},
        )
    write_json(
        output / "completed.json",
        {
            "passed": True,
            "source_tree_sha256": request["source_tree_sha256"],
            "plan_sha256": plan["plan_sha256"],
            "commitment_sha256": commitment_hash,
            "scores": scores,
            "inspection": inspection,
            "forecaster_fits": 0,
        },
    )
    print(
        json.dumps(
            {
                "parent_complete": True,
                "stream": task["stream"],
                "seed": task["seed"],
                "scored_checkpoints": 13,
            }
        )
    )


def review(request):
    from third_eye.experiments.confirmation_statistics import summarize

    plan = checked_plan(request)
    states, hashes = [], {}
    for task in plan["tasks"]:
        output = Path(task["output"])
        done = read(output / "completed.json")
        if (
            done.get("passed") is not True
            or done["plan_sha256"] != plan["plan_sha256"]
            or done["commitment_sha256"] != file_digest(output / "commitment.json")
        ):
            raise ValueError("All 32 source-bound complete parents required")
        decision = read(output / "commitment.json")
        if decision["actions"] != actions(
            task["stream"], decision["development_h1_heads"]
        ):
            raise ValueError("Policy actions differ from frozen development-only rules")
        rows, _ = inspect_task(
            {**task, "output": str(output / "collection")},
            [50_000_000, 100_000_000],
            request["source_tree_sha256"],
        )
        manifest, expected = load_final_manifest(
            task["final_manifest"], task["manifest"]
        )
        artifacts = {}
        for name, _, expected_hash in checkpoints(rows):
            folder = output / "scoring" / name
            for filename, key in [
                ("scores.json", "scores_sha256"),
                ("items.jsonl", "items_sha256"),
            ]:
                actual = file_digest(folder / filename)
                if actual != done["scores"][name][key]:
                    raise ValueError("Immutable scored artifact changed")
                hashes[str(folder / filename)] = actual
            metadata = read(folder / "scores.json")
            if (
                metadata["source_tree_sha256"] != request["source_tree_sha256"]
                or metadata["final_manifest_hash"] != digest(manifest)
                or metadata["plan_sha256"] != plan["plan_sha256"]
            ):
                raise ValueError("Final source/suite/plan provenance differs")
            artifacts[name] = read_scored_items(
                folder, expected, expected_hash, done["commitment_sha256"]
            )
        for k, row in enumerate(rows):
            for d, observation in enumerate(row["continuation_replicates"]):
                if not observation["continuation_available"]:
                    for role in expected:
                        if not np.array_equal(
                            artifacts[f"m2_{k}_{d}"][role], artifacts[f"m1_{k}"][role]
                        ):
                            raise ValueError(
                                "Terminal continuation final outcomes do not preserve M2=M1"
                            )
        states.append(
            {
                "stream": task["stream"],
                "seed": task["seed"],
                "actions": decision["actions"],
                "artifacts": artifacts,
                "ids": {
                    role: [(ex.id, ex.prompt_hash) for ex in examples]
                    for role, examples in expected.items()
                },
                "terminal_draws": sum(
                    1 - obs["continuation_available"]
                    for row in rows
                    for obs in row["continuation_replicates"]
                ),
            }
        )
        hashes[str(output / "commitment.json")] = done["commitment_sha256"]
    destination = Path(request["output"]) / "evidence"
    destination.mkdir(parents=True, exist_ok=False)
    summarize(
        states,
        destination,
        {
            "source_tree_sha256": request["source_tree_sha256"],
            "plan_sha256": plan["plan_sha256"],
            "declaration": DECLARED,
            "input_hashes": hashes,
            "design_power": plan["power"],
            "automatic_expansion": False,
        },
    )
    write_json(
        destination / "completed.json",
        {
            "passed": True,
            "report_sha256": file_digest(destination / "report.json"),
            "automatic_expansion": False,
            "all_experiments_complete": True,
        },
    )
    print(
        json.dumps(
            {
                "evidence_package_complete": True,
                "parents": 32,
                "forecaster_fits": 0,
                "automatic_expansion": False,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "code-preflight", "run", "review"))
    parser.add_argument("--request", required=True)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    request = checked_request(args.request)
    request["request_sha256"] = file_digest(args.request)
    if args.mode == "run":
        if args.index not in range(32):
            parser.error("Parent index must be 0..31")
        run(request, args.index)
    else:
        {"prepare": prepare, "code-preflight": code_preflight, "review": review}[
            args.mode
        ](request)


if __name__ == "__main__":
    main()
