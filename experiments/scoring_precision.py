"""Compute-only provenance audit and bounded inference timing, never confirmation."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from third_eye.cluster import require_compute_allocation, require_gpu_allocation
from third_eye.config import Config
from third_eye.data.benchmarks import canonical_prompt
from third_eye.data.splits import load_manifest
from third_eye.evaluation.final import load_final_manifest
from third_eye.experiments.scoring_precision import (
    DECLARATION,
    benchmark_checkpoints,
    identity_index,
    scan_exposure,
    timing_sample,
)
from third_eye.io import digest, file_digest, write_json
from third_eye.provenance import source_inventory


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked(path):
    require_compute_allocation()
    plan = read(path)
    if (
        plan["source_tree_sha256"] != source_inventory()["source_tree_sha256"]
        or plan["declaration"] != DECLARATION
    ):
        raise ValueError("Frozen source/declaration differs")
    validation = read(plan["validation"])
    if (
        validation.get("passed") is not True
        or validation["source_tree_sha256"] != plan["source_tree_sha256"]
    ):
        raise ValueError("Matching full CPU validation required")
    for key in ("pilot_plan", "pilot_report", "audit_report", "audit_completion"):
        if file_digest(plan[key]) != plan[key + "_sha256"]:
            raise ValueError("Bound development evidence changed: " + key)
    if read(plan["audit_completion"])["report_sha256"] != plan["audit_report_sha256"]:
        raise ValueError("Audit completion differs")
    return plan


def new_output(path):
    output = Path(path)
    output.mkdir(parents=True, exist_ok=False)
    return output


def exposure(plan):
    output = new_output(Path(plan["output"]) / "provenance")
    historical = read(plan["pilot_plan"])
    suites, hashes, metadata = {}, {}, {}
    for selection_path, audit in historical["manifest_audits"].items():
        final_path = audit["reserved_final_manifest"]
        if (
            file_digest(selection_path) != audit["sha256"]
            or file_digest(final_path) != audit["reserved_final_manifest_sha256"]
        ):
            raise ValueError("Frozen suite manifest changed")
        _, dev = load_manifest(selection_path)
        final_manifest, final = load_final_manifest(final_path, selection_path)
        suites[final_path] = final
        hashes[digest(final_manifest)] = {
            (final_path, role, ex.id)
            for role, examples in final.items()
            for ex in examples
        }
        metadata[final_path] = {
            "selection_manifest": selection_path,
            "selection_sha256": file_digest(selection_path),
            "final_sha256": file_digest(final_path),
            "roles": {
                role: {
                    "count": len(examples),
                    "benchmarks": dict(
                        Counter(
                            ex.metadata.get("benchmark", "unknown") for ex in examples
                        )
                    ),
                    "verifiers": dict(
                        Counter(ex.metadata.get("verifier", ex.task) for ex in examples)
                    ),
                    "prompt_characters_min_max": [
                        min(len(ex.prompt) for ex in examples),
                        max(len(ex.prompt) for ex in examples),
                    ],
                    "items": [
                        {
                            "id": ex.id,
                            "prompt_sha256": ex.prompt_hash,
                            "source_sha256": digest(canonical_prompt(ex)),
                        }
                        for ex in examples
                    ],
                }
                for role, examples in final.items()
            },
            "dev_benchmarks": {
                role: sorted(
                    {ex.metadata.get("benchmark", "unknown") for ex in examples}
                )
                for role, examples in dev.items()
            },
            "final_predictions_generated": False,
            "final_outcomes_extracted_or_used": False,
        }
    index = identity_index(suites)
    # Catch missing configs/checkpoints on CPUs before occupying a GPU.
    selected = [
        task
        for task in historical["tasks"]
        if task["seed"] == DECLARATION["benchmark_seed"]
    ]
    if len(selected) != 4 or len({task["stream"] for task in selected}) != 4:
        raise ValueError("Four fixed benchmark streams required")
    checkpoint_metadata = []
    for task in selected:
        for name in ("config", "manifest"):
            if file_digest(task[name]) != task[name + "_sha256"]:
                raise ValueError("Frozen benchmark input changed")
        cfg = Config.load(task["config"])
        rows = [
            json.loads(line)
            for line in (Path(task["output"]) / "meta_labels.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]
        for name, checkpoint, expected in benchmark_checkpoints(rows[0]):
            saved = read(checkpoint / "state.json")
            if (
                saved["adapter_hash"] != expected
                or saved["config_hash"] != cfg.fingerprint
            ):
                raise ValueError("Saved benchmark checkpoint identity differs")
            for filename, sha in saved["weights"].items():
                if file_digest(checkpoint / filename) != sha:
                    raise ValueError("Saved benchmark checkpoint weight changed")
            checkpoint_metadata.append(
                {
                    "stream": task["stream"],
                    "kind": name,
                    "checkpoint": str(checkpoint),
                    "state_sha256": file_digest(checkpoint / "state.json"),
                }
            )
    roots = [Path(path) for path in plan["history_roots"]]
    if any(not root.is_dir() for root in roots):
        raise ValueError("All declared artifact roots must be present")
    paths = {
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in {".json", ".jsonl"}
        and Path(plan["output"]) not in path.parents
    }
    hits, opaque, errors, inventory = scan_exposure(paths, index, hashes)
    for suite, value in metadata.items():
        for role, data in value["roles"].items():
            for row in data["items"]:
                row["recorded_exposure_files"] = sorted(
                    hits.get((suite, role, row["id"]), ())
                )
            data["recorded_exposed_count"] = sum(
                bool(row["recorded_exposure_files"]) for row in data["items"]
            )
    report = {
        "source_tree_sha256": plan["source_tree_sha256"],
        "declaration": DECLARATION,
        "history_roots": plan["history_roots"],
        "scanned_artifacts": len(inventory),
        "suites": metadata,
        "benchmark_checkpoint_metadata": checkpoint_metadata,
        "opaque_final_evaluations": opaque,
        "unreadable_artifacts": errors,
        "status": "scoped_recorded_exposure_audit_complete"
        if not errors and not opaque
        else "recorded_exposure_unresolved",
        "automatic_expansion": False,
        "limitation": "No absence-of-exposure guarantee: deleted, external, aggregate-only or unrecorded historical use and pretraining are not identified. Recorded hits disqualify untouched status; no hits only describe this search. No final predictions or accuracy outcomes were extracted.",
    }
    write_json(output / "report.json", report)
    write_json(output / "searched_artifacts.json", inventory)
    write_json(
        output / "completed.json",
        {
            "passed": True,
            "report_sha256": file_digest(output / "report.json"),
            "automatic_expansion": False,
        },
    )
    print(
        json.dumps(
            {
                "provenance_completed": True,
                "status": report["status"],
                "scanned_artifacts": len(inventory),
            }
        )
    )


def benchmark(plan, index):
    from forecasting_pilots import inspect_task
    from third_eye.evaluation.runner import generate_greedy, verify_completions
    from third_eye.evaluation.verifiers import VerifierRegistry
    from third_eye.training.hf_backend import HFBackend
    import torch

    require_gpu_allocation()
    receipt = read(Path(plan["output"]) / "provenance/completed.json")
    if not receipt["passed"] or receipt["report_sha256"] != file_digest(
        Path(plan["output"]) / "provenance/report.json"
    ):
        raise ValueError(
            "Completed provenance audit required; unresolved exposure does not authorize final scoring"
        )
    pilot = read(plan["pilot_plan"])
    selected = [
        task for task in pilot["tasks"] if task["seed"] == DECLARATION["benchmark_seed"]
    ]
    if len(selected) != 4 or len({task["stream"] for task in selected}) != 4:
        raise ValueError("Four predeclared benchmark parents required")
    task = sorted(selected, key=lambda value: value["stream"])[index]
    for name in ("config", "manifest"):
        if file_digest(task[name]) != task[name + "_sha256"]:
            raise ValueError("Frozen benchmark input changed")
    rows, _ = inspect_task(
        task, pilot["continuation_seed_offsets"], pilot["source_tree_sha256"]
    )
    # Existing row order and draw zero are frozen, never outcome-selected.
    row = rows[DECLARATION["candidate_row"]]
    checkpoint_specs = benchmark_checkpoints(row)
    cfg = Config.load(task["config"])
    _, splits = load_manifest(task["manifest"])
    samples = {
        role: timing_sample(splits[role])
        for role in ("target_dev", "ood_dev", "retention_dev")
    }
    output = new_output(Path(plan["output"]) / "benchmarks" / str(index))
    start = time.perf_counter()
    backend = HFBackend(cfg)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    verifier = VerifierRegistry.from_spec(
        "third_eye.evaluation.benchmarks:make_verifier"
    )
    observations = []
    for name, checkpoint, expected in checkpoint_specs:
        tick = time.perf_counter()
        saved = backend.load_checkpoint(checkpoint)
        if saved["adapter_hash"] != expected:
            raise ValueError("Saved adapter identity differs")
        torch.cuda.synchronize()
        restore_seconds = time.perf_counter() - tick
        before = backend.state_hash()
        # Fixed development-only warmup is separately timed, never hidden.
        tick = time.perf_counter()
        backend.generate(samples["target_dev"][0].prompt, 8, seed=cfg.protocol.seed)
        torch.cuda.synchronize()
        warmup_seconds = time.perf_counter() - tick
        for role, examples in samples.items():
            tick = time.perf_counter()
            completions = generate_greedy(
                backend,
                examples,
                cfg.protocol.max_new_tokens,
                [cfg.protocol.seed] * len(examples),
            )
            torch.cuda.synchronize()
            generation_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            verdicts = verify_completions(verifier, examples, completions)
            verification_seconds = time.perf_counter() - tick
            if len(verdicts) != len(examples):
                raise ValueError("Incomplete timed verification")
            # Outcomes/completions are deliberately neither stored nor summarized.
            observations.append(
                {
                    "checkpoint": name,
                    "adapter_hash": before,
                    "role": role,
                    "count": len(examples),
                    "items": [
                        {"id": ex.id, "prompt_sha256": ex.prompt_hash}
                        for ex in examples
                    ],
                    "generation_seconds": generation_seconds,
                    "verification_seconds": verification_seconds,
                    "generated_tokens": [
                        len(backend.tokenizer.encode(text, add_special_tokens=False))
                        for text in completions
                    ],
                    "restore_seconds": restore_seconds,
                    "warmup_seconds": warmup_seconds,
                }
            )
            write_json(output / "progress.json", {"observations": observations})
        if backend.state_hash() != before:
            raise RuntimeError("Inference benchmark changed the saved adapter")
    value = {
        "source_tree_sha256": plan["source_tree_sha256"],
        "plan_sha256": plan["plan_path_sha256"],
        "task": task,
        "config_sha256": task["config_sha256"],
        "resolved_revision": backend.resolved_revision,
        "model_source": backend.source_proof,
        "input_rendering": backend.input_rendering,
        "max_new_tokens": cfg.protocol.max_new_tokens,
        "generation_batch_size": cfg.protocol.generation_batch_size,
        "model_load_seconds": load_seconds,
        "total_seconds": time.perf_counter() - start,
        "observations": observations,
        "final_predictions_generated": False,
        "final_outcomes_extracted_or_used": False,
        "optimizer_steps": 0,
        "automatic_expansion": False,
        "limitation": "Length-stratified development timing only; unseen item lengths, completion lengths, verifier types, and hardware variation can change final inference cost.",
    }
    write_json(output / "report.json", value)
    write_json(
        output / "completed.json",
        {
            "passed": True,
            "report_sha256": file_digest(output / "report.json"),
            "automatic_expansion": False,
        },
    )
    print(
        json.dumps(
            {
                "benchmark_complete": True,
                "stream": task["stream"],
                "final_outcomes": False,
                "optimizer_steps": 0,
            }
        )
    )


def review(plan):
    output = new_output(Path(plan["output"]) / "review")
    provenance_path = Path(plan["output"]) / "provenance/report.json"
    provenance = read(provenance_path)
    records = []
    for index in range(4):
        path = Path(plan["output"]) / "benchmarks" / str(index)
        completed = read(path / "completed.json")
        if not completed["passed"] or completed["report_sha256"] != file_digest(
            path / "report.json"
        ):
            raise ValueError("Four complete immutable benchmark reports required")
        result = read(path / "report.json")
        if (
            result["source_tree_sha256"] != plan["source_tree_sha256"]
            or result["plan_sha256"] != plan["plan_path_sha256"]
        ):
            raise ValueError("Benchmark source/plan differs")
        task = result["task"]["stream"].split("/")[1]
        suite = next(
            v
            for v in provenance["suites"].values()
            if f"/{task}/" in v["selection_manifest"]
        )
        rates = {}
        for obs in result["observations"]:
            final_role = obs["role"].replace("_dev", "_test")
            rates[final_role] = max(
                rates.get(final_role, 0),
                (obs["generation_seconds"] + obs["verification_seconds"])
                / obs["count"],
            )
        records.append(
            {
                "stream": result["task"]["stream"],
                "development_seconds_per_item_max_over_checkpoints": rates,
                "conditional_single_checkpoint_final_suite_seconds_with_25pct_margin": 1.25
                * sum(
                    rates[role] * meta["count"] for role, meta in suite["roles"].items()
                ),
                "model_load_seconds": result["model_load_seconds"],
                "additional_limitations": "Per-checkpoint scaling only, not cost for all candidates/draws/parents. Code OOD is HumanEval versus MBPP development proxy; this extrapolation is not a validated price or final-suite precision.",
            }
        )
    value = {
        "source_tree_sha256": plan["source_tree_sha256"],
        "provenance_report_sha256": file_digest(provenance_path),
        "provenance_status": provenance["status"],
        "timing": records,
        "decision": "STOP_FOR_ITEM_PROVENANCE_POWER_AND_COST_REVIEW_NO_C_RELEASE",
        "automatic_expansion": False,
        "no_final_outcomes": True,
        "confirmation_not_authorized": True,
    }
    write_json(output / "report.json", value)
    write_json(
        output / "completed.json",
        {
            "passed": True,
            "report_sha256": file_digest(output / "report.json"),
            "automatic_expansion": False,
        },
    )
    print(json.dumps({"review_completed": True, "automatic_expansion": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("provenance", "benchmark", "review"))
    parser.add_argument("--plan", required=True)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    plan = checked(args.plan)
    plan["plan_path_sha256"] = file_digest(args.plan)
    if args.mode == "benchmark":
        if args.index not in range(4):
            parser.error("Benchmark index must be 0..3")
        benchmark(plan, args.index)
    elif args.mode == "provenance":
        exposure(plan)
    else:
        review(plan)


if __name__ == "__main__":
    main()
