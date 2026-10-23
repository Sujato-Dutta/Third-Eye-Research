"""Complete code timing and verify a runtime-only correction of the frozen cohort."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecasting_pilots import inspect_task
from third_eye.cluster import require_gpu_allocation
from third_eye.config import Config
from third_eye.evaluation.final import load_final_manifest
from third_eye.evaluation.runner import generate_greedy, verify_completions
from third_eye.evaluation.verifiers import VerifierRegistry
from third_eye.experiments.confirmation import checkpoints
from third_eye.experiments.confirmation_design import price
from third_eye.experiments.execution_cost import validate_limits, validate_observations
from third_eye.io import file_digest, write_json
from third_eye.provenance import source_inventory


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    require_gpu_allocation()
    request = read(args.request)
    current = source_inventory()
    validation = read(request["validation"])
    if (
        current["source_tree_sha256"] != request["execution_source_sha256"]
        or validation.get("passed") is not True
        or validation["source_tree_sha256"] != request["execution_source_sha256"]
    ):
        raise ValueError("Matching runtime-correction validation required")
    for path, sha in request["input_hashes"].items():
        if file_digest(path) != sha:
            raise ValueError("Bound original evidence changed")
    original_release = read(request["collection_release"])
    for path, sha in original_release["source_files"].items():
        if current["source_files"].get(path) != sha:
            raise ValueError(
                "Runtime correction must not alter the scientific implementation"
            )
    plan = read(request["cohort_plan"])
    plan_hash = file_digest(request["cohort_plan"])
    folder = Path(request["cohort_plan"]).parent
    if (
        plan["source_tree_sha256"] != original_release["source_tree_sha256"]
        or len(plan["tasks"]) != 32
    ):
        raise ValueError("Original frozen 32-parent cohort required")
    for value in plan["frozen_files"].values():
        if file_digest(value["path"]) != value["sha256"]:
            raise ValueError("Frozen scientific input changed")
    for task in plan["tasks"]:
        if Path(task["output"]).exists():
            raise ValueError(
                "Runtime correction requires no fresh parent to have started"
            )
    done = folder / "code_preflight/completed.json"
    if done.exists():
        raise FileExistsError("Code readiness already exists; no silent retry")
    calibration = read(folder / "code_calibration.json")
    if (
        file_digest(calibration["original_final_manifest"])
        != calibration["original_final_sha256"]
    ):
        raise ValueError("Code calibration source changed")
    code_sources = plan["code_preflight_sources"]
    _, suites = load_final_manifest(
        calibration["original_final_manifest"], code_sources[0]["manifest"]
    )
    examples = [
        next(
            ex
            for ex in suites["ood_test"]
            if ex.id == row["id"] and ex.prompt_hash == row["prompt_sha256"]
        )
        for row in calibration["items"]
    ]
    allowed = {
        (task["stream"], name) for task in code_sources for name in ("parent", "m2_0_0")
    }
    observations = list(read(request["failed_cost_report"])["observations"])
    measured = validate_observations(observations, allowed)
    original_request = read(request["cohort_request"])
    design = read(original_request["inputs"]["design_report"]["path"])
    pilot = read(design["inputs"]["pilot_plan"]["path"])
    verifier = VerifierRegistry.from_spec(
        "third_eye.evaluation.benchmarks:make_verifier"
    )
    import torch
    from third_eye.training.hf_backend import HFBackend

    for task in code_sources:
        missing = {
            name for stream, name in allowed - set(measured) if stream == task["stream"]
        }
        if not missing:
            continue
        if (
            file_digest(task["config"]) != task["config_sha256"]
            or file_digest(task["manifest"]) != task["manifest_sha256"]
        ):
            raise ValueError("Original code timing configuration changed")
        cfg = Config.load(task["config"])
        rows, _ = inspect_task(
            task, pilot["continuation_seed_offsets"], pilot["source_tree_sha256"]
        )
        backend = HFBackend(cfg)
        for name, checkpoint, expected in checkpoints(rows):
            if name not in missing:
                continue
            saved = backend.load_checkpoint(checkpoint)
            if saved["adapter_hash"] != expected:
                raise ValueError("Cost-calibration checkpoint differs")
            start = time.perf_counter()
            completions = generate_greedy(
                backend, examples, cfg.protocol.max_new_tokens, [cfg.protocol.seed] * 8
            )
            torch.cuda.synchronize()
            verdicts = verify_completions(verifier, examples, completions)
            seconds = time.perf_counter() - start
            if len(verdicts) != 8 or backend.state_hash() != expected:
                raise ValueError("Incomplete code verification or mutated checkpoint")
            observations.append(
                {
                    "stream": task["stream"],
                    "checkpoint": name,
                    "count": 8,
                    "seconds": seconds,
                    "seconds_per_item": seconds / 8,
                    "accuracy_recorded": False,
                }
            )
        del backend
        torch.cuda.empty_cache()
    measured = validate_observations(observations, allowed)
    if set(measured) != allowed:
        raise ValueError("All four direct timings required")
    rates = {}
    for task in code_sources:
        stream = task["stream"]
        actual = max(rate for (model, _), rate in measured.items() if model == stream)
        if actual > request["accepted_code_rate_bounds"][stream]:
            raise ValueError(
                "Direct timing exceeds corrected reservation; no fresh data"
            )
        rates[stream] = request["accepted_code_rate_bounds"][stream]
    benchmarks = [read(design["inputs"][f"benchmark_{i}"]["path"]) for i in range(4)]
    audit = read(design["inputs"]["audit_report"]["path"])
    counts = {
        task + "/" + role: size
        for task in ("math", "code")
        for role, size in (
            ("target_dev", 128),
            ("ood_dev", 128),
            ("retention_dev", 256),
        )
    }
    revised = price(
        32,
        counts,
        audit["cost"]["measured_max_with_25pct_margin_rounded_15min_seconds"],
        benchmarks,
        humaneval_rates=rates,
    )
    validate_limits(
        plan["cost"]["walltime_seconds_by_stream"],
        request["revised_limits"],
        revised["walltime_seconds_by_stream"],
        request["complete_stage_reservation_sus"],
    )
    report_path = folder / "code_preflight/reviewed_cost_report.json"
    write_json(
        report_path,
        {
            "observations": observations,
            "corrected_cost": revised,
            "scheduled_limits": request["revised_limits"],
            "scientific_plan_sha256": plan_hash,
            "scientific_source_tree_sha256": original_release["source_tree_sha256"],
            "cost_verifier_source_tree_sha256": current["source_tree_sha256"],
            "original_failure_preserved": True,
            "no_scientific_fields_changed": True,
            "forecaster_fits": 0,
        },
    )
    write_json(
        done,
        {
            "passed": True,
            "source_tree_sha256": original_release["source_tree_sha256"],
            "plan_sha256": plan_hash,
            "report": str(report_path),
            "report_sha256": file_digest(report_path),
            "cost_verifier_source_tree_sha256": current["source_tree_sha256"],
            "reviewed_runtime_only_correction": True,
        },
    )
    print(
        json.dumps(
            {
                "direct_code_cost_verified": True,
                "no_scientific_change": True,
                "fresh_parents": 32,
            }
        )
    )


if __name__ == "__main__":
    main()
