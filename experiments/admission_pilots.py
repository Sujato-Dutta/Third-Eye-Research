"""Prepare, submit and inspect the bounded fresh admission development pilot."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecasting_pilots import checked_plan, inspect_task, item_artifact
from review_admission import load_factors
from third_eye.cluster import require_compute_allocation
from third_eye.cluster_budget import reserve
from third_eye.config import Config
from third_eye.data.splits import load_manifest
from third_eye.evaluation.final import load_final_manifest
from third_eye.experiments.admission import effective_delta_norm
from third_eye.experiments.admission_pilot import (
    BUDGET_SU,
    CONTEXT,
    OFFSETS,
    SEEDS,
    native_arrays,
    review,
    runtime_limits,
)
from third_eye.io import file_digest, write_json
from third_eye.provenance import source_inventory


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked_request(path):
    request = read(path)
    if request["source_tree_sha256"] != source_inventory()["source_tree_sha256"]:
        raise ValueError("Admission request/source differs")
    for key in ("stage_a_report", "historical_plan"):
        if file_digest(request[key]) != request[key + "_sha256"]:
            raise ValueError("Bound development evidence changed")
    if request["stage_b_su_cap"] != BUDGET_SU:
        raise ValueError("Stage B ceiling changed")
    return request


def prepare(request):
    receipt = read(Path(request["output"]) / "validation/completed.json")
    if (
        receipt.get("passed") is not True
        or receipt["source_tree_sha256"] != request["source_tree_sha256"]
    ):
        raise ValueError("Matching full CPU validation required")
    previous = read(request["stage_a_report"])
    if (
        previous["decision"] != "REVIEW_B_development_signal_only_not_confirmation"
        or previous["pooling_sensitivity_eligible_development_only"]
    ):
        raise ValueError(
            "Stage A must warrant native development without a pooling arm"
        )
    historical = read(request["historical_plan"])
    if historical["continuation_seed_offsets"] != [*OFFSETS, 150_000_000, 200_000_000]:
        raise ValueError("Measured baseline must contain the declared five draws")
    # Scan run metadata on allocated CPUs; filenames/metrics never enter models.
    collisions, audited = [], []
    search_roots = [
        Path(request["root"]) / "runs",
        Path(request["root"]) / "amendments/a2/runs",
    ]
    paths = sorted({p for root in search_roots for p in root.rglob("run.json")})
    for path in paths:
        if Path(request["output"]) in path.parents:
            continue
        run = read(path)
        seed = run.get("config", {}).get("protocol", {}).get("seed")
        audited.append({"path": str(path), "sha256": file_digest(path), "seed": seed})
        if seed in SEEDS:
            collisions.append(str(path))
    if collisions:
        raise ValueError(f"Fresh development seed collision: {collisions}")
    measured, templates, manifest_audits = defaultdict(list), {}, {}
    for task in historical["tasks"]:
        if (
            file_digest(task["config"]) != task["config_sha256"]
            or file_digest(task["manifest"]) != task["manifest_sha256"]
        ):
            raise ValueError("Historical template/manifest changed")
        output = Path(task["output"])
        if read(output / "completed.json")["status"] != "complete":
            raise ValueError("Only complete runtime benchmarks may be used")
        duration = float(read(next(output.glob("states/*/summary.json")))["seconds"])
        rows = [
            json.loads(s)
            for s in (output / "meta_labels.jsonl").read_text().splitlines()
            if s.strip()
        ]
        if len(rows) != 3:
            raise ValueError("Measured pilot must have complete K=3 labels")
        for row in rows:
            draws = row["continuation_replicates"]
            if len(draws) != 5:
                raise ValueError("Measured pilot continuation count differs")
            for draw in draws[3:]:
                duration -= float(draw["pool_statistics"]["total_seconds"]) + float(
                    draw["training"]["seconds"]
                )
        measured[task["stream"]].append(duration)
        templates[task["stream"]] = task
        if task["manifest"] not in manifest_audits:
            _, splits = load_manifest(task["manifest"])
            expected_train = 306 if task["stream"].endswith("/code") else 1024
            if {r: len(v) for r, v in splits.items()} != {
                "train": expected_train,
                "target_dev": 64,
                "ood_dev": 64,
                "retention_dev": 256,
            }:
                raise ValueError("Frozen development harvest/evaluation sizes differ")
            final_path = Path(task["manifest"]).with_name("final.json")
            _, sealed = load_final_manifest(final_path, task["manifest"])
            manifest_audits[task["manifest"]] = {
                "sha256": file_digest(task["manifest"]),
                "sizes": {r: len(v) for r, v in splits.items()},
                "reserved_final_manifest": str(final_path),
                "reserved_final_manifest_sha256": file_digest(final_path),
                "reserved_final_sizes": {r: len(v) for r, v in sealed.items()},
                "final_predictions_read_or_generated": False,
                "historical_final_exposure_not_established_by_this_audit": True,
            }
    limits, maximum = runtime_limits(measured)
    output = Path(request["output"])
    tasks = []
    for stream in sorted(templates):
        template = templates[stream]
        for seed in SEEDS:
            cfg = read(template["config"])
            cfg["model"]["chat_kwargs"] = {
                **cfg["model"].get("chat_kwargs", {}),
                **CONTEXT,
            }
            cfg["protocol"]["seed"] = seed
            path = output / "configs" / f"{stream.replace('/', '_')}_{seed}.json"
            write_json(path, cfg)
            checked = Config.load(path)
            if (
                checked.protocol.correction_attempts != 8
                or checked.training.max_steps != 50
                or checked.protocol.candidates != 3
                or checked.protocol.candidate_size != 2
            ):
                raise ValueError("Frozen A2 training/harvesting budget differs")
            tasks.append(
                {
                    "index": len(tasks),
                    "stream": stream,
                    "seed": seed,
                    "config": str(path),
                    "config_sha256": file_digest(path),
                    "manifest": template["manifest"],
                    "manifest_sha256": template["manifest_sha256"],
                    "output": str(output / "pilots" / stream / f"seed{seed}"),
                    "generations": 1,
                }
            )
    plan = {
        "schema_version": 1,
        "purpose": "admission_development_pilots",
        "source_tree_sha256": request["source_tree_sha256"],
        "tasks": tasks,
        "continuation_seed_offsets": list(OFFSETS),
        "cpu_validation": str(output / "validation/completed.json"),
        "cuda_validation": str(output / "cuda_validation/completed.json"),
        "runtime_limits_by_stream": limits,
        "measured_three_draw_seconds_conservative": measured,
        "max_stage_reservation_sus": maximum,
        "stage_b_su_cap": BUDGET_SU,
        "manifest_audits": manifest_audits,
        "prior_seed_metadata_audit": audited,
        "features": [
            "h1_target",
            "h1_ood",
            "h1_retention",
            "log1p_effective_update_norm",
        ],
        "probe_exclusion": "Existing probes use the complete retention anchor, including scoring items; they are archived but excluded from gates.",
        "fixed_context": CONTEXT,
        "pooling_arm": False,
        "maximum_parallel_gpus": 16,
        "automatic_expansion": False,
        "confirmation_requires_new_review": True,
    }
    write_json(output / "plan.json", plan)
    print(
        json.dumps(
            {
                "prepared": True,
                "parents": 16,
                "max_reservation_sus": maximum,
                "limits": limits,
            }
        )
    )


def submit(request):
    """Allocated metadata controller: no retries or scientific stage expansion."""
    output = Path(request["output"])
    plan = checked_plan(output / "plan.json")
    if read(plan["cuda_validation"]).get("passed") is not True:
        raise ValueError("CUDA preflight must pass before pilot submission")
    ledger_path = Path(request["ledger"])
    ledger = read(ledger_path)
    receipt = read(output / "submissions.json")
    if "pilot_arrays" in receipt:
        raise ValueError(
            "Submission already attempted; preserve ambiguity and do not retry"
        )
    prefix = request["ledger_prefix"]
    for stream, seconds in plan["runtime_limits_by_stream"].items():
        reserve(ledger, prefix + stream.replace("/", "_"), "gh", seconds, tasks=4)
    reserve(ledger, prefix + "review", "gg", 3600)
    stage = sum(
        j.get("actual_sus", j["reserved_sus"])
        for k, j in ledger["jobs"].items()
        if k.startswith(prefix)
    )
    if stage > BUDGET_SU + 1e-9:
        raise ValueError("Total Stage B reservation exceeds declared ceiling")
    write_json(ledger_path, ledger)
    receipt["pilot_arrays"] = {}
    write_json(output / "submissions.json", receipt)

    def sbatch(name, script, partition, seconds, extra, env):
        command = [
            "sbatch",
            "--parsable",
            "--chdir",
            request["source"],
            "--partition",
            partition,
            "--time",
            str(seconds // 60),
            "--output",
            str(output / (name + "_%A_%a.out")),
            *extra,
            str(Path(request["source"]) / script),
        ]
        write_json(output / (name + "_request.json"), {"command": command})
        result = subprocess.run(command, env=env, text=True, capture_output=True)
        write_json(
            output / (name + "_response.json"),
            {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )
        ids = re.findall(r"^\s*(\d+)(?:;[A-Za-z0-9_.-]+)?\s*$", result.stdout, re.M)
        if result.returncode or len(ids) != 1:
            raise RuntimeError(
                "Submission failed/ambiguous; preserve reservation, no retry"
            )
        return ids[0]

    dependencies = []
    for offset, (stream, seconds) in enumerate(
        sorted(plan["runtime_limits_by_stream"].items())
    ):
        name = stream.replace("/", "_")
        env = dict(os.environ, THIRD_EYE_INDEX_BASE=str(offset * 4))
        job = sbatch(
            name,
            "experiments/jobs/forecast_pilots.slurm",
            "gh",
            seconds,
            ["--array=0-3%4"],
            env,
        )
        ledger["jobs"][prefix + name].update(job_id=job, status="submitted")
        write_json(ledger_path, ledger)
        receipt["pilot_arrays"][stream] = {
            "job": job,
            "index_base": offset * 4,
            "seconds": seconds,
        }
        write_json(output / "submissions.json", receipt)
        dependencies.append(job)
    job = sbatch(
        "review",
        "experiments/jobs/admission_pilot_review.slurm",
        "gg",
        3600,
        ["--dependency=afterany:" + ":".join(dependencies)],
        dict(os.environ),
    )
    ledger["jobs"][prefix + "review"].update(job_id=job, status="submitted")
    write_json(ledger_path, ledger)
    receipt["review"] = job
    write_json(output / "submissions.json", receipt)
    print(json.dumps(receipt))


def inspect(request):
    output = Path(request["output"])
    plan = checked_plan(output / "plan.json")
    states, coverage, hashes = [], [], {}
    for task in plan["tasks"]:
        try:
            rows, report = inspect_task(task, list(OFFSETS), plan["source_tree_sha256"])
            cfg = Config.load(task["config"])
            _, splits = load_manifest(task["manifest"])
            labels = Path(task["output"]) / "meta_labels.jsonl"
            hashes[str(labels)] = file_digest(labels)
            parent, immediate, future = None, [], []
            for row in rows:
                baseline = item_artifact(
                    row["evaluation_artifacts"]["parent"],
                    row["parent_adapter_hash"],
                    splits,
                )
                if parent is not None and parent != baseline:
                    raise ValueError("Repeated parent evaluation differs")
                parent = baseline
                rendering = baseline["input_rendering"]
                if (
                    rendering["chat_kwargs"] != cfg.model.chat_kwargs
                    or row["precommit"].get("input_rendering") != rendering
                ):
                    raise ValueError(
                        "Harvest/probe/evaluation rendering provenance differs"
                    )
                im = item_artifact(
                    row["evaluation_artifacts"]["t1"],
                    row["candidate_adapter_hash"],
                    splits,
                )
                if im["input_rendering"] != rendering:
                    raise ValueError("Immediate rendering differs")
                immediate.append(im)
                draws = []
                for draw in row["continuation_replicates"]:
                    item = item_artifact(
                        draw["items_path"], draw["continuation_adapter_hash"], splits
                    )
                    if item["input_rendering"] != rendering:
                        raise ValueError("Continuation rendering differs")
                    if draw["continuation_available"] == 0:
                        if (
                            draw["optimizer_steps"] != 0
                            or draw["continuation_adapter_hash"]
                            != row["candidate_adapter_hash"]
                            or draw["h2"] != row["labels"]["h1"]
                        ):
                            raise ValueError(
                                "Terminal continuation must preserve M2=M1"
                            )
                    elif draw["optimizer_steps"] != cfg.training.max_steps:
                        raise ValueError("Continuation optimizer budget differs")
                    draws.append(item)
                future.append(draws)
            parent_path = Path(rows[0]["checkpoints"]["t1"]).parent.parent / "parent"
            before = load_factors(
                parent_path, rows[0]["parent_adapter_hash"], cfg, hashes
            )
            norms = []
            for row in rows:
                after = load_factors(
                    row["checkpoints"]["t1"], row["candidate_adapter_hash"], cfg, hashes
                )
                norms.append(
                    effective_delta_norm(
                        before, after, cfg.training.alpha / cfg.training.rank
                    )
                )
                del after
            del before
            states.append(
                {
                    "trajectory_id": rows[0]["trajectory_id"],
                    "stream": task["stream"],
                    "parent": parent,
                    "panel": native_arrays(parent, immediate, future),
                    "effective_update_norms": norms,
                }
            )
            coverage.append(report)
        except (OSError, ValueError, KeyError, StopIteration) as exc:
            coverage.append(
                {
                    "stream": task["stream"],
                    "seed": task["seed"],
                    "complete": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )
    write_json(
        output / "pilot_inspection.json",
        {"tasks": coverage, "passed": len(states) == 16, "automatic_expansion": False},
    )
    if len(states) != 16:
        raise ValueError("Incomplete pilot cohort; preserve evidence and stop before C")
    report, fits = review(states)
    write_json(
        output / "evidence/report.json",
        {
            **report,
            "source_tree_sha256": plan["source_tree_sha256"],
            "plan_sha256": file_digest(output / "plan.json"),
            "input_hashes": hashes,
        },
    )
    write_json(output / "evidence/gate_fits.json", fits)
    write_json(
        output / "evidence/completed.json",
        {
            "passed": True,
            "automatic_expansion": False,
            "report_sha256": file_digest(output / "evidence/report.json"),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "submit", "inspect"))
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    require_compute_allocation()
    request = checked_request(args.request)
    {"prepare": prepare, "submit": submit, "inspect": inspect}[args.mode](request)


if __name__ == "__main__":
    main()
