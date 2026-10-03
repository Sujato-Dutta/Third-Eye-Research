"""Four-stream A1 calibration, bounded to four GPUs, then hold for Gate 1 review."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
from third_eye.config import Config  # noqa: E402
from third_eye.io import write_json, file_digest  # noqa: E402
from third_eye.provenance import source_inventory  # noqa: E402
from third_eye.cluster import scheduler_job_id  # noqa: E402
from third_eye.storage import check_storage  # noqa: E402
from continue_study import accounting, wait_success  # noqa: E402
from a1_inspection import inspect_pilots, review_allows_calibration  # noqa: E402

SEEDS = (1042, 2042, 3042, 4042, 5042)
STREAMS = (
    ("qwen3_4b", "math"),
    ("qwen3_4b", "code"),
    ("llama3_2_3b", "math"),
    ("llama3_2_3b", "code"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reviewed-pilots",
        help="SHA-256 of the inspected four-pilot report; requires --resume",
    )
    args = parser.parse_args()
    if args.reviewed_pilots and not args.resume:
        parser.error("Pilot review can only release a reviewed campaign resume")
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("A1 controller requires a compute allocation")
    gpu = json.loads((ROOT / "runs/a1_gpu_passed.json").read_text())
    cpu = json.loads((ROOT / "runs/a1_cpu_passed.json").read_text())
    inventory = source_inventory(ROOT)
    amendment = file_digest(ROOT / "docs/protocol_amendment_a1.md")
    if (
        cpu["status"] != "passed"
        or gpu["status"] != "passed"
        or cpu["source_tree_sha256"] != inventory["source_tree_sha256"]
        or gpu["source_tree_sha256"] != inventory["source_tree_sha256"]
        or cpu["amendment_sha256"] != amendment
    ):
        raise RuntimeError("A1 source or amendment changed after verification")
    root = ROOT / "runs/campaign_a1_20261003"
    if root.exists() and not args.resume:
        raise FileExistsError(
            "A1 campaign exists; use reviewed --resume to retain submitted IDs"
        )
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "campaign.json"
    plan_path = root / "plan.json"

    def event(status, **values):
        write_json(
            root / "status.json",
            {"status": status, "utc": datetime.now(timezone.utc).isoformat(), **values},
        )
        print(json.dumps({"status": status, **values}), flush=True)

    def backup():
        target = Path(os.environ["WORK"]) / "third_eye_results" / "campaign_a1_20261003"
        target.mkdir(parents=True, exist_ok=True)
        for name in (
            "src",
            "experiments",
            "cluster",
            "docs",
            "data/processed/a1",
            "runs",
        ):
            source = ROOT / name
            if source.exists():
                destination = target / name
                destination.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    ["rsync", "-a", str(source) + "/", str(destination) + "/"],
                    check=True,
                )

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["source_tree_sha256"] != inventory["source_tree_sha256"]
            or manifest["amendment_sha256"] != amendment
        ):
            raise RuntimeError("A1 resume source/amendment mismatch")
    else:
        prior = 0.0
        old_jobs = "1040277,1041762,1041764,1041766,1041768,1042902"
        for row in accounting(old_jobs):
            seconds = int(row[2])
            if seconds:
                prior += max(0.25, seconds / 3600)
        prior += max(0.25, sum(int(row[2]) for row in accounting(gpu["job_id"])) / 3600)
        tasks = []
        for model, task in STREAMS:
            template = Config.load(ROOT / f"experiments/configs/{model}_a1_pilot.json")
            for seed in SEEDS:
                from dataclasses import replace

                cfg = replace(template, protocol=replace(template.protocol, seed=seed))
                config = root / "configs" / f"{model}_{seed}.json"
                write_json(config, cfg.to_dict())
                output = root / "pilots" / f"{model}_{task}_{seed}"
                tasks.append(
                    {
                        "stream": f"{model}:{task}",
                        "seed": seed,
                        "output": str(output),
                        "commands": [
                            [
                                "experiments/run.py",
                                "--config",
                                str(config),
                                "--manifest",
                                str(ROOT / f"data/processed/a1/{task}/selection.json"),
                                "--output",
                                str(output),
                                "--pilot",
                                "--prune-branches",
                                "--keep-accepted",
                                "1",
                            ]
                        ],
                    }
                )
        write_json(
            plan_path,
            {
                "stages": {"a1_pilots": tasks},
                "status_root": str(root / "task_status"),
                "storage_quota_gb": 50,
            },
        )
        manifest = {
            "amendment": "A1",
            "amendment_sha256": amendment,
            **inventory,
            "seeds": SEEDS,
            "max_parallel": 4,
            "global_budget_h200_node_hours": 500,
            "prior_gpu_hours": prior,
            "submissions": [],
            "expansion_20_gpus": "held",
            "completion_deadline_utc": "2026-10-10T18:29:59+00:00",
        }
        write_json(manifest_path, manifest)
    plan = json.loads(plan_path.read_text())
    batches = [[0, 5, 10, 15]] + [[i, i + 5, i + 10, i + 15] for i in range(1, 5)]
    try:
        for phase, indices in enumerate(batches):
            known = next(
                (s for s in manifest["submissions"] if s["phase"] == phase), None
            )
            spent = manifest["prior_gpu_hours"] + sum(
                s.get("billed_hours", 0) for s in manifest["submissions"]
            )
            if known is None:
                check_storage(
                    Path(os.environ["THIRD_EYE_ORIGINAL_ROOT"]), 50, headroom_gb=4
                )
                if spent + len(indices) * 8 > 500:
                    raise RuntimeError(
                        "A1 pilot reservations exceed remaining global GPU budget"
                    )
                if (
                    datetime.now(timezone.utc).timestamp() + 8 * 3600
                    > datetime.fromisoformat(
                        manifest["completion_deadline_utc"]
                    ).timestamp()
                ):
                    raise RuntimeError("A1 pilot reservation exceeds declared deadline")
                environment = dict(
                    os.environ,
                    THIRD_EYE_PLAN=str(plan_path),
                    THIRD_EYE_STAGE="a1_pilots",
                )
                job = scheduler_job_id(
                    subprocess.check_output(
                        [
                            "sbatch",
                            "--parsable",
                            "--partition=gh",
                            "--account=IRI23021",
                            "--array=" + ",".join(map(str, indices)) + "%4",
                            "--time=08:00:00",
                            "experiments/jobs/vista_study.slurm",
                        ],
                        env=environment,
                        text=True,
                    )
                )
                known = {
                    "phase": phase,
                    "indices": indices,
                    "job_id": job,
                    "reserved_hours": len(indices) * 8,
                }
                manifest["submissions"].append(known)
                write_json(manifest_path, manifest)
            event(
                "waiting_for_a1_pilots",
                phase=phase,
                job_id=known["job_id"],
                max_parallel=4,
                expansion_20_gpus="held",
            )
            wait_success(known["job_id"], event)
            known["billed_hours"] = sum(
                max(0.25, int(row[2]) / 3600) for row in accounting(known["job_id"])
            )
            write_json(manifest_path, manifest)
            backup()
            if phase == 0:
                inspection = inspect_pilots(
                    [plan["stages"]["a1_pilots"][i] for i in indices]
                )
                inspection_path = root / "pilot_inspection.json"
                if inspection_path.exists():
                    if json.loads(inspection_path.read_text()) != inspection:
                        raise RuntimeError("Pilot evidence changed after inspection")
                else:
                    write_json(inspection_path, inspection)
                if not review_allows_calibration(
                    inspection, inspection_path, args.reviewed_pilots
                ):
                    event(
                        "pilot_review_required",
                        inspection_status=inspection["status"],
                        inspection_path=str(inspection_path),
                        inspection_sha256=file_digest(inspection_path),
                        terminal_fraction=inspection["terminal_fraction"],
                        all_terminal_streams=inspection["all_terminal_streams"],
                        expansion_20_gpus="held",
                        remaining_calibrations="held pending four-pilot inspection",
                    )
                    backup()
                    return
                manifest["pilot_review_sha256"] = args.reviewed_pilots
                write_json(manifest_path, manifest)
        labels = [
            Path(task["output"]) / "meta_labels.jsonl"
            for task in plan["stages"]["a1_pilots"]
        ]
        available = [str(p) for p in labels if p.exists()]
        missing = [str(p.parent) for p in labels if not p.exists()]
        report = root / "gate1"
        if available and not report.exists():
            subprocess.run(
                [
                    sys.executable,
                    "experiments/analyze.py",
                    "--labels",
                    *available,
                    "--output",
                    str(report),
                ],
                check=True,
            )
        evidence = (
            json.loads((report / "gate1.json").read_text())
            if (report / "gate1.json").exists()
            else {"status": "insufficient_evidence", "states": 0, "passed": False}
        )
        event(
            "gate1_review_required",
            gate1=evidence,
            correction_starved_runs=missing,
            expansion_20_gpus="held",
            note="No automatic forecaster, transfer or broad study submission",
        )
        backup()
    except Exception as error:
        event("stopped", reason=str(error), expansion_20_gpus="held")
        backup()
        raise


if __name__ == "__main__":
    main()
