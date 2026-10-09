# ruff: noqa: E402
"""Scheduled controller: validated twelve-state noise study, then review stop."""

import argparse
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools")]
from empirical_noise_v1 import BASE, frozen
from empirical_common_v1 import prepared
from empirical_relay_v1 import submit, atomic_json
from third_eye.cluster import scheduler_job_id
from third_eye.io import file_digest


def launch():
    scope = frozen()
    if datetime.now(timezone.utc) >= datetime.fromisoformat(scope["deadline_utc"]):
        raise RuntimeError("Empirical deadline passed")
    data = prepared()
    passed = json.loads((BASE / "gpu_verification/passed.json").read_text())
    if passed["noise_release_sha256"] != file_digest(BASE / "noise_release.json"):
        raise RuntimeError("GPU replay verification changed")
    cached = json.loads((BASE / "cached_analysis/completed.json").read_text())
    if cached["status"] != "cached_empirical_review_required":
        raise RuntimeError("Cached preparation/analysis must pass")
    target = BASE / "noise_submissions.json"
    if target.exists():
        raise RuntimeError(
            "Controller already attempted submissions: inspect receipts; never blindly resubmit"
        )
    record = dict(
        status="submitting",
        jobs=[],
        replication_released=False,
        noise_release_sha256=file_digest(BASE / "noise_release.json"),
    )
    atomic_json(target, record)
    os.environ["THIRD_EYE_EMPIRICAL_QUEUE"] = str(BASE / "submission_queue")
    # Submit early-generation balanced checks first; preserve the entire fixed sample.
    order = sorted(data["noise_tasks"], key=lambda t: (t["generation"], t["stream"]))
    for task in order:
        args = [
            "--parsable",
            "--account=IRI23021",
            "--nodes=1",
            "--ntasks=1",
            "--partition=gh",
            "--cpus-per-task=16",
            "--time=24:00:00",
            str(ROOT / "tools/empirical_noise_v1.slurm"),
            str(task["index"]),
        ]
        job = scheduler_job_id(submit(args))
        record["jobs"].append(
            dict(
                index=task["index"],
                job_id=job,
                stream=task["stream"],
                generation=task["generation"],
            )
        )
        atomic_json(target, record)
    dependencies = ":".join(t["job_id"] for t in record["jobs"])
    review = scheduler_job_id(
        submit(
            [
                "--parsable",
                "--account=IRI23021",
                "--nodes=1",
                "--ntasks=1",
                "--partition=gg",
                "--cpus-per-task=72",
                "--time=1:00:00",
                "--dependency=afterany:" + dependencies,
                str(ROOT / "tools/empirical_noise_control_v1.slurm"),
                "review",
            ]
        )
    )
    record.update(status="queued_review_stop", review_job=review)
    atomic_json(target, record)
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["validate", "launch", "review"])
    args = parser.parse_args()
    frozen()
    if args.mode == "validate":
        prepared()
        atomic_json(
            BASE / "noise_cpu_passed.json",
            dict(
                status="passed",
                noise_release_sha256=file_digest(BASE / "noise_release.json"),
            ),
        )
    elif args.mode == "launch":
        receipt = json.loads((BASE / "noise_cpu_passed.json").read_text())
        if receipt["noise_release_sha256"] != file_digest(BASE / "noise_release.json"):
            raise RuntimeError("CPU validation changed")
        launch()
    else:
        from empirical_noise_review_v1 import main as review

        review()


if __name__ == "__main__":
    main()
