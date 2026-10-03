"""Scheduled CPU verification profile using canonical development references."""

import json
import os
from pathlib import Path
import socket
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from third_eye.data.splits import load_manifest
from third_eye.evaluation.benchmarks import make_verifier
from third_eye.io import write_json


def main():
    assert os.environ.get("SLURM_JOB_ID") and "login" not in socket.gethostname()
    workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    verifier = make_verifier()
    report = {
        "job_id": os.environ["SLURM_JOB_ID"],
        "host": socket.gethostname(),
        "cpus": workers,
        "cases": [],
        "status": "passed",
    }
    for task, role in [("code", "train"), ("math", "ood_dev")]:
        _, splits = load_manifest(f"data/processed/v1/{task}/selection.json")
        examples = splits[role]
        completions = [
            ex.answer if task == "code" else "$\\boxed{" + ex.answer + "}$"
            for ex in examples
        ]
        results = []
        for count in [1, workers]:
            os.environ["THIRD_EYE_VERIFIER_WORKERS"] = str(count)
            os.environ["THIRD_EYE_SYMBOLIC_WORKERS"] = str(count)
            started = time.perf_counter()
            verdicts = verifier.verify_many(examples, completions)
            elapsed = time.perf_counter() - started
            assert all(verdicts), (
                f"Canonical {task} reference failed with {count} workers"
            )
            results.append(verdicts)
            case = {
                "task": task,
                "examples": len(examples),
                "workers": count,
                "seconds": elapsed,
            }
            report["cases"].append(case)
            print(json.dumps(case), flush=True)
        assert results[0] == results[1]
    report["recommended_symbolic_workers"] = (
        workers if report["cases"][3]["seconds"] < report["cases"][2]["seconds"] else 1
    )
    write_json("runs/verification_profile.json", report)


if __name__ == "__main__":
    main()
