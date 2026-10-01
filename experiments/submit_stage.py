"""Budget-check and submit one study stage through SLURM, serially."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from third_eye.io import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--partition", default="gpu_student")
    p.add_argument("--gres", required=True, help="Allowed GPU resource from sinfo")
    p.add_argument(
        "--estimated-task-hours",
        required=True,
        type=float,
        help="Estimate from measured pilot including generation/evaluation",
    )
    p.add_argument("--budget-hours", type=float, default=160)
    p.add_argument("--dependency", help="Successful predecessor job ID")
    p.add_argument(
        "--indices",
        nargs="*",
        type=int,
        help="Optional reviewed subset for staged pilots/model rotation",
    )
    p.add_argument(
        "--submit",
        action="store_true",
        help="Without this flag, print the concrete sbatch command",
    )
    a = p.parse_args()
    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    tasks = plan["stages"].get(a.stage)
    if not tasks:
        p.error("Unknown/empty study stage")
    indices = list(range(len(tasks))) if a.indices is None else sorted(set(a.indices))
    if not indices or any(i < 0 or i >= len(tasks) for i in indices):
        p.error("Task indices are invalid")
    if a.estimated_task_hours <= 0 or a.budget_hours <= 0:
        p.error("Runtime and budget must be positive")
    status = Path(plan["status_root"])
    spent = 0.0
    for path in status.glob("*/*.json"):
        if path.parent.name == "submissions":
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["status"] == "running":
            p.error(
                "An existing task is running; wait or account for the allocation before adding a stage"
            )
        spent += record.get("seconds", 0) / 3600
    reservations = 0.0
    for path in status.glob("submissions/*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        completed = {
            i
            for i in record["indices"]
            if (status / record["stage"] / f"{i}.json").exists()
        }
        reservations += (len(record["indices"]) - len(completed)) * record[
            "estimated_task_hours"
        ]
    if spent + reservations + len(indices) * a.estimated_task_hours > a.budget_hours:
        p.error(
            "Measured spend plus queued reservations and this stage exceed the GPU-hour ceiling"
        )
    for i in indices:
        if (status / a.stage / f"{i}.json").exists():
            p.error(
                "A selected task already has a status artifact; review it before creating a rerun plan"
            )
        for gate in tasks[i].get("requires_gates", []):
            if (
                json.loads(Path(gate).read_text(encoding="utf-8")).get("passed")
                is not True
            ):
                p.error("Required gate has not passed: " + gate)
    argv = [
        "sbatch",
        "--parsable",
        f"--partition={a.partition}",
        f"--gres={a.gres}",
        "--array=" + ",".join(map(str, indices)) + "%1",
        "--export=ALL,THIRD_EYE_PLAN="
        + str(Path(a.plan).resolve())
        + ",THIRD_EYE_STAGE="
        + a.stage,
    ]
    if a.dependency:
        argv.append("--dependency=afterok:" + a.dependency)
    argv.append("experiments/jobs/study.slurm")
    print(" ".join(argv))
    print(
        f"Spent {spent:.2f}h; queued reservations {reservations:.2f}h; proposed stage {len(indices) * a.estimated_task_hours:.2f}h"
    )
    if a.submit:
        job = subprocess.check_output(argv, text=True).strip().split(";")[0]
        write_json(
            status / "submissions" / f"{job}.json",
            {
                "job_id": job,
                "stage": a.stage,
                "indices": indices,
                "estimated_task_hours": a.estimated_task_hours,
                "budget_hours": a.budget_hours,
            },
        )
        print("Submitted job " + job)


if __name__ == "__main__":
    main()
