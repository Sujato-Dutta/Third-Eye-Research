"""Budget-check and submit a study stage with bounded SLURM parallelism."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from third_eye.io import write_json
from third_eye.cluster import scheduler_job_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--cluster", choices=("dgx", "vista"), default="dgx")
    p.add_argument("--partition")
    p.add_argument("--account", help="Verified allocation/project charged by SLURM")
    p.add_argument("--max-parallel", type=int, default=1)
    p.add_argument("--gres", help="Allowed DGX GPU resource; unsupported on Vista")
    p.add_argument(
        "--estimated-task-hours",
        required=True,
        type=float,
        help="Estimate from measured pilot including generation/evaluation",
    )
    p.add_argument("--budget-hours", type=float, default=160)
    p.add_argument("--dependency", help="Successful predecessor job ID")
    p.add_argument(
        "--time-limit", help="SLURM wall limit based on measured task runtime"
    )
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
    if a.max_parallel < 1:
        p.error("Parallelism must be positive and fit the verified account limits")
    if a.cluster == "dgx" and not a.gres:
        p.error("DGX requires an allowed --gres resource")
    if a.cluster == "vista" and a.gres:
        p.error("Vista allocates whole nodes and does not support --gres")
    if a.cluster == "vista" and not a.account:
        p.error("Vista requires an explicitly verified --account allocation")
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
        hours = record.get("seconds", 0) / 3600
        spent += max(0.25, hours) if a.cluster == "vista" else hours
    reservations = 0.0
    for path in status.glob("submissions/*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("cluster", "dgx") != a.cluster:
            p.error("A study plan must belong to one cluster; create a migration plan")
        if record["stage"] == a.stage and set(record["indices"]) & set(indices):
            p.error("Selected tasks already have a submission reservation")
        completed = {
            i
            for i in record["indices"]
            if (status / record["stage"] / f"{i}.json").exists()
        }
        reservations += (len(record["indices"]) - len(completed)) * record[
            "estimated_task_hours"
        ]
    estimate = (
        max(0.25, a.estimated_task_hours)
        if a.cluster == "vista"
        else a.estimated_task_hours
    )
    if spent + reservations + len(indices) * estimate > a.budget_hours:
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
        f"--partition={a.partition or ('gh' if a.cluster == 'vista' else 'gpu_student')}",
        "--array=" + ",".join(map(str, indices)) + f"%{a.max_parallel}",
    ]
    environment = dict(os.environ)
    environment.update(
        THIRD_EYE_PLAN=str(Path(a.plan).resolve()), THIRD_EYE_STAGE=a.stage
    )
    if a.cluster == "dgx":
        argv.extend(
            (
                f"--gres={a.gres}",
                "--export=ALL,THIRD_EYE_PLAN="
                + environment["THIRD_EYE_PLAN"]
                + ",THIRD_EYE_STAGE="
                + a.stage,
            )
        )
    if a.account:
        argv.append("--account=" + a.account)
    if a.dependency:
        argv.append("--dependency=afterok:" + a.dependency)
    if a.time_limit:
        argv.append("--time=" + a.time_limit)
    argv.append(
        "experiments/jobs/vista_study.slurm"
        if a.cluster == "vista"
        else "experiments/jobs/study.slurm"
    )
    print(" ".join(argv))
    print(
        f"Spent {spent:.2f}h; queued reservations {reservations:.2f}h; proposed stage {len(indices) * estimate:.2f}h"
    )
    if a.submit:
        job = scheduler_job_id(
            subprocess.check_output(argv, text=True, env=environment)
        )
        write_json(
            status / "submissions" / f"{job}.json",
            {
                "job_id": job,
                "stage": a.stage,
                "indices": indices,
                "estimated_task_hours": estimate,
                "budget_hours": a.budget_hours,
                "cluster": a.cluster,
                "account": a.account,
                "max_parallel": a.max_parallel,
            },
        )
        print("Submitted job " + job)


if __name__ == "__main__":
    main()
