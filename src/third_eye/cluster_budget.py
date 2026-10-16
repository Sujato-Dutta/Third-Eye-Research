"""Conservative Slurm reservations; completed failures also consume the cap."""

from decimal import Decimal


RATES = {"gh": Decimal("1"), "gh-dev": Decimal("1"), "gg": Decimal("0.33")}


def reservation(partition, seconds, tasks=1, nodes=1):
    if partition not in RATES or any(
        type(x) is not int or x <= 0 for x in (seconds, tasks, nodes)
    ):
        raise ValueError("Known partition and positive integer resources required")
    return float(RATES[partition] * Decimal(max(900, seconds)) / 3600 * tasks * nodes)


def reserve(ledger, name, partition, seconds, tasks=1, nodes=1):
    """Call before sbatch; submission uncertainty retains the reservation."""
    if name in ledger["jobs"]:
        raise ValueError("Duplicate job reservation; reconcile before retrying")
    cost = reservation(partition, seconds, tasks, nodes)
    charged = sum(
        job.get("actual_sus", job["reserved_sus"]) for job in ledger["jobs"].values()
    )
    if charged + cost > ledger["hard_cap_sus"] + 1e-9:
        raise ValueError("Additional SU cap would be exceeded")
    ledger["jobs"][name] = dict(
        partition=partition,
        seconds=seconds,
        tasks=tasks,
        nodes=nodes,
        reserved_sus=cost,
        status="reserved",
    )
    return cost


def reconcile(ledger, name, task_rows):
    """Only complete accounting of every array element can release headroom.

    task_rows excludes batch/extern steps. Cancelled/pending arrays must have
    individually resolved elements; missing accounting retains the full budget.
    """
    job = ledger["jobs"][name]
    terminal = {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
        "REVOKED",
    }
    if len(task_rows) != job["tasks"] or len({r["id"] for r in task_rows}) != len(
        task_rows
    ):
        raise ValueError("Complete unique task accounting required")
    total = 0.0
    for row in task_rows:
        if row["state"].split()[0].rstrip("+") not in terminal:
            raise ValueError("Active or unknown jobs retain their reservation")
        if type(row["elapsed_seconds"]) is not int or row["elapsed_seconds"] < 0:
            raise ValueError("Invalid elapsed accounting")
        if type(row["nodes"]) is not int or not 0 <= row["nodes"] <= job["nodes"]:
            raise ValueError("Invalid allocated node accounting")
        if row["nodes"]:
            total += reservation(
                job["partition"], max(1, row["elapsed_seconds"]), nodes=row["nodes"]
            )
    if total > job["reserved_sus"] + 1e-6:
        raise ValueError("Accounting exceeded reservation; stop and audit")
    job.update(actual_sus=total, status="accounted", accounting=task_rows)
    return total
