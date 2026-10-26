"""Recheck measured cost against increased Slurm reservations, without retraining.

An operational supplement to the immutable, already validated SmolLM release.
No candidate, training, scoring, seed, or inference setting is changed here.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path


def validate_limits(authorization, costs):
    bounds = authorization["seconds_by_task"]
    if (
        set(bounds) != {"code", "math"}
        or set(costs) != set(bounds)
        or authorization["parents_per_task"] != 8
        or authorization["additional_su_cap"] != 500
        or authorization["scientific_settings_changed"] is not False
    ):
        raise ValueError("Fixed sixteen-run scope and 500-SU cap required")
    revised = deepcopy(costs)
    for task, seconds in bounds.items():
        if type(seconds) is not int or not 1 <= seconds <= 172800:
            raise ValueError("Reservation must fit the verified 48-hour queue limit")
        if not costs[task]["includes_three_draws"]:
            raise ValueError("Complete three-draw cost required")
        if not 0 < costs[task]["required_seconds"] <= seconds:
            raise ValueError("Measured full runtime exceeds increased reservation")
        revised[task]["original_reserved_seconds"] = costs[task]["reserved_seconds"]
        revised[task]["reserved_seconds"] = seconds
    if 8 * sum(bounds.values()) / 3600 > 500:
        raise ValueError("Parent reservations exceed additional allowance")
    return revised


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args()

    # Imports resolve exclusively against the original source via Slurm PYTHONPATH.
    from smollm_confirmation import checked, plan_for
    from third_eye.cluster import require_compute_allocation
    from third_eye.cluster_budget import reservation
    from third_eye.io import file_digest, write_json

    require_compute_allocation()
    path = Path(args.authorization)
    auth = json.loads(path.read_text())
    if file_digest(__file__) != auth["operational_script_sha256"]:
        raise ValueError("Operational supplement changed after submission")
    for item in auth["frozen_inputs"].values():
        if file_digest(item["path"]) != item["sha256"]:
            raise ValueError("Previously validated timing or plan changed")
    request = checked(auth["request"])
    plan = plan_for(request)
    if (
        plan["plan_sha256"] != auth["plan_sha256"]
        or request["source_tree_sha256"] != auth["source_tree_sha256"]
        or len(plan["tasks"]) != 16
    ):
        raise ValueError("Original complete source-bound plan required")
    for task in ("code", "math"):
        item = auth["frozen_inputs"][task + "_preflight"]
        preflight = json.loads(Path(item["path"]).read_text())
        if (
            preflight["passed"] is not True
            or preflight["discarded_fixture"] is not True
            or preflight["source_tree_sha256"] != auth["source_tree_sha256"]
            or preflight["plan_sha256"] != auth["plan_sha256"]
        ):
            raise ValueError("Existing matching GPU checks must have passed")

    ledger = json.loads(Path(auth["ledger"]).read_text())
    used = sum(j.get("actual_sus", j["reserved_sus"]) for j in ledger["jobs"].values())
    if used - auth["revision_baseline_sus"] > auth["additional_su_cap"] + 1e-8:
        raise ValueError("Complete charged-plus-reserved revision exceeds cap")
    for task, seconds in auth["seconds_by_task"].items():
        entry = ledger["jobs"][auth["reservation_names"][task]]
        if (
            entry["partition"] != "gh"
            or entry["tasks"] != 8
            or entry["nodes"] != 1
            or entry["seconds"] != seconds
            or entry["reserved_sus"] != reservation("gh", seconds, tasks=8)
            or entry["job_id"] != auth["collection_jobs"][task]
        ):
            raise ValueError("Submitted parent reservation differs from measured plan")

    costs = json.loads(Path(auth["frozen_inputs"]["original_cost"]["path"]).read_text())
    revised = validate_limits(auth, costs)
    output = Path(request["output"]) / "preflight"
    if (output / "completed.json").exists():
        raise FileExistsError("Preserve existing readiness; no silent overwrite")
    report = output / "runtime_cost_report_20261026.json"
    if report.exists():
        raise FileExistsError("Operational runtime review is immutable")
    write_json(report, {"costs": revised, "authorization": auth})
    write_json(
        output / "completed.json",
        {
            "passed": True,
            "source_tree_sha256": auth["source_tree_sha256"],
            "plan_sha256": auth["plan_sha256"],
            "cost_report_path": str(report),
            "cost_report_sha256": file_digest(report),
            "operational_runtime_authorization_sha256": file_digest(path),
            "original_runtime_check_preserved": True,
            "scientific_settings_changed": False,
            "charged_plus_reserved_revision_sus": used - auth["revision_baseline_sus"],
        },
    )
    print(
        json.dumps(
            {"increased_reservations_cover_measured_cost": True, "costs": revised}
        )
    )


if __name__ == "__main__":
    main()
