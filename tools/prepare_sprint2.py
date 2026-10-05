"""Prepare a core-only plan after reviewed Gate 1, without submitting training."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))


def main():
    from plan_study import build_plan
    from third_eye.io import file_digest, write_json
    from third_eye.provenance import source_inventory

    campaign = ROOT / "runs/campaign_a1_20261003"
    review = json.loads((campaign / "gate1/evidence_review.json").read_text())
    gate = json.loads((campaign / "gate1/gate1.json").read_text())
    if (
        not gate["passed"]
        or gate["states"] != 20
        or review["status"] != "evidence_review_passed"
    ):
        raise ValueError("Gate 1 has not passed evidence review")
    if review.get("gate1_sha256") != file_digest(campaign / "gate1/gate1.json"):
        raise ValueError("Reviewed Gate 1 report hash changed")
    if any(file_digest(p) != h for p, h in gate["evidence_hashes"].items()):
        raise ValueError("Gate 1 evidence changed")
    if (
        source_inventory(ROOT)["source_tree_sha256"]
        != "f06285d80e3a248a8141448d51fbac55a095016cfef6036351f18741984abcaa"
    ):
        raise ValueError("Frozen scientific source changed")
    stage = ROOT / "runs/sprint2_a1_20261004"
    plan_path = build_plan(
        [
            str(campaign / "configs" / f"{m}_1042.json")
            for m in ["qwen3_4b", "llama3_2_3b"]
        ],
        str(ROOT / "data/processed/a1"),
        stage / "plan",
        stage / "results",
        seeds=(42, 43, 44),
        label_trajectories=3,
    )
    plan = json.loads(plan_path.read_text())
    # Retain the emitted plan verbatim; CPU placement is an explicit execution plan.
    cpu_plan = json.loads(json.dumps(plan))
    for job in cpu_plan["stages"]["forecasters"]:
        command = job["commands"][0]
        command[command.index("--device") + 1] = "cpu"
    cpu_plan["resource_policy"] = {
        "cluster": "vista",
        "max_parallel_gpu_nodes": 20,
        "planning_target_gpu_hours": 1000,
        "ceiling_gpu_hours": 3000,
        "original_planner_a100_budget_is_historical": True,
        "correction_attempts": 16,
        "eight_attempt_protocol_adopted": False,
        "requires_gate1_review_sha256": file_digest(
            campaign / "gate1/evidence_review.json"
        ),
        "8b_and_transfer": "held until Gate 2",
        "submitted_training_jobs": [],
    }
    assert len(plan["stages"]["labels"]) == 36
    assert len(plan["stages"]["forecasters"]) == 18
    assert len(plan["stages"]["online_core"]) == 96
    write_json(stage / "execution_plan_cpu_forecasters.json", cpu_plan)
    write_json(
        stage / "preparation.json",
        {
            "status": "prepared_not_submitted",
            "gate1_passed_and_reviewed": True,
            "label_trajectories": 36,
            "maximum_label_states": 180,
            "cpu_forecaster_fits": 18,
            "core_policy_trajectories": 96,
            "frozen_a1_correction_attempts": 16,
            "eight_attempt_adoption": False,
            "execution_plan_sha256": file_digest(
                stage / "execution_plan_cpu_forecasters.json"
            ),
            "reason_training_not_submitted": "Review retry-budget diagnostics before committing the expensive dataset; no automatic protocol change.",
        },
    )
    print(json.dumps({"status": "prepared_not_submitted", "path": str(stage)}))


if __name__ == "__main__":
    main()
