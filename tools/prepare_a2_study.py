"""Freeze an isolated A2 core plan, checking archived A1 evidence."""

from dataclasses import replace
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]


def main():
    from plan_study import build_plan
    from third_eye.config import Config
    from third_eye.io import file_digest, write_json
    from third_eye.provenance import source_inventory
    from third_eye.data.splits import load_manifest
    from third_eye.storage import check_storage

    if not os.environ.get("SLURM_JOB_ID") or "login" in os.uname().nodename:
        raise RuntimeError("Preparation requires a scheduled CPU compute allocation")
    archive = Path(os.environ["THIRD_EYE_A1_ROOT"])
    campaign = archive / "runs/campaign_a1_20261003"
    report_path = campaign / "gate1/gate1.json"
    report = json.loads(report_path.read_text())
    review = json.loads((campaign / "gate1/evidence_review.json").read_text())
    if (
        not report["passed"]
        or report["states"] != 20
        or review["status"] != "evidence_review_passed"
        or review["gate1_sha256"] != file_digest(report_path)
    ):
        raise RuntimeError("Reviewed A1 Gate 1 evidence is invalid")
    if any(file_digest(p) != h for p, h in report["evidence_hashes"].items()):
        raise RuntimeError("A1 evidence changed")
    if (
        source_inventory(archive)["source_tree_sha256"]
        != "f06285d80e3a248a8141448d51fbac55a095016cfef6036351f18741984abcaa"
    ):
        raise RuntimeError("Archived A1 scientific source changed")
    config_paths = []
    for name in ["qwen3_4b", "llama3_2_3b"]:
        old = Config.load(campaign / "configs" / f"{name}_1042.json")
        cfg = replace(
            old,
            protocol=replace(
                old.protocol,
                amendment="A2",
                correction_attempts=8,
                correction_seed_stride=16,
                deterministic_execution=True,
            ),
        )
        path = ROOT / "runs/a2/configs" / f"{name}.json"
        write_json(path, cfg.to_dict())
        config_paths.append(str(path))
    for task, count in [("math", 1024), ("code", 306)]:
        _, splits = load_manifest(ROOT / "data/processed/a1" / task / "selection.json")
        if len(splits["train"]) != count:
            raise RuntimeError("Frozen training manifest budget changed")
    stage = ROOT / "runs/a2"
    plan_path = build_plan(
        config_paths,
        str(ROOT / "data/processed/a1"),
        stage / "plan",
        stage / "study",
        seeds=(42, 43, 44),
        label_trajectories=3,
    )
    plan = json.loads(plan_path.read_text())
    gpu_gate = str(stage / "validation/gpu_review.json")
    for task in plan["stages"]["labels"]:
        task.setdefault("requires_gates", []).append(gpu_gate)
    for task in plan["stages"]["forecasters"]:
        cmd = task["commands"][0]
        cmd[cmd.index("--device") + 1] = "cpu"
        task.setdefault("requires_gates", []).append(
            str(stage / "study/phenomenon_a2/gate1.json")
        )
    plan.pop("compute_budget_a100_hours", None)
    plan.update(cluster="vista", storage_quota_gb=50)
    plan["resource_policy"] = {
        "max_parallel_gpu_nodes": 20,
        "planning_target_gpu_hours": 1000,
        "ceiling_gpu_hours": 3000,
        "prior_gpu_hours": 99.19666666666666,
        "amendment": "A2",
        "correction_attempts": 8,
        "correction_seed_stride": 16,
        "a1_gate1_sha256": file_digest(report_path),
        "amendment_sha256": file_digest(ROOT / "docs/protocol_amendment_a2.md"),
        "source_tree_sha256": source_inventory(ROOT)["source_tree_sha256"],
        "operational_files_sha256": {
            str(p.relative_to(ROOT)): file_digest(p)
            for p in sorted((ROOT / "tools").glob("*"))
            if p.is_file()
            and (
                p.name.startswith("a2_")
                or p.name
                in {"prepare_a2_study.py", "verify_a2_cuda.py", "review_a2_cuda.py"}
            )
        },
        "deadline_ist": "2026-10-10T23:59:59+05:30",
        "8b_and_transfer": "held until Gate 2 review",
    }
    assert len(plan["stages"]["labels"]) == 36
    assert len(plan["stages"]["forecasters"]) == 18
    assert len(plan["stages"]["online_core"]) == 96
    write_json(stage / "execution_plan.json", plan)
    write_json(
        stage / "validation/cpu.json",
        {
            "status": "passed",
            "passed": True,
            "job_id": os.environ["SLURM_JOB_ID"],
            "plan_sha256": file_digest(stage / "execution_plan.json"),
            "source_tree_sha256": source_inventory(ROOT)["source_tree_sha256"],
            "amendment_sha256": file_digest(ROOT / "docs/protocol_amendment_a2.md"),
            "a1_evidence_preserved": True,
            "storage": check_storage(ROOT, 50, headroom_gb=8),
        },
    )
    print(
        json.dumps(
            {"status": "prepared", "label_trajectories": 36, "correction_attempts": 8}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
