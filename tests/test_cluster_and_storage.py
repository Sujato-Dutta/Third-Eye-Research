import sys
import pytest
from third_eye.cluster import require_gpu_allocation
from third_eye.experiments.labeling import run_trajectory
from third_eye.io import write_json
from third_eye.storage import check_storage
from test_labeling import setup_labeler
from submit_stage import main as submit_main


def test_login_and_unscheduled_gpu_execution_rejected(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("socket.gethostname", lambda: "workstation")
    with pytest.raises(RuntimeError, match="SLURM"):
        require_gpu_allocation()
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr("socket.gethostname", lambda: "dgx-login01")
    with pytest.raises(RuntimeError, match="login"):
        require_gpu_allocation()
    monkeypatch.setattr("socket.gethostname", lambda: "dgxa")
    require_gpu_allocation()
    require_gpu_allocation("cpu")


def test_pruning_preserves_labels_and_latest_accepted_adapter(tmp_path, monkeypatch):
    labeler = setup_labeler(tmp_path, monkeypatch, depth=2)
    run_trajectory(labeler, prune=True, keep_accepted=1)
    assert len((tmp_path / "meta_labels.jsonl").read_text().splitlines()) == 6
    assert not (tmp_path / "accepted/generation_1").exists()
    assert (tmp_path / "accepted/generation_2/resume.json").exists()
    assert len(list(tmp_path.glob("states/*/labels.json"))) == 2
    assert not list(tmp_path.glob("states/*/k*/t1"))
    assert not list(tmp_path.glob("states/*/k*/t2"))


def test_budget_reservations_and_completed_status_are_accounted(
    tmp_path, monkeypatch, capsys
):
    root = tmp_path / "status"
    plan = tmp_path / "plan.json"
    write_json(
        plan, {"stages": {"forecasters": [{"commands": []}]}, "status_root": str(root)}
    )
    write_json(
        root / "submissions/100.json",
        {"stage": "labels", "indices": [0], "estimated_task_hours": 1.0},
    )
    argv = [
        "submit_stage.py",
        "--plan",
        str(plan),
        "--stage",
        "forecasters",
        "--gres",
        "gpu:a100:1",
        "--estimated-task-hours",
        "1",
        "--budget-hours",
        "1.5",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        submit_main()
    write_json(root / "labels/0.json", {"status": "complete", "seconds": 900})
    submit_main()
    assert "Spent 0.25h" in capsys.readouterr().out


def test_storage_headroom_rejects_overquota_project(tmp_path):
    (tmp_path / "large.bin").write_bytes(b"x" * 4096)
    with pytest.raises(RuntimeError, match="storage"):
        check_storage(tmp_path, quota_gb=0.000001, headroom_gb=0)
