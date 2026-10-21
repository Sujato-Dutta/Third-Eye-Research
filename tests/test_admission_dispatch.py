"""Scheduler dispatch uses login resources and stops on ambiguous submissions."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location(
    "admission_dispatch",
    Path(__file__).resolve().parents[1] / "cluster/admission_dispatch.py",
)
dispatch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dispatch)


def test_scheduler_dispatch_rejects_compute_allocations():
    with (
        patch.object(dispatch.socket, "gethostname", return_value="c642-071"),
        patch.dict(dispatch.os.environ, {}, clear=True),
    ):
        with pytest.raises(RuntimeError, match="login resource"):
            dispatch.login_only()
    with (
        patch.object(dispatch.socket, "gethostname", return_value="login1"),
        patch.dict(dispatch.os.environ, {"SLURM_JOB_ID": "123"}, clear=True),
    ):
        with pytest.raises(RuntimeError, match="login resource"):
            dispatch.login_only()
    with (
        patch.object(dispatch.socket, "gethostname", return_value="login1"),
        patch.dict(dispatch.os.environ, {}, clear=True),
    ):
        dispatch.login_only()


def test_scheduler_banner_is_not_a_job_id_and_attempts_are_not_repeated(tmp_path):
    path = tmp_path / "attempt.json"
    accepted = SimpleNamespace(
        returncode=0, stdout="[TACC] checks passed\n12345;vista\n", stderr=""
    )
    with patch.object(dispatch.subprocess, "run", return_value=accepted) as call:
        assert (
            dispatch.submit_job(["sbatch", "--parsable", "fixture.slurm"], {}, path)
            == "12345"
        )
        with pytest.raises(RuntimeError, match="already exists"):
            dispatch.submit_job(["sbatch"], {}, path)
        assert call.call_count == 1


def test_nonzero_or_ambiguous_scheduler_result_preserves_attempt(tmp_path):
    for index, result in enumerate(
        (
            SimpleNamespace(returncode=1, stdout="host rejected", stderr=""),
            SimpleNamespace(returncode=0, stdout="12345\n67890\n", stderr=""),
        )
    ):
        path = tmp_path / f"attempt{index}.json"
        with patch.object(dispatch.subprocess, "run", return_value=result):
            with pytest.raises(RuntimeError, match="failed/ambiguous"):
                dispatch.submit_job(["sbatch"], {}, path)
        assert path.exists()


def test_prepared_handoff_must_match_source_and_plan(tmp_path):
    path = tmp_path / "dispatch_ready.json"
    value = {
        "source_tree_sha256": "source",
        "plan_sha256": "plan",
        "parents": 16,
        "scheduler_dispatch_host": "login_resource_only",
    }
    path.write_text(json.dumps(value))
    assert dispatch.check_handoff(tmp_path, "source", "plan") == "prepared_handoff"
    with pytest.raises(ValueError, match="differs"):
        dispatch.check_handoff(tmp_path, "different", "plan")


def test_allocated_controller_prepares_handoff_without_submitting(tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import admission_pilots as pilot
    from third_eye.cluster_budget import reserve
    from third_eye.io import write_json

    output = tmp_path / "campaign"
    ledger_path = tmp_path / "ledger.json"
    prefix = "admission_b_fixture_"
    ledger = {"hard_cap_sus": 1200, "jobs": {}}
    reserve(ledger, prefix + "controller", "gg", 900)
    write_json(ledger_path, ledger)
    write_json(output / "submissions.json", {"jobs": {}})
    write_json(output / "cuda.json", {"passed": True})
    plan = {
        "source_tree_sha256": "fixture",
        "cuda_validation": str(output / "cuda.json"),
        "runtime_limits_by_stream": {str(i): 3600 for i in range(4)},
    }
    write_json(output / "plan.json", plan)
    request = {
        "output": str(output),
        "ledger": str(ledger_path),
        "ledger_prefix": prefix,
    }
    with (
        patch.object(pilot, "checked_plan", return_value=plan),
        patch(
            "subprocess.run",
            side_effect=AssertionError("Compute controller must not call sbatch"),
        ),
    ):
        pilot.submit(request)
        with pytest.raises(ValueError, match="already attempted"):
            pilot.submit(request)
    value = json.loads((output / "dispatch_ready.json").read_text())
    assert value["scheduler_dispatch_host"] == "login_resource_only"
    assert value["parents"] == 16
