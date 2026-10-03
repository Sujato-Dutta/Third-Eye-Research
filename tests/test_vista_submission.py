"""The login relay must reject arbitrary executables and inherited allocations."""

import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "vista_submission",
    Path(__file__).resolve().parents[1] / "cluster/vista_submission.py",
)
relay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relay)


def request(tmp_path):
    return {
        "arguments": [
            "--parsable",
            "--partition=gh",
            "--account=IRI23021",
            "--array=0,1%20",
            "experiments/jobs/vista_study.slurm",
        ],
        "cwd": str(tmp_path),
        "environment": {"THIRD_EYE_PLAN": "plan.json"},
    }


def test_project_submission_accepted(tmp_path):
    relay.validate(request(tmp_path), tmp_path)


@pytest.mark.parametrize(
    "mutation", ["script", "allocation", "environment", "directory", "argument"]
)
def test_unexpected_submission_rejected(tmp_path, mutation):
    record = request(tmp_path)
    if mutation == "script":
        record["arguments"][-1] = "/tmp/arbitrary.sh"
    elif mutation == "allocation":
        record["arguments"][2] = "--account=OTHER"
    elif mutation == "environment":
        record["environment"]["SLURM_JOB_ID"] = "123"
    elif mutation == "directory":
        record["cwd"] = str(tmp_path.parent)
    else:
        record["arguments"].insert(1, "--wrap=python train.py")
    with pytest.raises(ValueError):
        relay.validate(record, tmp_path)


def test_client_rejects_login_execution(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(relay.socket, "gethostname", lambda: "login2")
    with pytest.raises(RuntimeError, match="compute-node"):
        relay.submit(["--parsable"])
