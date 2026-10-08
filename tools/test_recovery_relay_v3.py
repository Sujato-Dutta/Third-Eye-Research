"""Recovery submissions use a bounded login relay, never compute-node sbatch."""

import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import pytest

import recovery_a2_relay_v3 as relay
import recovery_a2_controller_v3 as controller


def test_recovery_launchers_use_unix_line_endings():
    for name in ["recovery_a2_cpu_v3.slurm", "recovery_a2_gpu_v3.slurm"]:
        data = (Path(__file__).parent / name).read_bytes()
        assert data.startswith(b"#!/bin/bash\n") and b"\r" not in data


def request(root, script="gpu", mode=None):
    gpu = script == "gpu"
    return {
        "arguments": [
            "--parsable",
            "--account=IRI23021",
            f"--partition={'gh' if gpu else 'gg'}",
            "--nodes=1",
            "--ntasks=1",
            f"--cpus-per-task={16 if gpu else 4}",
            f"--time={48 if gpu else 4}:00:00",
            "--dependency=afterok:123,afterok:456",
            str(root / f"tools/recovery_a2_{script}_v3.slurm"),
            mode or ("20" if gpu else "finish"),
        ],
        "cwd": str(root),
        "environment": {},
        "controller_job": "123",
    }


def release(root):
    p = root / "runs/a2/recovery_20261008_v3"
    p.mkdir(parents=True)
    marker = root / "marker"
    marker.write_text("frozen operations")
    (p / "release_v3.json").write_text(
        json.dumps(
            {"files": {"marker": hashlib.sha256(marker.read_bytes()).hexdigest()}}
        )
    )


def test_bounded_requests_and_release_hashes(tmp_path):
    release(tmp_path)
    relay.validate(request(tmp_path), tmp_path)
    relay.validate(request(tmp_path, "cpu"), tmp_path)
    (tmp_path / "marker").write_text("modified")
    with pytest.raises(ValueError, match="hashes changed"):
        relay.validate(request(tmp_path), tmp_path)


def test_reject_invalid_modes_and_options(tmp_path):
    release(tmp_path)
    for r in [request(tmp_path, mode="36"), request(tmp_path, "cpu", "train")]:
        with pytest.raises(ValueError):
            relay.validate(r, tmp_path)
    for option in ["--export=ALL", "--nodelist=chosen", "--gres=gpu:1", "--array=0-19"]:
        r = request(tmp_path)
        r["arguments"].insert(0, option)
        with pytest.raises(ValueError):
            relay.validate(r, tmp_path)


def test_server_cannot_run_on_compute_node(monkeypatch, tmp_path):
    monkeypatch.setattr(relay.socket, "gethostname", lambda: "compute-node")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    with pytest.raises(RuntimeError, match="login node"):
        relay.serve(tmp_path, tmp_path, datetime.now(timezone.utc))


def test_controller_routes_scheduler_only_request(monkeypatch):
    observed = []
    monkeypatch.setattr(
        relay, "submit", lambda args: observed.append(args) or "12345\n"
    )
    monkeypatch.setenv("SLURM_JOB_ID", "caller")
    monkeypatch.setattr(
        controller.subprocess, "run", lambda *a, **k: pytest.fail("direct sbatch")
    )
    assert (
        controller.submit("recovery_a2_gpu_v3.slurm", [20], ["afterok:7"], "gh", 48)
        == "12345"
    )
    assert observed[0][-1] == "20"
    assert "--dependency=afterok:7" in observed[0]
    assert "--partition=gh" in observed[0]
