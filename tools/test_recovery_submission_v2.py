"""Submission must preserve workload settings and avoid inherited allocations."""

import os
from types import SimpleNamespace

import pytest
import recovery_a2_controller_v2 as controller


def test_scheduler_child_does_not_inherit_allocation(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "caller-allocation")
    monkeypatch.setenv("SBATCH_NODELIST", "caller-node")
    monkeypatch.setenv("THIRD_EYE_ORIGINAL_ROOT", "workload-root")
    observed = {}

    def run(argv, **kwargs):
        observed.update(argv=argv, **kwargs)
        return SimpleNamespace(returncode=0, stdout="12345\n", stderr="")

    monkeypatch.setattr(controller.subprocess, "run", run)
    assert (
        controller.submit(
            "recovery_a2_gpu_v2.slurm",
            dependencies=["afterok:7"],
            partition="gh",
            hours=48,
        )
        == "12345"
    )
    assert not any(k.startswith(("SLURM_", "SBATCH_")) for k in observed["env"])
    assert observed["env"]["THIRD_EYE_ORIGINAL_ROOT"] == "workload-root"
    assert os.environ["SLURM_JOB_ID"] == "caller-allocation"
    assert "--nodes=1" in observed["argv"] and "--ntasks=1" in observed["argv"]
    assert not any(
        arg.startswith(("--nodelist", "--export", "--gres")) for arg in observed["argv"]
    )


def test_scheduler_rejection_keeps_diagnostics(monkeypatch):
    monkeypatch.setattr(
        controller.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=1, stdout="site reason", stderr="scheduler reason"
        ),
    )
    with pytest.raises(RuntimeError, match="site reason.*scheduler reason"):
        controller.submit("recovery_a2_cpu_v2.slurm")
