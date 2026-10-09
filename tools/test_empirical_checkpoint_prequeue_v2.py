"""Synthetic scheduler integration; no cluster calls or scientific execution."""

import json
import pytest
import empirical_checkpoint_prequeue_v2 as queue


def setup_fixture(
    tmp_path, monkeypatch, *, controller="PENDING", validation_failed=False, nodes=3
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(queue.socket, "gethostname", lambda: "login1")
    out = tmp_path / "runs/a2/empirical_v1/checkpoint_fallback_v3"
    out.mkdir(parents=True)
    release = dict(
        files={},
        prerequisites={},
        scope=dict(deadline_utc="2099-10-11T18:29:59+00:00", gpu_cap=20),
    )
    (out / "release.json").write_text(json.dumps(release))
    (out / "validation_chain.json").write_text(
        json.dumps(
            dict(
                jobs=dict(
                    cpu_validation="100", cuda_validation="101", launch_controller="102"
                )
            )
        )
    )
    tasks = [
        dict(
            index=i,
            stream=f"stream_{i // 2}",
            seed=6142 + i % 2,
            output=str(tmp_path / f"output_{i}"),
        )
        for i in range(8)
    ]
    (out.parent / "prepared.json").write_text(json.dumps(dict(replication_tasks=tasks)))
    calls = []

    def scheduler(args):
        calls.append(args)
        if args[0].endswith("squeue"):
            if "-u" in args:
                return f"gh|{nodes}\n"
            return (
                controller if args[args.index("-j") + 1] == "102" else "PENDING"
            ) + "\n"
        if args[0].endswith("sbatch"):
            return str(199 + sum(c[0].endswith("sbatch") for c in calls)) + "\n"
        if args[0].endswith("sacct"):
            return (
                "100|FAILED\n101|PENDING\n"
                if validation_failed
                else "100|COMPLETED\n101|COMPLETED\n"
            )
        if args[0].endswith("scancel"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(queue, "scheduler", scheduler)
    return out, calls


def test_eight_jobs_block_on_both_validations_and_confirmation_waits_all(
    tmp_path, monkeypatch
):
    out, calls = setup_fixture(tmp_path, monkeypatch)
    queue.main()
    submissions = [c for c in calls if c[0].endswith("sbatch")]
    assert len(submissions) == 9
    assert all("--dependency=afterok:100:101" in c for c in submissions[:8])
    assert (
        "--dependency=afterany:" + ":".join(str(j) for j in range(200, 208))
        in submissions[-1]
    )
    assert [c[-1] for c in calls if c[0].endswith("scancel")] == ["102"]
    receipt = json.loads((out / "trajectory_submissions.json").read_text())
    assert len(receipt["jobs"]) == 8 and receipt["confirmation_job"] == "208"
    assert (
        json.loads((out / "validation_barrier_status.json").read_text())["status"]
        == "both_validations_completed"
    )


def test_failed_validation_cancels_only_blocked_collection_and_keeps_evidence(
    tmp_path, monkeypatch
):
    out, calls = setup_fixture(tmp_path, monkeypatch, validation_failed=True)
    queue.main()
    assert [c[-1] for c in calls if c[0].endswith("scancel")] == [
        "102",
        *map(str, range(200, 208)),
    ]
    assert (
        json.loads((out / "validation_barrier_status.json").read_text())["status"]
        == "validation_failed_no_collection"
    )


def test_running_controller_is_not_replaced(tmp_path, monkeypatch):
    out, calls = setup_fixture(tmp_path, monkeypatch, controller="RUNNING")
    with pytest.raises(RuntimeError, match="changed state"):
        queue.main()
    assert not (out / "trajectory_submissions.json").exists()
    assert not any(c[0].endswith("scancel") for c in calls)


def test_twenty_gpu_cap_prevents_any_additional_gpu_submission(tmp_path, monkeypatch):
    _, calls = setup_fixture(tmp_path, monkeypatch, nodes=20)
    with pytest.raises(RuntimeError, match="cap reached"):
        queue.main()
    assert not any(c[0].endswith("sbatch") for c in calls)
