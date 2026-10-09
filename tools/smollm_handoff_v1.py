"""One-time scheduler-only reconciliation after a rejected purged dependency.

The frozen S1 experiment code and scope remain unchanged. Preserve the failed
dispatcher record, verify accounting and matching validation receipts, and
queue only indices not previously submitted. Never retry an uncertain receipt.
"""

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

import smollm_dispatch_v1 as original


def completed_validation(state, exit_code, receipt, release_sha, tasks_sha):
    if (state, exit_code) != ("COMPLETED", "0:0"):
        raise RuntimeError("Validation must have completed successfully")
    if (
        receipt.get("status") != "passed"
        or receipt.get("release_sha256") != release_sha
        or receipt.get("tasks_sha256") != tasks_sha
    ):
        raise RuntimeError("Validation identity differs")
    # An already completed validation is enforced by its bound receipt and
    # accounting. Slurm can purge it from dependency lookup after completion.
    return None


def checked_history(raw, expected):
    actual = {line.strip().split("|")[0] for line in raw.splitlines() if line.strip()}
    if actual != set(expected):
        raise RuntimeError(f"Unexpected SmolLM submissions: {actual}")


def remaining_indices(jobs):
    indices = [j["index"] for j in jobs]
    if len(set(indices)) != len(indices) or any(i not in range(4) for i in indices):
        raise RuntimeError("Invalid existing grid")
    return [i for i in range(4) if i not in indices]


def main():
    if "login" not in socket.gethostname().lower() or os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Scheduler-only handoff requires login host")
    root = Path.cwd()
    out = root / "runs/a2/empirical_v1/smollm_supplement_v1"
    core = out.parent / "checkpoint_fallback_v3"
    repair = out / "handoff_v1"
    repair.mkdir(exist_ok=False)
    for name in ("smollm_handoff_v1.py", "test_smollm_handoff_v1.py"):
        (repair / name).write_bytes((root / "tools" / name).read_bytes())
    target = out / "submissions.json"
    prior = target.read_bytes()
    (repair / "previous_submissions.json").write_bytes(prior)
    record = json.loads(prior)
    assert record["status"] == "submitting_single_state"
    assert record["attempting_index"] == 1
    assert record["jobs"] == [dict(index=0, job_id="1058281")]
    release_sha = original.sha(out / "release.json")
    release = json.loads((out / "release.json").read_text())
    assert record["release_sha256"] == release_sha
    assert all(original.sha(root / p) == h for p, h in release["files"].items())
    assert original.sha(core / "release.json") == release["core_release_sha256"]
    for kind in ("cpu", "cuda"):
        assert original.passed_receipt(core / f"{kind}_passed.json", release["core_release_sha256"])
    path_repair = core / "path_repair_v1"
    assert original.passed_receipt(path_repair / "cpu_passed.json", original.sha(path_repair / "release.json"))
    tasks_sha = original.sha(out / "tasks.json")
    for kind in ("cpu", "cuda"):
        receipt = json.loads((out / f"{kind}_passed.json").read_text())
        assert receipt["tasks_sha256"] == tasks_sha
        assert original.passed_receipt(out / f"{kind}_passed.json", release_sha)
    # Read all matching accounting, including jobs that already left squeue.
    history = original.scheduler([
        "/usr/bin/sacct", "-u", "sujato_ts", "-S", "2026-10-08T00:00:00",
        "-X", "--name=third-eye-smollm-gpu", "--format=JobIDRaw", "-n", "-P",
    ])
    checked_history(history, {record["cuda_verification_job"], "1058281"})
    (repair / "pre_submit_accounting.txt").write_text(history)
    accounting = original.scheduler([
        "/usr/bin/sacct", "-j", record["cuda_verification_job"], "-X",
        "--format=JobIDRaw,State,ExitCode", "-n", "-P",
    ])
    rows = [line.strip().split("|") for line in accounting.splitlines() if line.strip()]
    assert len(rows) == 1 and rows[0][0] == record["cuda_verification_job"]
    completed_validation(rows[0][1], rows[0][2], json.loads((out / "cuda_passed.json").read_text()), release_sha, tasks_sha)
    provenance = dict(
        source_sha256=original.sha(Path(__file__).resolve()),
        original_release_sha256=release_sha,
        prior_submission_sha256=hashlib.sha256(prior).hexdigest(),
        validation_accounting=rows[0],
        removed_dependency="afterok:" + record["cuda_verification_job"],
        reason="Completed validation purged from Slurm dependency lookup; matching passed receipts and successful accounting retained",
        scientific_parameters_changed=False,
    )
    original.atomic(repair / "reconciliation.json", provenance)
    for index in remaining_indices(record["jobs"]):
        from datetime import datetime, timezone, timedelta
        if datetime.now(timezone.utc) >= datetime.fromisoformat(record["deadline_utc"]) - timedelta(hours=14):
            raise RuntimeError("Submission cutoff reached")
        queue = original.live_queue(original.scheduler([
            "/usr/bin/squeue", "-h", "-u", "sujato_ts", "-o", "%i|%P|%T|%D",
        ]))
        cap = original.capacity(core, queue)
        assert cap["free_after_reservation"] >= 1, cap
        record.update(status="handoff_submitting_single_state", attempting_index=index, capacity=cap)
        original.atomic(target, record)
        job = original.submit(root, "run", index, dependency=None)
        record["jobs"].append(dict(index=index, job_id=job))
        record.pop("attempting_index")
        record["status"] = "handoff_queued_single_state"
        original.atomic(target, record)
    record["status"] = "handoff_submitting_review"
    original.atomic(target, record)
    record["review_job"] = original.submit(root, "review", dependency=[j["job_id"] for j in record["jobs"]])
    record["status"] = "queued_four_states_and_review_stop"
    record["scheduler_reconciliation"] = str(repair / "reconciliation.json")
    original.atomic(target, record)
    original.atomic(repair / "success.json", record)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
