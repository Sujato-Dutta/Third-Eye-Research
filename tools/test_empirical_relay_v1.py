"""Operational safety and paired-noise analysis tested with synthetic fixtures."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import pytest
import empirical_relay_v1 as relay
import empirical_noise_review_v1 as review


def fixture(root):
    base = root / "runs/a2/empirical_v1"
    base.mkdir(parents=True)
    scope = base / "scope.json"
    scope.write_text(
        json.dumps(dict(deadline_utc="2100-01-01T00:00:00+00:00", gpu_cap=20))
    )
    cached = base / "release.json"
    cached.write_text(
        json.dumps(dict(scope_sha256=hashlib.sha256(scope.read_bytes()).hexdigest()))
    )
    noise = base / "noise_release.json"
    noise.write_text(
        json.dumps(
            dict(
                files={},
                cached_release_sha256=hashlib.sha256(cached.read_bytes()).hexdigest(),
            )
        )
    )
    verification = base / "gpu_verification"
    verification.mkdir()
    (verification / "passed.json").write_text(
        json.dumps(
            dict(noise_release_sha256=hashlib.sha256(noise.read_bytes()).hexdigest())
        )
    )


def request(root, gpu=True, mode=None):
    return dict(
        arguments=[
            "--parsable",
            "--account=IRI23021",
            "--nodes=1",
            "--ntasks=1",
            "--partition=" + ("gh" if gpu else "gg"),
            "--cpus-per-task=" + ("16" if gpu else "72"),
            "--time=" + ("24:00:00" if gpu else "1:00:00"),
            str(
                root
                / (
                    "tools/empirical_noise_v1.slurm"
                    if gpu
                    else "tools/empirical_noise_control_v1.slurm"
                )
            ),
            mode or ("11" if gpu else "review"),
        ],
        cwd=str(root),
        environment={},
    )


def test_scheduler_cap_and_validation_barrier(tmp_path, monkeypatch):
    fixture(tmp_path)
    monkeypatch.setattr(
        relay.subprocess, "check_output", lambda *a, **k: "gh|PENDING|1\n" * 19
    )
    relay.validate(request(tmp_path), tmp_path)
    relay.validate(request(tmp_path, False), tmp_path)
    monkeypatch.setattr(
        relay.subprocess, "check_output", lambda *a, **k: "gh|PENDING|1\n" * 20
    )
    with pytest.raises(ValueError, match="ceiling"):
        relay.validate(request(tmp_path), tmp_path)
    for mode in ["12", "-1", "0;train"]:
        with pytest.raises(ValueError):
            relay.validate(request(tmp_path, mode=mode), tmp_path)
    r = request(tmp_path)
    r["arguments"].insert(0, "--array=0-19")
    with pytest.raises(ValueError):
        relay.validate(r, tmp_path)


def test_relay_has_no_compute_service(tmp_path, monkeypatch):
    monkeypatch.setattr(relay.socket, "gethostname", lambda: "compute")
    with pytest.raises(RuntimeError, match="login node"):
        relay.serve(tmp_path, tmp_path, datetime.now(timezone.utc))
    for name in [
        "empirical_noise_v1.slurm",
        "empirical_noise_verify_v1.slurm",
        "empirical_noise_control_v1.slurm",
    ]:
        data = (Path(__file__).parent / name).read_bytes()
        assert data.startswith(b"#!/bin/bash\n") and b"\r" not in data


def noise_record():
    def items(bits):
        return {
            r: [
                dict(
                    id=str(i), prompt_sha256=str(i), prediction_sha256=str(b), correct=b
                )
                for i, b in enumerate(bits)
            ]
            for r in review.ROLES
        }

    def measured(bits, adapter):
        return dict(
            items=items(bits),
            adapter_hash=adapter,
            continuation_available=1,
            training=dict(optimizer_steps=50),
        )

    result = dict(
        state_id="s",
        trajectory_id="t",
        stream="synthetic/math",
        generation=0,
        parent=dict(items=items([0, 0, 0, 0]), metrics={h: 0.0 for h in review.HEADS}),
        candidates=[],
    )
    for i in range(3):
        one = measured([1] * i + [0] * (4 - i), f"m1-{i}")
        two = measured([1] * (2 - i) + [0] * (2 + i), f"m2-{i}")
        conditions = dict(
            original_m1=one,
            original_m2=two,
            optimizer_m1=deepcopy(one),
            optimizer_m2=deepcopy(two),
            fresh_m2_50000000=deepcopy(two),
            fresh_m2_100000000=deepcopy(two),
        )
        result["candidates"].append(
            dict(batch_hash=str(i), candidate_id=str(i), conditions=conditions)
        )
    return result


def test_rank_reversal_and_matched_noise_are_separate():
    r = review.inspect(noise_record())
    assert r["original_h1_h2_reversal"] == 1
    assert r["fixed_candidate_optimizer_reversal"] == 0
    assert r["fixed_continuation_optimizer_reversal"] == 0
    assert r["fresh_continuation_reversal"] == 0
    assert r["reversal_minus_fresh_continuation_variation"] == 1
    assert r["greedy_minus_random"] < 0
    assert review.aggregate([r])["original_h1_h2_reversal"]["mean"] == 1


def test_unaligned_outcomes_and_false_terminal_are_rejected():
    record = noise_record()
    record["candidates"][1]["conditions"]["original_m2"]["items"][review.ROLES[0]][0][
        "id"
    ] = "wrong"
    with pytest.raises(RuntimeError, match="aligned"):
        review.inspect(record)
    record = noise_record()
    record["candidates"][0]["conditions"]["fresh_m2_50000000"][
        "continuation_available"
    ] = 0
    with pytest.raises(RuntimeError, match="Terminal"):
        review.inspect(record)
