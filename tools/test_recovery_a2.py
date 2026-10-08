"""Synthetic interruption/reuse equivalence, never scientific results."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from third_eye.config import Config
from third_eye.experiments.labeling import LabelGenerator
from third_eye.evaluation.runner import Consequences
from third_eye.forecasting.encoding import encode
from third_eye.io import write_json
from recovery_a2_runtime import RecoveryRuntime

spec = importlib.util.spec_from_file_location(
    "fixture_labeling", ROOT / "tests/test_labeling.py"
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def build(tmp_path, monkeypatch):
    reference = fixture.setup_labeler(
        tmp_path / "original", monkeypatch, probe_steps=10
    )
    reference.config = replace(
        Config(),
        protocol=replace(
            Config().protocol,
            amendment="A2",
            candidate_sampling="stratified_distinct",
            terminal_continuation=True,
            correction_attempts=8,
            correction_seed_stride=16,
            deterministic_execution=True,
            candidate_size=2,
            probe_steps=10,
            depth=5,
        ),
    )
    reference.backend.config = reference.config
    rows = reference.label_state(4)
    state = reference.output / "states" / rows[0]["state_id"]
    context = {
        "parent_adapter_hash": rows[0]["parent_adapter_hash"],
        "generation": 4,
        "partial_state": str(state),
        "branches": [
            {
                "candidate_id": r["candidate_id"],
                "batch_hash": r["batch_hash"],
                "complete": i < 2,
                "t1_hash": r["candidate_adapter_hash"],
            }
            for i, r in enumerate(rows)
        ],
    }
    backend = fixture.FixtureBackend(reference.config)
    runtime = RecoveryRuntime(backend, context, tmp_path / "receipts.jsonl")
    return reference, rows, runtime


def test_completed_branches_reused_native_missing_branch_matches(tmp_path, monkeypatch):
    reference, expected, runtime = build(tmp_path, monkeypatch)
    with runtime.hooks():
        recovered = LabelGenerator(
            runtime,
            reference.config,
            reference.splits,
            reference.verifier,
            tmp_path / "recovered",
            "fixture-manifest",
            feature_extractor=runtime.extract_features,
            trajectory_id=reference.trajectory_id,
        ).label_state(4)
    for before, after in zip(expected, recovered):
        for field in [
            "labels",
            "evaluation",
            "parent_adapter_hash",
            "candidate_adapter_hash",
            "continuation_adapter_hash",
            "batch_hash",
            "trajectory_id",
            "state_id",
        ]:
            assert before[field] == after[field]
        for a, b in zip(encode(before), encode(after)):
            assert (a == b).all()
    assert len(runtime.backend.starts) == 3  # Only missing probe, T1 and continuation.
    receipts = [json.loads(line) for line in runtime.receipts.read_text().splitlines()]
    assert sum(r["kind"] == "completed_update_reused" for r in receipts) == 4
    assert (
        sum(r["kind"] == "unfinished_branch_t1_replay_matches" for r in receipts) == 1
    )


def test_bad_checkpoint_hash_rejected(tmp_path, monkeypatch):
    reference, rows, runtime = build(tmp_path, monkeypatch)
    key = next(iter(runtime.trains))
    cp, expected, log, source = runtime.trains[key]
    saved = json.loads((cp / "fixture.json").read_text())
    saved["state"] = [999]
    write_json(cp / "fixture.json", saved)
    batch = json.loads((cp.parent / "batch.json").read_text())
    from third_eye.data.schema import Correction, Example

    with pytest.raises(RuntimeError, match="checkpoint hash"):
        runtime.train(
            [
                Correction(Example(**c["example"]), c["completion"], c["attempt"])
                for c in batch
            ],
            key[2],
        )


def test_seed_mismatch_does_not_reuse_update(tmp_path, monkeypatch):
    reference, rows, runtime = build(tmp_path, monkeypatch)
    from third_eye.data.schema import Correction, Example

    batch = json.loads(
        (
            Path(runtime.context["partial_state"])
            / rows[0]["candidate_id"]
            / "batch.json"
        ).read_text()
    )
    runtime.train(
        [
            Correction(Example(**c["example"]), c["completion"], c["attempt"])
            for c in batch
        ],
        runtime.seed + 300_001,
    )
    assert len(runtime.backend.starts) == 1


def assembly_fixture(tmp_path, monkeypatch):
    from third_eye.experiments.labeling import run_trajectory

    reference = fixture.setup_labeler(
        tmp_path / "original", monkeypatch, probe_steps=10, depth=5
    )
    reference.config = replace(
        Config(),
        protocol=replace(
            Config().protocol,
            amendment="A2",
            candidate_sampling="stratified_distinct",
            terminal_continuation=True,
            correction_attempts=8,
            correction_seed_stride=16,
            deterministic_execution=True,
            candidate_size=2,
            probe_steps=10,
            depth=5,
        ),
    )
    reference.backend.config = reference.config
    monkeypatch.setattr(
        "third_eye.experiments.labeling.evaluate",
        lambda b, *args: Consequences(
            0.3 + sum(b.state) / 1000,
            0.4 + sum(b.state) / 1000,
            0.9 - sum(b.state) / 1000,
        ),
    )
    run_trajectory(reference, generations=5)
    original = reference.output
    lines = (original / "meta_labels.jsonl").read_text().splitlines()
    (original / "meta_labels.jsonl").write_text("\n".join(lines[:12]) + "\n")
    write_json(
        original / "run.json",
        {
            "trajectory_id": reference.trajectory_id,
            "config": reference.config.to_dict(),
        },
    )
    segment = tmp_path / "segment"
    segment.mkdir()
    (segment / "states").mkdir()
    (segment / "meta_labels.jsonl").write_text("\n".join(lines[12:]) + "\n")
    write_json(
        segment / "run.json",
        {
            "trajectory_id": reference.trajectory_id,
            "config": reference.config.to_dict(),
        },
    )
    state_id = json.loads(lines[12])["state_id"]
    # Directory links are part of the production assembler, tested on Linux CPU nodes.
    import shutil

    shutil.copytree(original / "states" / state_id, segment / "states" / state_id)
    from third_eye.io import file_digest

    context = {
        "original": str(original),
        "segment": str(segment),
        "view": str(tmp_path / "view"),
        "original_labels_sha256": file_digest(original / "meta_labels.jsonl"),
    }
    return reference, context


def test_assembly_preserves_five_states_one_trajectory(tmp_path, monkeypatch):
    from recovery_a2_controller import assemble
    from third_eye.forecasting.dataset import read_records

    reference, context = assembly_fixture(tmp_path, monkeypatch)
    view = assemble(context, reference.config)
    rows = read_records([view / "meta_labels.jsonl"])
    assert len(rows) == 15 and {r["generation"] for r in rows} == set(range(5))
    assert {r["trajectory_id"] for r in rows} == {reference.trajectory_id}


def test_assembly_rejects_new_trajectory_id(tmp_path, monkeypatch):
    from recovery_a2_controller import assemble

    reference, context = assembly_fixture(tmp_path, monkeypatch)
    path = Path(context["segment"]) / "meta_labels.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["trajectory_id"] = "wrong-trajectory"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(AssertionError):
        assemble(context, reference.config)
