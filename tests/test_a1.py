"""A1 terminal updates, overlapping compositions, and outcome-input isolation."""

from dataclasses import replace
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from third_eye.config import Config
from third_eye.data.schema import Correction, Example, ROLES
from third_eye.experiments.labeling import LabelGenerator, run_trajectory
from third_eye.evaluation.verifiers import VerifierRegistry
from third_eye.updates.candidates import batch_hash, sample_batches
from third_eye.updates.corrections import InsufficientCorrections


def amended(config):
    return replace(
        config,
        protocol=replace(
            config.protocol,
            amendment="A1",
            candidate_sampling="stratified_distinct",
            terminal_continuation=True,
        ),
    )


def test_original_fingerprint_unchanged():
    assert not {"amendment", "candidate_sampling", "terminal_continuation"} & set(
        Config().to_dict()["protocol"]
    )


def test_three_compositions_can_overlap_and_vary_difficulty():
    pool = [
        Correction(
            Example(str(i), f"question {i}", "2", "train", difficulty=str(i)), "2", 1
        )
        for i in range(3)
    ]
    batches = sample_batches(pool, 2, 3, 1042, "stratified_distinct")
    assert len({batch_hash(b) for b in batches}) == 3
    assert all(len({c.example.id for c in b}) == 2 for b in batches)
    assert sum(len(b) for b in batches) > len(pool)
    assert batches == sample_batches(pool, 2, 3, 1042, "stratified_distinct")
    with pytest.raises(InsufficientCorrections):
        sample_batches(pool[:2], 2, 3, 1042, "stratified_distinct")


def test_terminal_continuation_preserves_real_adapter(tmp_path):
    from verify_backend import make_tiny_backend

    backend = make_tiny_backend(
        device=os.environ.get("THIRD_EYE_A1_VERIFY_DEVICE", "cpu")
    )
    backend.config = amended(backend.config)
    cfg = backend.config
    splits = {
        role: [
            Example(f"{role}-{i}", "one plus one", "#### 2", role)
            for i in range(4 if role == "train" else 1)
        ]
        for role in ROLES
    }
    pool = [Correction(ex, "#### 2", 1) for ex in splits["train"]]
    parent = backend.state_hash()

    def harvest(b, *args):
        available = pool if b.state_hash() == parent else []
        return available, {
            "initial_failures": len(available),
            "training_prompts": 4,
            "verified_corrections": len(available),
            "verifier_pass_rate": 1.0,
            "harvest_complete": True,
        }

    labeler = LabelGenerator(
        backend, cfg, splits, VerifierRegistry(), tmp_path, "fixture"
    )
    with patch("third_eye.experiments.labeling.collect_corrections", harvest):
        records = labeler.label_state(0)
    assert backend.state_hash() == parent
    for r in records:
        assert r["continuation_available"] == 0
        assert r["labels"]["h2"] == r["labels"]["h1"]
        assert r["evaluation"]["t2"] == r["evaluation"]["t1"]
        assert r["runtime"]["candidate"]["optimizer_steps"] == cfg.training.max_steps
        assert r["runtime"]["continuation"]["optimizer_steps"] == 0
        assert "continuation_available" not in r["precommit"]
        assert backend.load_checkpoint(Path(r["checkpoints"]["t2"]))
        assert backend.state_hash() == r["candidate_adapter_hash"]
    assert len(records) == 3


def test_a1_root_scarcity_stops_without_fabricated_labels(tmp_path, monkeypatch):
    from test_labeling import setup_labeler

    labeler = setup_labeler(tmp_path, monkeypatch)
    labeler.config = amended(labeler.config)
    monkeypatch.setattr(
        "third_eye.experiments.labeling.collect_corrections",
        lambda *args: ([], {"verified_corrections": 0}),
    )
    assert run_trajectory(labeler) == []
    assert not (tmp_path / "meta_labels.jsonl").exists()
    failure = json.loads(next(tmp_path.glob("states/*/failure.json")).read_text())
    assert failure["status"] == "correction_scarcity"


def test_terminal_outcomes_cannot_enter_forecaster_inputs():
    import copy
    import numpy as np
    from test_forecasting import records
    from third_eye.forecasting.encoding import encode

    original = records()[0]
    changed = copy.deepcopy(original)
    changed.update(
        continuation_available=0,
        continuation={"pool_statistics": {"verified_corrections": 999}},
    )
    for before, after in zip(encode(original), encode(changed)):
        np.testing.assert_array_equal(before, after)


def test_original_and_a1_labels_cannot_be_pooled(tmp_path):
    from test_forecasting import records
    from third_eye.forecasting.dataset import read_records

    rows = records(trajectories=2)
    for r in rows[6:]:
        r["protocol_amendment"] = "A1"
    file = tmp_path / "mixed.jsonl"
    file.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="amendments"):
        read_records([file])


def test_insufficient_gate_evidence_is_not_a_failed_phenomenon():
    from test_forecasting import records
    from third_eye.statistics.report import phenomenon, gate1

    result = gate1(phenomenon(records(trajectories=2)))
    assert result["status"] == "insufficient_evidence"
    assert result["passed"] is False and result["sample_sufficient"] is False


def test_all_composition_ranks_are_unique():
    pool = [
        Correction(Example(str(i), f"question {i}", "2", "train"), "2", 1)
        for i in range(5)
    ]
    batches = sample_batches(pool, 2, 10, 42, "stratified_distinct")
    assert len({batch_hash(b) for b in batches}) == 10
