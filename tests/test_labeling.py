"""Branch invariants use a synthetic backend; these are never research results."""

from dataclasses import replace
import json
import re

import pytest

from third_eye.config import Config
from third_eye.data.schema import Example, ROLES
from third_eye.evaluation.runner import Consequences
from third_eye.evaluation.verifiers import VerifierRegistry
from third_eye.experiments.labeling import LabelGenerator, run_trajectory
from third_eye.io import digest, write_json


class FixtureBackend:
    def __init__(self, config, fail_call=None):
        self.config, self.state, self.starts, self.fail_call = config, (), [], fail_call

    def snapshot(self):
        return self.state

    def restore(self, snapshot):
        self.state = tuple(snapshot)

    def state_hash(self):
        return digest(self.state)

    def generate(self, prompt, *args, **kwargs):
        number = int(re.search(r"What is (\d+) plus 2", prompt).group(1))
        return f"#### {number + 2}" if "Reconsider" in prompt else "#### -1"

    def train(self, corrections, seed, max_steps=None, log_path=None):
        self.starts.append(self.state)
        self.state = (
            *self.state,
            sum(int(c.example.id.split("-")[-1]) + 1 for c in corrections),
        )
        if len(self.starts) == self.fail_call:
            raise RuntimeError("synthetic training failure")
        return {
            "optimizer_steps": max_steps or self.config.training.max_steps,
            "seconds": 0.001,
            "loss_first": 1.2,
            "loss_last": 1.0,
        }

    def measure_loss(self, examples):
        return 1.0 + sum(self.state) / 1000

    def save_checkpoint(self, path, metadata):
        path.mkdir(parents=True, exist_ok=False)
        write_json(path / "fixture.json", {"state": self.state, "metadata": metadata})

    def load_checkpoint(self, path):
        saved = json.loads((path / "fixture.json").read_text())
        self.restore(saved["state"])
        return saved


def setup_labeler(tmp_path, monkeypatch, probe_steps=0, fail_call=None, depth=1):
    cfg = Config()
    cfg = replace(
        cfg,
        protocol=replace(
            cfg.protocol, candidate_size=3, probe_steps=probe_steps, depth=depth
        ),
    )
    b = FixtureBackend(cfg, fail_call=fail_call)
    splits = {
        role: [
            Example(
                f"{role}-{i}",
                f"What is {offset + i} plus 2?",
                f"#### {offset + i + 2}",
                role,
                difficulty="even" if i % 2 else "odd",
            )
            for i in range(12 if role == "train" else 3)
        ]
        for role, offset in zip(ROLES, (0, 30, 60, 90))
    }

    def evaluate(backend, *args):
        v = sum(backend.state) / 100
        return Consequences(v, v / 2, -v / 3)

    monkeypatch.setattr("third_eye.experiments.labeling.evaluate", evaluate)
    return LabelGenerator(
        b, cfg, splits, VerifierRegistry(), tmp_path, "fixture-manifest"
    )


def test_candidate_branches_reset_and_h2_is_measured_from_parent(tmp_path, monkeypatch):
    labeler = setup_labeler(tmp_path, monkeypatch)
    b = labeler.backend
    b.state = (5,)
    parent_hash = b.state_hash()
    records = labeler.label_state(0)
    assert b.state_hash() == parent_hash
    assert b.starts[::2] == [(5,), (5,), (5,)]
    assert all(len(state) == 2 for state in b.starts[1::2])
    assert len(records) == 3
    assert len((tmp_path / "meta_labels.jsonl").read_text().splitlines()) == 3
    assert len({r["runtime"]["candidate"]["optimizer_steps"] for r in records}) == 1
    assert len({r["continuation"]["seed"] for r in records}) == 1
    for r in records:
        assert r["labels"]["h2"]["target"] == pytest.approx(
            r["evaluation"]["t2"]["target"] - 0.05
        )
        assert r["labels"]["h1"]["target"] == pytest.approx(
            r["evaluation"]["t1"]["target"] - 0.05
        )


def test_failed_branch_rolls_back_and_publishes_no_partial_state(tmp_path, monkeypatch):
    labeler = setup_labeler(tmp_path, monkeypatch, fail_call=3)
    parent = labeler.backend.state_hash()
    with pytest.raises(RuntimeError, match="synthetic training failure"):
        labeler.label_state(0)
    assert labeler.backend.state_hash() == parent
    assert not (tmp_path / "meta_labels.jsonl").exists()
    assert len(list(tmp_path.glob("states/*/failure.json"))) == 1


def test_probe_cannot_contaminate_full_candidate(tmp_path, monkeypatch):
    labeler = setup_labeler(tmp_path, monkeypatch, probe_steps=2)
    labeler.label_state(0)
    assert labeler.backend.starts[::3] == [(), (), ()]
    assert labeler.backend.starts[1::3] == [(), (), ()]
    assert labeler.backend.state == ()


def test_external_selector_never_receives_labels_and_commits_t1(tmp_path, monkeypatch):
    labeler = setup_labeler(tmp_path, monkeypatch, depth=2)
    inputs = []

    def selector(views):
        inputs.append(views)
        assert all(set(v) == {"state_id", "candidate_id", "precommit"} for v in views)
        assert all("labels" not in v["precommit"] for v in views)
        return 1

    accepted = run_trajectory(labeler, "external", selector=selector)
    assert len(accepted) == 2
    assert len(labeler.backend.state) == 2, (
        "Only one update per generation is committed"
    )
    assert inputs[1][0]["precommit"]["history"][0]["generation"] == 0
    assert (tmp_path / "accepted/generation_2/resume.json").exists()


def test_insufficient_continuation_is_not_fabricated(tmp_path, monkeypatch):
    from third_eye.updates.corrections import InsufficientCorrections

    labeler = setup_labeler(tmp_path, monkeypatch)
    original = labeler.backend.generate

    def generate(prompt, *args, **kwargs):
        if labeler.backend.state:
            number = int(re.search(r"What is (\d+) plus 2", prompt).group(1))
            return f"#### {number + 2}"  # t+1 has no failures; no continuation exists.
        return original(prompt, *args, **kwargs)

    labeler.backend.generate = generate
    with pytest.raises(InsufficientCorrections):
        labeler.label_state(0)
    assert labeler.backend.state == ()
    assert not (tmp_path / "meta_labels.jsonl").exists()
