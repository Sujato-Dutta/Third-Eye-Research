"""Synthetic partial-checkpoint reuse must not perform new optimizer work."""

from dataclasses import replace
import json
from pathlib import Path
import pytest
from third_eye.config import Config
from third_eye.data.schema import Correction, Example
from third_eye.io import file_digest, write_json
from third_eye.updates.candidates import batch_hash
from recovery_a2_runtime_v4 import RecoveryRuntime


class Backend:
    def __init__(self, cfg):
        self.config = cfg
        self.state = "parent"

    def state_hash(self):
        return self.state

    def load_checkpoint(self, path):
        self.state = json.loads((path / "state.json").read_text())["adapter_hash"]

    def train(self, *a, **k):
        raise AssertionError("Unexpected optimizer work")


def setup(root):
    cfg = Config(protocol=replace(Config().protocol, candidate_size=2))
    batch = [
        Correction(Example(str(i), f"problem {i}", "1", "train"), "#### 1", 1)
        for i in range(2)
    ]
    branch = root / "k2"
    checkpoint = branch / "t1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "weights.bin").write_bytes(b"unchanged trained adapter")
    write_json(
        checkpoint / "state.json",
        dict(
            config_hash=cfg.fingerprint,
            adapter_hash="saved_m1",
            weights={"weights.bin": file_digest(checkpoint / "weights.bin")},
        ),
    )
    write_json(
        root / "correction_pool.json",
        dict(items=[c.to_dict() for c in batch], stats={}),
    )
    (branch / "training_t1.jsonl").write_text(
        "".join(
            json.dumps(dict(step=i + 1, loss=0.1, gradient_norm=0.2)) + "\n"
            for i in range(50)
        )
    )
    context = dict(
        parent_adapter_hash="parent",
        generation=0,
        partial_state=str(root),
        branches=[
            dict(
                candidate_id="k2",
                batch_hash=batch_hash(batch),
                complete=False,
                t1_hash="saved_m1",
            )
        ],
    )
    return cfg, batch, context


def test_partial_m1_reuses_exact_saved_checkpoint(tmp_path):
    cfg, batch, context = setup(tmp_path)
    backend = Backend(cfg)
    runtime = RecoveryRuntime(backend, context, tmp_path / "receipts.jsonl")
    log = runtime.train(
        batch, cfg.protocol.seed + 300_000, log_path=tmp_path / "copied_training.jsonl"
    )
    assert backend.state_hash() == "saved_m1"
    assert log["optimizer_steps"] == 50 and log["executed_optimizer_steps"] == 0
    assert log["original_training_seconds"] is None and log["seconds"] >= 0
    assert (tmp_path / "copied_training.jsonl").read_bytes() == (
        tmp_path / "k2/training_t1.jsonl"
    ).read_bytes()
    assert (
        json.loads((tmp_path / "receipts.jsonl").read_text())["kind"]
        == "partial_t1_checkpoint_reused"
    )


def test_incomplete_training_log_and_changed_weights_stop_reuse(tmp_path):
    cfg, batch, context = setup(tmp_path)
    (tmp_path / "k2/t1/weights.bin").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="weight file"):
        RecoveryRuntime(Backend(cfg), context, tmp_path / "receipts.jsonl")
    cfg, batch, context = setup(tmp_path / "second")
    (tmp_path / "second/k2/training_t1.jsonl").write_text(
        json.dumps(dict(step=1, loss=0.1, gradient_norm=0.2)) + "\n"
    )
    with pytest.raises(RuntimeError, match="fifty"):
        RecoveryRuntime(Backend(cfg), context, tmp_path / "receipts.jsonl")


def test_v4_scripts_use_unix_line_endings():
    for name in [
        "recovery_a2_gpu_v4.slurm",
        "recovery_a2_cpu_v4.slurm",
        "recovery_a2_verify_v4.slurm",
        "recovery_a2_validate_v4.slurm",
    ]:
        data = (Path(__file__).parent / name).read_bytes()
        assert b"\r" not in data and data.startswith(b"#!/bin/bash\n")
