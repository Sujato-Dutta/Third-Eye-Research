from dataclasses import replace
import json

import pytest

from continue_study import decision, wall_limit
import continue_study
from types import SimpleNamespace
from third_eye.config import ModelConfig
from third_eye.io import file_digest, write_json
from third_eye.training.sources import resolve_source


def test_verified_source_pins_identity_and_metadata(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    config = runtime / "config.json"
    config.write_text('{"model_type":"llama"}')
    proof_path = tmp_path / "proof.json"
    proof = {
        "official": "meta-llama/Llama-3.2-3B-Instruct",
        "original_revision": "a" * 40,
        "source_repository": "public/model",
        "source_revision": "b" * 40,
        "runtime_directory": str(runtime),
        "verified": True,
        "verified_weight_sha256": {"model.safetensors": "c" * 64},
        "metadata_files": {"config.json": {"sha256": file_digest(config)}},
    }
    write_json(proof_path, proof)
    mappings = tmp_path / "sources.json"
    write_json(mappings, {proof["official"]: str(proof_path)})
    monkeypatch.setenv("THIRD_EYE_MODEL_SOURCES", str(mappings))
    model = ModelConfig(name=proof["official"], revision="a" * 40)
    source, revision, resolved = resolve_source(model)
    assert source == str(runtime) and revision == "b" * 40
    assert resolved["proof_sha256"] == file_digest(proof_path)
    with pytest.raises(ValueError, match="backbone"):
        resolve_source(replace(model, revision="d" * 40))
    config.write_text('{"model_type":"modified"}')
    with pytest.raises(ValueError, match="checksum"):
        resolve_source(model)
    assert resolve_source(ModelConfig())[2] is None


def test_campaign_halts_on_failed_or_wrong_gate(tmp_path):
    path = tmp_path / "gate.json"
    write_json(path, {"gate": 1, "passed": False, "sample_sufficient": False})
    with pytest.raises(RuntimeError, match="did not pass"):
        decision(path, 1)
    write_json(path, {"gate": 1, "passed": True})
    with pytest.raises(RuntimeError, match="Gate 2"):
        decision(path, 2)
    assert decision(path, 1) == json.loads(path.read_text())
    assert wall_limit(0.1) == "00:15:00"
    assert wall_limit(1.01) == "01:01:00"


def test_campaign_deadline_accounts_for_parallel_waves(monkeypatch):
    monkeypatch.setattr(continue_study.time, "time", lambda: 0)
    campaign = {"completion_deadline_utc": "1970-01-01T02:00:00+00:00"}
    continue_study.check_deadline(campaign, 4, 2, 1)
    with pytest.raises(RuntimeError, match="deadline"):
        continue_study.check_deadline(campaign, 5, 2, 1)
    with pytest.raises(ValueError, match="UTC offset"):
        continue_study.check_deadline(
            {"completion_deadline_utc": "1970-01-01T02:00:00"}, 1, 1, 1
        )


def test_completed_job_disappears_from_queue_but_accounting_remains(monkeypatch):
    query = SimpleNamespace(
        returncode=1,
        stdout="",
        stderr="slurm_load_jobs error: Invalid job id specified",
    )
    monkeypatch.setattr(continue_study.subprocess, "run", lambda *a, **k: query)
    monkeypatch.setattr(
        continue_study, "accounting", lambda job: [["123", "COMPLETED", "12", "0:0"]]
    )
    assert continue_study.wait_success("123", lambda *a, **k: None) == 12 / 3600
    monkeypatch.setattr(
        continue_study, "accounting", lambda job: [["123", "FAILED", "12", "1:0"]]
    )
    with pytest.raises(RuntimeError, match="failed"):
        continue_study.wait_success("123", lambda *a, **k: None)
    query.stderr = "Unable to contact scheduler"
    with pytest.raises(RuntimeError, match="Scheduler query failed"):
        continue_study.wait_success("123", lambda *a, **k: None)
