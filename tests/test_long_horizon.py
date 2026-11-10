from dataclasses import replace
import gzip
import json

import numpy as np
import pytest

from third_eye.config import Config
from third_eye.data.schema import Correction, Example
from third_eye.experiments.long_horizon import (
    DECLARATION, ENDPOINTS, MODELS, ROLES, configured, item_contrasts, learning_rate,
    step_seeds, summarize, train_step,
)
from third_eye.io import digest, file_digest, write_json


def test_cohort_has_four_paired_seeds_and_shared_horizons():
    assert len(MODELS) == 4
    assert all(len(set(m["seeds"])) == 4 for m in MODELS.values())
    assert len({s for m in MODELS.values() for s in m["seeds"]}) == 16
    assert DECLARATION["horizons"] == [3, 5] and 4 not in DECLARATION["scored_generations"]
    assert sum(4*m["gpu_hours"] for m in MODELS.values())+2 <= 450
    for model, spec in MODELS.items():
        cfg = configured(Config(), model, spec["seeds"][0])
        assert cfg.model.revision == spec["revision"]
        assert cfg.protocol.depth == 5 and cfg.training.max_steps == 25
        assert cfg.model.chat_kwargs["date_string"] == "11 Oct 2026"
    assert step_seeds(100, 1) == (100, 117, 201)
    assert step_seeds(100, 2) == (10100, 10119, 10131)
    assert learning_rate("native", 5) == 3e-5
    assert learning_rate("mitigated", 1) == 3e-5
    assert learning_rate("mitigated", 5) == 1.5e-5


def test_paired_endpoints_and_non_independence_of_generations():
    values = {"m0": [0, 0, 0, 0], "m1": [1, 1, 0, 0],
        "native_t3": [1, 0, 0, 0], "native_t5": [0, 0, 0, 0],
        "mitigated_t3": [1, 1, 0, 0], "mitigated_t5": [1, 0, 0, 0]}
    got = item_contrasts(values)
    assert got.mean(axis=1).tolist() == [.5, .25, 0, -.25, -.5, .25, .25, -.25]
    matrices = {r: np.stack([got]*4) for r in ROLES}
    report = summarize(matrices, "qwen3-4b", draws=200)
    assert report["paired_seeds"] == 4 and report["simultaneous_family_size"] == 32
    assert report["endpoints"]["native_h5_minus_h1"]["mean_pp"] == -50
    assert report == summarize(matrices, "qwen3-4b", draws=200)


def test_terminal_is_absorbing_and_no_duplicate_training_examples():
    class Backend:
        config = Config()

        def state_hash(self):
            return "unchanged"

        def train(self, *args, **kwargs):
            raise AssertionError("Terminal states must not train")

    backend = Backend()
    result = train_step(backend, backend.config, "mitigated", 5, None, 1, terminal=True)
    assert result["terminal"] and result["adapter_hash"] == result["start_hash"]
    batch = [Correction(Example(str(i), "p", "1", "train"), "#### 1", 1) for i in range(8)]
    with pytest.raises(ValueError, match="revive"):
        train_step(backend, backend.config, "native", 4, batch, 1, terminal=True)
    with pytest.raises(ValueError, match="unique"):
        train_step(backend, backend.config, "native", 4, batch[:7]+batch[:1], 1)


def test_imported_checkpoint_is_decompressed_into_new_directory_only(tmp_path):
    from long_horizon_trajectories import copy_checkpoint, saved_config
    source, dest = tmp_path/"source", tmp_path/"dest"
    source.mkdir()
    payload = b"saved adapter tensor bytes"
    raw = source/"adapter_model.safetensors"
    raw.write_bytes(payload)
    expected = file_digest(raw)
    with gzip.open(str(raw)+".gz", "wb") as zipped:
        zipped.write(payload)
    raw.unlink()
    cfg = configured(Config(), "qwen3-4b", MODELS["qwen3-4b"]["seeds"][0])
    cfg = replace(cfg, protocol=replace(cfg.protocol, depth=1))
    write_json(source/"state.json", {"config": cfg.to_dict(), "weights": {raw.name: expected}})
    state = copy_checkpoint(source, dest)
    assert (dest/raw.name).read_bytes() == payload and not raw.exists()
    assert saved_config(json.loads(json.dumps(state["config"]))).fingerprint == cfg.fingerprint
    assert file_digest(source/"state.json") == file_digest(dest/"state.json")


@pytest.mark.parametrize("scarcity", ["none", "initial", "third_step"])
def test_full_fresh_trajectory_seals_then_scores_and_detects_tampering(tmp_path, monkeypatch, scarcity):
    import long_horizon_trajectories as runner
    import third_eye.training.hf_backend as hf
    train = [Example(str(i), "p", "1", "train") for i in range(1024)]
    final = {r: [Example(r+str(i), "p", "1", r) for i in range(2)] for r in ROLES}
    monkeypatch.setitem(DECLARATION, "evaluation_items", {r: 2 for r in ROLES})
    monkeypatch.setattr(runner, "require_gpu_allocation", lambda: None)
    monkeypatch.setattr(runner, "load_manifest", lambda path: ({}, {"train": train}))
    monkeypatch.setattr(runner, "load_final_manifest", lambda *paths: ({}, final))

    class Backend:
        def __init__(self, config):
            self.config, self.state = config, "m0"

        def state_hash(self):
            return self.state

        def snapshot(self):
            return self.state

        def restore(self, snapshot):
            self.state = snapshot

        def _encode(self, *args):
            return {}

        def train(self, batch, seed, max_steps):
            self.state = digest([self.state, seed, self.config.training.learning_rate])
            return {"optimizer_steps": max_steps}

        def save_checkpoint(self, path, metadata):
            path.mkdir(parents=True, exist_ok=False)
            write_json(path/"state.json", {"adapter_hash": self.state})

        def generate_many(self, prompts, max_new_tokens, seeds):
            return ["#### 1"]*len(prompts)

    class Verifier:
        def verify(self, example, completion):
            return True

    calls = []

    def harvest(*args):
        calls.append(len(calls))
        absent = scarcity == "initial" or (scarcity == "third_step" and len(calls) == 3)
        return ([] if absent else [Correction(e, "#### 1", 1) for e in train[:8]]), {}

    monkeypatch.setattr(hf, "HFBackend", Backend)
    monkeypatch.setattr(runner.VerifierRegistry, "from_spec", lambda path: Verifier())
    monkeypatch.setattr(runner, "collect_corrections", harvest)
    model = "llama-3.2-3b-instruct"
    write_json(tmp_path/"template.json", Config().to_dict())
    request = {"output": str(tmp_path), "template": str(tmp_path/"template.json"),
        "manifest": "selection", "final_manifest": "final", "source_tree_sha256": "frozen-source"}
    write_json(tmp_path/"request.json", request)
    write_json(tmp_path/"compatibility"/model/"completed.json", {"passed": True, "source_tree_sha256": "frozen-source"})
    runner.run(request, model, 0)
    matrices = runner.inspect(request, model, 0)
    assert all(x.shape == (len(ENDPOINTS), 2) for x in matrices.values())
    folder = tmp_path/"parents"/model/str(MODELS[model]["seeds"][0])
    commitment = runner.read(folder/"commitment.json")
    assert len(commitment["states"]) == 10 and len(runner.read(folder/"completed.json")["scores"]) == 8
    if scarcity == "initial":
        assert len(calls) == 1
        assert all(x["adapter_hash"] == "m0" for x in commitment["states"].values())
    elif scarcity == "third_step":
        assert commitment["states"]["native_t5"]["adapter_hash"] == commitment["states"]["native_t2"]["adapter_hash"]
        assert not commitment["states"]["mitigated_t5"]["terminal"]
    write_json(folder/"donors/shared_t1.json", {"changed": True})
    with pytest.raises(ValueError, match="donor data"):
        runner.inspect(request, model, 0)
