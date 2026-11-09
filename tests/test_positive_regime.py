import numpy as np
import pytest

from third_eye.data.schema import Correction, Example
from third_eye.experiments.positive_regime import (
    DECLARATION, ENDPOINTS, ROLES, arm_plan, item_contrasts, summarize,
)


def test_matched_arms_and_terminal_identity():
    batch = [Correction(Example(str(i), "p", "1", "train", "math"), "#### 1", 1)
             for i in range(8)]
    plan = arm_plan("m0", "m1", batch, 71)
    assert {x["optimizer_seed"] for x in plan.values()} == {71}
    assert {tuple(x["batch_ids"]) for x in plan.values()} == {tuple(str(i) for i in range(8))}
    assert plan["native"]["start_hash"] == plan["mitigated"]["start_hash"] == "m1"
    assert plan["direct"]["start_hash"] == "m0"
    assert plan["mitigated"]["learning_rate"] == plan["native"]["learning_rate"]/2
    terminal = arm_plan("m0", "m1", None, 71)
    assert all(x["steps"] == 0 and not x["continuation_available"] for x in terminal.values())
    assert terminal["direct"]["start_hash"] == "m0"
    with pytest.raises(ValueError, match="distinct"):
        arm_plan("m0", "m1", batch[:7]+batch[:1], 71)


def test_increment_interaction_is_not_level_difference():
    # Positive first update; neutral native second update; positive direct update.
    m0 = [0, 0, 0, 1]
    m1 = [1, 0, 0, 1]
    draws = np.array([[m1, [0, 1, 0, 1], m1]]*3)
    got = dict(zip(ENDPOINTS, item_contrasts(m0, m1, draws).mean(axis=1)))
    assert got["g1"] == .25 and got["native_increment"] == 0
    assert got["native_minus_direct"] == 0 and got["interaction"] == -.25


def test_terminal_and_shared_item_resampling():
    m0, m1 = [0, 0, 1, 1], [1, 0, 1, 1]
    values = item_contrasts(m0, m1, np.array([[m1, m0, m1]]*3))
    matrices = {r: np.stack([values]*4) for r in ROLES}
    report = summarize(matrices, draws=200)
    for name in ("native_increment", "direct_gain", "interaction", "mitigation_gain"):
        assert report["endpoints"][name]["mean_pp"] == 0
        assert report["endpoints"][name]["ci_pp"] == [0, 0]
    assert report["parents"] == 4  # Three draws never become twelve independent parents.
    assert report["endpoints"]["g1"]["family_size"] == 4
    assert report["mitigation_retention"]["family_size"] == 2
    assert report == summarize(matrices, draws=200)


def test_declared_cohort_and_invalid_outcomes():
    assert len(set(DECLARATION["seeds"])) == 20
    assert not set(DECLARATION["seeds"]) & set(range(2026110700, 2026110704))
    with pytest.raises(ValueError, match="Binary"):
        item_contrasts([.5], [1], np.ones((3, 3, 1)))
    with pytest.raises(ValueError, match="Complete"):
        item_contrasts([0], [1], np.ones((2, 3, 1)))


def test_runner_restores_each_arm_and_does_not_train_terminal():
    from dataclasses import replace
    from third_eye.config import Config
    from positive_regime_confirmation import paired_updates

    class Backend:
        config = Config()
        calls = []

        def restore(self, value):
            self.state = value

        def state_hash(self):
            return self.state

        def train(self, batch, seed, max_steps):
            self.calls.append((self.state, tuple(c.example.id for c in batch), seed,
                               max_steps, self.config.training.learning_rate))
            self.state += "+updated"
            return {"optimizer_steps": max_steps}

    cfg = Config()
    cfg = replace(cfg, training=replace(cfg.training, learning_rate=3e-5, max_steps=25))
    backend = Backend()
    saved = {}
    batch = [Correction(Example(str(i), "p", "1", "train"), "#### 1", 1) for i in range(8)]
    paired_updates(backend, cfg, "m0", "m1", "m0", "m1", batch, 123,
                   lambda arm, state: saved.update({arm: state}))
    assert [call[0] for call in backend.calls] == ["m1", "m0", "m1"]
    assert [call[-1] for call in backend.calls] == [3e-5, 3e-5, 1.5e-5]
    assert all(call[2:4] == (123, 25) for call in backend.calls)
    calls = len(backend.calls)
    paired_updates(backend, cfg, "m0", "m1", "m0", "m1", None, 124,
                   lambda arm, state: saved.update({arm: state}))
    assert len(backend.calls) == calls
    assert saved["native"]["adapter_hash"] == saved["mitigated"]["adapter_hash"] == "m1"
    assert saved["direct"]["adapter_hash"] == "m0"
    assert backend.config == cfg


@pytest.mark.parametrize("scarcity", ["none", "initial", "continuation"])
def test_complete_artifact_flow_and_terminal_score_reuse(tmp_path, monkeypatch, scarcity):
    """Exercise real scoring, sealing and independent reload with a stub model."""
    from dataclasses import replace
    from third_eye.config import Config
    from third_eye.io import digest, write_json
    import third_eye.training.hf_backend as hf
    import positive_regime_confirmation as runner

    train = [Example(str(i), "p", "1", "train") for i in range(8)]
    final = {r: [Example(r+str(i), "p", "1", r) for i in range(2)] for r in ROLES}
    cfg = Config()
    cfg = replace(cfg, training=replace(cfg.training, learning_rate=3e-5, max_steps=25))
    monkeypatch.setitem(runner.DECLARATION, "training_prompts", 8)
    monkeypatch.setitem(runner.DECLARATION, "evaluation_items", {r: 2 for r in ROLES})
    monkeypatch.setattr(runner, "configured", lambda request, seed: cfg)
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

        def _encode(self, prompt, completion):
            return {}

        def train(self, batch, seed, max_steps):
            self.state = digest([self.state, seed, max_steps, self.config.training.learning_rate])
            return {"optimizer_steps": max_steps}

        def save_checkpoint(self, path, metadata):
            path.mkdir(parents=True, exist_ok=False)
            write_json(path/"state.json", {"adapter_hash": self.state, "metadata": metadata})

        def generate_many(self, prompts, max_new_tokens, seeds):
            return ["#### 1"]*len(prompts)

    class Verifier:
        def verify(self, example, completion):
            return completion in ("1", "#### 1")

    calls = []

    def harvest(*args):
        calls.append(len(calls))
        available = scarcity != "initial" and not (scarcity == "continuation" and len(calls) == 2)
        return ([Correction(e, "#### 1", 1) for e in train] if available else []), {"harvest_complete": True}

    monkeypatch.setattr(hf, "HFBackend", Backend)
    monkeypatch.setattr(runner.VerifierRegistry, "from_spec", lambda path: Verifier())
    monkeypatch.setattr(runner, "collect_corrections", harvest)
    request = {"output": str(tmp_path), "manifest": "selection", "final_manifest": "final",
               "final_manifest_sha256": "frozen-final", "source_tree_sha256": "frozen-source"}
    write_json(tmp_path/"request.json", request)
    runner.run_parent(request, 0)
    matrices = runner.inspect_parent(request, 0, final)
    assert all(x.shape == (len(ENDPOINTS), 2) for x in matrices.values())
    folder = tmp_path/"parents"/str(DECLARATION["seeds"][0])
    done = runner.read(folder/"completed.json")
    assert len(done["scores"]) == 11
    if scarcity == "initial":
        assert not done["initial_available"] and done["continuation_available"] == [0, 0, 0]
        assert len(calls) == 1
        assert runner.read(folder/"scoring/m1/scores.json")["identity_reuse"]
    elif scarcity == "continuation":
        assert done["continuation_available"] == [0, 1, 1]
        assert runner.read(folder/"scoring/d0_native/scores.json")["identity_reuse"]
    # Changed donor data must be detected on independent reload.
    write_json(folder/"donors/0.json", {"changed": True})
    with pytest.raises(ValueError, match="Donor commitment"):
        runner.inspect_parent(request, 0, final)
