"""Third-family collection keeps selection separate from future consequences."""

from copy import deepcopy
from dataclasses import asdict
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import smollm_confirmation as driver
from third_eye.config import Config
from third_eye.data.schema import Example
from third_eye.data.splits import create_manifest
from third_eye.experiments.smollm_confirmation import (
    DECLARATION,
    actions,
    bootstrap,
    matrix,
)
from third_eye.io import digest, file_digest, publish_jsonl_batch, write_json
from third_eye.evaluation.verifiers import VerifierRegistry


@pytest.mark.parametrize("task_name", ["code", "math"])
@pytest.mark.parametrize("gold_passes", [True, False])
def test_preflight_registry_api_checkpoint_rollback_and_rejected_gold(
    tmp_path, monkeypatch, task_name, gold_passes
):
    """Run the actual preflight control flow without model weights or a GPU."""
    import torch

    calls = {"verify": 0, "train": 0}

    def verify(example, completion):
        calls["verify"] += 1
        return gold_passes and completion == example.answer

    registry = VerifierRegistry(verify)
    monkeypatch.setattr(
        VerifierRegistry, "from_spec", classmethod(lambda cls, spec: registry)
    )

    class Backend:
        def __init__(self, cfg):
            self.state = "parent"

        def snapshot(self):
            return self.state

        def state_hash(self):
            return self.state

        def restore(self, snapshot):
            self.state = snapshot

        def train(self, fixtures, seed, log_path):
            calls["train"] += 1
            assert len(fixtures) == 2
            self.state = "trained"
            return {"optimizer_steps": 50, "seconds": 1.0}

        def save_checkpoint(self, path, metadata):
            assert self.state == "trained"
            assert metadata["purpose"] == "discarded_compatibility_fixture"

        def load_checkpoint(self, path):
            assert self.state == "parent"
            self.state = "trained"
            return {"adapter_hash": self.state}

        def generate_many(self, prompts, max_new_tokens, seeds):
            return ["2"] * len(prompts)

    monkeypatch.setitem(
        sys.modules,
        "third_eye.training.hf_backend",
        SimpleNamespace(HFBackend=Backend),
    )
    monkeypatch.setattr(driver, "require_gpu_allocation", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    cfg = Config()
    monkeypatch.setattr(Config, "load", classmethod(lambda cls, path: cfg))
    task = {"stream": "smollm3-3b/" + task_name, "config": "cfg", "manifest": "dev"}
    monkeypatch.setattr(
        driver, "plan_for", lambda request: {"tasks": [task], "plan_sha256": "plan"}
    )
    dev = {
        role: [Example(str(i), "Question " + str(i), "2", role) for i in range(8)]
        for role in ("train", "target_dev", "ood_dev", "retention_dev")
    }
    monkeypatch.setattr(driver, "load_manifest", lambda path: ({}, dev))
    final = [Example(str(i), "Final " + str(i), "2", "ood_test") for i in range(8)]
    monkeypatch.setattr(
        driver, "load_final_manifest", lambda *args: ({}, {"ood_test": final})
    )
    calibration = tmp_path / "items.json"
    write_json(calibration, {"items": [{"id": ex.id} for ex in final]})
    request = {
        "source_tree_sha256": "source",
        "output": str(tmp_path / "output"),
        "inputs": {
            "calibration_final": {"path": "final"},
            "calibration_items": {"path": str(calibration)},
        },
    }
    if not gold_passes:
        with pytest.raises(ValueError, match="Gold train compatibility"):
            driver.preflight(request, task_name)
        assert calls["train"] == 0
        return
    driver.preflight(request, task_name)
    receipt = driver.read(
        tmp_path / "output" / "preflight" / task_name / "completed.json"
    )
    assert receipt["passed"] and receipt["discarded_fixture"]
    assert len(receipt["observations"]) == (8 if task_name == "code" else 6)
    assert all(not row["accuracy_recorded"] for row in receipt["observations"])
    assert calls == {"verify": 66 if task_name == "code" else 50, "train": 1}


def test_smol_actions_do_not_transfer_family_gate_or_forecast():
    result = actions("smollm3-3b/code", [[0.1, 0, 0], [0.1, 0, 0], [-0.1, 0, 0]])
    assert set(result) == {"hold", "forced", "replay"}
    assert result["forced"] == [0.5, 0.5, 0]
    assert result["replay"] == [0.5, 0.5, 0]
    with pytest.raises(ValueError):
        actions("qwen3-8b/code", np.ones((3, 3)))


def sample_states():
    result = []
    for task in ("code", "math"):
        for seed in range(8):
            artifacts = {
                "parent": {r: np.ones(8) for r in driver.DECLARATION["scoring_items"]}
            }
            for k in range(3):
                artifacts[f"m1_{k}"] = {r: np.ones(8) for r in artifacts["parent"]}
                for d in range(3):
                    artifacts[f"m2_{k}_{d}"] = {
                        r: np.zeros(8) for r in artifacts["parent"]
                    }
            result.append(
                {
                    "stream": "smollm3-3b/" + task,
                    "seed": seed,
                    "artifacts": artifacts,
                    "actions": actions("smollm3-3b/" + task, np.zeros((3, 3))),
                    "ids": {
                        r: [(r + str(i), str(i)) for i in range(8)]
                        for r in artifacts["parent"]
                    },
                    "terminal_draws": 0,
                }
            )
    return result


def test_known_paired_loss_and_shared_item_uncertainty():
    p, b, *_ = bootstrap(sample_states(), draws=40)
    np.testing.assert_allclose(p, [0, -1, 0, -1, -1, -1, 0])
    np.testing.assert_allclose(b, np.broadcast_to(p, b.shape))
    state = sample_states()[0]
    saved = deepcopy(state["actions"])
    for d in range(3):
        state["artifacts"][f"m2_0_{d}"]["target_test"][:] = 1
    assert not np.allclose(
        matrix(state, "target_test")[1], matrix(sample_states()[0], "target_test")[1]
    )
    assert state["actions"] == saved


def test_complete_balanced_extension_required():
    with pytest.raises(ValueError):
        bootstrap(sample_states()[:-1], draws=10)


def test_prepare_freezes16_seeds_original_suites_and_nonthinking_context(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    tasks = []
    for task in ("math", "code"):
        folder = tmp_path / task
        folder.mkdir()
        paths = {}
        for role, count in {
            "train": 2,
            "target_dev": 64,
            "ood_dev": 64,
            "retention_dev": 256,
        }.items():
            p = folder / (role + ".jsonl")
            publish_jsonl_batch(
                p,
                [
                    asdict(
                        Example(
                            f"{task}/{role}/{i}",
                            f"{task} {role} unique question {i}",
                            "2",
                            role,
                            task,
                        )
                    )
                    for i in range(count)
                ],
            )
            paths[role] = p
        manifest = create_manifest(paths, folder / "selection.json")
        entries = {}
        for role, count in DECLARATION["scoring_items"].items():
            p = folder / (role + ".jsonl")
            publish_jsonl_batch(
                p,
                [
                    asdict(
                        Example(
                            f"{task}/{role}/{i}",
                            f"{task} {role} final question {i}",
                            "2",
                            role,
                            task,
                        )
                    )
                    for i in range(count)
                ],
            )
            entries[role] = {"path": p.name, "count": count, "sha256": file_digest(p)}
        write_json(
            folder / "final.json",
            {
                "schema_version": 1,
                "selection_hash": digest(manifest),
                "splits": entries,
            },
        )
        cfg = Config().to_dict()
        cfg["protocol"].update(
            candidate_size=2, correction_attempts=8, correction_seed_stride=16
        )
        write_json(folder / "config.json", cfg)
        row = {"stream": "qwen3-4b/" + task, "seed": 42}
        for key, name in [
            ("config", "config.json"),
            ("manifest", "selection.json"),
            ("final_manifest", "final.json"),
        ]:
            row[key] = str(folder / name)
            row[key + "_sha256"] = file_digest(folder / name)
        tasks.append(row)
    core = tmp_path / "core.json"
    write_json(core, {"tasks": tasks})
    weight = tmp_path / "fake_weight"
    weight.write_text("synthetic fixture, not model weights")
    identity = tmp_path / "identity.json"
    write_json(
        identity,
        {
            "model": DECLARATION["model"],
            "revision": DECLARATION["revision"],
            "file_sha256": {str(weight): file_digest(weight)},
        },
    )
    request = {
        "output": str(output),
        "inputs": {
            "core_plan": {"path": str(core)},
            "model_identity": {"path": str(identity)},
        },
        "history_roots": [],
        "source_tree_sha256": "synthetic",
    }
    driver.prepare(request)
    plan = driver.plan_for(request)
    assert len(plan["tasks"]) == 16
    seeds = []
    for task in plan["tasks"]:
        cfg = json.loads(open(task["config"]).read())
        seeds.append(cfg["protocol"]["seed"])
        assert cfg["model"]["name"] == DECLARATION["model"]
        assert cfg["model"]["chat_kwargs"] == {
            "enable_thinking": False,
            "date_string": "11 Oct 2026",
        }
        assert cfg["protocol"]["correction_attempts"] == 8
    assert len(set(seeds)) == 16
    p = plan["tasks"][0]["config"]
    cfg = json.loads(open(p).read())
    cfg["protocol"]["seed"] += 1
    write_json(p, cfg)
    with pytest.raises(ValueError, match="Frozen configuration"):
        driver.plan_for(request)
