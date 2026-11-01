from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from third_eye.config import Config, ModelConfig, ProtocolConfig
from third_eye.experiments.scale import (
    DECLARATION, MODELS, actions, compact_checkpoints,
    restore_archived_checkpoint, summarize, validate_authorization,
)
from third_eye.io import digest, file_digest, write_json


def test_vista_storage_uses_scratch_capacity_without_home_scan(monkeypatch):
    import scale_confirmation as driver

    monkeypatch.setattr(driver, "headroom", lambda *a: pytest.fail("DGX quota scan on Vista"))
    monkeypatch.setattr(driver.shutil, "disk_usage", lambda p: SimpleNamespace(used=100, free=4000))
    assert driver.storage_headroom({"execution_site": "vista", "root": "/scratch/example"}, 3000)["filesystem_free_bytes"] == 4000
    with pytest.raises(ValueError, match="headroom"):
        driver.storage_headroom({"execution_site": "vista", "root": "/scratch/example"}, 5000)
    with pytest.raises(ValueError, match="scratch"):
        driver.storage_headroom({"execution_site": "vista", "root": "/home/example"}, 1)


def test_vista_relocation_cannot_expand_to_qwen(tmp_path):
    import scale_confirmation as driver

    request = {"execution_site": "vista", "root": str(tmp_path), "output": str(tmp_path)}
    with pytest.raises(ValueError, match="Only.*Llama"):
        driver.prepare(request, "qwen3-8b")


def test_independent_vista_prepare_keeps_frozen_seeds_and_suites(tmp_path, monkeypatch):
    import hashlib
    import huggingface_hub
    import scale_confirmation as driver

    root = tmp_path
    output = root / "runs/campaign"
    output.mkdir(parents=True)
    payload = root / "payload"
    payload.mkdir()
    model = "llama3.1-8b"
    identity = MODELS[model]
    registry = {"files": {}, "tasks": []}

    def original(name, value):
        path = payload / name
        write_json(path, value)
        registry["files"][name] = {"sha256": file_digest(path)}
        return name

    metadata = {}
    for task in ("code", "math"):
        cfg = Config().to_dict()
        cfg["protocol"].update(candidate_size=2, correction_attempts=8, correction_seed_stride=16)
        config = original(task + "_config.json", cfg)
        dev = {"splits": {"train": {"path": original(task + "_train.json", [])}}}
        final = {"splits": {r: {"path": original(task + "_" + r + ".json", [])}
                            for r in DECLARATION["scoring_items"]}}
        manifest = original(task + "_dev.json", dev)
        final_manifest = original(task + "_final.json", final)
        metadata[str(payload / manifest)] = dev
        metadata[str(payload / final_manifest)] = final
        registry["tasks"].append({"stream": "qwen3-4b/" + task, "config": config,
                                  "manifest": manifest, "final_manifest": final_manifest})
    registry_path = root / "registry.json"
    write_json(registry_path, registry)
    blob = b"pinned weight fixture"
    weights = {"model.safetensors": {"sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}}
    catalog = {identity["official"]: {"revision": identity["revision"], "files": weights},
               identity["repository"]: {"revision": identity["source_revision"], "files": weights}}
    catalog_path = root / "catalog.json"
    write_json(catalog_path, catalog)

    def download(repo, name, *, revision, local_dir, token):
        assert repo == identity["repository"] and revision == identity["source_revision"] and token is False
        path = local_dir / name
        path.write_bytes(blob)
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(driver, "storage_headroom", lambda *a: {"free": 10**12})
    monkeypatch.setattr(driver, "load_manifest", lambda p: (metadata[str(p)], {}))
    monkeypatch.setattr(driver, "load_final_manifest", lambda p, m: (metadata[str(p)], {r: [None] * n for r, n in DECLARATION["scoring_items"].items()}))
    request = {"execution_site": "vista", "root": str(root), "output": str(output),
               "registry": str(registry_path), "catalog": str(catalog_path),
               "payload": str(payload), "source_tree_sha256": "source", "catalog_sha256": file_digest(catalog_path)}
    driver.prepare(request, model)
    plan = json.loads((output / model / "plan.json").read_text())
    assert len(plan["tasks"]) == 8
    assert {t["seed"] for t in plan["tasks"]} == {s for seeds in identity["seeds"].values() for s in seeds}
    assert plan["declaration"] == DECLARATION
    assert plan["execution_placement"]["site"] == "vista"
    assert not (output / "qwen3-8b").exists()


def test_scale_seed_grid_and_development_only_selection():
    seeds = [v for m in MODELS.values() for group in m["seeds"].values() for v in group]
    assert len(seeds) == len(set(seeds)) == 16
    assert all(len(v) == 4 for m in MODELS.values() for v in m["seeds"].values())
    x = np.array([[.2, .1, 0], [.1, .2, 0], [.5, -.4, 0]])
    result = actions("qwen3-8b/math", x)
    np.testing.assert_allclose(result["forced"], [.5, .5, 0])
    np.testing.assert_allclose(result["replay"], [.5, .5, 0])
    with pytest.raises(ValueError):
        actions("smollm3-3b/math", x)


def test_lossless_retention_roundtrip_preserves_scores_and_tensor_bytes(tmp_path):
    write_json(tmp_path / "completed.json", {"passed": True})
    write_json(tmp_path / "scoring/items.json", {"correct": True})
    before = file_digest(tmp_path / "scoring/items.json")
    data = b"exact float tensor fixture" * 1000
    for k in range(2):
        directory = tmp_path / str(k)
        directory.mkdir()
        tensor = directory / "adapter_model.safetensors"
        tensor.write_bytes(data)
        write_json(directory / "state.json", {"weights": {tensor.name: file_digest(tensor)}})
        (directory / "tokenizer.json").write_text("identical tokenizer")
    records = compact_checkpoints(tmp_path)
    assert len(records) == 2
    assert file_digest(tmp_path / "scoring/items.json") == before
    for k in range(2):
        directory = tmp_path / str(k)
        assert not (directory / "adapter_model.safetensors").exists()
        restore_archived_checkpoint(directory)
        assert (directory / "adapter_model.safetensors").read_bytes() == data
    with pytest.raises(FileExistsError):
        compact_checkpoints(tmp_path)


def test_incomplete_parents_cannot_be_compacted(tmp_path):
    write_json(tmp_path / "completed.json", {"passed": False})
    with pytest.raises(ValueError, match="complete"):
        compact_checkpoints(tmp_path)


def scale_states():
    states = []
    for task in ("code", "math"):
        for seed in range(4):
            artifacts = {"parent": {r: np.ones(6) * .5 for r in DECLARATION["scoring_items"]}}
            for k in range(3):
                artifacts[f"m1_{k}"] = {r: np.ones(6) * .52 for r in DECLARATION["scoring_items"]}
                for d in range(3):
                    artifacts[f"m2_{k}_{d}"] = {r: np.ones(6) * .49 for r in DECLARATION["scoring_items"]}
            states.append({"stream": "qwen3-8b/" + task, "seed": seed,
                           "actions": {"forced": [1 / 3] * 3, "replay": [0] * 3},
                           "artifacts": artifacts,
                           "ids": {r: [(task + str(i), str(i)) for i in range(6)] for r in DECLARATION["scoring_items"]}})
    return states


def test_balanced_hierarchical_inference_retains_known_constant_horizon_loss(tmp_path):
    states = scale_states()
    report = summarize(states, "qwen3-8b", tmp_path / "evidence", {}, draws=30)
    np.testing.assert_allclose(report["aggregate"]["selected_change"]["mean_pp"], -3)
    np.testing.assert_allclose(report["aggregate"]["selected_change"]["ci975_pp"], [-3, -3])
    with pytest.raises(ValueError, match="eight"):
        summarize(states[:-1], "qwen3-8b", tmp_path / "bad", {}, draws=30)
    mismatched = deepcopy(states)
    mismatched[0]["ids"]["target_test"] = mismatched[0]["ids"]["target_test"][::-1]
    with pytest.raises(ValueError, match="shared"):
        summarize(mismatched, "qwen3-8b", tmp_path / "bad", {}, draws=30)


def test_empirical_authorization_does_not_fake_a_forecasting_pass(tmp_path, monkeypatch):
    import third_eye.provenance as provenance

    monkeypatch.setattr(provenance, "source_inventory", lambda: {"source_tree_sha256": "source"})
    identity = MODELS["qwen3-8b"]
    cfg = Config(model=ModelConfig(name=identity["official"], revision=identity["revision"]),
                 protocol=ProtocolConfig(seed=identity["seeds"]["code"][0]))
    config, manifest = tmp_path / "cfg.json", tmp_path / "manifest.json"
    write_json(config, cfg.to_dict())
    write_json(manifest, {"frozen": True})
    ready = tmp_path / "ready.json"
    write_json(ready, {"passed": True, "source_tree_sha256": "source"})
    auth = tmp_path / "authorization.json"
    write_json(auth, {"purpose": "frozen_empirical_scale_confirmation", "model": "qwen3-8b",
                      "declaration_sha256": digest(DECLARATION), "source_tree_sha256": "source",
                      "preflight": str(ready), "preflight_sha256": file_digest(ready),
                      "tasks": [{"seed": cfg.protocol.seed, "config_hash": cfg.fingerprint,
                                 "manifest_sha256": file_digest(manifest)}],
                      "forecasting_gate_passed_claimed": False})
    kwargs = dict(mode="labels", policy="random", generations=1,
                  offsets=[50_000_000, 100_000_000], capture=True)
    assert validate_authorization(auth, config, manifest, **kwargs)["forecasting_gate_passed_claimed"] is False
    for key, value in [("mode", "online"), ("policy", "direct"), ("capture", False)]:
        with pytest.raises(ValueError):
            validate_authorization(auth, config, manifest, **{**kwargs, key: value})
    write_json(ready, {"passed": False, "source_tree_sha256": "source"})
    with pytest.raises(ValueError):
        validate_authorization(auth, config, manifest, **kwargs)


@pytest.mark.parametrize("valid_gold", [True, False])
def test_real_preflight_flow_uses_registry_rollback_and_checkpoint_reload(tmp_path, monkeypatch, valid_gold):
    import sys
    import torch
    import scale_confirmation as driver
    from third_eye.data.schema import Example

    model = "qwen3-8b"
    (tmp_path / model).mkdir()
    cfg = Config()
    tasks = [{"stream": model + "/" + name, "config": name, "manifest": name,
              "final_manifest": name, "manifest_sha256": "manifest", "seed": i}
             for i, name in enumerate(("code", "math"))]
    monkeypatch.setattr(driver, "plan_for", lambda r, m: {"tasks": tasks, "plan_sha256": "plan"})
    monkeypatch.setattr(Config, "load", classmethod(lambda cls, p: cfg))
    dev = {r: [Example(str(i), "Question " + str(i), "2", r) for i in range(32)]
           for r in ["train", "target_dev", "ood_dev", "retention_dev"]}
    final = {r: [Example(str(i), "Question " + str(i), "2", r) for i in range(8)]
             for r in DECLARATION["scoring_items"]}
    monkeypatch.setattr(driver, "load_manifest", lambda p: ({}, dev))
    monkeypatch.setattr(driver, "load_final_manifest", lambda *a: ({}, final))
    monkeypatch.setattr(driver, "require_gpu_allocation", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(driver, "collect_corrections", lambda *a: ([], {"total_seconds": 1}))
    monkeypatch.setattr(driver, "generate_greedy", lambda b, e, *a: ["2"] * len(e))
    monkeypatch.setattr(driver, "verify_completions", lambda v, e, c: [valid_gold] * len(e))
    calls = []

    class Backend:
        def __init__(self, c): self.state = "parent"
        def snapshot(self): return self.state
        def state_hash(self): return self.state
        def restore(self, state): self.state = state
        def train(self, *a):
            self.state = "updated"
            calls.append("train50")
            return {"optimizer_steps": 50, "seconds": 1}
        def save_checkpoint(self, path, metadata): calls.append("saved")
        def load_checkpoint(self, path):
            self.state = "updated"
            return {"adapter_hash": self.state}

    monkeypatch.setitem(sys.modules, "third_eye.training.hf_backend", SimpleNamespace(HFBackend=Backend))
    registry = driver.VerifierRegistry(lambda e, c: True)
    monkeypatch.setattr(driver.VerifierRegistry, "from_spec", classmethod(lambda cls, spec: registry))
    request = {"output": str(tmp_path), "source_tree_sha256": "source", "per_parent_seconds": 604800}
    if valid_gold:
        driver.preflight(request, model)
        assert calls == ["train50", "saved"] * 2
        assert json.loads((tmp_path / model / "authorized_scale.json").read_text())["forecasting_gate_passed_claimed"] is False
    else:
        with pytest.raises(ValueError, match="gold"):
            driver.preflight(request, model)
        assert calls == []
