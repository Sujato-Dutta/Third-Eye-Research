"""S1 contracts: bounded grid, unchanged scientific budgets and reserved capacity."""

from dataclasses import asdict
import json
from types import SimpleNamespace
import pytest
import smollm_supplement_v1 as s1
import smollm_dispatch_v1 as dispatcher
from third_eye.config import Config, ProtocolConfig


def original():
    return Config(
        protocol=ProtocolConfig(
            seed=6142,
            candidate_size=2,
            correction_attempts=8,
            depth=5,
            status="frozen",
            generation_batch_size=32,
            sampled_batch_size=16,
            multiple_choice_scoring="conditional_likelihood",
            amendment="A2",
            candidate_sampling="stratified_distinct",
            terminal_continuation=True,
            correction_seed_stride=16,
            deterministic_execution=True,
        )
    )


def test_model_swap_preserves_training_protocol_and_original_config():
    before = original()
    cfg = s1.make_config(before, "a" * 40, 6143)
    assert cfg.model.name == s1.MODEL and cfg.model.chat_kwargs == {
        "enable_thinking": False
    }
    assert asdict(cfg.training) == asdict(before.training)
    p = asdict(before.protocol)
    p["seed"] = 6143
    assert asdict(cfg.protocol) == p and before.protocol.seed == 6142
    assert (
        cfg.protocol.depth == 5
    )  # Invocation is bounded to one state, not a changed protocol.


def test_unpinned_or_new_seed_is_rejected():
    with pytest.raises(ValueError):
        s1.make_config(original(), "main", 6142)
    with pytest.raises(ValueError):
        s1.make_config(original(), "a" * 40, 6144)


def core_fixture(tmp_path, jobs=8):
    core = tmp_path / "core"
    (core / "path_repair_v1").mkdir(parents=True)
    (core / "noise_submissions").mkdir()
    (core / "noise_inputs").mkdir()
    items = [dict(index=i, job_id=str(100 + i)) for i in range(jobs)]
    (core / "path_repair_v1/submissions.json").write_text(json.dumps(dict(jobs=items)))
    return core


def test_reserving_eight_future_controls_prevents_stealing_core_slots(tmp_path):
    core = core_fixture(tmp_path)
    queued = {
        str(100 + i): dict(partition="gh", state="PENDING", nodes=1) for i in range(8)
    }
    queued.update(
        {str(200 + i): dict(partition="gh", state="PENDING", nodes=1) for i in range(3)}
    )
    cap = dispatcher.capacity(core, queued)
    assert cap == dict(
        occupied_gpu_nodes=11, reserved_core_controls=8, free_after_reservation=1
    )
    queued["300"] = dict(partition="gh-dev", state="RUNNING", nodes=1)
    assert dispatcher.capacity(core, queued)["free_after_reservation"] == 0


def test_submitted_controls_and_finished_parents_release_reservations(tmp_path):
    core = core_fixture(tmp_path)
    queued = {
        str(100 + i): dict(partition="gh", state="RUNNING", nodes=1) for i in range(8)
    }
    (core / "noise_submissions/task_0.json").write_text(
        json.dumps(dict(status="queued", job_id="900"))
    )
    queued["900"] = dict(partition="gh", state="PENDING", nodes=1)
    assert dispatcher.reserved_core_slots(core, queued) == 7
    queued.pop("101")
    assert dispatcher.reserved_core_slots(core, queued) == 6
    # A published state awaiting submission continues to reserve its control slot.
    (core / "noise_inputs/task_1.json").write_text("{}")
    assert dispatcher.reserved_core_slots(core, queued) == 7


def test_queue_counts_pending_nodes_and_both_gpu_partitions():
    parsed = dispatcher.live_queue(
        "123|gh|PENDING|2\n124|gh-dev|RUNNING|1\n125|gg|RUNNING|1\n"
    )
    assert parsed["123"]["nodes"] == 2 and parsed["124"]["partition"] == "gh-dev"


def test_generated_supplement_jobs_use_correct_scripts_and_validation_dependencies(
    monkeypatch, tmp_path
):
    calls = []
    monkeypatch.setattr(
        dispatcher, "scheduler", lambda args: calls.append(args) or "12345\n"
    )
    assert dispatcher.submit(tmp_path, "verify") == "12345"
    assert "--dependency=afterok:1057730" in calls[-1]
    assert "smollm_gpu_v1.slurm" in calls[-1][-2]
    dispatcher.submit(tmp_path, "run", 3, ["12345"])
    assert "--dependency=afterok:12345" in calls[-1] and "--time=12:00:00" in calls[-1]
    dispatcher.submit(tmp_path, "review", dependency=["1", "2", "3", "4"])
    assert "--dependency=afterany:1:2:3:4" in calls[-1]
    assert "smollm_cpu_v1.slurm" in calls[-1][-2]


def test_single_state_runner_never_prunes_or_requests_more_generations(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(s1, "OUT", tmp_path)
    monkeypatch.setattr(s1, "frozen", lambda: {})
    monkeypatch.setattr(s1, "require_gpu_allocation", lambda: None)
    monkeypatch.setattr(s1.core, "arm_deadline", lambda scope: None)
    (tmp_path / "release.json").write_text("{}")
    (tmp_path / "tasks.json").write_text("{}")
    from third_eye.io import file_digest

    receipt = dict(
        status="passed",
        release_sha256=file_digest(tmp_path / "release.json"),
        tasks_sha256=file_digest(tmp_path / "tasks.json"),
    )
    (tmp_path / "cuda_passed.json").write_text(json.dumps(receipt))
    task = dict(
        config="config", manifest="manifest", output=str(tmp_path / "new_labels")
    )
    monkeypatch.setattr(s1, "tasks", lambda check_weights=False: [task] * 4)

    def run(path, run_name):
        argv = s1.sys.argv
        assert argv[argv.index("--generations") + 1] == "1"
        assert argv[argv.index("--keep-accepted") + 1] == "5"
        assert "--prune-branches" not in argv

    monkeypatch.setattr(s1.runpy, "run_path", run)
    old = s1.sys.argv
    try:
        s1.run(0)
    finally:
        s1.sys.argv = old


def test_failed_cuda_validation_prevents_any_collection(tmp_path, monkeypatch):
    monkeypatch.setattr(s1, "OUT", tmp_path)
    monkeypatch.setattr(s1, "frozen", lambda: {})
    monkeypatch.setattr(s1, "require_gpu_allocation", lambda: None)
    monkeypatch.setattr(s1.core, "arm_deadline", lambda scope: None)
    (tmp_path / "cuda_passed.json").write_text(json.dumps(dict(status="failed")))
    with pytest.raises(RuntimeError, match="CUDA validation"):
        s1.run(0)


@pytest.mark.parametrize("corrupt", [False, True])
def test_setup_pins_and_verifies_official_weights_before_collection(
    tmp_path, monkeypatch, corrupt
):
    import hashlib
    import huggingface_hub
    import transformers
    import third_eye.data.splits as splits_module
    import third_eye.training.tokenization as tokenization
    from third_eye.io import file_digest
    from third_eye.data.schema import Example

    out = tmp_path / "supplement"
    out.mkdir()
    monkeypatch.setattr(s1, "OUT", out)
    monkeypatch.setattr(s1, "frozen", lambda: {})
    (out / "release.json").write_text("{}")
    cfg = tmp_path / "original.json"
    cfg.write_text(json.dumps(original().to_dict()))
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    templates = [
        dict(
            stream="qwen3-4b/" + f,
            config=str(cfg),
            manifest=str(manifest),
            manifest_sha256=file_digest(manifest),
        )
        for f in ("code", "math")
    ]
    monkeypatch.setattr(s1, "prepared", lambda: dict(replication_tasks=templates))
    model = tmp_path / "model"
    model.mkdir()
    weight = model / "model.safetensors"
    weight.write_bytes(b"synthetic-weight-fixture")
    expected = "0" * 64 if corrupt else hashlib.sha256(weight.read_bytes()).hexdigest()
    info = SimpleNamespace(
        gated=False,
        sha="a" * 40,
        siblings=[
            SimpleNamespace(rfilename=weight.name, lfs=SimpleNamespace(sha256=expected))
        ],
    )
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda: SimpleNamespace(model_info=lambda *a, **kw: info),
    )
    calls = []
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *a, **kw: calls.append(kw) or str(model),
    )
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *a, **kw: SimpleNamespace(model_type="smollm3"),
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *a, **kw: SimpleNamespace(is_fast=True),
    )
    fixture = [Example(str(i), "prompt" + str(i), "2", "train") for i in range(2)]
    monkeypatch.setattr(
        splits_module, "load_manifest", lambda path: ({}, dict(train=fixture))
    )
    monkeypatch.setattr(
        tokenization,
        "encode_completion",
        lambda *a: dict(input_ids=[1, 2], labels=[-100, 2]),
    )
    if corrupt:
        with pytest.raises(RuntimeError, match="checksum differs"):
            s1.setup()
        assert not (out / "cpu_passed.json").exists()
        assert (out / "setup/failure.json").exists()
    else:
        s1.setup()
        assert len(s1.tasks()) == 4
        identity = json.loads((out / "model_identity.json").read_text())
        assert identity["revision"] == "a" * 40
        assert identity["official_lfs_sha256"][str(weight)] == expected
        assert all(t["generations"] == 1 for t in s1.tasks())
    assert len(calls) == 2 and all(c["token"] is False for c in calls)
