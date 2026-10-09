"""E2 synthetic integration tests: saved-state controls and bounded orchestration."""

import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import empirical_checkpoint_v2 as e2
import empirical_checkpoint_noise_v2 as noise
import empirical_checkpoint_relay_v2 as relay
from third_eye.data.schema import Example
from third_eye.io import file_digest


def test_fixed_generations_and_budget_arguments():
    assert e2.generation_for(6142) == 0
    assert e2.generation_for(6143) == 4
    with pytest.raises(ValueError):
        e2.generation_for(6144)
    assert "--time=48:00:00" in e2.arguments("trajectory", 0)
    assert "--time=24:00:00" in e2.arguments("noise", 7)
    assert "--dependency=afterany:123:456" in e2.arguments(
        "review", dependency=[123, 456]
    )


class Backend:
    config = SimpleNamespace(
        protocol=SimpleNamespace(multiple_choice_scoring="generation")
    )

    def state_hash(self):
        return "saved"

    def generate_many(self, prompts, max_new_tokens, seeds):
        return ["correct" for _ in prompts]


class Verifier:
    def __init__(self):
        self.calls = 0

    def verify_many(self, examples, texts):
        self.calls += 1
        return [self.calls % 3 != 2 for _ in examples]


def test_identical_code_verifier_variability_is_separate_from_regeneration():
    splits = {
        r: [
            Example("one", "p", "a", r, task="code" if r != "retention_dev" else "math")
        ]
        for r in noise.ROLES
    }
    result = noise.collect_items(
        Backend(), splits, Verifier(), SimpleNamespace(max_new_tokens=512), 42
    )
    assert (
        result["fixed_prediction_verifier_repeats"]["target_dev"]["fluctuating_items"]
        == 1
    )
    assert result["adapter_hash"] == "saved"
    noise.difference(result, dict(target=0, ood=0, retention=0))
    assert result["reevaluation_minus_original"]["target"] == 1


def test_actual_m1_m2_loads_never_train_or_reconstruct(monkeypatch):
    class Saved:
        value = "parent"
        loaded = []

        def load_checkpoint(self, path):
            self.loaded.append(path)
            self.value = str(path)

        def state_hash(self):
            return self.value

        def snapshot(self):
            return self.value

        def train(self, *args, **kwargs):
            raise AssertionError("Original checkpoint loading attempted training")

    b = Saved()
    row = dict(
        checkpoints=dict(t1="m1", t2="m1"),
        candidate_adapter_hash="m1",
        continuation_adapter_hash="m1",
        continuation_available=0,
    )
    assert noise.original_m1(b, {}, row, None, None) == "m1"
    noise.original_m2(b, {}, row, None)
    assert b.loaded == ["m1", "m1"]
    row["continuation_adapter_hash"] = "wrong"
    with pytest.raises(RuntimeError, match="Exact replay failed"):
        noise.original_m2(b, {}, row, None)


def make_state(tmp_path, monkeypatch):
    monkeypatch.setattr(e2, "OUT", tmp_path / "e2")
    (e2.OUT / "noise_inputs").mkdir(parents=True)
    (e2.OUT / "release.json").write_text("{}")
    task = dict(
        index=0, seed=6142, stream="model/code", output=str(tmp_path / "trajectory")
    )
    state = Path(task["output"]) / "states/s"
    rows = []
    for i in range(3):
        checkpoints = {h: str(state / f"k{i}" / h) for h in ("t1", "t2")}
        rows.append(
            dict(
                state_id="s",
                generation=0,
                candidate_id=f"k{i}",
                batch_hash=f"b{i}",
                trajectory_id="t",
                checkpoints=checkpoints,
            )
        )
    for folder in [
        state / "parent",
        *[Path(r["checkpoints"][h]) for r in rows for h in ("t1", "t2")],
    ]:
        folder.mkdir(parents=True)
        (folder / "state.json").write_text("{}")
        (folder / "adapter_model.safetensors").write_bytes(b"fixture")
    return task, rows


def test_state_receipt_binds_retained_weights_and_rejects_mutation(
    tmp_path, monkeypatch
):
    task, rows = make_state(tmp_path, monkeypatch)
    e2.publish_noise_input(task, rows)
    monkeypatch.setattr(e2, "prepared", lambda: dict(replication_tasks=[task]))
    assert e2.load_noise_input(0)["state_id"] == "s"
    (Path(rows[0]["checkpoints"]["t1"]) / "adapter_model.safetensors").write_bytes(
        b"changed"
    )
    with pytest.raises(RuntimeError, match="files changed"):
        e2.load_noise_input(0)


def test_pruned_checkpoint_and_duplicate_composition_stop(tmp_path, monkeypatch):
    task, rows = make_state(tmp_path, monkeypatch)
    weight = Path(rows[0]["checkpoints"]["t1"]) / "adapter_model.safetensors"
    weight.unlink()
    with pytest.raises(RuntimeError, match="Retained checkpoint missing"):
        e2.publish_noise_input(task, rows)
    rows[1]["batch_hash"] = rows[0]["batch_hash"]
    with pytest.raises(RuntimeError, match="composition mismatch"):
        e2.publish_noise_input(task, rows)


def relay_fixture(tmp_path, monkeypatch):
    out = tmp_path / "runs/a2/empirical_v1/checkpoint_fallback_v2"
    out.mkdir(parents=True)
    (out / "release.json").write_text(
        json.dumps(
            dict(
                files={},
                prerequisites={},
                scope=dict(deadline_utc="2099-10-11T18:29:59+00:00", gpu_cap=20),
            )
        )
    )
    for kind in ("cpu", "cuda"):
        (out / f"{kind}_passed.json").write_text(
            json.dumps(
                dict(status="passed", release_sha256=file_digest(out / "release.json"))
            )
        )
    args = e2.arguments("trajectory", 0)
    args[-3] = str(tmp_path / "tools/empirical_checkpoint_gpu_v2.slurm")
    request = dict(arguments=args, cwd=str(tmp_path), environment={})
    monkeypatch.setattr(
        relay.subprocess, "check_output", lambda *a, **kw: "gh|PENDING|19\n"
    )
    return out, request


def test_relay_counts_recoveries_pending_jobs_and_rejects_over_cap(
    tmp_path, monkeypatch
):
    _, request = relay_fixture(tmp_path, monkeypatch)
    relay.validate(request, tmp_path)
    monkeypatch.setattr(
        relay.subprocess,
        "check_output",
        lambda *a, **kw: "gh|PENDING|19\ngh-dev|RUNNING|1\n",
    )
    with pytest.raises(ValueError, match="cap reached"):
        relay.validate(request, tmp_path)


def test_relay_rejects_failed_validation_and_unbounded_options(tmp_path, monkeypatch):
    out, request = relay_fixture(tmp_path, monkeypatch)
    request["arguments"].insert(0, "--wrap=python bad.py")
    with pytest.raises(ValueError, match="Bounded"):
        relay.validate(request, tmp_path)
    request["arguments"].pop(0)
    (out / "cuda_passed.json").write_text(
        json.dumps(
            dict(status="failed", release_sha256=file_digest(out / "release.json"))
        )
    )
    with pytest.raises(ValueError, match="validations required"):
        relay.validate(request, tmp_path)


def test_wrapper_preserves_native_labeling_and_launches_only_fixed_generation(
    tmp_path, monkeypatch
):
    import third_eye.experiments.labeling as labeling

    task = dict(
        index=0,
        seed=6142,
        config="config",
        manifest="manifest",
        output=str(tmp_path / "new"),
    )
    monkeypatch.setattr(e2, "OUT", tmp_path)
    (tmp_path / "release.json").write_text("{}")
    (tmp_path / "noise_submissions").mkdir()
    monkeypatch.setattr(e2, "frozen", lambda: {})
    monkeypatch.setattr(e2, "passed", lambda k: None)
    monkeypatch.setattr(e2, "require_gpu_allocation", lambda: None)
    monkeypatch.setattr(e2, "arm_deadline", lambda s: None)
    monkeypatch.setattr(e2, "prepared", lambda: dict(replication_tasks=[task]))
    events = []

    class Original:
        def label_state(self, generation, history=()):
            events.append(("native", generation, history))
            return [generation]

    monkeypatch.setattr(labeling, "LabelGenerator", Original)
    monkeypatch.setattr(
        e2, "publish_noise_input", lambda t, rs: events.append(("publish", rs))
    )
    monkeypatch.setattr(relay, "submit", lambda args: "12345\n")

    def native_run(path, run_name):
        labeler = labeling.LabelGenerator()
        assert labeler.label_state(0, ["h"]) == [0]
        assert labeler.label_state(1) == [1]
        assert "--prune-branches" not in e2.sys.argv and e2.sys.argv[-2:] == [
            "--keep-accepted",
            "5",
        ]

    monkeypatch.setattr(e2.runpy, "run_path", native_run)
    old_argv = e2.sys.argv
    try:
        e2.run(0)
    finally:
        e2.sys.argv = old_argv
    assert events == [("native", 0, ["h"]), ("publish", [0]), ("native", 1, ())]
    assert labeling.LabelGenerator is Original
    assert (
        json.loads((tmp_path / "noise_submissions/task_0.json").read_text())["job_id"]
        == "12345"
    )


def test_full_noise_measurement_terminal_controls_keep_m2_equal_m1(
    tmp_path, monkeypatch
):
    from empirical_noise_review_v1 import inspect

    monkeypatch.setattr(noise, "OUT", tmp_path)
    (tmp_path / "release.json").write_text("{}")
    monkeypatch.setattr(
        noise,
        "frozen",
        lambda: dict(
            fixed_optimizer_seed_offset=50000,
            fresh_continuation_seed_offsets=[50000000, 100000000],
        ),
    )
    monkeypatch.setattr(noise, "passed", lambda k: None)
    monkeypatch.setattr(noise, "arm_deadline", lambda s: None)
    monkeypatch.setattr(noise, "require_gpu_allocation", lambda: None)
    rows = [
        dict(
            candidate_id=f"k{i}",
            batch_hash=f"b{i}",
            candidate_adapter_hash=f"m1_{i}",
            continuation_adapter_hash=f"m1_{i}",
            continuation_available=0,
            checkpoints=dict(t1=f"m1_{i}", t2=f"m1_{i}"),
            runtime=dict(candidate=dict(seed=300000)),
            continuation=dict(seed=600000),
            evaluation=dict(
                parent=dict(target=0, ood=0, retention=0),
                t1=dict(target=0, ood=0, retention=0),
                t2=dict(target=0, ood=0, retention=0),
            ),
        )
        for i in range(3)
    ]
    task = dict(
        index=0,
        stream="model/code",
        generation=0,
        trajectory_id="t",
        state_id="s",
        selected_records=rows,
    )
    monkeypatch.setattr(noise, "load_noise_input", lambda i: task)

    class Saved:
        value = "parent"
        steps = 0

        def snapshot(self):
            return self.value

        def restore(self, value):
            self.value = value

        def state_hash(self):
            return self.value

        def load_checkpoint(self, value):
            self.value = value

        def save_checkpoint(self, path, metadata):
            path.mkdir()

        def train(self, items, seed, log_path):
            assert items, "Terminal condition executed an optimizer update"
            self.steps += 50
            self.value = "updated"
            return dict(optimizer_steps=50)

    b = Saved()
    p = SimpleNamespace(
        seed=6142, candidate_size=2, candidate_sampling="stratified_distinct"
    )
    monkeypatch.setattr(
        noise,
        "initialize",
        lambda t, o: (SimpleNamespace(protocol=p), {"train": []}, None, b),
    )
    monkeypatch.setattr(
        noise,
        "batch",
        lambda t, r, continuation=False: [] if continuation else ["a", "b"],
    )
    monkeypatch.setattr(
        noise,
        "collect_corrections",
        lambda *args: (
            [],
            dict(
                harvest_complete=True,
                correction_attempt_limit=8,
                correction_seed_stride=16,
            ),
        ),
    )

    def collect(backend, *args):
        return dict(
            adapter_hash=backend.state_hash(),
            metrics=dict(target=0, ood=0, retention=0),
            items={
                r: [
                    dict(
                        id="one", prompt_sha256="p", prediction_sha256="pred", correct=0
                    )
                ]
                for r in noise.ROLES
            },
        )

    monkeypatch.setattr(noise, "collect_items", collect)
    noise.measure(0)
    record = json.loads((tmp_path / "noise/task_0/completed.json").read_text())
    reviewed = inspect(record)
    assert b.steps == 150 and reviewed["fresh_terminal_fraction"] == 1
    assert reviewed["fixed_continuation_optimizer_reversal"] == 0
    for c in record["candidates"]:
        assert len(c["conditions"]) == 6
        assert (
            c["conditions"]["optimizer_m2"]["adapter_hash"]
            == c["conditions"]["original_m1"]["adapter_hash"]
        )
