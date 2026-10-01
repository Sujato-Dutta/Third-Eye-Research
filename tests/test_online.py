import pytest
from test_labeling import setup_labeler
from third_eye.evaluation.runner import Consequences
from third_eye.experiments.online import run_online


@pytest.mark.parametrize(
    "policy,updates",
    [("no_update", 0), ("random", 1), ("heuristic", 1), ("greedy_h1", 3)],
)
def test_policy_executes_only_its_actual_update_budget(
    tmp_path, monkeypatch, policy, updates
):
    labeler = setup_labeler(tmp_path, monkeypatch)

    def evaluation(backend, *args):
        value = sum(backend.state) / 100
        return Consequences(value, value / 2, -value / 3)

    monkeypatch.setattr("third_eye.experiments.online.evaluate", evaluation)
    result = run_online(
        labeler.backend,
        labeler.config,
        labeler.splits,
        labeler.verifier,
        tmp_path,
        labeler.manifest_hash,
        policy,
    )
    assert len(labeler.backend.starts) == updates
    assert len(labeler.backend.state) == int(policy != "no_update")
    assert result["complete"]
    assert not (tmp_path / "meta_labels.jsonl").exists()
    assert (tmp_path / "accepted/generation_1/resume.json").exists()


def test_online_error_restores_parent(tmp_path, monkeypatch):
    labeler = setup_labeler(tmp_path, monkeypatch, fail_call=1)
    monkeypatch.setattr(
        "third_eye.experiments.online.evaluate", lambda *args: Consequences(0, 0, 0)
    )
    with pytest.raises(RuntimeError, match="synthetic training failure"):
        run_online(
            labeler.backend,
            labeler.config,
            labeler.splits,
            labeler.verifier,
            tmp_path,
            labeler.manifest_hash,
            "random",
        )
    assert labeler.backend.state == ()
    assert not (tmp_path / "accepted").exists()


def test_resume_keeps_complete_baseline_trajectory(tmp_path, monkeypatch):
    import json

    labeler = setup_labeler(tmp_path / "part1", monkeypatch, depth=3)

    def evaluation(backend, *args):
        value = sum(backend.state) / 100
        return Consequences(value, value / 2, -value / 3)

    monkeypatch.setattr("third_eye.experiments.online.evaluate", evaluation)
    first = run_online(
        labeler.backend,
        labeler.config,
        labeler.splits,
        labeler.verifier,
        tmp_path / "part1",
        labeler.manifest_hash,
        "random",
        generations=1,
    )
    resume = json.loads(
        (tmp_path / "part1/accepted/generation_1/resume.json").read_text()
    )
    second = run_online(
        labeler.backend,
        labeler.config,
        labeler.splits,
        labeler.verifier,
        tmp_path / "part2",
        labeler.manifest_hash,
        "random",
        start_generation=1,
        history=resume["history"],
        generations=2,
        prior_points=first["points"],
    )
    assert [p["generation"] for p in second["points"]] == [0, 1, 2, 3]
    assert second["points"][0] == first["points"][0]
    assert len(labeler.backend.state) == 3


def test_learned_policy_connects_forecaster_without_future_labels(
    tmp_path, monkeypatch
):
    import torch
    from test_forecasting import records
    from third_eye.forecasting.dataset import split_records
    from third_eye.forecasting.training import train_forecaster

    torch.set_num_threads(1)
    f = train_forecaster(
        split_records(records()), tmp_path / "forecaster", epochs=1, hidden=8
    )
    labeler = setup_labeler(tmp_path / "online", monkeypatch)
    monkeypatch.setattr(
        "third_eye.experiments.online.evaluate",
        lambda *args: Consequences(0.4, 0.2, 0.6),
    )
    result = run_online(
        labeler.backend,
        labeler.config,
        labeler.splits,
        labeler.verifier,
        tmp_path / "online",
        labeler.manifest_hash,
        "direct",
        forecaster=f,
    )
    assert result["complete"] and len(labeler.backend.starts) == 1
    assert (tmp_path / "online/online/generation_0/forecast.json").exists()
    f.metadata["horizon"] = 1
    with pytest.raises(ValueError, match="horizon"):
        run_online(
            labeler.backend,
            labeler.config,
            labeler.splits,
            labeler.verifier,
            tmp_path / "invalid",
            labeler.manifest_hash,
            "direct",
            forecaster=f,
        )
