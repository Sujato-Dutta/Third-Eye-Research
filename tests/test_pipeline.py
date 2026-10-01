"""Offline end-to-end benchmark, planning and reporting fixtures."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from third_eye.config import Config
from third_eye.data.benchmarks import prepare, MATH_SUBJECTS
from third_eye.data.schema import Example
from third_eye.data.splits import load_manifest
from third_eye.evaluation.final import load_final_manifest
from third_eye.evaluation.benchmarks import BenchmarkVerifier
from third_eye.experiments.features import extract_features
from third_eye.io import write_json
from plan_study import build_plan
from verify_backend import make_tiny_backend


def fixture_dataset(repo, config, split, revision):
    assert revision == "a" * 40
    prefix = f"{repo} {config} {split}"
    if repo == "openai/gsm8k":
        return [
            {"question": f"{prefix} distinct problem {i}", "answer": "#### 2"}
            for i in range(10)
        ]
    if repo == "EleutherAI/hendrycks_math":
        return [
            {
                "problem": f"{prefix} distinct symbol problem {i}",
                "solution": r"\boxed{\frac{1}{2}}",
                "type": config,
                "level": "Level 2",
            }
            for i in range(3)
        ]
    if repo == "HuggingFaceH4/MATH-500":
        return [
            {
                "problem": f"{prefix} distinct final math {i}",
                "answer": "1/2",
                "subject": "algebra",
            }
            for i in range(3)
        ]
    if repo == "google-research-datasets/mbpp":
        offset = {"train": 0, "validation": 100, "test": 200}[split]
        return [
            {
                "task_id": offset + i,
                "text": f"{prefix} distinct code {i}",
                "code": "def f(x): return x+1",
                "test_list": ["assert f(1)==2"],
                "test_setup_code": "",
            }
            for i in range(10)
        ]
    if repo == "openai/openai_humaneval":
        return [
            {
                "task_id": str(i),
                "prompt": f'def f{i}(x):\n    """Return x."""\n',
                "canonical_solution": "    return x",
                "test": "def check(candidate):\n    assert candidate(1)==1",
                "entry_point": f"f{i}",
            }
            for i in range(3)
        ]
    return [
        {
            "question": f"{prefix} distinct retention {i}",
            "choices": ["yes", "no", "maybe", "none"],
            "answer": 0,
            "subject": "knowledge",
        }
        for i in range(10)
    ]


@pytest.mark.parametrize("task", ["math", "code"])
def test_prepare_portable_suites_and_final_isolation(tmp_path, monkeypatch, task):
    from types import SimpleNamespace

    monkeypatch.setattr("datasets.load_dataset", fixture_dataset)
    monkeypatch.setattr(
        "huggingface_hub.HfApi.dataset_info",
        lambda *a, **k: SimpleNamespace(sha="a" * 40),
    )
    out = tmp_path / "original"
    selection = prepare(out, task, train_size=4, dev_size=2, retention_size=3)
    manifest, splits = load_manifest(selection)
    assert set(splits) == {"train", "target_dev", "ood_dev", "retention_dev"}
    assert all(not Path(v["path"]).is_absolute() for v in manifest["splits"].values())
    final, tests = load_final_manifest(out / "final.json", selection)
    assert set(tests) == {"target_test", "ood_test", "retention_test"}
    assert all(ex.split.endswith("_test") for rows in tests.values() for ex in rows)
    relocated = tmp_path / "relocated"
    shutil.copytree(out, relocated)
    load_manifest(relocated / "selection.json")
    load_final_manifest(relocated / "final.json", relocated / "selection.json")
    with pytest.raises(ValueError, match="split manifest"):
        load_manifest(out / "final.json")
    assert len(MATH_SUBJECTS) == 7


def test_actual_symbolic_math_equivalence():
    ex = Example(
        "m",
        "math prompt",
        r"\frac{1}{2}",
        "ood_dev",
        metadata={"verifier": "symbolic_math"},
    )
    verifier = BenchmarkVerifier()
    assert verifier.verify(ex, r"\boxed{0.5}")
    assert not verifier.verify(ex, r"\boxed{2}")


def test_actual_diagnostics_and_extended_probe_restore_parent(tmp_path):
    from third_eye.data.schema import Correction

    b = make_tiny_backend()
    batch = [
        Correction(Example("a", "one plus one", "#### 2", "train"), "#### 2", 1),
        Correction(Example("b", "one plus two", "#### 3", "train"), "#### 3", 2),
    ]
    anchor = [Example("anchor", "two plus two", "#### 4", "retention_dev")]
    before = b.state_hash()
    features = extract_features(
        b, batch, anchor, b.config.protocol, 42, tmp_path / "probe.jsonl"
    )
    assert b.state_hash() == before
    assert features["completion_tokens_mean"] > 0
    assert -1.0001 <= features["retention_gradient_cosine"] <= 1.0001
    assert features["probe_anchor_kl"] >= 0
    assert features["probe_adapter_delta_norm"] > 0


def test_frozen_study_plan_and_transfer_gates(tmp_path):
    paths = []
    for i, model in enumerate(
        ("Qwen/Qwen3-4B", "meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen3-8B")
    ):
        cfg = Config()
        cfg = replace(
            cfg,
            model=replace(cfg.model, name=model, revision="a" * 40),
            protocol=replace(cfg.protocol, status="frozen", depth=5),
        )
        path = tmp_path / f"config{i}.json"
        write_json(path, cfg.to_dict())
        paths.append(path)
    plan_path = build_plan(
        paths[:2],
        "data/processed/v1",
        tmp_path / "plan",
        "runs/fixture",
        label_trajectories=1,
        transfer_configs=paths[2:],
    )
    plan = json.loads(plan_path.read_text())
    assert len(plan["stages"]["labels"]) == 12
    assert len(plan["stages"]["online_core"]) == 96
    assert all(t["requires_gates"] for t in plan["stages"]["online_transfer"])
    assert all("--gate2" in t["commands"][0] for t in plan["stages"]["transfer_labels"])
    assert any(
        "--pool-order-seed" in t["commands"][0] for t in plan["stages"]["stress"]
    )
    assert len(plan["stages"]["reports"][0]["commands"]) == 2


def test_result_cli_separates_final_evidence_and_creates_plots(tmp_path):
    runs = []
    finals = []
    cfg = Config()
    cfg = replace(cfg, protocol=replace(cfg.protocol, status="frozen", depth=1))
    for seed in (42, 43, 44):
        for policy, gain in (("greedy_h1", 0.01), ("direct", 0.03)):
            directory = tmp_path / f"{policy}_{seed}"
            metadata = {
                "config": cfg.to_dict(),
                "resolved_revision": "a" * 40,
                "pilot": False,
            }
            points = [
                {
                    "generation": 0,
                    "scores": {"target": 0.4, "ood": 0.3, "retention": 0.6},
                    "utility": 0.433333333,
                },
                {
                    "generation": 1,
                    "scores": {
                        "target": 0.4 + gain,
                        "ood": 0.3 + gain,
                        "retention": 0.6,
                    },
                    "utility": 0.433333333 + gain,
                    "seconds": 1.0,
                },
            ]
            write_json(directory / "run.json", metadata)
            write_json(
                directory / "trajectory.json",
                {
                    "points": points,
                    "complete": True,
                    "seed": seed,
                    "policy": policy,
                    "model": "fixture-model",
                    "manifest_hash": "fixture-data",
                },
            )
            final = directory / "final"
            write_json(
                final / "scores.json",
                {
                    "scores": {
                        "target_test": 0.5 + gain,
                        "ood_test": 0.4 + gain,
                        "retention_test": 0.6,
                    },
                    "seed": seed,
                    "policy": policy,
                    "model": "fixture-model",
                    "final_manifest_hash": "fixture-final",
                    "config_hash": "cfg",
                    "resolved_revision": "a" * 40,
                    "utility_weights": [1 / 3] * 3,
                },
            )
            items = [
                {"id": role, "role": role, "passed": policy == "direct"}
                for role in ("target_test", "ood_test", "retention_test")
            ]
            (final / "items.jsonl").write_text(
                "\n".join(json.dumps(r) for r in items), encoding="utf-8"
            )
            runs.append(str(directory))
            finals.append(str(final))
    out = tmp_path / "report"
    subprocess.run(
        [
            sys.executable,
            "experiments/report_results.py",
            "--runs",
            *runs,
            "--final-evaluations",
            *finals,
            "--output",
            str(out),
        ],
        check=True,
    )
    report = json.loads((out / "report.json").read_text())
    assert report["final_benchmark_macro_comparisons"]["direct"]["mean"] > 0
    assert len(report["final_item_tests"]) == 9
    assert (out / "final_benchmarks.csv").exists()
    assert (out / "trajectories.png").exists()


def test_forecast_report_cli_compares_identical_held_out_states(tmp_path):
    import torch
    from test_forecasting import records
    from third_eye.forecasting.dataset import split_records
    from third_eye.forecasting.training import train_forecaster

    torch.set_num_threads(1)
    parts = split_records(records())
    direct = tmp_path / "direct"
    h1 = tmp_path / "one_step"
    train_forecaster(parts, direct, epochs=1, hidden=8)
    train_forecaster(parts, h1, kind="one_step", horizon=1, epochs=1, hidden=8)
    output = tmp_path / "forecast_report"
    subprocess.run(
        [
            sys.executable,
            "experiments/report_forecasts.py",
            "--models",
            str(direct),
            str(h1),
            "--baseline",
            str(direct),
            "--output",
            str(output),
        ],
        check=True,
    )
    report = json.loads((output / "report.json").read_text())
    assert report["evaluation_horizon"] == 2
    assert len(report["rows"]) == 2
    assert report["independent_units"] == "recursive trajectories"
    assert (output / "forecasting.csv").exists()
    assert (output / "consequences.png").exists()
