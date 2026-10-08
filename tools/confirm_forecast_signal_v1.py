"""One frozen held-out confirmation; no neural training or policy launch."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools")]
import numpy as np
import torch

import diagnose_forecast_signal_v1 as study
from extensions.forecasting_f2.model import build_model
from third_eye.forecasting.dataset import read_records, split_records
from third_eye.io import file_digest, write_json
from fit_forecaster_f2_v1 import SNAPSHOT, frozen

OUT = ROOT / "runs/a2/signal_confirmation_v1"


def predict(folder, rows):
    meta = json.loads((folder / "metadata.json").read_text())
    if meta["role"] != "diagnostic_only_not_an_online_forecaster":
        raise ValueError("Unknown diagnostic artifact")
    if file_digest(folder / "diagnostic_weights.pt") != meta["weights_sha256"]:
        raise ValueError("Diagnostic weight checksum differs")
    arrays, ix = study.group_inputs(rows)
    arrays = list(arrays)
    arrays[1] = np.clip(
        (arrays[1] - np.array(meta["feature_mean"], dtype=np.float32))
        / np.array(meta["feature_scale"], dtype=np.float32),
        -20,
        20,
    )
    model = build_model("independent")
    model.load_state_dict(
        torch.load(folder / "diagnostic_weights.pt", weights_only=True)
    )
    model.eval()
    with torch.inference_mode():
        p = model(*[torch.tensor(a, dtype=torch.float32) for a in arrays])[
            "prediction"
        ].numpy()
    p = p * np.array(meta["target_scale"], dtype=np.float32)
    result = np.empty((len(rows), 3))
    result[ix.reshape(-1)] = p.reshape(-1, 3)
    return result


def assert_held_out(development, test):
    if not test:
        raise ValueError("No held-out population")
    for key in ["trajectory_id", "state_id"]:
        if {r[key] for r in development} & {r[key] for r in test}:
            raise ValueError("Held-out selection leakage")


def main():
    torch.set_num_threads(2)
    frozen()
    own = json.loads((ROOT / "runs/a2/signal_confirmation_release_v1.json").read_text())
    assert all(file_digest(ROOT / p) == h for p, h in own["files"].items())
    assert all(
        file_digest(ROOT / p) == h for p, h in own["diagnostic_artifact_files"].items()
    )
    study.checked_parts()
    assert not OUT.exists(), "Held-out study cannot be silently repeated"
    diagnostic = study.OUT
    development = json.loads((diagnostic / "review.json").read_text())
    assert development["status"] == "signal_review_required"
    manifest = json.loads((SNAPSHOT / "snapshot_manifest.json").read_text())
    paths = [
        SNAPSHOT / "snapshot" / f"task_{t['index']}" / "meta_labels.jsonl"
        for t in manifest["trajectories"]
    ]
    all_parts = split_records(read_records(paths), 42)
    assert {
        p: sorted({r["trajectory_id"] for r in rs}) for p, rs in all_parts.items()
    } == manifest["trajectory_assignments"]
    assert_held_out(all_parts["train"] + all_parts["validation"], all_parts["test"])
    # Annotate using the same immutable stream mapping; do not fit any parameters.
    all_parts = study.annotate(all_parts)
    rows = all_parts["test"]
    assert len(study.groups(rows)) == 40
    OUT.mkdir(parents=True)
    tasks = [
        {"target": target, "seed": seed, "trajectories": 20}
        for target in study.TARGETS
        for seed in study.SEEDS
    ]
    weights = {
        study.task_name(t): file_digest(
            diagnostic / "fits" / study.task_name(t) / "diagnostic_weights.pt"
        )
        for t in tasks
    }
    # Commit scope and frozen weights before any numerical held-out evaluation.
    write_json(
        OUT / "predeclared_confirmation.json",
        {
            "release": own,
            "scope": "Nine full-training independent GRUs, six fixed ridge controls, two heuristics",
            "tasks": tasks,
            "weights": weights,
            "test_trajectory_ids": sorted({r["trajectory_id"] for r in rows}),
            "snapshot_sha256": file_digest(SNAPSHOT / "snapshot_manifest.json"),
            "development_report_sha256": file_digest(diagnostic / "review.json"),
            "new_neural_fits": 0,
            "deterministic_ridge_replays": 6,
            "method_selection_from_test": False,
            "gate2_unchanged_and_development_gate_still_failed": True,
        },
    )
    results = {}
    for t in tasks:
        name = study.task_name(t)
        folder = diagnostic / "fits" / name
        meta = json.loads((folder / "metadata.json").read_text())
        assert meta["target"] == t["target"] and meta["seed"] == t["seed"]
        assert not set(meta["training_trajectory_ids"]) & {
            r["trajectory_id"] for r in rows
        }
        saved = np.array(
            json.loads((folder / "validation_predictions.json").read_text())
        )
        assert np.array_equal(saved, predict(folder, all_parts["validation"])), (
            "Validation reload differs before test evaluation"
        )
        pred = predict(folder, rows)
        results[name] = study.evaluate_fit(rows, pred, t["target"])
        write_json(OUT / f"{name}_predictions.json", pred.tolist())
    for target in study.TARGETS:
        for relative in [False, True]:
            name = f"ridge_{'relative' if relative else 'absolute'}_{target}_n20"
            pred = study.ridge(all_parts["train"], rows, target, relative)
            results[name] = study.evaluate_fit(rows, pred, target)
    for name, pred in [
        ("random_tied", np.zeros((len(rows), 3))),
        (
            "heuristic_low_nll",
            np.repeat(
                np.array(
                    [-r["precommit"]["features"].get("pre_update_nll", 0) for r in rows]
                )[:, None],
                3,
                axis=1,
            ),
        ),
    ]:
        results[name] = study.evaluate_fit(rows, pred, "future")
    write_json(
        OUT / "signal_decomposition.json", study.diagnostic_data({"heldout": rows})
    )
    paired = study.paired_review({"validation": rows}, results)
    write_json(
        OUT / "review.json",
        {
            "status": "heldout_signal_review_required",
            "fits": results,
            "paired_comparisons": paired,
            "new_neural_fits": 0,
            "deterministic_ridge_replays": 6,
            "test_evaluated": True,
            "method_selection_from_test": False,
            "development_gate2_still_failed": True,
            "online_submitted": False,
            "limitations": [
                "Eight held-out trajectories",
                "Conditional small-model protocol",
                "Aggregate phenomenon previously reported on original label population",
                "No new independent continuation draws",
                "No universal noise ceiling",
            ],
        },
    )
    print(
        json.dumps(
            {"status": "heldout_signal_review_required", "paired_comparisons": paired}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
