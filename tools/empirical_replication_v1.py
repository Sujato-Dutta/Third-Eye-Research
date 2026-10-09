# ruff: noqa: E402
"""Freeze all existing methods before the eight balanced held-out trajectories."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools")]
from empirical_common_v1 import BASE, SNAPSHOT, checked, prepared, snapshot_parts
from empirical_noise_v1 import frozen as noise_frozen
from empirical_comparator_v1 import frozen as comparator_frozen
import diagnose_forecast_signal_v1 as study
import confirm_forecast_signal_v1 as confirmation
from third_eye.config import Config
from third_eye.cluster import require_gpu_allocation
from third_eye.io import file_digest, write_json
from third_eye.statistics.metrics import ranking


def frozen():
    checked()
    own = json.loads((BASE / "replication_release.json").read_text())
    for name, sha in own["files"].items():
        if file_digest(ROOT / name) != sha:
            raise RuntimeError("Replication release changed")
    for name, sha in own["prerequisite_release_sha256"].items():
        if file_digest(BASE / name) != sha:
            raise RuntimeError("Empirical prerequisite changed")
    noise_frozen()
    comparator_frozen()
    return own


def ridge_fit(rows, target, relative):
    arrays, ix = study.group_inputs(rows)
    x = np.concatenate([a.reshape(len(ix), 3, -1) for a in arrays], 2).astype(float)
    mean, scale = x.mean((0, 1)), np.maximum(x.std((0, 1)), 1e-6)
    x = np.clip((x - mean) / scale, -20, 20)
    y = study.targets(rows, target)[ix]
    base = y.mean((0, 1))
    if relative:
        x = x - x.mean(1, keepdims=True)
        y = y - y.mean(1, keepdims=True)
    else:
        y = y - base
    x = x.reshape(-1, x.shape[-1])
    beta = np.linalg.solve(x.T @ x + np.eye(x.shape[1]), x.T @ y.reshape(-1, 3))
    return dict(
        target=target,
        relative=relative,
        mean=mean.tolist(),
        scale=scale.tolist(),
        base=base.tolist(),
        beta=beta.tolist(),
    )


def ridge_predict(meta, rows):
    arrays, ix = study.group_inputs(rows)
    x = np.concatenate([a.reshape(len(ix), 3, -1) for a in arrays], 2).astype(float)
    x = np.clip((x - np.array(meta["mean"])) / np.array(meta["scale"]), -20, 20)
    if meta["relative"]:
        x = x - x.mean(1, keepdims=True)
    p = x @ np.array(meta["beta"]) + np.array(meta["base"])
    result = np.empty((len(rows), 3))
    result[ix.reshape(-1)] = p.reshape(-1, 3)
    return result


def freeze_methods():
    own = frozen()
    if any(Path(t["output"]).exists() for t in prepared()["replication_tasks"]):
        raise RuntimeError("Methods must be frozen before any replication outcome")
    completed = json.loads((BASE / "comparator/completed.json").read_text())
    if completed["status"] != "comparator_frozen_for_fresh_replication":
        raise RuntimeError("Comparator fitting must finish before replication")
    output = BASE / "methods"
    output.mkdir(exist_ok=False)
    _, parts = snapshot_parts()
    models = []
    artifact_hashes = {}
    populations = [
        ("original", SNAPSHOT / "study/forecasters", "*/metadata.json", 18),
        ("f2", ROOT / "runs/a2/forecaster_f2_v1/models", "*/metadata.json", 15),
        ("diagnostic", study.OUT / "fits", "gru_*_n20_seed*/metadata.json", 9),
    ]
    for family, parent, pattern, count in populations:
        files = sorted(parent.glob(pattern))
        if len(files) != count:
            raise RuntimeError(
                f"Expected {count} existing {family} fits, found {len(files)}"
            )
        for path in files:
            meta = json.loads(path.read_text())
            folder = path.parent
            files_to_bind = [
                path,
                folder
                / ("diagnostic_weights.pt" if family == "diagnostic" else "weights.pt"),
            ]
            if family == "f2":
                files_to_bind.append(folder / "artifact_manifest.json")
            if family == "diagnostic":
                files_to_bind.append(folder / "validation_predictions.json")
            for p in files_to_bind:
                artifact_hashes[str(p)] = file_digest(p)
            models.append(
                dict(
                    name=family + "_" + folder.name,
                    family=family,
                    folder=str(folder),
                    target=meta.get(
                        "target", "immediate" if meta.get("horizon") == 1 else "future"
                    ),
                    scalar=bool(meta.get("scalar", False)),
                )
            )
    ridge = {}
    for target in study.TARGETS:
        for relative in [False, True]:
            name = f"ridge_{'relative' if relative else 'absolute'}_{target}"
            meta = ridge_fit(parts["train"], target, relative)
            if not np.array_equal(
                ridge_predict(meta, parts["validation"]),
                study.ridge(parts["train"], parts["validation"], target, relative),
            ):
                raise RuntimeError("Fixed ridge coefficient replay differs")
            ridge[name] = meta
    write_json(output / "ridge.json", ridge)
    artifact_hashes[str(output / "ridge.json")] = file_digest(output / "ridge.json")
    for target in study.TARGETS:
        folder = BASE / "comparator/models" / target
        for p in sorted(folder.iterdir()):
            if p.is_file():
                artifact_hashes[str(p)] = file_digest(p)
    manifest = dict(
        status="methods_frozen_before_replication",
        models=models,
        artifact_sha256=artifact_hashes,
        ridge_models=sorted(ridge),
        tree_targets=list(study.TARGETS),
        primary_comparison="original_direct_seed42 versus original_matched_h1_seed42 on future utility",
        secondary="All existing fits, seeds and ablations; no selection using replication outcomes",
        release_sha256=file_digest(BASE / "replication_release.json"),
        base_source_sha256=own["base_source_sha256"],
    )
    write_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            dict(
                status=manifest["status"],
                existing_neural_fits=len(models),
                ridge_models=6,
                tree_models=3,
            )
        ),
        flush=True,
    )


def methods():
    frozen()
    result = json.loads((BASE / "methods/manifest.json").read_text())
    if result["release_sha256"] != file_digest(
        BASE / "replication_release.json"
    ) or any(file_digest(p) != h for p, h in result["artifact_sha256"].items()):
        raise RuntimeError("Frozen method artifact changed")
    return result


def run(index):
    require_gpu_allocation()
    methods()
    receipt = json.loads((BASE / "replication_science_review.json").read_text())
    report = BASE / "preliminary_noise_review/review.json"
    if (
        receipt["decision"] != "release_balanced_replication"
        or receipt["noise_report_sha256"] != file_digest(report)
        or receipt["methods_manifest_sha256"]
        != file_digest(BASE / "methods/manifest.json")
    ):
        raise RuntimeError("A recorded preliminary science review is required")
    task = prepared()["replication_tasks"][index]
    if Path(task["output"]).exists():
        raise RuntimeError("Replication output must be new; retain any failed run")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments/run.py"),
            "--config",
            task["config"],
            "--manifest",
            task["manifest"],
            "--output",
            task["output"],
            "--mode",
            "labels",
            "--policy",
            "random",
            "--generations",
            "5",
        ],
        check=True,
    )


def stratified_interval(values, rows):
    by_stream = defaultdict(lambda: defaultdict(list))
    for value, g in zip(values, study.groups(rows)):
        if value is not None and np.isfinite(value):
            r = rows[g[0]]
            by_stream[r["analysis_stream"]][r["trajectory_id"]].append(float(value))
    rng = np.random.default_rng(42)
    boots = []
    means = []
    for stream, trajs in sorted(by_stream.items()):
        unit = np.array([np.mean(v) for v in trajs.values()])
        means.append(unit.mean())
        boots.append(rng.choice(unit, (5000, len(unit)), replace=True).mean(1))
    if not boots:
        return dict(mean=None, ci95=None)
    return dict(
        mean=float(np.mean(means)),
        ci95=np.quantile(np.mean(boots, axis=0), [0.025, 0.975]).tolist(),
        covered_streams=len(by_stream),
        covered_trajectories=sum(len(t) for t in by_stream.values()),
        resampling="5000 within-stream trajectory bootstrap draws, equal stream means, seed42",
    )


def future_selection(rows, prediction):
    one = study.targets(rows, "immediate").mean(1)
    two = study.targets(rows, "future").mean(1)
    score = prediction[:, 0] if prediction.shape[1] == 1 else prediction.mean(1)
    values = defaultdict(list)
    for g in study.groups(rows):
        s = np.round(score[g], 12)
        selected = np.flatnonzero(s == s.max())
        future = float(two[g][selected].mean())
        random = float(two[g].mean())
        values["future_utility"].append(future)
        values["greedy_policy_minus_random"].append(future - random)
        values["forced_future_oracle_regret"].append(float(two[g].max() - future))
        values["negative_future_utility_rate"].append(
            float(np.mean(two[g][selected] < 0))
        )
        values["h1_observed_utility"].append(float(one[g][selected].mean()))
    return {k: stratified_interval(v, rows) for k, v in values.items()}


def confirm():
    import torch
    from a2_after_labels import audit_labels
    from third_eye.forecasting.training import Forecaster
    from extensions.forecasting_f2.training import ContrastForecaster
    from empirical_comparator_v1 import predict as tree_predict
    from empirical_cached_v1 import selection, mechanism

    torch.set_num_threads(2)
    manifest = methods()
    tasks = prepared()["replication_tasks"]
    output = BASE / "replication_confirmation"
    output.mkdir(exist_ok=False)
    rows = []
    coverage = []
    for task in tasks:
        out = Path(task["output"])
        p = out / "completed.json"
        if not p.exists():
            coverage.append(
                dict(
                    index=task["index"],
                    stream=task["stream"],
                    status="incomplete",
                    states=0,
                )
            )
            continue
        completed = json.loads(p.read_text())
        cfg = Config.load(task["config"])
        rs = audit_labels(out, cfg, completed)
        for r in rs:
            r["analysis_stream"] = task["stream"]
        rows.extend(rs)
        coverage.append(
            dict(
                index=task["index"],
                stream=task["stream"],
                status=completed["status"],
                states=len(rs) // 3,
            )
        )
    complete = (
        len(rows) == 120
        and len({r["trajectory_id"] for r in rows}) == 8
        and all(c["status"] == "complete" and c["states"] == 5 for c in coverage)
    )
    write_json(
        output / "coverage.json",
        dict(complete_balanced_cohort=complete, tasks=coverage, states=len(rows) // 3),
    )
    if not complete:
        raise RuntimeError(
            "Balanced cohort incomplete: retain all failures; do not substitute trajectories"
        )
    _, parts = snapshot_parts()
    study.assert_disjoint(parts["train"] + parts["validation"] + parts["test"], rows)
    results = {}
    predictions = {}
    for spec in manifest["models"]:
        folder = Path(spec["folder"])
        if spec["family"] == "diagnostic":
            prediction = confirmation.predict(folder, rows)
        elif spec["family"] == "f2":
            prediction = ContrastForecaster.load(folder).predict(rows)
        else:
            prediction = Forecaster.load(folder).predict(rows)
        if prediction.shape[1] == 1:
            truth = study.targets(rows, spec["target"]).mean(1)
            measures = [ranking(truth[g], prediction[g, 0]) for g in study.groups(rows)]
            measured = dict(
                scalar_only=True,
                own_utility_spearman=stratified_interval(
                    [m["spearman"] for m in measures], rows
                ),
            )
        else:
            measured = study.evaluate_fit(rows, prediction, spec["target"])
        measured["future_selection"] = future_selection(rows, prediction)
        results[spec["name"]] = measured
        predictions[spec["name"]] = prediction.tolist()
    ridge = json.loads((BASE / "methods/ridge.json").read_text())
    for name, meta in ridge.items():
        prediction = ridge_predict(meta, rows)
        results[name] = study.evaluate_fit(rows, prediction, meta["target"])
        results[name]["future_selection"] = future_selection(rows, prediction)
        predictions[name] = prediction.tolist()
    for target in study.TARGETS:
        prediction = tree_predict(target, rows)
        name = "tuneahead_style_" + target
        results[name] = study.evaluate_fit(rows, prediction, target)
        results[name]["future_selection"] = future_selection(rows, prediction)
        predictions[name] = prediction.tolist()
    for name, prediction in [
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
        results[name] = study.evaluate_fit(rows, prediction, "future")
        results[name]["future_selection"] = future_selection(rows, prediction)
        predictions[name] = prediction.tolist()
    paired = {}
    for horizon in ["immediate", "future", "continuation"]:
        for seed in [42, 43, 44]:
            name = f"gru_{horizon}_n20_seed{seed}"
            paired[name] = results["diagnostic_" + name]
    primary_names = ["original_direct_seed42", "original_matched_h1_seed42"]
    if any(n not in predictions for n in primary_names):
        raise RuntimeError("Frozen primary comparison missing")
    # Pair actual policy value at each state, retaining all three consequence heads.
    values = []
    two = study.targets(rows, "future").mean(1)
    for g in study.groups(rows):
        v = []
        for name in primary_names:
            p = np.array(predictions[name])[g].mean(1)
            p = np.round(p, 12)
            v.append(two[g][p == p.max()].mean())
        values.append(float(v[0] - v[1]))
    write_json(output / "predictions.json", predictions)
    write_json(output / "selection_values.json", selection(rows))
    write_json(output / "mechanism_replication.json", mechanism(rows))
    write_json(
        output / "decomposition.json", study.diagnostic_data({"replication": rows})
    )
    write_json(
        output / "review.json",
        dict(
            status="balanced_replication_science_review_required",
            states=40,
            trajectories=8,
            fits=results,
            primary_direct_h2_minus_matched_h1_future_value=stratified_interval(
                values, rows
            ),
            diagnostic_three_horizon_contrasts=study.paired_review(
                {"validation": rows}, paired
            ),
            new_neural_fits=0,
            method_selection_from_replication=False,
            original_gate2_unchanged=True,
            online_or_scaling_submitted=False,
            methods_manifest_sha256=file_digest(BASE / "methods/manifest.json"),
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["freeze", "run", "confirm"])
    p.add_argument("--index", type=int, choices=range(8))
    args = p.parse_args()
    if args.mode == "run" and args.index is None:
        p.error("Need fixed replication task index")
    if args.mode == "freeze":
        freeze_methods()
    elif args.mode == "run":
        run(args.index)
    else:
        confirm()


if __name__ == "__main__":
    main()
