# ruff: noqa: E402
"""Fixed tabular adaptation and independent-GRU synthetic controls, CPU only."""

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools")]
import numpy as np
import diagnose_forecast_signal_v1 as study
from empirical_common_v1 import BASE, checked, snapshot_parts
from third_eye.io import file_digest, write_json

OUT = BASE / "comparator"
PARAMS = dict(
    objective="regression",
    learning_rate=0.05,
    num_leaves=4,
    n_estimators=140,
    subsample=0.6,
    subsample_freq=1,
    colsample_bytree=0.6,
    min_child_samples=20,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=4,
    deterministic=True,
    force_col_wise=True,
    verbosity=-1,
)


def frozen():
    checked()
    release = json.loads((BASE / "comparator_release.json").read_text())
    if file_digest(BASE / "release.json") != release["cached_release_sha256"]:
        raise RuntimeError("Cached release changed")
    for name, sha in release["files"].items():
        if file_digest(ROOT / name) != sha:
            raise RuntimeError("Comparator release changed")
    # These imports reuse frozen diagnostic fitting and encoding, without modifying them.
    study.checked_parts()
    return release


def flat(rows):
    arrays, ix = study.group_inputs(rows)
    result = np.empty((len(rows), sum(int(np.prod(a.shape[2:])) for a in arrays)))
    result[ix.reshape(-1)] = np.concatenate(
        [a.reshape(len(ix), 3, -1) for a in arrays], 2
    ).reshape(len(rows), -1)
    if not np.isfinite(result).all():
        raise RuntimeError("Invalid pre-commit features")
    return result


def fit_tree(train, val, target, folder):
    import lightgbm as lgb

    study.assert_disjoint(train, val)
    x, z = flat(train), flat(val)
    y, vy = study.targets(train, target), study.targets(val, target)
    folder.mkdir(parents=True, exist_ok=False)
    prediction = np.empty_like(vy)
    artifacts = {}
    best_iterations = []
    for head in range(3):
        model = lgb.LGBMRegressor(**PARAMS)
        model.fit(
            x,
            y[:, head],
            eval_set=[(z, vy[:, head])],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        prediction[:, head] = model.predict(z)
        name = f"head_{head}.txt"
        model.booster_.save_model(
            str(folder / name), num_iteration=model.best_iteration_
        )
        restored = lgb.Booster(model_file=str(folder / name))
        if not np.array_equal(prediction[:, head], restored.predict(z)):
            raise RuntimeError("Tabular comparator reload differs")
        artifacts[name] = file_digest(folder / name)
        best_iterations.append(model.best_iteration_)
    meta = dict(
        target=target,
        seed=42,
        training_trajectory_ids=sorted({r["trajectory_id"] for r in train}),
        validation_trajectory_ids=sorted({r["trajectory_id"] for r in val}),
        feature_columns=x.shape[1],
        feature_source="original encode allowlist: parent state, candidate descriptors/masks, history/masks",
        parameters=PARAMS,
        early_stopping_rounds=50,
        best_iterations=best_iterations,
        artifact_sha256=artifacts,
        lightgbm_version=importlib.metadata.version("lightgbm"),
        role="TuneAhead-style cached-feature adaptation; not a faithful TuneAhead reproduction",
        probe_steps=10,
        reference_probe_steps=100,
        reload_exact=True,
        test_evaluated=False,
    )
    write_json(folder / "metadata.json", meta)
    write_json(folder / "validation_metrics.json", study.scores(val, vy, prediction))
    return prediction


def synthetic_rows(train, val, noise, seed):
    x = np.array([r["precommit"]["features"].get("pre_update_nll", 0) for r in train])
    z = np.array([r["precommit"]["features"].get("pre_update_nll", 0) for r in val])
    mean, scale = x.mean(), max(float(x.std()), 1e-6)
    x, z = (x - mean) / scale, (z - mean) / scale
    for values, rows in [(x, train), (z, val)]:
        ix = study.groups(rows)
        values[ix] -= values[ix].mean(1, keepdims=True)
    rng = np.random.default_rng(seed)
    a, b = copy.deepcopy(train), copy.deepcopy(val)
    for rows, signal in [(a, x), (b, z)]:
        injected = 0.01 * (signal + noise * rng.normal(size=len(signal)))
        for i, r in enumerate(rows):
            r["labels"]["h2"] = {h: float(injected[i]) for h in study.HEADS}
    return a, b


def fit():
    frozen()
    if importlib.metadata.version("lightgbm") != "4.6.0":
        raise RuntimeError("Comparator requires pinned LightGBM 4.6.0")
    _, parts = snapshot_parts()
    # The existing exposed test never reaches this fitter.
    train, val = parts["train"], parts["validation"]
    OUT.mkdir(exist_ok=False)
    results = {}
    for target in study.TARGETS:
        pred = fit_tree(train, val, target, OUT / "models" / target)
        results[target] = study.scores(val, study.targets(val, target), pred)
    controls = []
    for noise in [0.0, 0.5, 1.0]:
        for seed in [42, 43, 44]:
            a, b = synthetic_rows(train, val, noise, seed)
            pred = study.fit_gru(
                a, b, "future", seed, OUT / "synthetic" / f"noise{noise}_seed{seed}"
            )
            controls.append(
                dict(
                    noise_sd=noise,
                    seed=seed,
                    synthetic=True,
                    real_forecasting_success=False,
                    metrics=study.scores(b, study.targets(b, "future"), pred),
                )
            )
    write_json(
        OUT / "completed.json",
        dict(
            status="comparator_frozen_for_fresh_replication",
            validation=results,
            independent_gru_positive_controls=controls,
            test_evaluated=False,
            new_protocol=False,
            release_sha256=file_digest(BASE / "comparator_release.json"),
        ),
    )


def predict(target, rows):
    import lightgbm as lgb

    folder = OUT / "models" / target
    meta = json.loads((folder / "metadata.json").read_text())
    if any(file_digest(folder / n) != h for n, h in meta["artifact_sha256"].items()):
        raise RuntimeError("Comparator weights changed")
    if {r["trajectory_id"] for r in rows} & set(
        meta["training_trajectory_ids"] + meta["validation_trajectory_ids"]
    ):
        raise RuntimeError("Fresh confirmation overlaps development")
    return np.column_stack(
        [
            lgb.Booster(model_file=str(folder / f"head_{h}.txt")).predict(flat(rows))
            for h in range(3)
        ]
    )


def preliminary():
    frozen()
    from empirical_noise_v1 import frozen as noise_frozen
    from empirical_noise_review_v1 import inspect, aggregate
    from empirical_common_v1 import prepared

    noise_frozen()
    selected = [t for t in prepared()["noise_tasks"] if t["generation"] == 0]
    results = []
    for t in selected:
        p = BASE / "noise" / f"task_{t['index']}/completed.json"
        if not p.exists():
            raise RuntimeError(
                "Preliminary noise wave incomplete; retain failure evidence"
            )
        r = json.loads(p.read_text())
        if r["noise_release_sha256"] != file_digest(BASE / "noise_release.json"):
            raise RuntimeError("Preliminary noise release differs")
        results.append(inspect(r))
    if len(results) != 4:
        raise RuntimeError("Need one early parent per stream")
    out = BASE / "preliminary_noise_review"
    out.mkdir(exist_ok=False)
    write_json(
        out / "review.json",
        dict(
            status="preliminary_noise_science_review_required",
            states=4,
            parent_results=results,
            all=aggregate(results),
            replication_released=False,
            scope="Four early-generation independent parents; not final twelve-state reliability evidence",
            no_automatic_science_threshold=True,
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["fit", "preliminary"])
    args = p.parse_args()
    fit() if args.mode == "fit" else preliminary()


if __name__ == "__main__":
    main()
