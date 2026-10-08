"""Frozen development-only signal study; no test evaluation or policy launch."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools")]
import numpy as np
import torch

from extensions.forecasting_f2.model import build_model
from extensions.forecasting_f2.training import group_inputs, objective
from third_eye.forecasting.encoding import FEATURE_NAMES, HEADS, labels
from third_eye.io import digest, file_digest, write_json
from third_eye.statistics.inference import holm, paired_comparison
from third_eye.statistics.metrics import ranking
from fit_forecaster_f2_v1 import frozen, SNAPSHOT

OUT = ROOT / "runs/a2/signal_diagnostic_v1"
WEIGHTS = np.ones(3) / 3
TARGETS = ("immediate", "future", "continuation")
LEVELS = (5, 10, 15, 20)
SEEDS = (42, 43, 44)


def targets(rows, target):
    if target not in TARGETS:
        raise ValueError("Unknown diagnostic target")
    one = np.stack([labels(r, 1) for r in rows])
    two = np.stack([labels(r, 2) for r in rows])
    return one if target == "immediate" else two if target == "future" else two - one


def groups(rows):
    _, ix = group_inputs(rows)
    return ix


def annotate(parts):
    manifest = json.loads((SNAPSHOT / "snapshot_manifest.json").read_text())
    streams = {}
    for item in manifest["trajectories"]:
        p = Path(item["source"]).parts
        i = p.index("labels")
        streams[item["trajectory_id"]] = "/".join(p[i + 1 : i + 3])
    return {
        p: [dict(r, analysis_stream=streams[r["trajectory_id"]]) for r in rs]
        for p, rs in parts.items()
    }


def training_order(rows):
    by_stream = defaultdict(set)
    for r in rows:
        by_stream[r["analysis_stream"]].add(r["trajectory_id"])
    ordered = {
        s: sorted(ids, key=lambda t: digest(["signal_v1", 42, t]))
        for s, ids in sorted(by_stream.items())
    }
    return [
        ordered[s][i]
        for i in range(max(map(len, ordered.values())))
        for s in ordered
        if i < len(ordered[s])
    ]


def subset(rows, n):
    order = training_order(rows)
    if n > len(order):
        raise ValueError("Training trajectory budget exceeds available sample")
    selected = set(order[:n])
    return [r for r in rows if r["trajectory_id"] in selected]


def interval(values, rows):
    ix = groups(rows)
    if len(values) != len(ix):
        raise ValueError("One metric per complete state is required")
    clustered = defaultdict(list)
    finite = []
    for value, g in zip(values, ix):
        if value is not None and np.isfinite(value):
            finite.append(float(value))
            clustered[rows[g[0]]["trajectory_id"]].append(float(value))
    unit = np.array([np.mean(v) for v in clustered.values()])
    if not len(unit):
        return {
            "state_mean": None,
            "trajectory_mean": None,
            "ci95": None,
            "states": 0,
            "trajectories": 0,
        }
    boot = (
        np.random.default_rng(42).choice(unit, (5000, len(unit)), replace=True).mean(1)
    )
    return {
        "state_mean": float(np.mean(finite)),
        "trajectory_mean": float(unit.mean()),
        "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "states": len(finite),
        "trajectories": len(unit),
        "resampling": "equal-weight trajectory means, 5000 draws, seed42",
    }


def scores(rows, y, prediction):
    if y.shape != prediction.shape or y.shape != (len(rows), 3):
        raise ValueError("Matched physical three-head predictions required")
    ix = groups(rows)
    result = {}
    for head, column in [("utility", None), *[(h, i) for i, h in enumerate(HEADS)]]:
        a = y @ WEIGHTS if column is None else y[:, column]
        b = prediction @ WEIGHTS if column is None else prediction[:, column]
        measures = [ranking(a[g], b[g]) for g in ix]
        result[head] = {
            "spearman": interval([m["spearman"] for m in measures], rows),
            "top1": interval(
                [m["top1"] if m["chance_top1"] < 1 else None for m in measures], rows
            ),
            "chance_top1": interval(
                [m["chance_top1"] if m["chance_top1"] < 1 else None for m in measures],
                rows,
            ),
            "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
        }
    result["per_state_utility"] = {
        rows[g[0]]["state_id"]: ranking(y[g] @ WEIGHTS, prediction[g] @ WEIGHTS)
        for g in ix
    }
    return result


def decomposition(rows):
    ix = groups(rows)
    one, two = targets(rows, "immediate"), targets(rows, "future")
    inc = two - one
    u1, u2 = one @ WEIGHTS, two @ WEIGHTS
    measures = defaultdict(list)
    for g in ix:
        a, b = np.round(u1[g], 12), np.round(u2[g], 12)
        w1, w2 = set(np.flatnonzero(a == a.max())), set(np.flatnonzero(b == b.max()))
        margin = np.sort(b)[-1] - np.sort(b)[-2]
        measures["h1_h2_spearman"].append(ranking(b, a)["spearman"])
        measures["strict_winner_reversal"].append(float(not (w1 & w2)))
        measures["immediate_oracle_future_regret"].append(
            float(b.max() - b[list(w1)].mean())
        )
        measures["h2_winner_margin"].append(float(margin))
        measures["exact_winner_tie"].append(float(len(w2) > 1))
        measures["margin_le_retention_one_item"].append(
            float(margin <= 1 / 768 + 1e-12)
        )
        measures["margin_le_target_one_item"].append(float(margin <= 1 / 192 + 1e-12))
        measures["terminal_candidate_fraction"].append(
            float(np.mean([rows[i].get("continuation_available", 1) == 0 for i in g]))
        )
        for threshold in [0, -1 / 768, -1 / 192, -0.02]:
            measures[f"h1_utility_below_{threshold:.8f}"].append(
                float(np.mean(a < threshold))
            )
    centered = [a[ix] - a[ix].mean(1, keepdims=True) for a in [one, inc, two]]
    variance = {}
    for head, col in [("utility", None), *[(h, i) for i, h in enumerate(HEADS)]]:
        arrays = [a @ WEIGHTS if col is None else a[:, :, col] for a in centered]
        v1, vc, v2 = [float(np.mean(a**2)) for a in arrays]
        cov = float(np.mean(arrays[0] * arrays[1]))
        assert np.isclose(v2, v1 + vc + 2 * cov, atol=1e-12)
        variance[head] = {
            "within_parent_h1_variance": v1,
            "within_parent_continuation_variance": vc,
            "within_parent_h2_variance": v2,
            "h1_continuation_covariance": cov,
        }
    return {
        "states": len(ix),
        "trajectories": len({r["trajectory_id"] for r in rows}),
        "metrics": {k: interval(v, rows) for k, v in measures.items()},
        "variance_decomposition": variance,
        "interpretation": "Observed variation, not irreducible-noise estimates",
    }


def diagnostic_data(parts):
    report = {}
    for part, rows in parts.items():
        report[part] = {"all": decomposition(rows), "streams": {}, "generations": {}}
        for stream in sorted({r["analysis_stream"] for r in rows}):
            report[part]["streams"][stream] = decomposition(
                [r for r in rows if r["analysis_stream"] == stream]
            )
        for gen in sorted({r["generation"] for r in rows}):
            report[part]["generations"][str(gen)] = decomposition(
                [r for r in rows if r["generation"] == gen]
            )
        arrays, _ = group_inputs(rows)
        f = arrays[1]
        report[part]["feature_audit"] = {}
        for i, name in enumerate(FEATURE_NAMES):
            v = f[:, :, i]
            report[part]["feature_audit"][name] = {
                "available_fraction": float(f[:, :, i + len(FEATURE_NAMES)].mean()),
                "global_std": float(v.std()),
                "within_parent_std": float((v - v.mean(1, keepdims=True)).std()),
                "constant_within_parent_fraction": float(
                    np.mean(np.ptp(v, axis=1) <= 1e-12)
                ),
                "unique_values": len(np.unique(v)),
                "min": float(v.min()),
                "max": float(v.max()),
            }
    return report


def assert_disjoint(train_rows, val_rows):
    for field in ["state_id", "trajectory_id"]:
        if {r[field] for r in train_rows} & {r[field] for r in val_rows}:
            raise ValueError("Diagnostic partition leakage")


def ridge(train_rows, val_rows, target, relative):
    assert_disjoint(train_rows, val_rows)
    a, ti = group_inputs(train_rows)
    b, vi = group_inputs(val_rows)

    def flat(inputs):
        return np.concatenate([x.reshape(x.shape[0], 3, -1) for x in inputs], 2).astype(
            float
        )

    x, z = flat(a), flat(b)
    mean, scale = x.mean((0, 1)), np.maximum(x.std((0, 1)), 1e-6)
    x, z = np.clip((x - mean) / scale, -20, 20), np.clip((z - mean) / scale, -20, 20)
    y = targets(train_rows, target)[ti]
    base = y.mean((0, 1))
    if relative:
        x, z = x - x.mean(1, keepdims=True), z - z.mean(1, keepdims=True)
        y = y - y.mean(1, keepdims=True)
    else:
        y = y - base
    x = x.reshape(-1, x.shape[-1])
    beta = np.linalg.solve(x.T @ x + np.eye(x.shape[1]), x.T @ y.reshape(-1, 3))
    p = z @ beta + base
    result = np.empty((len(val_rows), 3))
    result[vi.reshape(-1)] = p.reshape(-1, 3)
    return result


def fit_gru(train_rows, val_rows, target, seed, folder):
    """Original independent GRU, F2 fitting rules, explicit diagnostic targets."""
    assert_disjoint(train_rows, val_rows)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    a, ti = group_inputs(train_rows)
    b, vi = group_inputs(val_rows)
    mean, std = a[1].mean((0, 1)), np.maximum(a[1].std((0, 1)), 1e-6)
    a, b = list(a), list(b)
    a[1], b[1] = (
        np.clip((a[1] - mean) / std, -20, 20),
        np.clip((b[1] - mean) / std, -20, 20),
    )
    h1 = targets(train_rows, "immediate")[ti].astype(np.float32)
    scale = np.maximum(
        (h1 - h1.mean(1, keepdims=True)).std((0, 1)),
        np.array([1 / 64, 1 / 64, 1 / 256], dtype=np.float32),
    )
    u1 = h1 @ WEIGHTS
    temp = max(float((u1 - u1.mean(1, keepdims=True)).std()), 0.001)
    x = [torch.tensor(v, dtype=torch.float32) for v in a]
    vx = [torch.tensor(v, dtype=torch.float32) for v in b]
    y = torch.tensor(
        targets(train_rows, target)[ti].astype(np.float32) / scale, dtype=torch.float32
    )
    hs = torch.tensor(h1 / scale)
    model = build_model("independent")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    true_val = targets(val_rows, target)
    curve, best, best_score, stale = [], None, -float("inf"), 0
    started = time.perf_counter()
    for epoch in range(200):
        model.train()
        for batch in torch.randperm(len(ti)).split(16):
            optimizer.zero_grad(set_to_none=True)
            out = model(*(v[batch] for v in x))
            loss = objective(
                out,
                y[batch],
                hs[batch],
                torch.tensor(WEIGHTS, dtype=torch.float32),
                torch.tensor(scale),
                temp,
                2,
                "independent",
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite diagnostic training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            p = model(*vx)["prediction"].numpy() * scale
        prediction = np.empty((len(val_rows), 3))
        prediction[vi.reshape(-1)] = p.reshape(-1, 3)
        measures = [ranking(true_val[g] @ WEIGHTS, prediction[g] @ WEIGHTS) for g in vi]
        defined = [m["spearman"] for m in measures if m["spearman"] is not None]
        informative = [m["top1"] for m in measures if m["chance_top1"] < 1]
        rho, top = (
            float(np.mean(defined)) if defined else -1,
            float(np.mean(informative)) if informative else 0,
        )
        rmse = float(np.sqrt(np.mean(((prediction - true_val) @ WEIGHTS) ** 2)))
        score = 0.5 * (rho + top) - 0.01 * rmse
        curve.append({"epoch": epoch + 1, "score": score, "spearman": rho, "top1": top})
        if score > best_score + 1e-8:
            best, best_score, stale = copy.deepcopy(model.state_dict()), score, 0
        else:
            stale += 1
        if stale >= 40:
            break
    model.load_state_dict(best)
    model.eval()
    with torch.inference_mode():
        saved = model(*vx)["prediction"].numpy() * scale
    prediction[vi.reshape(-1)] = saved.reshape(-1, 3)
    folder.mkdir(parents=True, exist_ok=False)
    torch.save(best, folder / "diagnostic_weights.pt")
    restored = build_model("independent")
    restored.load_state_dict(
        torch.load(folder / "diagnostic_weights.pt", weights_only=True)
    )
    restored.eval()
    with torch.inference_mode():
        assert np.array_equal(saved, restored(*vx)["prediction"].numpy() * scale)
    write_json(
        folder / "metadata.json",
        {
            "role": "diagnostic_only_not_an_online_forecaster",
            "target": target,
            "seed": seed,
            "epochs": len(curve),
            "best_epoch": max(curve, key=lambda r: r["score"])["epoch"],
            "feature_mean": mean.tolist(),
            "feature_scale": std.tolist(),
            "target_scale": scale.tolist(),
            "training_trajectory_ids": sorted({r["trajectory_id"] for r in train_rows}),
            "seconds": time.perf_counter() - started,
            "reload_exact": True,
            "weights_sha256": file_digest(folder / "diagnostic_weights.pt"),
        },
    )
    write_json(folder / "learning_curve.json", curve)
    return prediction


def specifications():
    return [
        {"target": target, "seed": seed, "trajectories": n}
        for n in LEVELS
        for target in TARGETS
        for seed in SEEDS
    ]


def task_name(task):
    return f"gru_{task['target']}_n{task['trajectories']}_seed{task['seed']}"


def checked_parts():
    release, parts = frozen()
    own = json.loads((ROOT / "runs/a2/signal_diagnostic_release_v1.json").read_text())
    if any(file_digest(ROOT / p) != h for p, h in own["files"].items()):
        raise RuntimeError("Diagnostic release changed")
    parts = annotate(parts)
    assert set(parts) == {"train", "validation"}
    assert len(training_order(parts["train"])) == 20
    assert_disjoint(parts["train"], parts["validation"])
    return own, parts


def evaluate_fit(rows, prediction, target):
    measured = {"own_target": scores(rows, targets(rows, target), prediction)}
    if target != "continuation":
        measured["future_h2"] = scores(rows, targets(rows, "future"), prediction)
    measured["streams"] = {}
    for stream in sorted({r["analysis_stream"] for r in rows}):
        ix = [i for i, r in enumerate(rows) if r["analysis_stream"] == stream]
        rs = [rows[i] for i in ix]
        measured["streams"][stream] = scores(rs, targets(rs, target), prediction[ix])
    return measured


def paired_review(parts, fitted):
    rows = parts["validation"]
    contrasts = [
        (
            "immediate_own_minus_future_own",
            "immediate",
            "own_target",
            "future",
            "own_target",
        ),
        (
            "future_own_minus_immediate_future",
            "future",
            "own_target",
            "immediate",
            "future_h2",
        ),
        (
            "immediate_own_minus_continuation_own",
            "immediate",
            "own_target",
            "continuation",
            "own_target",
        ),
    ]
    result = {}
    for name, a, ak, b, bk in contrasts:
        trajectory_differences = defaultdict(list)
        for g in groups(rows):
            sid, tid = rows[g[0]]["state_id"], rows[g[0]]["trajectory_id"]
            differences = []
            for seed in SEEDS:
                x = fitted[f"gru_{a}_n20_seed{seed}"][ak]["per_state_utility"][sid][
                    "spearman"
                ]
                y = fitted[f"gru_{b}_n20_seed{seed}"][bk]["per_state_utility"][sid][
                    "spearman"
                ]
                if x is not None and y is not None:
                    differences.append(x - y)
            if differences:
                trajectory_differences[tid].append(float(np.mean(differences)))
        vals = np.array([np.mean(v) for v in trajectory_differences.values()])
        if len(vals):
            result[name] = paired_comparison(vals, np.zeros_like(vals))
    adjusted = holm({k: v["paired_permutation_p"] for k, v in result.items()})
    for name in result:
        result[name]["holm_p_three_predeclared_contrasts"] = adjusted[name]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, choices=range(36))
    args = parser.parse_args()
    release, parts = checked_parts()
    if args.index is not None:
        task = specifications()[args.index]
        pred = fit_gru(
            subset(parts["train"], task["trajectories"]),
            parts["validation"],
            task["target"],
            task["seed"],
            OUT / "fits" / task_name(task),
        )
        write_json(
            OUT / "fits" / task_name(task) / "validation_predictions.json",
            pred.tolist(),
        )
        write_json(
            OUT / "fits" / task_name(task) / "metrics.json",
            evaluate_fit(parts["validation"], pred, task["target"]),
        )
        return
    assert not OUT.exists(), "Preserve previous diagnostic output"
    OUT.mkdir(parents=True)
    (OUT / "logs").mkdir()
    write_json(
        OUT / "predeclared_study.json",
        {
            "release": release,
            "gru_fits": specifications(),
            "ridge_penalty": 1,
            "ridge_modes": ["absolute", "relative"],
            "parts": {
                p: {
                    "rows": len(rs),
                    "states": len(groups(rs)),
                    "trajectories": len({r["trajectory_id"] for r in rs}),
                    "record_digest": digest(rs),
                }
                for p, rs in parts.items()
            },
            "nested_training_order": training_order(parts["train"]),
            "test_evaluated": False,
            "scientific_thresholds_unchanged": True,
        },
    )
    write_json(OUT / "signal_decomposition.json", diagnostic_data(parts))
    print("Signal decomposition and feature audit published", flush=True)
    fitted = {}
    for n in LEVELS:
        for target in TARGETS:
            for relative in [False, True]:
                pred = ridge(
                    subset(parts["train"], n), parts["validation"], target, relative
                )
                name = f"ridge_{'relative' if relative else 'absolute'}_{target}_n{n}"
                fitted[name] = evaluate_fit(parts["validation"], pred, target)
    for name, pred in [
        ("random_tied", np.zeros((len(parts["validation"]), 3))),
        (
            "heuristic_low_nll",
            np.repeat(
                np.array(
                    [
                        -r["precommit"]["features"].get("pre_update_nll", 0)
                        for r in parts["validation"]
                    ]
                )[:, None],
                3,
                axis=1,
            ),
        ),
    ]:
        fitted[name] = evaluate_fit(parts["validation"], pred, "future")
    write_json(OUT / "simple_controls.json", fitted)

    def launch(index):
        with (OUT / "logs" / f"fit_{index:02d}.log").open("w") as log:
            subprocess.run(
                [sys.executable, str(Path(__file__)), "--index", str(index)],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        return index

    with ThreadPoolExecutor(
        max_workers=min(12, int(os.environ["SLURM_CPUS_PER_TASK"]) // 2)
    ) as pool:
        for f in as_completed([pool.submit(launch, i) for i in range(36)]):
            i = f.result()
            name = task_name(specifications()[i])
            fitted[name] = json.loads(
                (OUT / "fits" / name / "metrics.json").read_text()
            )
            print(f"Fixed diagnostic {name} complete", flush=True)
    comparisons = paired_review(parts, fitted)
    write_json(
        OUT / "review.json",
        {
            "status": "signal_review_required",
            "fits": fitted,
            "paired_comparisons": comparisons,
            "test_evaluated": False,
            "gate2_redefined": False,
            "online_submitted": False,
            "limitations": [
                "Previously used development population",
                "Eight validation trajectories",
                "No irreducible-noise claim",
                "No new independent GPU replication",
            ],
        },
    )
    print(
        json.dumps(
            {"status": "signal_review_required", "paired_comparisons": comparisons}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
