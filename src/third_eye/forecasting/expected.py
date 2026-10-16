"""Fit Third Eye's expected-value objective on repeated continuation labels."""

import copy
import time

import numpy as np
import torch
from torch.nn import functional as F

from third_eye.forecasting.continuation import (
    TARGET_VERSION,
    reference_h1,
    candidate_inputs,
    repeated_targets,
    state_groups,
)
from third_eye.forecasting.encoding import FEATURE_NAMES, SCHEMA_VERSION
from third_eye.forecasting.models import build_model
from third_eye.io import digest, file_digest, write_json
from third_eye.provenance import source_inventory, versions


def expected_metrics(rows, prediction, weights):
    # Reuse the ranking contracts but never relabel historical native H2 rows.
    from third_eye.forecasting.training import evaluate_predictions

    mean, _, _ = repeated_targets(rows)
    views = [
        dict(
            r,
            labels={
                **r["labels"],
                "h2": dict(zip(("target", "ood", "retention"), y.tolist())),
            },
        )
        for r, y in zip(rows, mean)
    ]
    measured = evaluate_predictions(views, prediction, weights, 2)
    measured["estimand"] = (
        "empirical mean of declared continuation draws, conditional on M1 and evaluation items"
    )
    return measured


def _ranking_loss(prediction, groups, draws, weights, device):
    """Rank reliable within-parent gaps using paired continuation differences."""
    losses = []
    utility = prediction @ weights
    for indices in groups:
        outcomes = (
            torch.tensor(
                np.stack([draws[i] for i in indices]),
                dtype=torch.float32,
                device=device,
            )
            @ weights
        )
        for a in range(3):
            for b in range(a + 1, 3):
                gap = outcomes[a] - outcomes[b]
                target = gap.mean()
                if target.abs() <= 1e-8:
                    continue
                se = gap.std(unbiased=True) / gap.numel() ** 0.5
                confidence = (target.abs() / (se + 0.005)).tanh().detach()
                difference = utility[indices[a]] - utility[indices[b]]
                losses.append(
                    confidence * F.softplus(-target.sign() * difference / 0.05)
                )
    return torch.stack(losses).mean() if losses else prediction.sum() * 0


def fit_expected(
    parts,
    output,
    weights,
    ablations,
    seed,
    epochs,
    patience,
    hidden,
    learning_rate,
    device,
    risk_penalty=0.0,
    allow_abstain=False,
):
    """Called after the shared forecaster's protocol/split checks.

    Early stopping uses validation only. Test inputs/outcomes are not decoded
    until the best checkpoint has been selected. Hyperparameters and decision
    settings must be chosen before examining fresh test outcomes.
    """
    from third_eye.forecasting.training import Forecaster

    if not np.isfinite(risk_penalty) or risk_penalty < 0:
        raise ValueError("Risk penalty must be finite and nonnegative")
    if not isinstance(allow_abstain, bool):
        raise ValueError("allow_abstain must be boolean")
    started = time.perf_counter()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    x = {p: candidate_inputs(parts[p], ablations) for p in ("train", "validation")}
    mean, scale = x["train"].mean(0), x["train"].std(0)
    scale[scale < 1e-6] = 1.0
    tensors = {
        p: torch.tensor(
            np.clip((v - mean) / scale, -20, 20), dtype=torch.float32, device=device
        )
        for p, v in x.items()
    }
    target, covariance, draws = repeated_targets(parts["train"])
    repeated_targets(parts["validation"])
    truth = torch.tensor(target, dtype=torch.float32, device=device)
    cov = torch.tensor(covariance, dtype=torch.float32, device=device)
    h1 = torch.tensor(
        np.stack([reference_h1(r) for r in parts["train"]]),
        dtype=torch.float32,
        device=device,
    )
    weight = torch.tensor(weights, dtype=torch.float32, device=device)
    groups = state_groups(parts["train"])
    model = build_model("expected", hidden).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    best, best_score, stale, curve = None, -float("inf"), 0, []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(groups))
        losses = []
        for start in range(0, len(order), 16):
            selected = sum([groups[i] for i in order[start : start + 16]], [])
            optimizer.zero_grad(set_to_none=True)
            out = model(tensors["train"][selected])
            loss = F.mse_loss(out["h1"], h1[selected])
            loss = loss + F.mse_loss(
                out["continuation"], truth[selected] - h1[selected]
            )
            loss = loss + F.mse_loss(out["prediction"], truth[selected])
            # Gaussian covariance objective on unbiased centered draw covariance.
            # Centering at the sample mean keeps mean-fit error out of the
            # aleatoric target. A small floor handles deterministic terminals.
            sample_cov = cov[selected] + torch.eye(3, device=device) * 1e-8
            solved = torch.cholesky_solve(sample_cov, out["factor"])
            variance_loss = (
                solved.diagonal(dim1=-2, dim2=-1).sum(-1)
                + 2 * out["factor"].diagonal(dim1=-2, dim2=-1).log().sum(-1)
            ).mean()
            local_groups = [list(range(i, i + 3)) for i in range(0, len(selected), 3)]
            local_draws = [draws[i] for i in selected]
            loss = (
                loss
                + 0.0005 * variance_loss
                + 0.01
                * _ranking_loss(
                    out["prediction"], local_groups, local_draws, weight, device
                )
            )
            if not torch.isfinite(loss):
                raise ValueError("Non-finite expected-value training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            prediction = model(tensors["validation"])["prediction"].cpu().numpy()
        metrics = expected_metrics(parts["validation"], prediction, weights)
        score = metrics["ranking"]["top1"] - 0.1 * metrics["utility"]["rmse"]
        curve.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "validation_score": score,
            }
        )
        if score > best_score + 1e-8:
            best, best_score, stale = copy.deepcopy(model.state_dict()), score, 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best)
    model.eval()
    output.mkdir(parents=True)
    torch.save({k: v.cpu() for k, v in best.items()}, output / "weights.pt")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "kind": "expected",
        "horizon": 2,
        "hidden": hidden,
        "scalar": False,
        "target_version": TARGET_VERSION,
        "minimum_continuation_replicates": 3,
        "input_version": "candidate_relative_v1",
        "ablations": list(ablations),
        "seed": seed,
        "utility_weights": list(weights),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "epochs_completed": len(curve),
        "best_validation_score": best_score,
        "seconds": time.perf_counter() - started,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "weights_sha256": file_digest(output / "weights.pt"),
        "uncertainty": "conditional continuation draw covariance; excludes epistemic and item uncertainty",
        "decision": {
            "risk_penalty": risk_penalty,
            "allow_abstain": allow_abstain,
            "no_update_utility": 0.0,
        },
        "training_trajectory_ids": sorted({r["trajectory_id"] for r in parts["train"]}),
        "selection_trajectory_ids": sorted(
            {r["trajectory_id"] for p in ("train", "validation") for r in parts[p]}
        ),
        "selection_state_ids": sorted(
            {r["state_id"] for p in ("train", "validation") for r in parts[p]}
        ),
        "split_hashes": {p: digest(rows) for p, rows in parts.items()},
        "training_config": {
            "epochs": epochs,
            "patience": patience,
            "learning_rate": learning_rate,
            "ranking_weight": 0.01,
            "covariance_weight": 0.0005,
        },
        "software_versions": versions(),
        **source_inventory(),
    }
    write_json(output / "metadata.json", metadata)
    write_json(output / "learning_curve.json", curve)
    forecaster = Forecaster(model, metadata, mean, scale, device)
    measured = {}
    for p, rows in parts.items():
        result = forecaster.predict_distribution(rows)
        measured[p] = expected_metrics(rows, result["mean"], weights)
        true_mean, true_cov, _ = repeated_targets(rows)
        true_var = np.einsum("i,nij,j->n", weights, true_cov, weights)
        measured[p]["draw_variance_mae"] = float(
            np.mean(np.abs(result["utility_std"] ** 2 - true_var))
        )
        measured[p]["selection"] = selection_metrics(
            rows, result, weights, metadata["decision"]
        )
        if p == "test":
            write_json(
                output / "test_predictions.json",
                [
                    {
                        "state_id": r["state_id"],
                        "candidate_id": r["candidate_id"],
                        "trajectory_id": r["trajectory_id"],
                        "truth_h1": reference_h1(r).tolist(),
                        "truth_expected_h2": true_mean[i].tolist(),
                        "prediction": result["mean"][i].tolist(),
                        "continuation_covariance": result["covariance"][i].tolist(),
                    }
                    for i, r in enumerate(rows)
                ],
            )
    write_json(output / "metrics.json", measured)
    write_json(
        output / "split_audit.json",
        {
            p: [
                {
                    k: r[k]
                    for k in ("state_id", "candidate_id", "trajectory_id", "model")
                }
                for r in rows
            ]
            for p, rows in parts.items()
        },
    )
    return forecaster


def selection_metrics(rows, distribution, weights, decision):
    """Tie-averaged held-out selector value; measured oracle is descriptive.

    Independent continuation seeds/items for privileged-oracle comparisons are
    handled by the independent-scoring report, not this native diagnostic.
    """
    mean, _, _ = repeated_targets(rows)
    actual = mean @ weights
    h1 = np.stack([reference_h1(r) for r in rows]) @ weights
    forecast = distribution["mean"] @ weights
    risk = forecast - decision["risk_penalty"] * distribution["utility_std"]
    results = []
    for indices in state_groups(rows):
        chosen = np.isclose(risk[indices], max(risk[indices]), atol=1e-12, rtol=0)
        abstain = decision["allow_abstain"] and max(risk[indices]) <= 0
        value = 0.0 if abstain else float(actual[indices][chosen].mean())
        greedy = np.isclose(h1[indices], max(h1[indices]), atol=1e-12, rtol=0)
        results.append(
            {
                "state_id": rows[indices[0]]["state_id"],
                "trajectory_id": rows[indices[0]]["trajectory_id"],
                "abstained": bool(abstain),
                "forecast_future_utility": value,
                "immediate_future_utility": float(actual[indices][greedy].mean()),
                "random_future_utility": float(actual[indices].mean()),
                "no_update_utility": 0.0,
            }
        )
    return {
        "per_state": results,
        "interpretation": "Held-out conditional mean outcomes; immediate comparator uses observed H1 and pays for full branches. No claim about independent-oracle regret or uncertainty coverage.",
    }
