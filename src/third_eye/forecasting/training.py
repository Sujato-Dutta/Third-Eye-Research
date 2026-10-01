"""Train-only normalization, whole-state batches, and validation-only stopping."""

from collections import defaultdict
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from third_eye.forecasting.encoding import encode, labels, FEATURE_NAMES, SCHEMA_VERSION
from third_eye.forecasting.models import build_model
from third_eye.io import digest, file_digest, write_json
from third_eye.provenance import source_inventory, versions
from third_eye.statistics.metrics import ranking, regression


def stack_inputs(records, ablations=()):
    rows = [encode(r, ablations) for r in records]
    return tuple(np.stack([r[i] for r in rows]) for i in range(4))


def evaluate_predictions(records, prediction, weights, horizon=2):
    y = np.stack([labels(r, horizon) for r in records])
    utility = y @ np.asarray(weights)
    if prediction.shape[1] == 1:
        predicted = prediction[:, 0]
        heads = None
    else:
        predicted = prediction @ np.asarray(weights)
        heads = {
            name: regression(y[:, i], prediction[:, i])
            for i, name in enumerate(("target", "ood", "retention"))
        }
    states = defaultdict(list)
    for i, r in enumerate(records):
        states[r["state_id"]].append(i)
    scores = {
        s: ranking(utility[index], predicted[index]) for s, index in states.items()
    }
    summary = {}
    for key in ("spearman", "kendall", "ndcg_at_3", "top1", "chance_top1"):
        values = [v[key] for v in scores.values() if v[key] is not None]
        summary[key] = float(np.mean(values)) if values else None
    informative = [v for v in scores.values() if v["chance_top1"] < 1]
    summary["informative_top1"] = (
        float(np.mean([v["top1"] for v in informative])) if informative else None
    )
    summary["informative_chance_top1"] = (
        float(np.mean([v["chance_top1"] for v in informative])) if informative else None
    )
    return {
        "horizon": horizon,
        "utility": regression(utility, predicted),
        "ranking": summary,
        "per_state": scores,
        "heads": heads,
        "states": len(states),
        "nonconstant_states": sum(v["spearman"] is not None for v in scores.values()),
        "tradeoff_accuracy": float(
            np.mean(
                ((prediction[:, 0] > 0) & (prediction[:, 2] < 0))
                == ((y[:, 0] > 0) & (y[:, 2] < 0))
            )
        )
        if prediction.shape[1] == 3
        else None,
    }


class Forecaster:
    def __init__(self, model, metadata, mean, scale, device="cpu"):
        self.model = model.to(device).eval()
        self.metadata, self.mean, self.scale, self.device = (
            metadata,
            mean,
            scale,
            device,
        )

    def predict(self, views):
        arrays = list(stack_inputs(views, self.metadata["ablations"]))
        arrays[1] = np.clip((arrays[1] - self.mean) / self.scale, -20, 20)
        with torch.inference_mode():
            tensors = [
                torch.as_tensor(a, dtype=torch.float32, device=self.device)
                for a in arrays
            ]
            return self.model(*tensors)["prediction"].cpu().numpy()

    def select(self, views):
        prediction = self.predict(views)
        utility = (
            prediction[:, 0]
            if prediction.shape[1] == 1
            else prediction @ np.asarray(self.metadata["utility_weights"])
        )
        return int(np.argmax(utility))

    @classmethod
    def load(cls, path, device="cpu"):
        path = Path(path)
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if metadata["schema_version"] != SCHEMA_VERSION or metadata[
            "feature_names"
        ] != list(FEATURE_NAMES):
            raise ValueError("Forecaster feature schema differs")
        if file_digest(path / "weights.pt") != metadata["weights_sha256"]:
            raise ValueError("Forecaster checksum mismatch")
        model = build_model(metadata["kind"], metadata["hidden"], metadata["scalar"])
        model.load_state_dict(
            torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
        )
        return cls(
            model,
            metadata,
            np.asarray(metadata["mean"], dtype=np.float32),
            np.asarray(metadata["scale"], dtype=np.float32),
            device,
        )


def train_forecaster(
    parts,
    output,
    kind="direct",
    horizon=2,
    weights=(1 / 3, 1 / 3, 1 / 3),
    ablations=(),
    seed=42,
    epochs=200,
    patience=20,
    hidden=64,
    learning_rate=1e-3,
    scalar=False,
    device="cpu",
):
    if kind in {"one_step", "matched_h1"} and horizon != 1:
        raise ValueError("H=1 baselines must be trained on H=1 targets")
    if kind == "dynamics" and horizon != 2:
        raise ValueError("Dynamics predicts H=2")
    if kind == "direct" and horizon != 2:
        raise ValueError(
            "Direct predicts H=2; use matched_h1 for the H=1 temporal comparison"
        )
    if min(epochs, patience, hidden) < 1:
        raise ValueError("Training budgets must be positive")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Forecaster learning rate must be finite and positive")
    for rows in parts.values():
        if not rows:
            raise ValueError(
                "Meta-training, validation and test partitions must be nonempty"
            )
        for record in rows:
            if record.get("protocol_status") == "pilot":
                raise ValueError(
                    "Forecaster fitting requires frozen-protocol label records"
                )
            for h in (1, 2):
                outcome = labels(record, h)
                if not np.isfinite(outcome).all() or (np.abs(outcome) > 1).any():
                    raise ValueError(
                        "Consequence labels must be finite accuracy-fraction deltas"
                    )
            if "utility_weights" in record and not np.allclose(
                record["utility_weights"], weights
            ):
                raise ValueError(
                    "Training utility weights differ from the declared label protocol"
                )
    if (
        len(weights) != 3
        or any(w < 0 for w in weights)
        or not np.isclose(sum(weights), 1)
    ):
        raise ValueError("Utility weights must be a probability vector")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new forecaster output directory")
    trajectories = [
        set(r["trajectory_id"] for r in parts[p])
        for p in ("train", "validation", "test")
    ]
    states = [
        set(r["state_id"] for r in parts[p]) for p in ("train", "validation", "test")
    ]
    if any(
        trajectories[i] & trajectories[j] or states[i] & states[j]
        for i in range(3)
        for j in range(i)
    ):
        raise ValueError("Meta-split trajectory/state leakage")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    arrays = {p: list(stack_inputs(rows, ablations)) for p, rows in parts.items()}
    mean = arrays["train"][1].mean(axis=0)
    scale = np.maximum(arrays["train"][1].std(axis=0), 1e-6)
    for arr in arrays.values():
        arr[1] = np.clip((arr[1] - mean) / scale, -20, 20)
    inputs = {
        p: [torch.tensor(a, dtype=torch.float32, device=device) for a in arr]
        for p, arr in arrays.items()
    }
    targets = {
        p: torch.tensor(
            np.stack([labels(r, horizon) for r in rows]),
            dtype=torch.float32,
            device=device,
        )
        for p, rows in parts.items()
    }
    h1 = torch.tensor(
        np.stack([labels(r, 1) for r in parts["train"]]),
        dtype=torch.float32,
        device=device,
    )
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    model = build_model(kind, hidden, scalar).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    groups = defaultdict(list)
    for i, r in enumerate(parts["train"]):
        groups[r["state_id"]].append(i)
    if any(len(v) != 3 for v in groups.values()):
        raise ValueError("Train batches require whole K=3 states")
    groups = list(groups.values())
    best_score = -float("inf")
    best = None
    stale = 0
    curve = []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        random.shuffle(groups)
        losses = []
        for start in range(0, len(groups), 16):
            selected = sum(groups[start : start + 16], [])
            optimizer.zero_grad(set_to_none=True)
            out = model(*(a[selected] for a in inputs["train"]))
            truth = targets["train"][selected]
            expected = (truth @ weight_tensor)[:, None] if scalar else truth
            loss = F.mse_loss(out["prediction"], expected)
            pu = (
                out["prediction"][:, 0] if scalar else out["prediction"] @ weight_tensor
            )
            yu = truth @ weight_tensor
            # Pairwise logistic loss only for non-tied labels within one state.
            pdiff = pu.reshape(-1, 3)[:, :, None] - pu.reshape(-1, 3)[:, None, :]
            ydiff = yu.reshape(-1, 3)[:, :, None] - yu.reshape(-1, 3)[:, None, :]
            valid = ydiff.abs() > 1e-8
            if valid.any():
                loss = (
                    loss
                    + 0.01
                    * F.softplus(-ydiff[valid].sign() * pdiff[valid] / 0.05).mean()
                )
            if kind == "dynamics":
                loss = loss + 0.5 * F.mse_loss(out["h1"], h1[selected])
                with torch.no_grad():
                    teacher1 = model.encode_state(
                        inputs["train"][0][selected] + h1[selected],
                        out["context"].detach(),
                    )
                    teacher2 = model.encode_state(
                        inputs["train"][0][selected] + truth, out["context"].detach()
                    )
                loss = loss + 0.05 * (
                    F.mse_loss(out["z1"], teacher1) + F.mse_loss(out["z2"], teacher2)
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            prediction = model(*inputs["validation"])["prediction"].cpu().numpy()
        validation = evaluate_predictions(
            parts["validation"], prediction, weights, horizon
        )
        score = validation["ranking"]["top1"] - 0.1 * validation["utility"]["rmse"]
        curve.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "validation_score": score,
            }
        )
        if score > best_score + 1e-8:
            best_score, best, stale = score, copy.deepcopy(model.state_dict()), 0
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
        "kind": kind,
        "horizon": horizon,
        "hidden": hidden,
        "scalar": scalar,
        "ablations": list(ablations),
        "seed": seed,
        "utility_weights": list(weights),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "weights_sha256": file_digest(output / "weights.pt"),
        "seconds": time.perf_counter() - started,
        "epochs_completed": len(curve),
        "best_validation_score": best_score,
        "split_hashes": {
            p: digest([(r["state_id"], r["candidate_id"]) for r in rows])
            for p, rows in parts.items()
        },
        "training_trajectory_ids": sorted(trajectories[0]),
        "selection_trajectory_ids": sorted(trajectories[0] | trajectories[1]),
        "selection_state_ids": sorted(states[0] | states[1]),
        "training_config": {
            "epochs": epochs,
            "patience": patience,
            "learning_rate": learning_rate,
            "ranking_weight": 0.01,
        },
    }
    metadata.update(source_inventory())
    metadata["software_versions"] = versions()
    write_json(output / "metadata.json", metadata)
    write_json(output / "learning_curve.json", curve)
    forecaster = Forecaster(model, metadata, mean, scale, device)
    measured = {
        p: evaluate_predictions(rows, forecaster.predict(rows), weights, horizon)
        for p, rows in parts.items()
    }
    for p, rows in parts.items():
        measured[p]["future_h2"] = evaluate_predictions(
            rows, forecaster.predict(rows), weights, 2
        )
        if kind == "dynamics":
            with torch.inference_mode():
                out = model(*inputs[p])
                true1 = torch.tensor(
                    np.stack([labels(r, 1) for r in rows]),
                    dtype=torch.float32,
                    device=device,
                )
                true2 = targets[p]
                teacher1 = model.encode_state(inputs[p][0] + true1, out["context"])
                teacher2 = model.encode_state(inputs[p][0] + true2, out["context"])
                measured[p]["latent_rollout"] = {
                    "h1_mse": float(F.mse_loss(out["z1"], teacher1)),
                    "h2_mse": float(F.mse_loss(out["z2"], teacher2)),
                    "reference": "future capability encoding with fixed current-history context",
                }
                measured[p]["intermediate_h1"] = evaluate_predictions(
                    rows, out["h1"].cpu().numpy(), weights, 1
                )
    write_json(output / "metrics.json", measured)
    test_prediction = forecaster.predict(parts["test"])
    write_json(
        output / "test_predictions.json",
        [
            {
                "state_id": r["state_id"],
                "candidate_id": r["candidate_id"],
                "trajectory_id": r["trajectory_id"],
                "truth_h1": labels(r, 1).tolist(),
                "truth_h2": labels(r, 2).tolist(),
                "prediction": test_prediction[i].tolist(),
            }
            for i, r in enumerate(parts["test"])
        ],
    )
    write_json(
        output / "split_audit.json",
        {
            p: [
                {
                    "state_id": r["state_id"],
                    "candidate_id": r["candidate_id"],
                    "trajectory_id": r["trajectory_id"],
                    "model": r["model"],
                }
                for r in rows
            ]
            for p, rows in parts.items()
        },
    )
    return forecaster
