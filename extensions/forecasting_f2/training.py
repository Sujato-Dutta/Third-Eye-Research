"""Train candidate-relative predictions without opening the test partition."""

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
from third_eye.forecasting.training import evaluate_predictions
from third_eye.io import digest, file_digest, write_json
from third_eye.provenance import source_inventory, versions
from third_eye.statistics.metrics import ranking
from extensions.forecasting_f2.model import build_model

EXTENSION = "third_eye_contrast_f2_v1"


def extension_inventory():
    root = Path(__file__).resolve().parents[2]
    paths = [*Path(__file__).parent.glob("*.py"), root / "extensions/__init__.py"]
    files = {p.relative_to(root).as_posix(): file_digest(p) for p in sorted(paths)}
    return {
        "extension": EXTENSION,
        "extension_source_sha256": digest(files),
        "extension_files": files,
    }


def group_inputs(records, ablations=()):
    groups = defaultdict(list)
    for i, row in enumerate(records):
        groups[row["state_id"]].append(i)
    indices = list(groups.values())
    for group in indices:
        if len(group) != 3 or len({records[i]["candidate_id"] for i in group}) != 3:
            raise ValueError(
                "Each inference/training state must have three unique candidates"
            )
        context = []
        for i in group:
            p = records[i]["precommit"]
            context.append(
                digest(
                    {
                        k: p.get(k)
                        for k in ["state", "history", "pool_statistics", "generation"]
                    }
                )
            )
        if len(set(context)) != 1:
            raise ValueError(
                "Candidates do not share the same pre-commit parent context"
            )
    if not indices:
        raise ValueError("No complete candidate states")
    encoded = [encode(r, ablations) for r in records]
    arrays = tuple(
        np.stack([[encoded[i][field] for i in g] for g in indices])
        for field in range(4)
    )
    return arrays, np.asarray(indices, dtype=np.int64)


def group_targets(records, indices, horizon):
    return np.stack([[labels(records[i], horizon) for i in g] for g in indices]).astype(
        np.float32
    )


def centered(x):
    return x - x.mean(1, keepdim=True)


def head_rankings(records, prediction, horizon):
    groups = defaultdict(list)
    for i, row in enumerate(records):
        groups[row["state_id"]].append(i)
    truth = np.stack([labels(r, horizon) for r in records])
    result = {}
    for head, column in [("target", 0), ("ood", 1), ("retention", 2)]:
        scores = [
            ranking(truth[ix, column], prediction[ix, column]) for ix in groups.values()
        ]
        defined = [r["spearman"] for r in scores if r["spearman"] is not None]
        informative = [r for r in scores if r["chance_top1"] < 1]
        result[head] = {
            "states": len(scores),
            "defined_rank_states": len(defined),
            "spearman": float(np.mean(defined)) if defined else None,
            "informative_top1": float(np.mean([r["top1"] for r in informative]))
            if informative
            else None,
            "chance_top1": float(np.mean([r["chance_top1"] for r in informative]))
            if informative
            else None,
        }
    return result


def vector_loss(predicted, truth, comparative=True):
    if not comparative:
        return F.mse_loss(predicted, truth)
    return F.mse_loss(centered(predicted), centered(truth)) + 0.1 * F.mse_loss(
        predicted.mean(1), truth.mean(1)
    )


def objective(out, truth, immediate, weights, scale, utility_scale, horizon, variant):
    # Model outputs are normalized deltas; aggregate physical consequence units.
    comparative = variant != "no_comparative_loss"
    loss = vector_loss(out["prediction"], truth, comparative)
    if "h1" in out and variant != "no_temporal_aux":
        residual = truth - immediate if horizon == 2 else torch.zeros_like(truth)
        loss = loss + 0.25 * vector_loss(out["h1"], immediate, comparative)
        loss = loss + 0.25 * vector_loss(out["continuation"], residual, comparative)
    if comparative:
        predicted_u = (out["prediction"] * scale) @ weights
        true_u = (truth * scale) @ weights
        pdiff = predicted_u[:, :, None] - predicted_u[:, None, :]
        tdiff = true_u[:, :, None] - true_u[:, None, :]
        valid = tdiff.abs() > 1e-8
        if valid.any():
            loss = (
                loss
                + 0.25
                * F.softplus(-tdiff[valid].sign() * pdiff[valid] / utility_scale).mean()
            )
    return loss


class ContrastForecaster:
    def __init__(self, model, metadata, device="cpu"):
        self.model, self.metadata, self.device = (
            model.to(device).eval(),
            metadata,
            device,
        )

    def predict(self, records):
        arrays, indices = group_inputs(records, self.metadata.get("ablations", []))
        mean, std = (
            np.asarray(self.metadata["feature_mean"]),
            np.asarray(self.metadata["feature_scale"]),
        )
        arrays = list(arrays)
        arrays[1] = np.clip((arrays[1] - mean) / std, -20, 20)
        with torch.inference_mode():
            tensors = [
                torch.as_tensor(a, dtype=torch.float32, device=self.device)
                for a in arrays
            ]
            prediction = self.model(*tensors)["prediction"].cpu().numpy() * np.asarray(
                self.metadata["target_scale"]
            )
        result = np.empty((len(records), 3), dtype=np.float64)
        result[indices.reshape(-1)] = prediction.reshape(-1, 3)
        return result

    def select(self, views):
        if len({r["state_id"] for r in views}) != 1:
            raise ValueError(
                "Online selection requires one shared-parent candidate set"
            )
        return int(
            np.argmax(
                self.predict(views) @ np.asarray(self.metadata["utility_weights"])
            )
        )

    def save(self, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        torch.save(
            {k: v.cpu() for k, v in self.model.state_dict().items()},
            output / "weights.pt",
        )
        self.metadata["weights_sha256"] = file_digest(output / "weights.pt")
        write_json(output / "metadata.json", self.metadata)
        write_json(
            output / "artifact_manifest.json",
            {
                "metadata_sha256": file_digest(output / "metadata.json"),
                "weights_sha256": self.metadata["weights_sha256"],
            },
        )

    @classmethod
    def load(cls, output, device="cpu"):
        output = Path(output)
        artifact = json.loads((output / "artifact_manifest.json").read_text())
        if file_digest(output / "metadata.json") != artifact["metadata_sha256"]:
            raise ValueError("F2 metadata checksum mismatch")
        metadata = json.loads((output / "metadata.json").read_text())
        if (
            metadata.get("extension") != EXTENSION
            or metadata["schema_version"] != SCHEMA_VERSION
            or metadata["feature_names"] != list(FEATURE_NAMES)
        ):
            raise ValueError("Unknown F2 artifact or feature schema")
        if (
            metadata["extension_source_sha256"]
            != extension_inventory()["extension_source_sha256"]
        ):
            raise ValueError("F2 implementation changed after fitting")
        if metadata["source_tree_sha256"] != source_inventory()["source_tree_sha256"]:
            raise ValueError("Frozen base source changed after F2 fitting")
        if (
            file_digest(output / "weights.pt") != metadata["weights_sha256"]
            or artifact["weights_sha256"] != metadata["weights_sha256"]
        ):
            raise ValueError("F2 weights checksum mismatch")
        model = build_model(metadata["variant"], metadata["hidden"])
        model.load_state_dict(
            torch.load(output / "weights.pt", map_location="cpu", weights_only=True)
        )
        return cls(model, metadata, device)


def train(
    train_rows,
    validation_rows,
    output,
    horizon=2,
    variant="full",
    seed=42,
    epochs=200,
    patience=40,
    hidden=16,
    weights=(1 / 3, 1 / 3, 1 / 3),
    learning_rate=1e-3,
    ablations=(),
):
    if horizon not in {1, 2} or min(epochs, patience, hidden) < 1:
        raise ValueError("Invalid forecast horizon or fitting budget")
    if (
        len(weights) != 3
        or not np.isclose(sum(weights), 1)
        or any(w < 0 for w in weights)
    ):
        raise ValueError("Fixed physical utility weights required")
    if learning_rate <= 0 or not np.isfinite(learning_rate):
        raise ValueError("Invalid learning rate")
    if Path(output).exists():
        raise FileExistsError("Use a new immutable F2 output directory")
    train_ids = {r["trajectory_id"] for r in train_rows}
    val_ids = {r["trajectory_id"] for r in validation_rows}
    if train_ids & val_ids or {r["state_id"] for r in train_rows} & {
        r["state_id"] for r in validation_rows
    }:
        raise ValueError("Training/validation trajectory leakage")
    if {r.get("protocol_amendment") for r in train_rows + validation_rows} != {"A2"}:
        raise ValueError("F2 requires unmixed frozen A2 labels")
    if any(r.get("protocol_status") == "pilot" for r in train_rows + validation_rows):
        raise ValueError("F2 does not accept unfrozen pilot labels")
    if any(
        "utility_weights" in r and not np.allclose(r["utility_weights"], weights)
        for r in train_rows + validation_rows
    ):
        raise ValueError("Forecaster utility weights differ from frozen labels")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    arrays, indices = group_inputs(train_rows, ablations)
    group_inputs(validation_rows, ablations)
    mean = arrays[1].mean(axis=(0, 1))
    std = np.maximum(arrays[1].std(axis=(0, 1)), 1e-6)
    arrays = list(arrays)
    arrays[1] = np.clip((arrays[1] - mean) / std, -20, 20)
    primary = group_targets(train_rows, indices, horizon)
    immediate = group_targets(train_rows, indices, 1)
    if (
        not np.isfinite(primary).all()
        or not np.isfinite(immediate).all()
        or np.abs(primary).max() > 1
        or np.abs(immediate).max() > 1
    ):
        raise ValueError("Targets must be finite accuracy-fraction deltas")
    # Both arms use the same train-only H=1 scales; H=1 never reads H=2 targets.
    resolution = np.asarray([1 / 64, 1 / 64, 1 / 256], dtype=np.float32)
    scale = np.maximum(
        (immediate - immediate.mean(1, keepdims=True)).std(axis=(0, 1)), resolution
    )
    u1 = immediate @ np.asarray(weights)
    utility_scale = max(float((u1 - u1.mean(1, keepdims=True)).std()), 1e-3)
    inputs = [torch.as_tensor(a, dtype=torch.float32) for a in arrays]
    truth, h1 = torch.tensor(primary / scale), torch.tensor(immediate / scale)
    scale_t, weight_t = torch.tensor(scale), torch.tensor(weights, dtype=torch.float32)
    model = build_model(variant, hidden)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    metadata = {
        "extension": EXTENSION,
        "schema_version": SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "architecture": "contrastive_continuation"
        if variant != "independent"
        else "original_gru_comparative_objective",
        "kind": "direct" if horizon == 2 else "matched_h1",
        "kind_semantics": "online policy role; architecture is explicit",
        "method": "Third Eye-Contrast"
        if variant != "independent"
        else "Original GRU with comparative fitting",
        "horizon": horizon,
        "variant": variant,
        "hidden": 64 if variant == "independent" else hidden,
        "seed": seed,
        "scalar": False,
        "ablations": list(ablations),
        "feature_mean": mean.tolist(),
        "feature_scale": std.tolist(),
        "target_scale": scale.tolist(),
        "utility_weights": list(weights),
        "training_trajectory_ids": sorted(train_ids),
        "selection_trajectory_ids": sorted(train_ids | val_ids),
        "selection_state_ids": sorted(
            {r["state_id"] for r in train_rows + validation_rows}
        ),
        "split_hashes": {
            p: digest([(r["state_id"], r["candidate_id"]) for r in rows])
            for p, rows in [("train", train_rows), ("validation", validation_rows)]
        },
        "training_config": {
            "epochs": epochs,
            "patience": patience,
            "learning_rate": learning_rate,
            "rank_weight": 0.25,
            "temporal_aux_weight": 0.25,
            "mean_weight": 0.1,
            "utility_temperature_train_h1": utility_scale,
            "checkpoint_score": "0.5*(own-horizon Spearman+informative_top1) - 0.01*utility_RMSE",
        },
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "test_evaluated": False,
        "protocol_amendment": "A2",
        **source_inventory(),
        **extension_inventory(),
        "software_versions": versions(),
    }
    forecaster = ContrastForecaster(model, metadata)
    best, best_score, stale, curve = None, -float("inf"), 0, []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(indices))
        losses = []
        for batch in order.split(16):
            optimizer.zero_grad(set_to_none=True)
            out = model(*(a[batch] for a in inputs))
            loss = objective(
                out,
                truth[batch],
                h1[batch],
                weight_t,
                scale_t,
                utility_scale,
                horizon,
                variant,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite F2 loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        measured = evaluate_predictions(
            validation_rows, forecaster.predict(validation_rows), weights, horizon
        )
        rho = measured["ranking"]["spearman"]
        top1 = measured["ranking"]["informative_top1"]
        score = (
            0.5 * ((rho if rho is not None else -1) + (top1 if top1 is not None else 0))
            - 0.01 * measured["utility"]["rmse"]
        )
        curve.append(
            {
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "validation_score": score,
                "validation_spearman": rho,
                "validation_top1": top1,
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
    metadata.update(
        seconds=time.perf_counter() - started,
        epochs_completed=len(curve),
        best_validation_score=best_score,
        best_epoch=max(curve, key=lambda r: r["validation_score"])["epoch"],
    )
    output = Path(output)
    forecaster.save(output)
    write_json(output / "learning_curve.json", curve)
    metrics = {}
    for part, rows in [("train", train_rows), ("validation", validation_rows)]:
        predicted = forecaster.predict(rows)
        metrics[part] = evaluate_predictions(rows, predicted, weights, horizon)
        metrics[part]["future_h2"] = evaluate_predictions(rows, predicted, weights, 2)
        metrics[part]["head_state_rankings"] = head_rankings(rows, predicted, horizon)
    write_json(output / "metrics.json", metrics)
    # H=1 future-H=2 metrics are reporting only, never checkpoint selection.
    write_json(
        output / "split_audit.json",
        {
            p: [
                {
                    "state_id": r["state_id"],
                    "candidate_id": r["candidate_id"],
                    "trajectory_id": r["trajectory_id"],
                }
                for r in rs
            ]
            for p, rs in [("train", train_rows), ("validation", validation_rows)]
        },
    )
    return forecaster
