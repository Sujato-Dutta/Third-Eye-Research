"""Repeated-continuation targets and candidate-set inputs for Third Eye.

Outcome repetitions are supervision only. They never enter the input encoder.
The estimand is conditional on the saved M1 and the declared evaluation items;
finite-item uncertainty and epistemic uncertainty are not continuation variance.
"""

from collections import defaultdict

import numpy as np

from third_eye.forecasting.encoding import HEADS, encode, labels


TARGET_VERSION = 1


def reference_h1(row):
    """Use matched reevaluation supervision without rewriting native labels."""
    reference = row.get("continuation_reference")
    if reference is None:
        return labels(row, 1)
    if (
        reference.get("schema_version") != 1
        or reference.get("source") != "matched_control_reevaluation"
    ):
        raise ValueError("Unknown continuation reference schema")
    for field in ("candidate_adapter_hash", "parent_adapter_hash", "manifest_hash"):
        if reference.get(field) != row.get(field):
            raise ValueError("Continuation reference checkpoint/manifest differs")
    value = np.asarray([reference["h1"][h] for h in HEADS], dtype=float)
    if not np.isfinite(value).all() or (np.abs(value) > 1).any():
        raise ValueError("Invalid matched immediate consequence")
    return value


def state_groups(records):
    groups = defaultdict(list)
    for i, row in enumerate(records):
        groups[row["state_id"]].append(i)
    for indices in groups.values():
        rows = [records[i] for i in indices]
        if len(rows) != 3 or len({r["candidate_id"] for r in rows}) != 3:
            raise ValueError(
                "Expected-value forecasting requires whole distinct K=3 states"
            )
        for field in ("trajectory_id", "parent_adapter_hash", "manifest_hash"):
            if len({r.get(field) for r in rows}) != 1:
                raise ValueError(f"Unmatched {field} within candidate set")
    return list(groups.values())


def candidate_inputs(records, ablations=()):
    """Absolute and centered features, with availability-aware centering.

    Center only within the current parent, never over trajectories or splits.
    Missing features retain their mask and do not contribute a synthetic zero
    to the mean of observed siblings. Candidate identities only define groups.
    """
    arrays = [np.stack(a) for a in zip(*(encode(r, ablations) for r in records))]
    state, features, history, mask = arrays
    n = features.shape[1] // 2
    values, available = features[:, :n], features[:, n:]
    relative = np.zeros_like(values)
    for indices in state_groups(records):
        observed = available[indices]
        center = (values[indices] * observed).sum(0) / np.maximum(1, observed.sum(0))
        relative[indices] = (values[indices] - center) * observed
    return np.concatenate(
        [state, features, relative, history.reshape(len(records), -1), mask], axis=1
    ).astype(np.float32)


def repeated_targets(records, minimum_replicates=3):
    """Validate independent harvest seeds for exactly the same candidate M1.

    Returns per-candidate means, unbiased covariance, and the raw draws. Never
    interpret an unreplicated historical H2 label as a zero-variance target.
    All candidates in a parent must share one predeclared seed grid and budget.
    """
    if minimum_replicates < 3:
        raise ValueError("At least three continuation draws are required")
    targets, covariances, draws, grids, budgets = [], [], [], [], []
    for row in records:
        immediate = reference_h1(row)
        observations = row.get("continuation_replicates", [])
        if len(observations) < minimum_replicates:
            raise ValueError(
                "Missing repeated-continuation supervision; single H2 is insufficient"
            )
        seeds = [r["seed"] for r in observations]
        if any(type(seed) is not int or seed < 0 for seed in seeds) or len(
            set(seeds)
        ) != len(seeds):
            raise ValueError("Continuation seeds must be distinct nonnegative integers")
        m1 = row.get("candidate_adapter_hash")
        if not m1:
            raise ValueError("Repeated labels must bind the candidate M1 hash")
        values, reserved = [], set()
        for observation in observations:
            if observation["candidate_adapter_hash"] != m1:
                raise ValueError("Continuation draw starts from a different M1")
            if observation["evaluation_manifest_hash"] != row["manifest_hash"]:
                raise ValueError("Repeated labels use a different evaluation manifest")
            value = np.asarray([observation["h2"][h] for h in HEADS], dtype=float)
            if not np.isfinite(value).all() or (np.abs(value) > 1).any():
                raise ValueError("Invalid continuation consequence")
            available = observation["continuation_available"]
            steps = observation["optimizer_steps"]
            budget = observation["reserved_optimizer_steps"]
            if available not in (0, 1) or type(budget) is not int or budget <= 0:
                raise ValueError(
                    "Invalid continuation availability or optimizer budget"
                )
            if available:
                if steps != budget:
                    raise ValueError("Nonterminal continuation budget differs")
            elif (
                steps != 0
                or observation["continuation_adapter_hash"] != m1
                or not np.allclose(value, immediate, atol=1e-12, rtol=0)
            ):
                raise ValueError("Terminal continuation must preserve M2=M1")
            if not observation["continuation_adapter_hash"]:
                raise ValueError("Missing continuation checkpoint hash")
            reserved.add(budget)
            values.append(value)
        if len(reserved) != 1:
            raise ValueError("Unequal reserved continuation budgets")
        order = np.argsort(seeds)
        values = np.asarray(values)[order]
        targets.append(values.mean(0))
        covariances.append(np.cov(values, rowvar=False, ddof=1))
        draws.append(values)
        grids.append(tuple(sorted(seeds)))
        budgets.append(next(iter(reserved)))
    for indices in state_groups(records):
        if (
            len({grids[i] for i in indices}) != 1
            or len({budgets[i] for i in indices}) != 1
        ):
            raise ValueError(
                "Candidates must share continuation seeds and reserved budgets"
            )
    return np.asarray(targets), np.asarray(covariances), draws


def reliability_audit(records, weights=(1 / 3, 1 / 3, 1 / 3)):
    """Descriptive signal/variation audit; not a population claim or a gate."""
    mean, covariance, draws = repeated_targets(records)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (3,) or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError("Utility weights must sum to one")
    utility = mean @ weights
    within = np.einsum("i,nij,j->n", weights, covariance, weights)
    rows = []
    for indices in state_groups(records):
        count = len(draws[indices[0]])
        between = float(np.var(utility[indices], ddof=1))
        raw = np.stack([draws[i] @ weights for i in indices])
        centered_covariance = np.cov(raw - raw.mean(0), ddof=1)
        noise_of_mean = float(np.trace(centered_covariance) / (2 * count))
        # Uniform mass over tied winners; compare independent draw pairs.
        winner = np.isclose(raw, raw.max(0), atol=1e-12, rtol=0)
        mass = winner / winner.sum(0)
        agreement = [
            float(mass[:, a] @ mass[:, b])
            for a in range(count)
            for b in range(a + 1, count)
        ]
        rows.append(
            {
                "state_id": records[indices[0]]["state_id"],
                "trajectory_id": records[indices[0]]["trajectory_id"],
                "continuation_draws": count,
                "between_candidate_mean_variance": between,
                "within_candidate_draw_variance": float(np.mean(within[indices])),
                "mean_measurement_variance": noise_of_mean,
                "noise_corrected_between_variance": between - noise_of_mean,
                "independent_draw_winner_agreement": float(np.mean(agreement)),
            }
        )
    return {
        "target_version": TARGET_VERSION,
        "states": rows,
        "interpretation": "Conditional descriptive audit, not a gate. Mean-noise correction retains matched-seed covariance across candidates; shared finite scoring items and small draw counts still limit interpretation.",
    }
