"""Frozen empirical confirmation: choose on development, score independently."""

import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

from third_eye.experiments.mechanism import choose


DECLARED = {
    "version": "final_empirical_confirmation_20261022",
    "parents": 32,
    "parents_per_stream": 8,
    "selection_orientation": 0,
    "scoring_items": {"target_test": 128, "ood_test": 128, "retention_test": 256},
    "continuation_seed_offsets": [50_000_000, 100_000_000],
    "seed_grid": {
        "llama-3.2-3b-instruct/code": list(range(2026102200, 2026102208)),
        "llama-3.2-3b-instruct/math": list(range(2026102300, 2026102308)),
        "qwen3-4b/code": list(range(2026102400, 2026102408)),
        "qwen3-4b/math": list(range(2026102500, 2026102508)),
    },
    "primary": "qwen_minus_llama_forced_candidate_plus_continuation_utility",
    "primary_test": "two_sided_95pct_joint_parent_shared_item_bootstrap_CI",
    "planning": "at_least_80pct_conditional_power_at_75pct_pilot_effect_and_1.25_SD_inflation",
    "secondary_family": [
        ["family_admission", "hold"],
        ["family_admission", "forced"],
        ["family_admission", "replay"],
        ["replay", "hold"],
        ["replay", "forced"],
        ["forced_future", "forced_immediate"],
    ],
    "policies": ["hold", "forced", "replay", "family_admission"],
    "family_admission": "development-derived fixed baseline: commit Qwen, hold Llama; no trained predictor or test-time fitting",
    "forecaster_fits": 0,
    "cost_calibration_items": 8,
    "scored_checkpoints_per_parent": 13,
    "max_parallel_gpus": 20,
    "stage_su_cap": 560,
    "bootstrap_draws": 5000,
    "automatic_expansion": False,
    "scope": "two fixed backbones, two tasks, initial parents, K3 LoRA size2 updates, H2; no general-model or forecasting-success claim from primary",
    "terminal": "preserve M2=M1; no fabricated updates or favorable replacement seeds",
    "completion": "one fixed cohort then evidence freeze; no automatic follow-up, scaling or architecture sweep",
}


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: plain(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(child) for child in value]
    return value


def actions(stream, immediate):
    immediate = np.asarray(immediate, float)
    if immediate.shape != (3, 3) or not np.isfinite(immediate).all():
        raise ValueError("Complete finite development H1 heads required")
    if not stream.startswith(("qwen", "llama")):
        raise ValueError("Family baseline is scoped to the two frozen backbones")
    greedy = choose(immediate.mean(1))
    result = {
        "hold": np.zeros(3),
        "forced": greedy,
        "replay": greedy
        * ((immediate[:, 0] > 0) & (immediate[:, 1] >= 0) & (immediate[:, 2] >= 0)),
        "family_admission": greedy if stream.startswith("qwen") else np.zeros(3),
    }
    return {key: value.tolist() for key, value in result.items()}


def checkpoints(rows):
    if len(rows) != 3 or any(len(row["continuation_replicates"]) != 3 for row in rows):
        raise ValueError("Complete K3 by three native continuations required")
    output = [
        (
            "parent",
            Path(rows[0]["checkpoints"]["t1"]).parent.parent / "parent",
            rows[0]["parent_adapter_hash"],
        )
    ]
    for index, row in enumerate(rows):
        output.append(
            (
                f"m1_{index}",
                Path(row["checkpoints"]["t1"]),
                row["candidate_adapter_hash"],
            )
        )
        for draw, observation in enumerate(row["continuation_replicates"]):
            path = (
                Path(row["checkpoints"]["t2"])
                if draw == 0
                else Path(row["checkpoints"]["t1"]).parent
                / "continuation_replicates"
                / str(observation["seed"])
                / "t2"
            )
            output.append(
                (f"m2_{index}_{draw}", path, observation["continuation_adapter_hash"])
            )
    return output


def primary_power(point, boot):
    effect = point[6] * 0.75
    noise = (boot[:, 6] - point[6]) * 1.25
    sd = boot[:, 6].std(ddof=1) * 1.25
    critical = NormalDist().inv_cdf(0.975)
    return {
        "conditional_planned_power": float(np.mean(effect + noise - critical * sd > 0)),
        "shrunk_effect_pp": float(effect * 100),
        "inflated_sd_pp": float(sd * 100),
        "not_confirmed_power": True,
    }


def read_scored_items(directory, expected, adapter_hash, commitment_hash):
    directory = Path(directory)
    meta = json.loads((directory / "scores.json").read_text())
    if (
        meta["adapter_hash"] != adapter_hash
        or meta["commitment_sha256"] != commitment_hash
    ):
        raise ValueError("Final checkpoint or pre-scoring commitment differs")
    rows = [
        json.loads(line)
        for line in (directory / "items.jsonl").read_text().splitlines()
        if line.strip()
    ]
    output = {}
    for role, examples in expected.items():
        selected = [row for row in rows if row["role"] == role]
        if [row["id"] for row in selected] != [ex.id for ex in examples] or any(
            type(row["passed"]) is not bool for row in selected
        ):
            raise ValueError("Final paired item identities or binary outcomes differ")
        output[role] = np.asarray([row["passed"] for row in selected], float)
        if abs(output[role].mean() - meta["scores"][role]) > 1e-12:
            raise ValueError("Final aggregate/item scores differ")
    if len(rows) != sum(len(examples) for examples in expected.values()):
        raise ValueError("Unexpected final rows")
    return output
