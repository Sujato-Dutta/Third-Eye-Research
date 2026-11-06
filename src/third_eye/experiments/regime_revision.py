"""Saved-outcome sensitivities and frozen development rules for the next round."""

from decimal import Decimal, InvalidOperation
import random
import re

import numpy as np

from third_eye.evaluation.verifiers import NUMBER, extract_numeric


RECIPES = (
    {"id": "r0", "learning_rate": 1e-5, "steps": 10, "batch_size": 8},
    {"id": "r1", "learning_rate": 3e-5, "steps": 25, "batch_size": 8},
    {"id": "r2", "learning_rate": 1e-5, "steps": 25, "batch_size": 32},
    {"id": "r3", "learning_rate": 3e-5, "steps": 10, "batch_size": 32},
)


def lenient_numeric(text):
    """Reference-blind final-line sensitivity; never search for the gold value.

    Preserve any unambiguous strict prediction, even when incorrect. Otherwise
    accept a single final numeric value with a fixed answer/conclusion prefix
    and optional familiar unit. Arbitrary numbers inside reasoning are rejected.
    """
    strict = extract_numeric(text)
    if strict is not None:
        return strict
    body = text.split("</think>")[-1].strip()
    if not body:
        return None
    line = body.splitlines()[-1].strip()
    line = re.sub(r"^\s*(?:#{1,4}\s*|(?:the\s+)?(?:final\s+answer|answer)\s*(?:is|:|=)\s*|"
                  r"(?:therefore|thus|hence|so)[,:]?\s*)", "", line, flags=re.I)
    line = line.strip().strip("*").strip()
    units = (r"(?:dollars?|cents?|apples?|people|persons?|students?|books?|"
             r"minutes?|hours?|days?|weeks?|years?|miles?|meters?|metres?|"
             r"kilometers?|kilometres?|feet|inches|cm|kg|units?|%)")
    match = re.fullmatch(rf"\$?\s*({NUMBER})\s*(?:{units})?\s*[.!]?", line, re.I)
    if not match:
        return None
    try:
        value = Decimal(match.group(1).replace(",", ""))
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def factor_pairs(weights):
    """Normalize PEFT adapter keys without conflating distinct projection layers."""
    output = {}
    for key, value in weights.items():
        match = re.fullmatch(r"(.+)\.lora_([AB])(?:\.[^.]+)?\.weight", key)
        if not match:
            raise ValueError("Unsupported adapter parameter: " + key)
        layer, side = match.groups()
        pair = output.setdefault(layer, {})
        if side in pair:
            raise ValueError("Multiple adapters for one layer")
        pair[side] = np.asarray(value, dtype=np.float64)
    for pair in output.values():
        if set(pair) != {"A", "B"} or any(x.ndim != 2 for x in pair.values()):
            raise ValueError("Complete matrix-valued LoRA factors required")
        if pair["A"].shape[0] != pair["B"].shape[1]:
            raise ValueError("LoRA rank dimensions differ")
        if not all(np.isfinite(x).all() for x in pair.values()):
            raise ValueError("Nonfinite LoRA factors")
    if not output:
        raise ValueError("No LoRA factors")
    return output


def effective_distance(left, right, scale_left, scale_right):
    """Frobenius norm of effective weight difference, using low-rank Gram matrices."""
    if set(left) != set(right) or not all(
        np.isfinite(s) and s > 0 for s in (scale_left, scale_right)
    ):
        raise ValueError("Compatible layers and positive scales required")
    total, layers = 0.0, {}
    for name in left:
        a0, b0 = left[name]["A"], left[name]["B"]
        a1, b1 = right[name]["A"], right[name]["B"]
        if a0.shape[1] != a1.shape[1] or b0.shape[0] != b1.shape[0]:
            raise ValueError("Effective weight dimensions differ")
        n0 = scale_left**2 * float(np.sum((b0.T @ b0) * (a0 @ a0.T).T))
        n1 = scale_right**2 * float(np.sum((b1.T @ b1) * (a1 @ a1.T).T))
        cross = scale_left * scale_right * float(
            np.sum((b0.T @ b1) * (a1 @ a0.T).T)
        )
        squared = n0 + n1 - 2 * cross
        if squared < -1e-9 * max(n0 + n1, 1):
            raise ArithmeticError("Invalid negative squared effective distance")
        layers[name] = float(np.sqrt(max(0, squared)))
        total += max(0, squared)
    return {"frobenius_norm": float(np.sqrt(total)), "layers": layers}


def unique_batch(pool, size, seed):
    by_id = {}
    for correction in pool:
        key = correction.example.id
        if key in by_id:
            raise ValueError("Correction pool contains repeated problem identities")
        by_id[key] = correction
    if len(by_id) < size:
        return None
    ordered = [by_id[k] for k in sorted(by_id)]
    random.Random(seed).shuffle(ordered)
    return tuple(ordered[:size])


def select_regime(records):
    """Use all four declared development parents; no final outcomes enter."""
    eligible = []
    expected = {
        "llama-3.2-3b-instruct": set(range(2026110600, 2026110604)),
        "qwen3-4b": set(range(2026110700, 2026110704)),
    }
    for model, seeds in expected.items():
        for recipe in RECIPES:
            rows = [r for r in records if r["model"] == model and
                    r["recipe"] == recipe["id"]]
            if len(rows) != 4 or {r["seed"] for r in rows} != seeds:
                continue
            if any(not r["available"] for r in rows):
                continue
            gains = np.asarray([r["development_head_gains"] for r in rows], float)
            if gains.shape != (4, 3) or not np.isfinite(gains).all():
                raise ValueError("Finite complete development head gains required")
            mean = float(gains.mean())
            retention = float(gains[:, 2].mean())
            if mean >= .01 - 1e-12 and retention >= -.005 - 1e-12:
                eligible.append({"model": model, "recipe": recipe["id"],
                                 "mean_g1": mean, "mean_retention": retention})
    return max(eligible, key=lambda r: (r["mean_g1"],
        -list(expected).index(r["model"]), -int(r["recipe"][1:]))) if eligible else None


def paired_bootstrap(states, matrices, names, draws=5000):
    """Descriptive model-specific and original-core intervals, shared item draws."""
    if not states or len(states) != len(matrices):
        raise ValueError("Complete paired states and matrices required")
    roles = ("target_test", "ood_test", "retention_test")
    rng = np.random.default_rng(20261106)
    groups = sorted({s["stream"] for s in states})
    shared, seen, result = {}, {}, {}
    for group in groups:
        indices = [i for i, s in enumerate(states) if s["stream"] == group]
        n = len(indices)
        choices = rng.integers(n, size=(draws, n))
        point = np.zeros(len(names))
        boot = np.zeros((draws, len(names)))
        for j, index in enumerate(indices):
            state = states[index]
            weight = (choices == j).sum(1) / n
            for role in roles:
                identity = tuple(tuple(x) for x in state["ids"][role])
                if any(set(x) & set(identity) and x != identity
                       for x in seen.get(role, set())):
                    raise ValueError("Partially shared item identity pools")
                seen.setdefault(role, set()).add(identity)
                x = np.asarray(matrices[index][role], float)
                if x.shape != (len(names), len(identity)) or not np.isfinite(x).all():
                    raise ValueError("Complete finite paired outcome matrices required")
                key = role, identity
                if key not in shared:
                    shared[key] = rng.integers(len(identity), size=(draws, len(identity)))
                point += x.mean(1) / (3 * n)
                boot += x[:, shared[key]].mean(2).T * weight[:, None] / 3
        result[group] = point, boot

    def summarize(group_names):
        p = np.mean([result[g][0] for g in group_names], axis=0)
        b = np.mean([result[g][1] for g in group_names], axis=0)
        return {name: {"mean_pp": float(p[i] * 100),
            "ci95_pp": (np.quantile(b[:, i], [.025, .975]) * 100).tolist()}
            for i, name in enumerate(names)}

    output = {"streams": {g: summarize([g]) for g in groups}, "models": {}}
    for model in sorted({g.rsplit("/", 1)[0] for g in groups}):
        gs = [g for g in groups if g.rsplit("/", 1)[0] == model]
        if len(gs) == 2:
            output["models"][model] = summarize(gs)
    core = [g for g in groups if g.rsplit("/", 1)[0] in
            {"qwen3-4b", "llama-3.2-3b-instruct"}]
    if len(core) == 4:
        output["original_32_run_core"] = summarize(core)
    return output
