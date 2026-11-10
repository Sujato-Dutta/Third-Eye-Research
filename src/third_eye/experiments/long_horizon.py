"""Predeclared paired trajectories: two families, two scales, T=3 and T=5."""
from dataclasses import replace

import numpy as np

from third_eye.experiments.scale import MODELS as SCALE_MODELS

MODELS = {
    "qwen3-4b": {"official": "Qwen/Qwen3-4B",
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "seeds": list(range(2026110900, 2026110904)), "reuse_draw": 0, "gpu_hours": 24},
    "llama-3.2-3b-instruct": {"official": "meta-llama/Llama-3.2-3B-Instruct",
        "revision": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
        "seeds": list(range(2026111000, 2026111004)), "gpu_hours": 24},
    "qwen3-8b": {**{k: SCALE_MODELS["qwen3-8b"][k] for k in ("official", "revision")},
        "seeds": list(range(2026111100, 2026111104)), "gpu_hours": 32},
    "llama3.1-8b": {**{k: SCALE_MODELS["llama3.1-8b"][k] for k in ("official", "revision")},
        "seeds": list(range(2026111200, 2026111204)), "gpu_hours": 32},
}
POLICIES = ("native", "mitigated")
ROLES = ("target_test", "ood_test", "retention_test")
ENDPOINTS = ("g1", "native_h3_gain", "native_h5_gain", "native_h3_minus_h1",
             "native_h5_minus_h1", "mitigation_h3", "mitigation_h5", "native_h5_minus_h3")
DECLARATION = {
    "version": "long_horizon_20261109", "models": MODELS, "task": "math",
    "seeds_per_model": 4, "policies": list(POLICIES), "max_generation": 5,
    "scored_generations": [0, 1, 2, 3, 5], "horizons": [3, 5],
    "batch_size": 8, "steps": 25, "learning_rate": 3e-5,
    "mitigated_learning_rate": 1.5e-5, "rank": 16, "alpha": 32,
    "correction_attempts": 8, "training_prompts": 1024,
    "evaluation_items": {"target_test": 1319, "ood_test": 500, "retention_test": 1024},
    "qwen_reuse": "first four primary parents; canonical draw 0; no outcome-based choice",
    "donor_policy": "shared M1 donor at t2; own-state harvesting from t3, with paired RNG seeds",
    "terminal": "once correction-starved, preserve state for all remaining generations",
    "hold": "same pretrained checkpoint and fixed benchmark at every generation",
    "bootstrap_replicates": 5000, "bootstrap_seed": 20261109,
    "inference": "exploratory parent/shared-item intervals; simultaneous family across all endpoints/models",
    "scope": "four fixed backbones in two families; four paired seeds/model; one math task stream",
    "larger_primary_confirmation_unchanged": True, "forecaster_fits": 0, "vista_sus": 0,
    "worker_gpu_hour_ceiling": 450, "continuous_two_gpu_project_ceiling": 960,
    "calendar_days": 20, "additional_architecture_search": False,
}


def configured(template, model, seed):
    spec = MODELS[model]
    if seed not in spec["seeds"]:
        raise ValueError("Seed not in fixed trajectory cohort")
    chat = {"date_string": "11 Oct 2026"}
    if model.startswith("qwen"):
        chat["enable_thinking"] = False
    return replace(template,
        model=replace(template.model, name=spec["official"], revision=spec["revision"], chat_kwargs=chat),
        training=replace(template.training, learning_rate=3e-5, max_steps=25),
        protocol=replace(template.protocol, seed=seed, depth=5, candidate_size=8))


def step_seeds(seed, generation):
    if not 1 <= generation <= 5:
        raise ValueError("Trajectory generation outside fixed range")
    if generation == 1:
        return seed, seed+17, seed+101
    harvest = seed+10_000 if generation == 2 else seed+generation*1_000_000
    return harvest, harvest+19, harvest+31


def learning_rate(policy, generation):
    if policy not in POLICIES or not 1 <= generation <= 5:
        raise ValueError("Invalid policy/generation")
    return 1.5e-5 if policy == "mitigated" and generation >= 2 else 3e-5


def train_step(backend, config, policy, generation, batch, optimizer_seed, terminal=False):
    """No batch shrinkage, duplicates or optimizer reuse; terminal is absorbing."""
    if batch is not None and (len(batch) != 8 or len({c.example.id for c in batch}) != 8 or
                             any(c.example.split != "train" for c in batch)):
        raise ValueError("Eight unique training corrections required")
    if terminal and batch is not None:
        raise ValueError("Do not revive a terminal trajectory")
    before = backend.state_hash()
    backend.config = replace(config, training=replace(config.training,
        learning_rate=learning_rate(policy, generation)))
    available = batch is not None and not terminal
    training = backend.train(batch, optimizer_seed, max_steps=25) if available else None
    after = backend.state_hash()
    if training is not None and training["optimizer_steps"] != 25:
        raise ValueError("Matched optimizer budget changed")
    if not available and before != after:
        raise ValueError("Terminal transition changed parameters")
    return {"start_hash": before, "adapter_hash": after, "training": training,
        "optimizer_seed": optimizer_seed, "generation": generation, "policy": policy,
        "learning_rate": learning_rate(policy, generation), "continuation_available": int(available),
        "terminal": not available, "batch_ids": [c.example.id for c in batch] if batch is not None else []}


def item_contrasts(values):
    needed = ("m0", "m1", "native_t3", "native_t5", "mitigated_t3", "mitigated_t5")
    if any(k not in values for k in needed):
        raise ValueError("Missing fixed trajectory checkpoint")
    arrays = {k: np.asarray(values[k], dtype=float) for k in needed}
    shape = arrays["m0"].shape
    if len(shape) != 1 or not shape[0] or any(x.shape != shape or not np.isin(x, [0, 1]).all()
                                            for x in arrays.values()):
        raise ValueError("Aligned binary shared-item outcomes required")
    b, first, n3, n5, m3, m5 = (arrays[k] for k in needed)
    return np.stack((first-b, n3-b, n5-b, n3-first, n5-first, m3-n3, m5-n5, n5-n3))


def summarize(matrices, model, draws=5000):
    if model not in MODELS or set(matrices) != set(ROLES) or draws < 100:
        raise ValueError("Declared model, roles and bootstrap budget required")
    values = {r: np.asarray(matrices[r], dtype=float) for r in ROLES}
    if any(x.ndim != 3 or x.shape[:2] != (4, len(ENDPOINTS)) or x.shape[2] < 1 or
           not np.isfinite(x).all() for x in values.values()):
        raise ValueError("Four complete paired seeds per model required")
    rng = np.random.default_rng(DECLARATION["bootstrap_seed"]+list(MODELS).index(model))
    pw = rng.multinomial(4, np.full(4, .25), size=draws)/4
    points, boots = [], []
    for role in ROLES:
        data = values[role]
        j = data.shape[2]
        boot = np.empty((draws, len(ENDPOINTS)))
        for first in range(0, draws, 64):
            stop = min(first+64, draws)
            iw = rng.multinomial(j, np.full(j, 1/j), size=stop-first)/j
            averaged = np.einsum("bn,nej->bej", pw[first:stop], data, optimize=True)
            boot[first:stop] = np.einsum("bej,bj->be", averaged, iw, optimize=True)
        points.append(data.mean(axis=(0, 2)))
        boots.append(boot)
    point, boot = np.mean(points, axis=0), np.mean(boots, axis=0)
    alpha = .05/(len(MODELS)*len(ENDPOINTS))
    result = {}
    for i, name in enumerate(ENDPOINTS):
        result[name] = {"mean_pp": float(100*point[i]),
            "exploratory_ci95_pp": (100*np.quantile(boot[:, i], [.025, .975])).tolist(),
            "simultaneous_ci95_pp": (100*np.quantile(boot[:, i], [alpha/2, 1-alpha/2])).tolist(),
            "roles": {r: {"mean_pp": float(100*points[k][i]),
                "exploratory_ci95_pp": (100*np.quantile(boots[k][:, i], [.025, .975])).tolist()}
                for k, r in enumerate(ROLES)}}
    return {"model": model, "paired_seeds": 4, "endpoints": result,
            "scope": DECLARATION["scope"], "simultaneous_family_size": len(MODELS)*len(ENDPOINTS),
            "exploratory_extension": True, "independent_observations_are_seeds": True}
