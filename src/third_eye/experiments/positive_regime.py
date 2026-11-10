"""Frozen positive-regime confirmation and paired parent/item inference."""

import numpy as np


DECLARATION = {
    "version": "positive_regime_20261109",
    "model": "qwen3-4b", "task": "math", "recipe": "r1",
    "seeds": list(range(2026110900, 2026110920)),
    "batch_size": 8, "steps": 25, "learning_rate": 3e-5,
    "mitigation_learning_rate": 1.5e-5, "continuation_draws": 3,
    "prespecified_initial_updates_per_parent": 1,
    "initial_batch_seed_offset": 17, "initial_optimizer_seed_offset": 101,
    "draw_seed_offsets": [10000, 110000, 210000],
    "donor_batch_seed_offset": 19, "continuation_optimizer_seed_offset": 31,
    "evaluation_items": {"target_test": 1319, "ood_test": 500, "retention_test": 1024},
    "bootstrap_replicates": 5000, "bootstrap_seed": 20261109,
    "primary_family": ["g1", "native_increment", "native_minus_direct", "interaction"],
    "mitigation_family": ["mitigation_gain", "mitigation_retention"],
    "retention_noninferiority_margin": -.005,
    "initial_scarcity": "identity M1=M0; all continuation arms identity; retain parent",
    "continuation_scarcity": "native=mitigated=M1; direct=M0; retain draw",
    "correction_attempts": 8, "training_prompts": 1024,
    "max_new_tokens": 512, "sft_max_tokens": 1024,
    "primary_utility": "equal mean of target, OOD and retention accuracy",
    "draw_aggregation": "average three paired draws within parent before resampling",
    "inference": "joint parent/shared-item bootstrap; Bonferroni within each declared family",
    "missingness": "no replacement or exclusion; incomplete cohort has no confirmatory claim",
    "mitigation_rationale": "isolated second-step LR sensitivity; not selected for observed H2 benefit",
    "forecaster_fits": 0, "vista_sus": 0, "gpu_concurrency": 1,
}

ROLES = ("target_test", "ood_test", "retention_test")
ENDPOINTS = ("g1", "native_g2", "native_increment", "direct_gain",
             "native_minus_direct", "interaction", "mitigation_gain")
ARMS = ("native", "direct", "mitigated")


def arm_plan(parent_hash, m1_hash, batch, optimizer_seed):
    """One donor batch, order and optimizer seed; only starting state/LR differ."""
    if batch is not None and (len(batch) != DECLARATION["batch_size"] or
            len({c.example.id for c in batch}) != len(batch)):
        raise ValueError("Exactly eight distinct verified corrections required")
    return {arm: {"start_hash": parent_hash if arm == "direct" else m1_hash,
        "learning_rate": DECLARATION["mitigation_learning_rate"] if arm == "mitigated"
                         else DECLARATION["learning_rate"],
        "steps": DECLARATION["steps"] if batch is not None else 0,
        "optimizer_seed": optimizer_seed,
        "batch_ids": [c.example.id for c in batch] if batch is not None else [],
        "continuation_available": int(batch is not None)} for arm in ARMS}


def item_contrasts(parent, m1, draws):
    """Binary item outcomes -> paired endpoints, with draws nested in parent."""
    parent, m1, draws = (np.asarray(x, dtype=float) for x in (parent, m1, draws))
    if (parent.ndim != 1 or m1.shape != parent.shape or
            draws.shape != (3, 3, len(parent)) or len(parent) == 0):
        raise ValueError("Complete aligned parent, M1 and three three-arm draws required")
    if any(not np.isin(x, [0, 1]).all() for x in (parent, m1, draws)):
        raise ValueError("Binary finite item outcomes required")
    native, direct, mitigated = draws.mean(axis=0)
    g1 = m1-parent
    increment = native-m1
    return np.stack((g1, native-parent, increment, direct-parent,
                     native-direct, increment-(direct-parent), mitigated-native))


def summarize(matrices, draws=5000, seed=20261109):
    """Resample parents and the same items across all arms and parents per role.

    Continuation randomness is part of each parent's three-draw measurement;
    draws do not inflate the number of independent parent experiments.
    """
    if set(matrices) != set(ROLES) or draws < 100:
        raise ValueError("Three roles and at least 100 bootstrap replicates required")
    data = {r: np.asarray(matrices[r], dtype=float) for r in ROLES}
    n = data[ROLES[0]].shape[0]
    if n < 2 or any(x.ndim != 3 or x.shape[:2] != (n, len(ENDPOINTS)) or
                   x.shape[2] < 1 or not np.isfinite(x).all() for x in data.values()):
        raise ValueError("Aligned finite parent/endpoint/item tensors required")
    rng = np.random.default_rng(seed)
    pw = rng.multinomial(n, np.full(n, 1/n), size=draws)/n
    role_point, role_boot = {}, {}
    for role, values in data.items():
        j = values.shape[2]
        boot = np.empty((draws, len(ENDPOINTS)))
        for first in range(0, draws, 64):
            stop = min(first+64, draws)
            iw = rng.multinomial(j, np.full(j, 1/j), size=stop-first)/j
            averaged = np.einsum("bn,nej->bej", pw[first:stop], values, optimize=True)
            boot[first:stop] = np.einsum("bej,bj->be", averaged, iw, optimize=True)
        role_point[role] = values.mean(axis=(0, 2))
        role_boot[role] = boot
    point = np.mean(list(role_point.values()), axis=0)
    boot = np.mean(list(role_boot.values()), axis=0)

    def estimate(value, samples, family_size=1):
        alpha = .05/family_size
        return {"mean_pp": float(value*100),
                "ci_pp": (np.quantile(samples, [alpha/2, 1-alpha/2])*100).tolist(),
                "family_size": family_size, "family_coverage": .95}

    primary = DECLARATION["primary_family"]
    result = {"parents": n, "draws_per_parent": 3, "bootstrap_replicates": draws,
              "endpoints": {}, "roles": {}, "forecaster_fits": 0}
    for i, name in enumerate(ENDPOINTS):
        size = len(primary) if name in primary else 2 if name == "mitigation_gain" else 1
        result["endpoints"][name] = estimate(point[i], boot[:, i], size)
        result["roles"][name] = {r: estimate(role_point[r][i], role_boot[r][:, i]) for r in ROLES}
    i = ENDPOINTS.index("mitigation_gain")
    result["mitigation_retention"] = estimate(
        role_point["retention_test"][i], role_boot["retention_test"][:, i], 2)
    result["interpretation"] = {
        "confirmed_positive_g1": result["endpoints"]["g1"]["ci_pp"][0] > 0,
        "confirmed_negative_increment": result["endpoints"]["native_increment"]["ci_pp"][1] < 0,
        "confirmed_negative_interaction": result["endpoints"]["interaction"]["ci_pp"][1] < 0,
        "mitigation_utility_improves": result["endpoints"]["mitigation_gain"]["ci_pp"][0] > 0,
        "mitigation_retention_noninferior": result["mitigation_retention"]["ci_pp"][0] >
            100*DECLARATION["retention_noninferiority_margin"],
        "mechanism": "Matched donor data controls donor differences, not every effect of a prior update",
    }
    return result
