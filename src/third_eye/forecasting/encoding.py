"""Versioned allowlist: post-update artifacts never enter model inputs."""

import numpy as np

HEADS = ("target", "ood", "retention")
FEATURE_GROUPS = {
    "data": (
        "completion_words_mean",
        "completion_words_std",
        "completion_tokens_mean",
        "completion_tokens_std",
        "pre_update_nll",
        "confidence_mean",
        "revision_attempt_mean",
        "verifier_pass_rate",
    ),
    "diversity": ("duplicate_completion_rate", "embedding_diversity"),
    "gradient": (
        "gradient_norm",
        "retention_gradient_cosine",
        "gradient_sign_agreement",
        "layer_gradient_norm_mean",
        "layer_gradient_norm_std",
    ),
    "probe": (
        "probe_steps",
        "probe_loss_change_per_step",
        "probe_anchor_loss_change",
        "probe_candidate_loss_change",
        "probe_anchor_kl",
        "probe_adapter_delta_norm",
    ),
    "pool": ("failure_rate", "correction_rate", "pool_verifier_pass_rate"),
    "context": ("generation",),
}
FEATURE_NAMES = tuple(name for group in FEATURE_GROUPS.values() for name in group)
SCHEMA_VERSION = 1


def encode(view, ablations=()):
    """Return state, masked candidate features, and at most three past states.

    Missing optional measurements have explicit availability bits. Runtime,
    model names, IDs and dataset identity are excluded to prevent shortcuts.
    """
    p = view["precommit"]
    state = [float(p["state"][h]) for h in HEADS]
    f = dict(p["features"])
    if "generation" in p:
        f["generation"] = p["generation"]
    pool = p.get("pool_statistics", {})
    count = max(1, pool.get("training_prompts", 0))
    f.update(
        failure_rate=pool.get("initial_failures", 0) / count,
        correction_rate=pool.get("verified_corrections", 0) / count,
        pool_verifier_pass_rate=pool.get("verifier_pass_rate", 0),
    )
    removed = {
        name
        for group in ablations
        if group in FEATURE_GROUPS
        for name in FEATURE_GROUPS[group]
    }
    if "retention" in ablations:
        state[2] = 0.0
        removed.update(
            (
                "retention_gradient_cosine",
                "gradient_sign_agreement",
                "probe_anchor_loss_change",
                "probe_anchor_kl",
            )
        )
    values = [
        float(f.get(name, 0)) if name not in removed else 0.0 for name in FEATURE_NAMES
    ]
    masks = [float(name in f and name not in removed) for name in FEATURE_NAMES]
    history = np.zeros((3, 6), dtype=np.float32)
    past = [] if "history" in ablations else p.get("history", [])[-3:]
    for i, entry in enumerate(past, start=3 - len(past)):
        history[i] = [float(entry["consequences"][h]) for h in HEADS] + [
            float(entry["delta"][h]) for h in HEADS
        ]
        if "retention" in ablations:
            history[i, [2, 5]] = 0
    result = (
        np.asarray(state, dtype=np.float32),
        np.asarray(values + masks, dtype=np.float32),
        history,
        np.asarray([0] * (3 - len(past)) + [1] * len(past), dtype=np.float32),
    )
    if any(not np.isfinite(a).all() for a in result):
        raise ValueError("Non-finite pre-commit features")
    return result


def labels(record, horizon):
    return np.asarray(
        [record["labels"][f"h{horizon}"][h] for h in HEADS], dtype=np.float64
    )
