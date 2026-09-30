"""Cheap pre-commit features and a reversible short training probe.

This is a systems-level logging adapter. The modeling component can inject
additional gradient alignment, diversity, confidence, and KL features later.
"""

from collections import Counter
from dataclasses import replace
import statistics

from third_eye.io import digest


def extract_features(backend, batch, anchor, protocol, seed, log_path):
    lengths = [len(item.completion.split()) for item in batch]
    completions = {digest(item.completion) for item in batch}
    result = {
        "completion_words_mean": statistics.mean(lengths),
        "completion_words_std": statistics.pstdev(lengths),
        "duplicate_completion_rate": 1 - len(completions) / len(batch),
        "difficulty_counts": dict(Counter(item.example.difficulty for item in batch)),
    }
    examples = [replace(item.example, answer=item.completion) for item in batch]
    result["pre_update_nll"] = backend.measure_loss(examples)
    if not protocol.probe_steps:
        return result
    parent, parent_hash = backend.snapshot(), backend.state_hash()
    anchor_before = backend.measure_loss(anchor)
    try:
        probe = backend.train(
            batch, seed, max_steps=protocol.probe_steps, log_path=log_path
        )
        result.update(
            probe_steps=protocol.probe_steps,
            probe_loss_change_per_step=(probe["loss_last"] - probe["loss_first"])
            / max(1, protocol.probe_steps - 1),
            probe_anchor_loss_change=backend.measure_loss(anchor) - anchor_before,
            probe_seconds=probe["seconds"],
        )
    finally:
        backend.restore(parent)
        if backend.state_hash() != parent_hash:
            raise RuntimeError("Probe rollback failed")
    return result
