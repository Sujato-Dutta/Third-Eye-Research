"""Matched difficulty quotas with reproducible composition differences."""

from collections import Counter, defaultdict
import random

from third_eye.io import digest
from third_eye.updates.corrections import InsufficientCorrections


def batch_hash(batch):
    return digest(sorted((item.example.id, item.completion) for item in batch))


def sample_batches(pool, size, count, seed):
    if len(pool) < size or len({item.example.id for item in pool}) != len(pool):
        raise InsufficientCorrections(
            f"Need {size} unique verified examples; have {len(pool)}"
        )
    groups = defaultdict(list)
    for item in pool:
        groups[item.example.difficulty].append(item)
    # Fix quotas once, then vary examples within strata for every candidate.
    rng = random.Random(seed)
    quota = Counter(item.example.difficulty for item in rng.sample(pool, size))
    batches, seen = [], set()
    for _ in range(1000):
        batch = []
        for group, n in sorted(quota.items()):
            batch.extend(rng.sample(groups[group], n))
        rng.shuffle(batch)
        fingerprint = batch_hash(batch)
        if fingerprint in seen:
            continue
        batches.append(tuple(batch))
        seen.add(fingerprint)
        if len(batches) == count:
            return batches
    raise InsufficientCorrections(
        "Not enough distinct matched batches; increase the correction pool or repilot the fixed batch size"
    )
