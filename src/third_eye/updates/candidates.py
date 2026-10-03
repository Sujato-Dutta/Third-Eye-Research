"""Matched difficulty quotas with reproducible composition differences."""

from collections import Counter, defaultdict
from math import comb, prod
import random

from third_eye.io import digest
from third_eye.updates.corrections import InsufficientCorrections


def batch_hash(batch):
    return digest(sorted((item.example.id, item.completion) for item in batch))


def sample_batches(pool, size, count, seed, mode="matched_strata"):
    if len(pool) < size or len({item.example.id for item in pool}) != len(pool):
        raise InsufficientCorrections(
            f"Need {size} unique verified examples; have {len(pool)}"
        )
    groups = defaultdict(list)
    for item in pool:
        groups[item.example.difficulty].append(item)
    # Fix quotas once, then vary examples within strata for every candidate.
    rng = random.Random(seed)
    if mode == "stratified_distinct":
        # Uniform composition ranks induce pool-proportional stratum quotas.
        # Quotas may differ across candidates; examples may overlap across
        # candidates, while every composition and every within-batch ID is unique.
        ordered = sorted(pool, key=lambda x: (x.example.difficulty, x.example.id))
        capacity = comb(len(ordered), size)
        if capacity < count:
            raise InsufficientCorrections(
                "Not enough distinct compositions at the fixed size"
            )
        ranks = set()
        # Floyd sampling works even when the combination count exceeds sys.maxsize.
        for upper in range(capacity - count, capacity):
            draw = rng.randrange(upper + 1)
            ranks.add(upper if draw in ranks else draw)
        ranks = sorted(ranks)
        rng.shuffle(ranks)
        batches = []
        for rank in ranks:
            batch, start = [], 0
            for left in range(size, 0, -1):
                for index in range(start, len(ordered) - left + 1):
                    ways = comb(len(ordered) - index - 1, left - 1)
                    if rank < ways:
                        batch.append(ordered[index])
                        start = index + 1
                        break
                    rank -= ways
            rng.shuffle(batch)
            batches.append(tuple(batch))
        return batches
    if mode != "matched_strata":
        raise ValueError("Unknown candidate sampling policy")
    quota = Counter(item.example.difficulty for item in rng.sample(pool, size))
    if prod(comb(len(groups[group]), n) for group, n in quota.items()) < count:
        # A random quota can consume an entire small stratum even when another
        # quota supports K distinct batches. Find a feasible quota using pool
        # counts only; preserve the original seeded choice whenever feasible.
        feasible = {0: (1, Counter())}
        for group in sorted(groups):
            updated = {}
            for used, (ways, chosen) in feasible.items():
                for n in range(min(len(groups[group]), size - used) + 1):
                    capacity = ways * comb(len(groups[group]), n)
                    if capacity > updated.get(used + n, (0, None))[0]:
                        selected = chosen.copy()
                        if n:
                            selected[group] = n
                        updated[used + n] = (capacity, selected)
            feasible = updated
        capacity, quota = feasible.get(size, (0, None))
        if capacity < count:
            raise InsufficientCorrections(
                "Not enough distinct difficulty-matched batches at the fixed size"
            )
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
