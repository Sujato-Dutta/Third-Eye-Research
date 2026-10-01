"""Paired resampling over independent units, never individual candidate rows."""

import numpy as np


def bootstrap(values, seed=42, draws=5000):
    x = np.asarray(values, dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Bootstrap needs finite independent unit values")
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [rng.choice(x, len(x), replace=True).mean() for _ in range(draws)]
    )
    return {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "ci95": [float(v) for v in np.quantile(means, [0.025, 0.975])],
        "units": len(x),
    }


def paired_comparison(a, b, seed=42, draws=10000):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Paired samples must have matching shapes")
    d = a - b
    result = bootstrap(d, seed, draws)
    if len(d) <= 16:
        from itertools import product

        perm = [
            abs(np.mean(d * np.asarray(s))) for s in product((-1, 1), repeat=len(d))
        ]
        pvalue = float(np.mean(np.asarray(perm) >= abs(d.mean()) - 1e-12))
    else:
        rng = np.random.default_rng(seed)
        exceed = sum(
            abs(np.mean(d * rng.choice((-1, 1), len(d)))) >= abs(d.mean()) - 1e-12
            for _ in range(draws)
        )
        pvalue = (exceed + 1) / (draws + 1)
    return {**result, "paired_permutation_p": pvalue}


def holm(pvalues):
    order = sorted(pvalues, key=pvalues.get)
    result, previous = {}, 0.0
    for i, name in enumerate(order):
        previous = max(previous, min(1.0, (len(order) - i) * pvalues[name]))
        result[name] = previous
    return result


def mcnemar(a, b):
    """Exact two-sided binomial test for paired item correctness."""
    import math

    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("Item IDs/order must be paired before McNemar")
    n01, n10 = int(np.sum(~a & b)), int(np.sum(a & ~b))
    n = n01 + n10
    p = (
        min(1.0, 2 * sum(math.comb(n, k) for k in range(min(n01, n10) + 1)) / 2**n)
        if n
        else 1.0
    )
    return {"n01": n01, "n10": n10, "exact_p": p}
