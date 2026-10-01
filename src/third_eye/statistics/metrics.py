"""Tie-aware metrics; undefined correlations are explicit nulls."""

import math
import numpy as np


def average_ranks(x):
    x = np.round(np.asarray(x, dtype=float), 12)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2 + 1
        i = j
    return ranks


def spearman(a, b):
    a, b = average_ranks(a), average_ranks(b)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def kendall(a, b):
    a, b = np.round(np.asarray(a), 12), np.round(np.asarray(b), 12)
    concordant = discordant = tied_a = tied_b = 0
    for i in range(len(a)):
        da, db = np.sign(a[i] - a[i + 1 :]), np.sign(b[i] - b[i + 1 :])
        concordant += int(np.sum(da * db > 0))
        discordant += int(np.sum(da * db < 0))
        tied_a += int(np.sum((da == 0) & (db != 0)))
        tied_b += int(np.sum((db == 0) & (da != 0)))
    pairs = concordant + discordant
    denom = math.sqrt((pairs + tied_a) * (pairs + tied_b))
    return (concordant - discordant) / denom if denom else None


def ranking(truth, prediction):
    """NDCG uses within-state shifted gains to handle negative update values.

    Predicted ties average discount positions; top-1 is expected correctness
    under uniform tie breaking, not an optimistic choice of candidate order.
    """
    y, p = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    y, p = np.round(y, 12), np.round(p, 12)
    if len(y) != 3 or p.shape != y.shape:
        raise ValueError("Ranking requires a matched K=3 state")
    winners = y == y.max()
    selected = p == p.max()
    gains = y - y.min()
    discount = 1 / np.log2(np.arange(2, len(y) + 2))
    ideal = float(np.dot(np.sort(gains)[::-1], discount))
    order = np.argsort(-p, kind="stable")
    dcg, i = 0.0, 0
    while i < len(y):
        j = i + 1
        while j < len(y) and p[order[j]] == p[order[i]]:
            j += 1
        dcg += float(gains[order[i:j]].mean() * discount[i:j].sum())
        i = j
    return {
        "spearman": spearman(y, p),
        "kendall": kendall(y, p),
        "ndcg_at_3": dcg / ideal if ideal else 1.0,
        "top1": float(winners[selected].mean()),
        "chance_top1": float(winners.mean()),
        "truth_tied": int(winners.sum()) > 1,
    }


def regression(truth, prediction):
    y, p = np.asarray(truth), np.asarray(prediction)
    error = p - y
    bins = []
    for indices in np.array_split(np.argsort(p, kind="stable"), min(10, len(p))):
        if len(indices):
            bins.append(
                {
                    "count": len(indices),
                    "predicted_mean": float(p[indices].mean()),
                    "observed_mean": float(y[indices].mean()),
                }
            )
    nonzero = y != 0
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(error.mean()),
        "spearman": spearman(y, p),
        "kendall": kendall(y, p),
        "sign_accuracy": float(np.mean(np.sign(y) == np.sign(p))),
        "nonzero_sign_accuracy": float(
            np.mean(np.sign(y[nonzero]) == np.sign(p[nonzero]))
        )
        if nonzero.any()
        else None,
        "calibration_bins": bins,
    }
