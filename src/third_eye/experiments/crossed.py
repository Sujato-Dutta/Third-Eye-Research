"""Fixed-checkpoint by fixed-correction-batch counterfactual controls."""

import numpy as np

from third_eye.experiments.mechanism import ROLES, choose, folds, matrix
from third_eye.experiments.rendering import FIXED_CONTEXT
from third_eye.statistics.inference import holm, paired_comparison


DECLARATION = {
    "version": "crossed_corrections_20261015",
    "development_only": True,
    "automatic_expansion": False,
    "offsets": [50_000_000, 100_000_000, 150_000_000],
    "parents": 8,
    "candidates": 3,
    "draws": 3,
    "primary": "native_pairing_surplus",
    "selection_draws": [0, 1],
    "scoring_draw": 2,
    "terminal_donor": "identity_on_recipient",
    "training": "unchanged_original_continuation_config_and_seed",
    "harvesting": "none_saved_verified_train_batches_only",
    "fold_namespace": "third-eye-lr-mechanism-20261013",
    "tie_policy": "uniform_maximum_rounded_12_decimals",
    "independent_unit": "parent_trajectory",
    "input_context_correction": "C1_temporal_template_context",
    "fixed_prompt_context": FIXED_CONTEXT,
    "diagonal_outcomes": "fresh_matched_context_remeasurement",
    "baseline_outcomes": "fresh_matched_context_parent_and_M1",
}


def cells():
    """Array sharding changes execution only; donors/draws are never selected."""
    return [(i, j, s) for i in range(3) for j in range(3) for s in range(3)]


def decompose(values):
    """Balanced recipient, donor and interaction effects on an observed matrix.

    Squares include finite-evaluation noise; they are not noise-corrected causal
    variance components or evidence of excess signal by themselves.
    """
    x = np.asarray(values, float)
    if x.shape != (3, 3) or not np.isfinite(x).all():
        raise ValueError("Finite complete 3x3 matrix required")
    grand = x.mean()
    row, col = x.mean(1) - grand, x.mean(0) - grand
    interaction = x - grand - row[:, None] - col[None, :]
    return {
        "grand_mean": float(grand),
        "recipient_effects": row.tolist(),
        "donor_effects": col.tolist(),
        "interaction": interaction.tolist(),
        "recipient_mean_square": float(np.mean(row**2)),
        "donor_mean_square": float(np.mean(col**2)),
        "interaction_mean_square": float(np.mean(interaction**2)),
        "native_pairing_surplus": float(np.trace(x) / 3 - grand),
    }


def state_review(parent, immediate, cube):
    """cube[recipient][donor][draw] holds checkpoint-bound item artifacts.

    Native pairing surplus is a signed linear contrast, not a noisy maximum.
    Selection uses two draws/one item half; utility uses the remaining draw and
    other half. Item-fold orientations stay repeated measurements of one parent.
    All lookahead selectors here have privileged future observations.
    """
    if (
        len(immediate) != 3
        or len(cube) != 3
        or any(len(row) != 3 or any(len(draws) != 3 for draws in row) for row in cube)
    ):
        raise ValueError("Complete K=3 by K=3 by three draws required")
    assignment = folds(parent)
    by_fold, decomposition = [], []
    for fold in (0, 1):
        # Tensor axes: draw, recipient, donor, metric head.
        tensor = np.asarray(
            [
                [
                    matrix([cube[i][j][s] for j in range(3)], parent, assignment, fold)
                    for i in range(3)
                ]
                for s in range(3)
            ]
        )
        heldout = np.asarray(
            [matrix([cube[i][i][2] for i in range(3)], parent, assignment, 1 - fold)]
        )[0]
        h1 = matrix(immediate, parent, assignment, fold)
        selection = tensor[:2].mean(axis=(0, 3))
        native = choose(np.diag(selection))
        common = choose(selection.mean(1))
        greedy = choose(h1.mean(1))
        # Holding a recipient fixed, compare its own donor to uniform donors.
        all_matrix = tensor.mean(axis=(0, 3))
        details = decompose(all_matrix)
        continuation = all_matrix - h1.mean(1)[:, None]
        gain_details = decompose(continuation)
        row = {
            "native_pairing_surplus": details["native_pairing_surplus"],
            "common_donor_selection_minus_native": float(
                (common - native) @ heldout.mean(1)
            ),
            "common_donor_selection_minus_immediate": float(
                (common - greedy) @ heldout.mean(1)
            ),
            "native_selection_minus_immediate": float(
                (native - greedy) @ heldout.mean(1)
            ),
            "common_native_choice_disagreement": float(1 - common @ native),
        }
        for head, role in enumerate(ROLES):
            head_matrix = tensor[:, :, :, head].mean(0)
            row["pairing_surplus_" + role] = decompose(head_matrix)[
                "native_pairing_surplus"
            ]
        by_fold.append(row)
        decomposition.append(
            {"total_future": details, "continuation_gain": gain_details}
        )
    return {
        "comparison": {k: float(np.mean([r[k] for r in by_fold])) for k in by_fold[0]},
        "fold_decomposition": decomposition,
        "squared_components_include_noise": True,
        "selectors_use_privileged_future_outcomes": True,
    }


def aggregate(states):
    if len(states) != 8 or len({s["trajectory_id"] for s in states}) != 8:
        raise ValueError("Eight distinct development parents required")
    streams = {s["stream"] for s in states}
    if len(streams) != 4 or any(
        sum(s["stream"] == g for s in states) != 2 for g in streams
    ):
        raise ValueError("Balanced four-stream parents required")
    results = {
        key: paired_comparison(
            [s["comparison"][key] for s in states], [0] * 8, seed=20261015
        )
        for key in states[0]["comparison"]
        if key != "common_native_choice_disagreement"
    }
    adjusted = holm(
        {
            k: v["paired_permutation_p"]
            for k, v in results.items()
            if k != DECLARATION["primary"]
        }
    )
    for key, value in adjusted.items():
        results[key]["secondary_holm_p"] = value
    return {
        "declaration": DECLARATION,
        "comparisons": results,
        "independent_parents": 8,
        "gate2_passed": None,
        "automatic_expansion": False,
        "limitations": "Previously inspected parents, shared finite evaluation items, small N; donor transfer is a counterfactual intervention. Not a learned forecast, natural mediation effect, noise-corrected variance estimate or deployed policy benefit.",
    }
