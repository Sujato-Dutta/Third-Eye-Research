"""Runtime-only correction checks; scientific cohort manifests stay immutable."""

import math


def validate_observations(observations, allowed, count=8):
    result = {}
    for row in observations:
        key = (row["stream"], row["checkpoint"])
        if (
            key not in allowed
            or key in result
            or row["count"] != count
            or row["accuracy_recorded"] is not False
        ):
            raise ValueError("Unexpected or duplicated cost-only observation")
        rate = row["seconds_per_item"]
        if (
            not math.isfinite(rate)
            or rate <= 0
            or not math.isclose(row["seconds"] / count, rate, rel_tol=1e-9)
        ):
            raise ValueError("Invalid measured timing")
        result[key] = rate
    return result


def validate_limits(original, proposed, required, total_sus, ceiling=560):
    if set(original) != set(proposed) or set(proposed) != set(required):
        raise ValueError("Runtime correction cannot add or drop streams")
    if any(
        type(proposed[key]) is not int
        or proposed[key] < max(original[key], required[key])
        for key in original
    ):
        raise ValueError("Every stream must retain adequate runtime coverage")
    if total_sus > ceiling + 1e-9:
        raise ValueError("Corrected execution exceeds original stage ceiling")
