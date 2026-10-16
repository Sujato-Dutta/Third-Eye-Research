"""Selection-weighted effects must not be substituted by averages over candidates."""

import numpy as np
import pytest

from review_mechanism_precision import panel


def test_average_improvement_can_hide_a_loss_for_the_fixed_selected_candidate():
    h1 = np.asarray([[0.3] * 3, [0.2] * 3, [0.1] * 3])
    baseline = np.zeros((3, 3, 3))
    small = np.broadcast_to(
        np.asarray([[-0.03] * 3, [0.06] * 3, [0.06] * 3]), (3, 3, 3)
    )
    result = panel(h1, baseline, small)
    assert result["selected_mass"] == [1, 0, 0]
    np.testing.assert_allclose(result["mean_all_candidate_effect_by_head"], [0.03] * 3)
    np.testing.assert_allclose(result["same_selected_effect_by_head"], [-0.03] * 3)
    # Future outcomes cannot retrospectively choose a better candidate.
    changed = panel(h1, baseline, -small)
    assert result["selected_mass"] == changed["selected_mass"]


def test_incomplete_or_nonfinite_panels_fail_instead_of_becoming_extra_samples():
    with pytest.raises(ValueError, match="K=3"):
        panel(np.zeros((2, 3)), np.zeros((3, 3, 3)), np.zeros((3, 3, 3)))
    with pytest.raises(ValueError, match="Finite"):
        panel(np.full((3, 3), np.nan), np.zeros((3, 3, 3)), np.zeros((3, 3, 3)))
