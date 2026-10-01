import numpy as np
import pytest
from third_eye.statistics.metrics import ranking, spearman, kendall, regression
from third_eye.statistics.inference import paired_comparison, holm, mcnemar


def test_tie_aware_ranking_and_negative_gains():
    scores = ranking([-0.3, -0.2, -0.1], [-0.3, -0.2, -0.1])
    assert scores["top1"] == scores["ndcg_at_3"] == scores["spearman"] == 1
    tied = ranking([0, 1, 2], [0, 0, 0])
    assert tied["top1"] == pytest.approx(1 / 3)
    assert tied["spearman"] is None
    all_tied = ranking([0, 0, 0], [1, 2, 3])
    assert all_tied["chance_top1"] == 1
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)
    assert kendall([1, 2, 2], [1, 2, 2]) == pytest.approx(1)
    assert ranking([0.1 + 0.2, 0.3, 0.1], [1, 2, 3])["chance_top1"] == pytest.approx(
        2 / 3
    )


def test_error_units_and_paired_exact_tests():
    assert regression([0, 0], [0.1, -0.1])["rmse"] == pytest.approx(0.1)
    comparison = paired_comparison(np.ones(6), np.zeros(6))
    assert comparison["ci95"] == [1.0, 1.0]
    assert comparison["paired_permutation_p"] == pytest.approx(2 / 64)
    assert holm({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}
    assert mcnemar([True] * 6, [False] * 6)["exact_p"] == pytest.approx(2 / 64)
