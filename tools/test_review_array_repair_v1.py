import numpy as np
import pytest
from empirical_review_array_repair_v1 import list_safe_regret, original_regret


def test_native_record_lists_use_declared_regret_expression():
    with pytest.raises(TypeError):
        original_regret([.2,.1,0], [0,.3,.1])
    assert list_safe_regret([.2,.1,0], [0,.3,.1]) == pytest.approx(.3)


def test_existing_array_comparisons_are_unchanged():
    for one,two in (([.2,.1,0],[0,.3,.1]),([0,0,1],[0,1,2]),([0,0,0],[-1,0,1])):
        one,two=np.array(one),np.array(two)
        assert list_safe_regret(one,two) == original_regret(one,two)


def test_tied_immediate_winners_are_averaged_without_mutation():
    one,two=[1,1,0],[0,.4,.5]
    assert list_safe_regret(one,two) == pytest.approx(.3)
    assert one==[1,1,0] and two==[0,.4,.5]
