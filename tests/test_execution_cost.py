import pytest

from third_eye.experiments.execution_cost import validate_limits, validate_observations


def test_runtime_correction_cannot_drop_streams_shorten_required_time_or_exceed_cap():
    validate_limits(
        {"code": 10, "math": 20},
        {"code": 15, "math": 20},
        {"code": 15, "math": 20},
        533,
    )
    with pytest.raises(ValueError):
        validate_limits({"code": 10, "math": 20}, {"code": 15}, {"code": 15}, 533)
    with pytest.raises(ValueError):
        validate_limits({"code": 10}, {"code": 14}, {"code": 15}, 533)
    with pytest.raises(ValueError):
        validate_limits({"code": 10}, {"code": 15}, {"code": 15}, 561)


def test_measured_code_timings_must_be_complete_unique_and_outcome_blind():
    allowed = {("qwen/code", "parent")}
    row = {
        "stream": "qwen/code",
        "checkpoint": "parent",
        "count": 8,
        "seconds": 40,
        "seconds_per_item": 5,
        "accuracy_recorded": False,
    }
    assert validate_observations([row], allowed) == {("qwen/code", "parent"): 5}
    for records in (
        [row, row],
        [{**row, "accuracy_recorded": True}],
        [{**row, "seconds_per_item": 6}],
    ):
        with pytest.raises(ValueError):
            validate_observations(records, allowed)
