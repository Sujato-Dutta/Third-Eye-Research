import pytest
from smollm_handoff_v1 import checked_history, completed_validation, remaining_indices


def test_completed_purged_validation_requires_matching_receipts():
    receipt = dict(status="passed", release_sha256="release", tasks_sha256="tasks")
    assert completed_validation("COMPLETED", "0:0", receipt, "release", "tasks") is None
    with pytest.raises(RuntimeError):
        completed_validation("FAILED", "1:0", receipt, "release", "tasks")
    with pytest.raises(RuntimeError):
        completed_validation("COMPLETED", "0:0", receipt, "release", "other")


def test_accounting_rejects_unrecorded_submissions():
    checked_history("1057855|\n1058281|\n", {"1057855", "1058281"})
    with pytest.raises(RuntimeError):
        checked_history("1057855|\n1058281|\n999|\n", {"1057855", "1058281"})


def test_existing_experiment_is_not_resubmitted():
    assert remaining_indices([dict(index=0, job_id="1058281")]) == [1, 2, 3]
    with pytest.raises(RuntimeError):
        remaining_indices([dict(index=0), dict(index=0)])
