"""Failures, array accounting and ambiguous submissions cannot evade the cap."""

import pytest

from third_eye.cluster_budget import reservation, reserve, reconcile


def test_reservation_includes_array_nodes_and_minimum():
    assert reservation("gh", 86400, tasks=8) == 192
    assert reservation("gg", 1) == pytest.approx(0.0825)
    assert reservation("gh", 3600, tasks=2, nodes=3) == 6
    with pytest.raises(ValueError):
        reservation("gh", -1)


def test_failures_count_and_missing_accounting_cannot_release_headroom():
    ledger = dict(hard_cap_sus=3, jobs={})
    reserve(ledger, "array", "gh", 3600, tasks=2)
    rows = [dict(id="1_0", state="FAILED", elapsed_seconds=1800, nodes=1)]
    with pytest.raises(ValueError, match="Complete"):
        reconcile(ledger, "array", rows)
    with pytest.raises(ValueError, match="cap"):
        reserve(ledger, "retry", "gh", 7200)
    rows.append(dict(id="1_1", state="COMPLETED", elapsed_seconds=3600, nodes=1))
    assert reconcile(ledger, "array", rows) == 1.5
    reserve(ledger, "retry", "gh", 3600)
    with pytest.raises(ValueError, match="Duplicate"):
        reserve(ledger, "retry", "gh", 3600)
    assert (
        sum(j.get("actual_sus", j["reserved_sus"]) for j in ledger["jobs"].values())
        == 2.5
    )


def test_active_unknown_or_duplicate_tasks_keep_full_reservation():
    ledger = dict(hard_cap_sus=3, jobs={})
    reserve(ledger, "job", "gh", 3600)
    row = dict(id="1", state="RUNNING", elapsed_seconds=60, nodes=1)
    with pytest.raises(ValueError, match="Active"):
        reconcile(ledger, "job", [row])
    row.update(state="FAILED", elapsed_seconds=7200)
    with pytest.raises(ValueError, match="exceeded"):
        reconcile(ledger, "job", [row])
    assert "actual_sus" not in ledger["jobs"]["job"]
