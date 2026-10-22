"""Protect final identities, outcome blindness and checkpoint/sample provenance."""

from dataclasses import replace
import json

import pytest

from third_eye.data.schema import Example
from third_eye.experiments.scoring_precision import (
    benchmark_checkpoints,
    identity_index,
    record_matches,
    scan_exposure,
    synthetic_fixture_records,
    timing_sample,
)


def final():
    return Example(
        "sealed",
        "question with wrapper",
        "2",
        "target_test",
        metadata={"source_prompt": "RAW question"},
    )


def test_exposure_matches_changed_id_and_wrapper_using_source_identity():
    index = identity_index({"suite": {"target_test": [final()]}})
    changed = {
        "id": "other",
        "prompt": "different wrapper",
        "answer": "unrelated",
        "split": "train",
        "metadata": {"source_prompt": " raw   QUESTION "},
    }
    hits, opaque = record_matches({"items": [{"example": changed}]}, index, {})
    assert hits == {("suite", "target_test", "sealed")} and not opaque


def test_scoring_hash_marks_suite_but_planned_reference_does_not():
    keys = {("suite", "target_test", "sealed")}
    assert record_matches({"final_manifest_hash": "hash"}, {}, {"hash": keys}) == (
        set(),
        False,
    )
    for score in (0, 1):
        assert record_matches(
            {
                "final_manifest_hash": "hash",
                "scores": {"target_test": score},
                "evaluation": "greedy",
            },
            {},
            {"hash": keys},
        ) == (keys, False)
    assert record_matches(
        {"final_manifest_hash": "unknown", "scores": {}, "evaluation": "greedy"}, {}, {}
    ) == (set(), True)


def test_saved_item_matches_without_reading_prediction_or_correctness():
    index = identity_index({"suite": {"target_test": [final()]}})
    for correct in (0, 1):
        hits, _ = record_matches(
            {
                "items": {
                    "target_dev": [
                        {
                            "id": "different",
                            "prompt_sha256": final().prompt_hash,
                            "correct": correct,
                            "prediction_sha256": "ignored",
                        }
                    ]
                }
            },
            index,
            {},
        )
        assert hits == {("suite", "target_test", "sealed")}


def test_unreadable_artifact_and_opaque_final_are_not_silently_dropped(tmp_path):
    bad = tmp_path / "partial.json"
    bad.write_text("{not json", encoding="utf-8")
    opaque = tmp_path / "opaque.jsonl"
    opaque.write_text(
        json.dumps({"role": "target_test", "passed": True}) + "\n", encoding="utf-8"
    )
    _, unknown, errors, inventory = scan_exposure([bad, opaque], {}, {})
    assert unknown == [str(opaque)]
    assert errors == [{"path": str(bad), "error": "JSONDecodeError"}]
    assert len(inventory) == 2


def test_timing_is_outcome_independent_includes_extremes_and_forbids_final():
    examples = [Example(str(i), "a" * (i + 1), "2", "target_dev") for i in range(64)]
    sample = timing_sample(examples)
    assert len({ex.id for ex in sample}) == 16
    assert sample[0].id == "0" and sample[-1].id == "63"
    altered = [
        replace(ex, answer="different", metadata={"correct": i % 2})
        for i, ex in enumerate(reversed(examples))
    ]
    assert [ex.id for ex in timing_sample(altered)] == [ex.id for ex in sample]
    with pytest.raises(ValueError, match="final"):
        timing_sample([replace(ex, split="target_test") for ex in examples])


def test_native_future_uses_recorded_t2_path_not_evaluation_path(tmp_path):
    t1, t2 = tmp_path / "candidate/t1", tmp_path / "candidate/t2"
    row = {
        "checkpoints": {"t1": str(t1), "t2": str(t2)},
        "parent_adapter_hash": "parent",
        "continuation_replicates": [
            {
                "items_path": "/unrelated/evaluation.json",
                "continuation_adapter_hash": "future",
            }
        ],
    }
    assert benchmark_checkpoints(row) == [
        ("parent", tmp_path / "parent", "parent"),
        ("future", t2, "future"),
    ]


def test_fixture_path_alone_cannot_exclude_real_research_exposure():
    path = "/runs/campaign/validation/fixtures/test_prepare_portable_suites_a0/original/target_test.jsonl"
    row = {
        "id": "gsm8k:test:0",
        "prompt": "real question",
        "metadata": {"source_revision": "b" * 40},
    }
    assert not synthetic_fixture_records(path, [row])
    row = {
        "id": "gsm8k:test:0",
        "prompt": "openai/gsm8k main test distinct problem 0\nSolve",
        "metadata": {
            "source_revision": "a" * 40,
            "source_repository": "openai/gsm8k",
            "source_prompt": "openai/gsm8k main test distinct problem 0",
        },
    }
    assert synthetic_fixture_records(path, [row])
    assert not synthetic_fixture_records(
        path.replace("/validation/fixtures/", "/real_runs/"), [row]
    )


def test_mock_score_classification_uses_identity_markers_not_score_values():
    path = "/runs/campaign/cpu_validation/fixtures/test_result_cli_separates_fina0/direct_42/final/scores.json"
    row = {
        "model": "fixture-model",
        "final_manifest_hash": "fixture-final",
        "config_hash": "cfg",
        "resolved_revision": "a" * 40,
    }
    assert synthetic_fixture_records(path, [row])
    assert synthetic_fixture_records(path, [{**row, "scores": {"target_test": 0.99}}])
    assert not synthetic_fixture_records(path, [{**row, "final_manifest_hash": "real"}])
