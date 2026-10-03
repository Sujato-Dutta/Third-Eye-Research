from collections import Counter
from dataclasses import replace
import json

import pytest

from third_eye.config import Config
from third_eye.data.schema import Correction, Example, ROLES
from third_eye.data.splits import create_manifest, load_manifest, validate_splits
from third_eye.evaluation.verifiers import extract_numeric
from third_eye.updates.candidates import batch_hash, sample_batches
from third_eye.updates.corrections import InsufficientCorrections, collect_corrections


def examples():
    return {
        role: [
            Example(f"{role}-{i}", f"{role} question {i}", "#### 2", role)
            for i in range(4)
        ]
        for role in ROLES
    }


def test_cross_split_prompt_leakage_is_detected_even_with_new_ids():
    splits = examples()
    splits["target_dev"][0] = replace(
        splits["target_dev"][0], prompt="TRAIN   question 0"
    )
    with pytest.raises(ValueError, match="leakage"):
        validate_splits(splits)


def test_manifest_detects_mutation(tmp_path):
    paths = {}
    for role in ROLES:
        path = tmp_path / f"{role}.jsonl"
        path.write_text(
            json.dumps(
                {
                    "id": role,
                    "prompt": f"{role} unique prompt",
                    "answer": "#### 2",
                    "split": role,
                }
            )
            + "\n"
        )
        paths[role] = path
    manifest = tmp_path / "manifest.json"
    create_manifest(paths, manifest)
    load_manifest(manifest)
    paths["train"].write_text(paths["train"].read_text().replace("#### 2", "#### 3"))
    with pytest.raises(ValueError, match="changed"):
        load_manifest(manifest)
    with pytest.raises(FileExistsError):
        create_manifest(paths, manifest)


def test_final_test_role_is_rejected():
    with pytest.raises(ValueError, match="Final test"):
        Example("x", "prompt", "answer", "test")


def test_candidates_match_sizes_strata_and_reproduce():
    pool = [
        Correction(
            Example(
                str(i),
                f"problem {i}",
                "#### 2",
                "train",
                difficulty="hard" if i % 2 else "easy",
            ),
            "#### 2",
            1,
        )
        for i in range(20)
    ]
    batches = sample_batches(pool, 6, 3, 42)
    assert len({batch_hash(b) for b in batches}) == 3
    assert all(len(b) == 6 for b in batches)
    assert all(
        Counter(x.example.difficulty for x in b)
        == Counter(x.example.difficulty for x in batches[0])
        for b in batches
    )
    assert [batch_hash(b) for b in batches] == [
        batch_hash(b) for b in sample_batches(pool, 6, 3, 42)
    ]


def test_insufficient_pool_does_not_silently_shrink_candidates():
    with pytest.raises(InsufficientCorrections):
        sample_batches([], 32, 3, 42)
    pool = [Correction(Example(str(i), str(i), "2", "train"), "2", 1) for i in range(3)]
    with pytest.raises(InsufficientCorrections, match="distinct"):
        sample_batches(pool, 3, 3, 42)


def test_infeasible_random_quota_does_not_reject_a_sufficient_pool():
    pool = [
        Correction(
            Example(
                str(i),
                f"problem {i}",
                "2",
                "train",
                difficulty="short" if i < 2 else "long",
            ),
            "2",
            1,
        )
        for i in range(5)
    ]
    # Seed 1 picks both short items initially, producing only one combination.
    batches = sample_batches(pool, 2, 3, 1)
    assert len({batch_hash(batch) for batch in batches}) == 3
    assert all(len(batch) == 2 for batch in batches)
    assert Counter(x.example.difficulty for x in batches[0]) == {"short": 1, "long": 1}
    assert all(
        Counter(x.example.difficulty for x in batch)
        == Counter(x.example.difficulty for x in batches[0])
        for batch in batches
    )
    assert [batch_hash(batch) for batch in batches] == [
        batch_hash(batch) for batch in sample_batches(pool, 2, 3, 1)
    ]


def test_stratum_singletons_cannot_be_fabricated_into_matched_candidates():
    pool = [
        Correction(
            Example(str(i), f"problem {i}", "2", "train", difficulty=str(i)), "2", 1
        )
        for i in range(3)
    ]
    with pytest.raises(InsufficientCorrections, match="distinct"):
        sample_batches(pool, 2, 3, 1)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("reasoning\n#### 1,200", 1200),
        ("\\boxed{2.5}", 2.5),
        ("Final answer: -3", -3),
        (".5", 0.5),
        ("1e3", 1000),
    ],
)
def test_numeric_formats(text, expected):
    assert float(extract_numeric(text)) == expected


@pytest.mark.parametrize(
    "text",
    ["I considered 2 and 3", "#### 1/2", "#### NaN", "#### 2 or 3", "#### 12,34"],
)
def test_ambiguous_math_fails_closed(text):
    assert extract_numeric(text) is None


def test_correction_prompt_never_contains_reference_answer():
    from third_eye.evaluation.verifiers import VerifierRegistry

    prompts = []

    class Backend:
        def generate(self, prompt, *args, **kwargs):
            prompts.append(prompt)
            return "#### 987654" if "Reconsider" in prompt else "#### -1"

    ex = Example("x", "A deliberately opaque training problem", "#### 987654", "train")
    pool, stats = collect_corrections(
        Backend(),
        [ex],
        VerifierRegistry(),
        replace(Config().protocol, correction_attempts=1),
        42,
    )
    assert len(pool) == 1 and stats["verified_corrections"] == 1
    assert all("987654" not in prompt for prompt in prompts)


def test_invalid_config_rejected():
    base = Config()
    with pytest.raises(ValueError):
        replace(base, protocol=replace(base.protocol, candidates=4))
    with pytest.raises(ValueError):
        replace(base, protocol=replace(base.protocol, utility_weights=(1, 1, 1)))
    with pytest.raises(ValueError):
        replace(base, model=replace(base.model, device="cpu", quantization="nf4"))
