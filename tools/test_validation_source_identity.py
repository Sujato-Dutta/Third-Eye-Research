"""Regression coverage for the Qwen None-proof logging failure."""

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validation_source_identity import model_source_identity


def fixture(proof=None, revision="a" * 40):
    cfg = SimpleNamespace(
        model=SimpleNamespace(name="Qwen/Qwen3-4B", revision="a" * 40)
    )
    backend = SimpleNamespace(resolved_revision=revision, source_proof=proof)
    return cfg, backend


def test_direct_source_needs_no_mirror_proof():
    value = model_source_identity(*fixture())
    assert value["load_route"] == "pinned_hugging_face"
    assert value["mirror_proof_sha256"] is None
    assert value["resolved_revision"] == "a" * 40


def test_verified_mirror_proof_is_retained():
    proof = {
        "verified": True,
        "official": "Qwen/Qwen3-4B",
        "original_revision": "a" * 40,
        "proof_sha256": "b" * 64,
    }
    value = model_source_identity(*fixture(proof))
    assert value["load_route"] == "verified_mirror"
    assert value["mirror_proof_sha256"] == "b" * 64


@pytest.mark.parametrize("revision", ["main", "b" * 40, "z" * 40])
def test_unpinned_or_different_revision_rejected(revision):
    with pytest.raises(ValueError, match="pinned"):
        model_source_identity(*fixture(revision=revision))


def test_unverified_mirror_rejected():
    with pytest.raises(ValueError, match="mirror"):
        model_source_identity(*fixture({"verified": False}))
