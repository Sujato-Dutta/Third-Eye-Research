from copy import deepcopy
from types import SimpleNamespace

import pytest

from third_eye.data.schema import Correction, Example
from third_eye.experiments.revision import (
    check_portable_identity,
    pooled_corrections,
    reference_corrections,
)


def test_python310_digest_streams_weights_without_unbounded_read(monkeypatch):
    import hashlib
    from io import BytesIO
    import third_eye.io as io_module

    content = b"model weight fixture" * 100

    class BoundedStream(BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 8 * 1024**2
            return super().read(size)

    class Input:
        def __init__(self, value):
            pass

        def open(self, mode):
            assert mode == "rb"
            return BoundedStream(content)

    expected = hashlib.sha256(content).hexdigest()
    monkeypatch.delattr(io_module.hashlib, "file_digest", raising=False)
    monkeypatch.setattr(io_module, "Path", Input)
    assert io_module.file_digest("fixture") == expected


def test_portable_restore_changes_only_physical_location():
    proof = {
        "official": "model",
        "original_revision": "a" * 40,
        "verified_weight_sha256": {"weight": "hash"},
        "metadata_files": {"config": "hash"},
        "runtime_directory": "/old",
        "proof_sha256": "old",
    }
    saved = {
        "config_hash": "cfg",
        "resolved_revision": "rev",
        "input_rendering": {"template": "same", "date": "fixed"},
        "model_source": proof,
    }
    current = {**proof, "runtime_directory": "/new", "proof_sha256": "new"}
    backend = SimpleNamespace(
        config=SimpleNamespace(fingerprint="cfg"),
        resolved_revision="rev",
        input_rendering=saved["input_rendering"],
        source_proof=current,
    )
    check_portable_identity(saved, backend)
    for field, value in [
        ("verified_weight_sha256", {"weight": "different"}),
        ("metadata_files", {"config": "different"}),
        ("original_revision", "b" * 40),
    ]:
        backend.source_proof = {**current, field: value}
        with pytest.raises(ValueError, match="differs"):
            check_portable_identity(saved, backend)
    backend.source_proof = current
    bad = deepcopy(saved)
    bad["input_rendering"]["date"] = "drift"
    with pytest.raises(ValueError, match="differs"):
        check_portable_identity(bad, backend)


def test_pool_deduplicates_problems_with_order_independent_content_rule():
    ex = Example("id", "question", "worked solution\n#### 2", "train")
    a, b = Correction(ex, "2", 1), Correction(ex, "#### 2", 2)
    assert pooled_corrections([(a,), (b,)]) == pooled_corrections([(b,), (a,)])
    assert len(pooled_corrections([(a,), (b,)])) == 1
    different = Correction(Example("id", "different question", "2", "train"), "2", 1)
    with pytest.raises(ValueError, match="different examples"):
        pooled_corrections([(a,), (different,)])


def test_gold_uses_complete_reference_and_preserves_problem_identity():
    ex = Example("id", "question", "2+2=4\n#### 4", "train")
    original = Correction(ex, "#### 4", 3)
    reference = reference_corrections([original])[0]
    assert reference.example is ex
    assert reference.completion == ex.answer
    assert original.completion == "#### 4"


def test_paired_control_contrasts_subtract_each_receivers_own_baseline():
    import numpy as np
    from third_eye.experiments.review_statistics import control_matrix

    cells = {
        "m0_before": {"target_test": np.ones(4)},
        "m1_before": {"target_test": np.full(4, 0.5)},
        "m1_pooled_50": {"target_test": np.ones(4)},
    }
    for d in (0, 2):
        for receiver, donor, value in [
            ("m0", "d0", 0.8),
            ("m0", "d1", 0.7),
            ("m1", "d0", 0.4),
            ("m1", "d1", 0.2),
        ]:
            cells[f"draw{d}_{receiver}_{donor}_50"] = {"target_test": np.full(4, value)}
        for name, value in [("d1_10", 0.6), ("d1_25", 0.4), ("gold_50", 0.8)]:
            cells[f"draw{d}_m1_{name}"] = {"target_test": np.full(4, value)}
    expected = [-0.3, -0.2, 0, 0.1, -0.1, 0.4, 0.2, 0.6, 0.8]
    result = control_matrix(cells, [0, 2], "target_test")
    np.testing.assert_allclose(
        result, np.broadcast_to(np.asarray(expected)[:, None], result.shape), atol=1e-12
    )


def test_joint_role_bootstrap_preserves_constant_known_losses():
    import numpy as np
    from third_eye.experiments.review_statistics import ROLES, bootstrap_heads

    states, matrices = [], []
    for family in ("qwen", "llama"):
        for task in ("math", "code"):
            for seed in range(2):
                states.append(
                    {
                        "stream": family + "/" + task,
                        "seed": seed,
                        "ids": {
                            r: [(task + str(i), str(i)) for i in range(4)]
                            for r in ROLES
                        },
                    }
                )
                matrices.append(
                    {
                        r: np.full((2, 4), [-0.1, 0.2][j]) * np.ones((2, 4))
                        for j, r in enumerate(ROLES[:2])
                    }
                    | {ROLES[2]: np.full((2, 4), -0.3)}
                )
    point, boot, *_ = bootstrap_heads(states, matrices, draws=20)
    np.testing.assert_allclose(boot, np.broadcast_to(point, boot.shape), atol=1e-12)
    np.testing.assert_allclose(point[:, 0], [-0.1, 0.2, -0.3])
