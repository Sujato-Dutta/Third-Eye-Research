from dataclasses import replace
import subprocess
from types import SimpleNamespace
from pathlib import Path
import re
import pytest

from third_eye.data.schema import Example, Correction
from third_eye.data.benchmarks import boxed_answer, audit_overlap, deduplicate
from third_eye.evaluation.benchmarks import BenchmarkVerifier
from third_eye.evaluation.sandbox import DockerSandbox, SandboxUnavailable, extract_code


def test_balanced_math_box_and_multiple_choice():
    assert (
        boxed_answer(r"earlier \boxed{0}, final \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    )
    ex = Example(
        "x",
        "Question",
        "C",
        "retention_dev",
        "exact",
        metadata={"verifier": "multiple_choice"},
    )
    verifier = BenchmarkVerifier()
    assert verifier.verify(ex, "Final answer: C")
    assert not verifier.verify(ex, "A or C")
    assert not verifier.verify(ex, "D")


def test_source_prompt_audit_and_final_priority():
    ex = Example(
        "x",
        "question with instruction",
        "2",
        "train",
        metadata={"source_prompt": "The raw question"},
    )
    final = replace(
        ex, id="other", prompt="different instructions", split="target_test"
    )
    with pytest.raises(ValueError, match="Duplicate"):
        audit_overlap({"train": [ex], "target_test": [final]})
    suites = {
        role: []
        for role in (
            "train",
            "target_dev",
            "ood_dev",
            "retention_dev",
            "target_test",
            "ood_test",
            "retention_test",
        )
    }
    suites["train"], suites["target_test"] = [ex], [final]
    cleaned, removed = deduplicate(suites)
    assert cleaned["train"] == [] and cleaned["target_test"] == [final]
    assert removed[0]["role"] == "train"
    with pytest.raises(ValueError, match="training"):
        Correction(final, "2", 1)


def test_code_verifier_has_no_host_execution_fallback(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        stdout = b""
        if argv[1] == "run":
            mount = argv[argv.index("--mount") + 1]
            directory = Path(mount.split("source=", 1)[1].split(",target=", 1)[0])
            marker = re.search(
                r"THIRD_EYE_TESTS_PASSED_[0-9a-f]+",
                (directory / "runner.py").read_text(),
            ).group(0)
            stdout = marker.encode() + b"\n"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        "third_eye.evaluation.sandbox.run_bounded", lambda argv, timeout: fake_run(argv)
    )
    sandbox = DockerSandbox("python@sha256:" + "a" * 64)
    ex = Example(
        "c",
        "code prompt",
        "def f(): return 1",
        "train",
        "code",
        metadata={"test_list": ["assert f()==1"]},
    )
    assert sandbox.verify(ex, "```python\ndef f(): return 1\n```")
    command = calls[-1]
    assert command[:2] == ["docker", "run"]
    assert (
        "--network=none" in command
        and "--read-only" in command
        and "--cap-drop=ALL" in command
    )
    assert "--pids-limit=32" in command
    monkeypatch.setattr(
        "third_eye.evaluation.sandbox.run_bounded",
        lambda *a, **k: SimpleNamespace(
            returncode=125, stdout=b"", stderr=b"daemon unavailable"
        ),
    )
    with pytest.raises(SandboxUnavailable):
        sandbox.verify(ex, "bad code")
    assert extract_code("```python\nx=1\n```") == "x=1"
