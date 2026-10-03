"""Deterministic benchmark dispatch; symbolic math uses math-verify."""

import os
import re
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from third_eye.evaluation.verifiers import VerifierRegistry


class BenchmarkVerifier:
    def __init__(self, sandbox=None):
        self.sandbox = sandbox
        self.numeric = VerifierRegistry()

    def verify(self, example, completion):
        kind = example.metadata.get("verifier", example.task)
        if example.task == "code":
            if self.sandbox is None:
                raise RuntimeError(
                    "Code verification requires an isolated sandbox configuration"
                )
            return self.sandbox.verify(example, completion)
        if kind == "symbolic_math":
            # Use an outer process timeout where native signal timeouts cannot
            # work. The pinned library's Windows nested target is unpicklable.
            if os.name == "nt" or threading.current_thread() != threading.main_thread():
                try:
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).with_name("math_worker.py")),
                        ],
                        input=json.dumps(
                            {"reference": example.answer, "completion": completion}
                        ),
                        text=True,
                        capture_output=True,
                        timeout=8,
                    )
                except subprocess.TimeoutExpired:
                    return False
                if result.returncode:
                    raise RuntimeError(
                        "Symbolic verifier worker failed: " + result.stderr[:300]
                    )
                return bool(json.loads(result.stdout))
            from third_eye.evaluation.math_worker import verify_math

            return verify_math(example.answer, completion)
        if kind == "multiple_choice":
            text = completion.split("</think>")[-1].strip()
            marked = re.findall(
                r"(?:final answer|answer)\s*(?:is|:|=)\s*\(?([A-D])\)?", text, re.I
            )
            bare = re.fullmatch(r"\(?([A-D])\)?[. ]*", text)
            choice = marked[-1].upper() if marked else bare.group(1) if bare else None
            return choice == example.answer.strip()
        return self.numeric.verify(example, completion)

    def __call__(self, example, completion):
        return self.verify(example, completion)

    def verify_many(self, examples, completions):
        if len(examples) != len(completions):
            raise ValueError("Verifier inputs must match")
        workers = int(os.environ.get("THIRD_EYE_VERIFIER_WORKERS", "1"))
        allocation = int(os.environ.get("SLURM_CPUS_PER_TASK", str(workers)))
        if not 1 <= workers <= allocation:
            raise ValueError("Verifier workers must fit the allocated CPUs")
        symbolic_workers = int(os.environ.get("THIRD_EYE_SYMBOLIC_WORKERS", "1"))
        if not 1 <= symbolic_workers <= allocation:
            raise ValueError("Symbolic workers must fit the allocated CPUs")
        if examples and all(
            ex.metadata.get("verifier") == "symbolic_math" for ex in examples
        ):
            workers = symbolic_workers
        if workers == 1:
            return [self.verify(ex, text) for ex, text in zip(examples, completions)]
        # Numeric/letter checks are cheap; expensive symbolic and sandbox checks
        # can run independently, with deterministic result order and exceptions.
        results, expensive = [None] * len(examples), []
        for i, (ex, text) in enumerate(zip(examples, completions)):
            if ex.task == "code" or (
                symbolic_workers > 1 and ex.metadata.get("verifier") == "symbolic_math"
            ):
                expensive.append((i, ex, text))
            else:
                results[i] = self.verify(ex, text)

        def check(item):
            i, ex, text = item
            return i, self.verify(ex, text)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, passed in pool.map(check, expensive):
                results[i] = passed
        return results


def make_verifier():
    path = os.environ.get("THIRD_EYE_SANDBOX_CONFIG")
    if path:
        from third_eye.evaluation.sandbox import sandbox_from_config

        sandbox = sandbox_from_config(path)
    else:
        sandbox = None
    return BenchmarkVerifier(sandbox)
