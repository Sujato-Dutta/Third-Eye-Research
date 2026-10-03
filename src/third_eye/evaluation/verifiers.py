"""Minimal numeric verifier and an injectable external code-verifier contract."""

from decimal import Decimal, InvalidOperation
import importlib
import re

NUMBER = r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"


def extract_numeric(text):
    # Ignore an explicit reasoning channel when the model provides one.
    text = text.split("</think>")[-1]
    if "####" in text:
        tail = text.rsplit("####", 1)[1].strip()
    else:
        boxes = re.findall(r"\\boxed\{([^{}]+)\}", text)
        if boxes:
            tail = boxes[-1].strip()
        else:
            marked = re.findall(
                r"(?:final answer|answer)\s*(?:is|:|=)\s*(.+)", text, re.I
            )
            tail = marked[-1].strip() if marked else text.strip()
    # No 'last number in arbitrary reasoning' shortcut: ambiguous answers fail.
    match = re.fullmatch(rf"\$?({NUMBER})\s*\.?", tail)
    if not match:
        return None
    try:
        value = Decimal(match.group(1).replace(",", ""))
        return value if value.is_finite() else None
    except InvalidOperation:
        return None


class VerifierRegistry:
    def __init__(self, external=None):
        self.external = external

    @classmethod
    def from_spec(cls, spec):
        """Trusted local module:factory returning callable(example, completion)."""
        if not spec:
            return cls()
        module, factory = spec.split(":", 1)
        return cls(getattr(importlib.import_module(module), factory)())

    def verify(self, example, completion):
        if self.external is not None:
            return bool(self.external(example, completion))
        if example.task == "math":
            predicted, reference = (
                extract_numeric(completion),
                extract_numeric(example.answer),
            )
            return (
                predicted is not None
                and reference is not None
                and predicted == reference
            )
        if example.task == "exact":
            return completion.strip() == example.answer.strip()
        raise RuntimeError(
            "Code verification requires the evaluation team's sandboxed verifier plugin"
        )

    def verify_many(self, examples, completions):
        if len(examples) != len(completions):
            raise ValueError("Verifier inputs must match")
        if self.external is not None and hasattr(self.external, "verify_many"):
            return self.external.verify_many(examples, completions)
        return [self.verify(ex, text) for ex, text in zip(examples, completions)]
