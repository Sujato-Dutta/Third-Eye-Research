"""Portable symbolic scoring worker; the parent enforces process timeout."""

import json
import sys


def verify_math(reference, completion, timeout=5):
    from math_verify import LatexExtractionConfig, ExprExtractionConfig, parse, verify

    gold = parse(
        "$" + reference + "$",
        extraction_config=[LatexExtractionConfig()],
        parsing_timeout=timeout,
        raise_on_error=True,
    )
    if not gold:
        raise ValueError("Unable to parse symbolic benchmark reference")
    prediction = parse(
        completion.split("</think>")[-1],
        extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
        parsing_timeout=timeout,
    )
    return bool(verify(gold, prediction, timeout_seconds=timeout))


if __name__ == "__main__":
    payload = json.load(sys.stdin)
    print(
        json.dumps(
            verify_math(payload["reference"], payload["completion"], timeout=None)
        )
    )
