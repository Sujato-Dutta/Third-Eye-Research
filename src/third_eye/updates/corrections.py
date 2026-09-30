from third_eye.data.schema import Correction


class InsufficientCorrections(RuntimeError):
    """No silent candidate-size/budget changes are permitted."""


def collect_corrections(backend, train_examples, verifier, protocol, seed):
    if any(ex.split != "train" for ex in train_examples):
        raise ValueError("Correction generation accepts training prompts only")
    pool, failures, attempted = [], 0, 0
    for i, example in enumerate(train_examples):
        initial = backend.generate(
            example.prompt, protocol.max_new_tokens, seed=seed + i
        )
        if verifier.verify(example, initial):
            continue
        failures += 1
        # Reference answers and verifier feedback are never exposed to the model.
        retry_prompt = (
            example.prompt
            + "\n\nYour earlier proposed solution was:\n"
            + initial
            + "\n\nReconsider the problem independently and provide a corrected solution. "
            + (
                "End with #### followed by the numeric answer."
                if example.task == "math"
                else "Return the solution."
            )
        )
        for attempt in range(protocol.correction_attempts):
            attempted += 1
            completion = backend.generate(
                retry_prompt,
                protocol.max_new_tokens,
                seed=seed + 100_000 + i * protocol.correction_attempts + attempt,
                temperature=protocol.correction_temperature,
            )
            if verifier.verify(example, completion):
                pool.append(Correction(example, completion, attempt + 1))
                break
    return pool, {
        "training_prompts": len(train_examples),
        "initial_failures": failures,
        "revision_attempts": attempted,
        "verified_corrections": len(pool),
        "verifier_pass_rate": len(pool) / attempted if attempted else 0.0,
    }
