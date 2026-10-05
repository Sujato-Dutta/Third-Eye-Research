from third_eye.data.schema import Correction
from third_eye.evaluation.runner import generate_greedy, verify_completions
import time


class InsufficientCorrections(RuntimeError):
    """No silent candidate-size/budget changes are permitted."""


def collect_corrections(backend, train_examples, verifier, protocol, seed):
    if any(ex.split != "train" for ex in train_examples):
        raise ValueError("Correction generation accepts training prompts only")
    started = time.perf_counter()
    seed_stride = protocol.correction_seed_stride or protocol.correction_attempts
    pool, failures, attempted = [], 0, 0
    generated_words, generated_tokens = 0, 0
    tokenizer = getattr(backend, "tokenizer", None)
    stamp = time.perf_counter()
    initial_completions = generate_greedy(
        backend,
        train_examples,
        protocol.max_new_tokens,
        [seed + i for i in range(len(train_examples))],
    )
    timings = {
        "greedy_generation_seconds": time.perf_counter() - stamp,
        "revision_generation_seconds": 0.0,
        "revision_verification_seconds": 0.0,
    }
    stamp = time.perf_counter()
    initial_verdicts = verify_completions(verifier, train_examples, initial_completions)
    timings["initial_verification_seconds"] = time.perf_counter() - stamp
    pending = []
    retry_prompts = {}
    for i, (example, initial, passed) in enumerate(
        zip(train_examples, initial_completions, initial_verdicts)
    ):
        generated_words += len(initial.split())
        if tokenizer is not None:
            generated_tokens += len(tokenizer.encode(initial, add_special_tokens=False))
        if passed:
            continue
        failures += 1
        # Reference answers and verifier feedback are never exposed to the model.
        retry_prompts[i] = (
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
        pending.append(i)

    batched = protocol.sampled_batch_size > 1 and hasattr(
        backend, "generate_sampled_many"
    )
    print(
        f"Correction pool: {failures} initial failures; {protocol.correction_attempts} retries; sampled batch {protocol.sampled_batch_size}",
        flush=True,
    )
    if batched:
        completed = {}
        for attempt in range(protocol.correction_attempts):
            if not pending:
                break
            attempted += len(pending)
            seeds = [seed + 100_000 + i * seed_stride + attempt for i in pending]
            stamp = time.perf_counter()
            completions = backend.generate_sampled_many(
                [retry_prompts[i] for i in pending],
                protocol.max_new_tokens,
                seeds,
                protocol.correction_temperature,
            )
            timings["revision_generation_seconds"] += time.perf_counter() - stamp
            if len(completions) != len(pending):
                raise ValueError("Correction generation returned an incomplete batch")
            stamp = time.perf_counter()
            verdicts = verify_completions(
                verifier, [train_examples[i] for i in pending], completions
            )
            timings["revision_verification_seconds"] += time.perf_counter() - stamp
            remaining = []
            for i, completion, passed in zip(pending, completions, verdicts):
                generated_words += len(completion.split())
                if tokenizer is not None:
                    generated_tokens += len(
                        tokenizer.encode(completion, add_special_tokens=False)
                    )
                if passed:
                    completed[i] = Correction(
                        train_examples[i], completion, attempt + 1
                    )
                else:
                    remaining.append(i)
            pending = remaining
        pool = [completed[i] for i in sorted(completed)]
    else:
        for i in pending:
            example = train_examples[i]
            for attempt in range(protocol.correction_attempts):
                attempted += 1
                stamp = time.perf_counter()
                completion = backend.generate(
                    retry_prompts[i],
                    protocol.max_new_tokens,
                    seed=seed + 100_000 + i * seed_stride + attempt,
                    temperature=protocol.correction_temperature,
                )
                timings["revision_generation_seconds"] += time.perf_counter() - stamp
                generated_words += len(completion.split())
                if tokenizer is not None:
                    generated_tokens += len(
                        tokenizer.encode(completion, add_special_tokens=False)
                    )
                stamp = time.perf_counter()
                passed = verifier.verify(example, completion)
                timings["revision_verification_seconds"] += time.perf_counter() - stamp
                if passed:
                    pool.append(Correction(example, completion, attempt + 1))
                    break
    return pool, {
        "training_prompts": len(train_examples),
        "initial_failures": failures,
        "revision_attempts": attempted,
        "verified_corrections": len(pool),
        "correction_attempt_limit": protocol.correction_attempts,
        **(
            {"correction_seed_stride": seed_stride}
            if protocol.correction_seed_stride
            else {}
        ),
        "unresolved_failures": failures - len(pool),
        "harvest_complete": True,
        "verifier_pass_rate": len(pool) / attempted if attempted else 0.0,
        "generated_completions": len(train_examples) + attempted,
        "generated_words": generated_words,
        **({"generated_tokens": generated_tokens} if tokenizer is not None else {}),
        **timings,
        "total_seconds": time.perf_counter() - started,
    }
