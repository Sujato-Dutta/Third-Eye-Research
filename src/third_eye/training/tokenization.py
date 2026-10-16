"""Assistant-only SFT masking; reject truncation rather than train bad targets."""


def frozen_chat_kwargs(tokenizer, configured, context=None):
    """A wall-clock-dependent system prefix is part of the model input."""
    context = dict(context or {})
    if set(context) - {"date_string"}:
        raise ValueError("Unknown rendering-context field")
    if any(k in configured and configured[k] != v for k, v in context.items()):
        raise ValueError("Rendering context conflicts with the frozen configuration")
    kwargs = {**configured, **context}
    template = tokenizer.chat_template
    templates = template.values() if isinstance(template, dict) else [template or ""]
    if any("strftime_now" in t for t in templates):
        date = kwargs.get("date_string")
        if not isinstance(date, str) or not date.strip():
            raise ValueError(
                "Date-dependent chat template requires an explicit frozen date_string"
            )
    return kwargs


def format_prompt(tokenizer, prompt, chat_kwargs):
    chat_kwargs = frozen_chat_kwargs(tokenizer, chat_kwargs)
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **chat_kwargs,
        )
    return f"Question: {prompt}\nAnswer: "


def encode_completion(tokenizer, prompt, completion, max_length, chat_kwargs):
    chat_kwargs = frozen_chat_kwargs(tokenizer, chat_kwargs)
    prefix = format_prompt(tokenizer, prompt, chat_kwargs)
    if tokenizer.chat_template:
        full = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            tokenize=False,
            add_generation_prompt=False,
            **chat_kwargs,
        )
        if not full.startswith(prefix):
            raise ValueError(
                "Chat template's assistant prefix differs from generation prefix"
            )
    else:
        full = prefix + completion + (tokenizer.eos_token or "")
    # Offset mappings handle a BPE token that crosses the prompt/answer boundary.
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if len(ids) > max_length:
        raise ValueError(
            "SFT example exceeds max_sequence_length; change the frozen budget before running"
        )
    labels = [
        token if end > len(prefix) and start >= len(prefix) else -100
        for token, (start, end) in zip(ids, offsets)
    ]
    if not any(label != -100 for label in labels[1:]):
        raise ValueError("No assistant tokens remain for causal loss")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def collate(rows, pad_id):
    import torch

    length = max(len(row["input_ids"]) for row in rows)
    output = {}
    for key, padding in (
        ("input_ids", pad_id),
        ("attention_mask", 0),
        ("labels", -100),
    ):
        output[key] = torch.tensor(
            [row[key] + [padding] * (length - len(row[key])) for row in rows],
            dtype=torch.long,
        )
    return output
