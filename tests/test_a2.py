"""A2 keeps historical fingerprints and the audited correction draw prefix."""

from dataclasses import replace
from pathlib import Path

import pytest

from third_eye.config import Config
from third_eye.io import digest


def a2(config):
    return replace(
        config,
        protocol=replace(
            config.protocol,
            amendment="A2",
            correction_attempts=8,
            correction_seed_stride=16,
            deterministic_execution=True,
            candidate_sampling="stratified_distinct",
            terminal_continuation=True,
        ),
    )


def test_historical_config_fingerprints_survive():
    import json

    for path in (Path(__file__).resolve().parents[1] / "experiments/configs").glob(
        "*.json"
    ):
        raw = json.loads(path.read_text())
        assert Config.load(path).fingerprint == digest(raw), path


@pytest.mark.parametrize(
    "changes",
    [
        {"correction_attempts": 16},
        {"correction_seed_stride": 8},
        {"deterministic_execution": False},
        {"terminal_continuation": False},
    ],
)
def test_invalid_a2_rejected(changes):
    cfg = a2(Config())
    with pytest.raises(ValueError):
        replace(cfg, protocol=replace(cfg.protocol, **changes))


@pytest.mark.parametrize("batched", [False, True])
def test_native_a2_collector_preserves_full_budget_draw_prefix(batched):
    from third_eye.data.schema import Example
    from third_eye.updates.corrections import collect_corrections

    class Backend:
        def generate(self, prompt, max_new_tokens, seed, temperature=0.0):
            if temperature == 0:
                return "bad"
            i, attempt = divmod(seed - 100_042, 16)
            return "ok" if attempt + 1 == [2, 9, 16][i] else "bad"

        def generate_many(self, prompts, max_new_tokens, seeds):
            return ["bad"] * len(prompts)

        def generate_sampled_many(self, prompts, max_new_tokens, seeds, temperature):
            return [
                self.generate(p, max_new_tokens, s, temperature)
                for p, s in zip(prompts, seeds)
            ]

    class Verifier:
        def verify(self, example, completion):
            return completion == "ok"

    cfg = replace(
        Config(),
        protocol=replace(Config().protocol, sampled_batch_size=16 if batched else 1),
    )
    examples = [Example(str(i), f"problem {i}", "ok", "train") for i in range(3)]
    full, _ = collect_corrections(
        Backend(),
        examples,
        Verifier(),
        replace(cfg.protocol, correction_attempts=16),
        42,
    )
    short, stats = collect_corrections(
        Backend(), examples, Verifier(), a2(cfg).protocol, 42
    )
    assert [c.to_dict() for c in short] == [c.to_dict() for c in full if c.attempt <= 8]
    assert stats["revision_attempts"] == 18
    assert stats["correction_seed_stride"] == 16


def test_a1_and_a2_cannot_be_pooled(tmp_path):
    import json
    from test_forecasting import records
    from third_eye.forecasting.dataset import read_records

    rows = records(trajectories=2)
    for r in rows:
        r["protocol_amendment"] = (
            "A1" if r["trajectory_id"] == rows[0]["trajectory_id"] else "A2"
        )
    path = tmp_path / "mixed.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="amendments"):
        read_records([path])
