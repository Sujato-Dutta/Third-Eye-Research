"""Time-dependent tokenizer inputs must be explicit, stable and checkpointed."""

import pytest
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace

from third_eye.training.tokenization import frozen_chat_kwargs, format_prompt


class ClockTemplate:
    chat_template = "{{ strftime_now('%d %b %Y') }}"

    def apply_chat_template(self, messages, **kwargs):
        return kwargs["date_string"] + " / " + messages[0]["content"]


def test_volatile_template_cannot_silently_use_todays_date():
    with pytest.raises(ValueError, match="explicit frozen"):
        format_prompt(ClockTemplate(), "problem", {})
    assert (
        format_prompt(ClockTemplate(), "problem", {"date_string": "11 Oct 2026"})
        == "11 Oct 2026 / problem"
    )
    assert format_prompt(
        ClockTemplate(), "problem", {"date_string": "12 Oct 2026"}
    ) != format_prompt(ClockTemplate(), "problem", {"date_string": "11 Oct 2026"})


def test_context_cannot_override_an_explicit_protocol_date():
    with pytest.raises(ValueError, match="conflicts"):
        frozen_chat_kwargs(
            ClockTemplate(),
            {"date_string": "11 Oct 2026"},
            {"date_string": "12 Oct 2026"},
        )
    with pytest.raises(ValueError, match="Unknown"):
        frozen_chat_kwargs(ClockTemplate(), {}, {"hidden_override": 1})
    configured = {"enable_thinking": False}
    result = frozen_chat_kwargs(
        ClockTemplate(), configured, {"date_string": "11 Oct 2026"}
    )
    assert configured == {"enable_thinking": False}
    assert result == {"enable_thinking": False, "date_string": "11 Oct 2026"}


def test_static_template_does_not_require_date_and_dictionary_templates_are_checked():
    tokenizer = ClockTemplate()
    tokenizer.chat_template = "static"
    assert frozen_chat_kwargs(tokenizer, {"enable_thinking": False}) == {
        "enable_thinking": False
    }
    tokenizer.chat_template = {"default": "static", "tool": "{{ strftime_now('x') }}"}
    with pytest.raises(ValueError, match="explicit frozen"):
        frozen_chat_kwargs(tokenizer, {})


def test_original_log_time_reconstruction_rejects_midnight_ambiguity(monkeypatch):
    import third_eye.experiments.rendering as rendering

    monkeypatch.setattr(
        rendering, "ZoneInfo", lambda name: timezone(timedelta(hours=-5))
    )
    # 03:04 UTC on Oct12 was Oct11 at the original Vista runtime timezone.
    timestamp = datetime(2026, 10, 12, 3, 4, tzinfo=timezone.utc).timestamp()
    log = SimpleNamespace(stat=lambda: SimpleNamespace(st_mtime=timestamp))
    assert rendering.historical_training_date(log, 132)["date_string"] == "11 Oct 2026"
    ending_after_midnight = datetime(
        2026, 10, 12, 5, 0, 30, tzinfo=timezone.utc
    ).timestamp()
    log = SimpleNamespace(stat=lambda: SimpleNamespace(st_mtime=ending_after_midnight))
    assert rendering.historical_training_date(log, 132)["date_string"] == "11 Oct 2026"
    boundary = datetime(2026, 10, 12, 5, 2, 30, tzinfo=timezone.utc).timestamp()
    log = SimpleNamespace(stat=lambda: SimpleNamespace(st_mtime=boundary))
    with pytest.raises(ValueError, match="ambiguous"):
        rendering.historical_training_date(log, 132)
