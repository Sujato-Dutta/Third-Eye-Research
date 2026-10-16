"""Recover historical rendering metadata without tuning on outcome values."""

from datetime import datetime
from zoneinfo import ZoneInfo


FIXED_CONTEXT = {"date_string": "11 Oct 2026"}


def historical_training_date(log_path, seconds):
    """Use original log timestamps, not a date chosen to improve performance.

    Training rows are encoded before timing starts. A 60-second margin around
    encoding/synchronization rejects midnight ambiguity rather than guessing.
    Exact saved-adapter replay remains necessary to verify reconstruction.
    """
    if seconds < 0:
        raise ValueError("Nonnegative original training duration required")
    ended = log_path.stat().st_mtime
    zone = ZoneInfo("America/Chicago")
    encoded_at = ended - seconds
    first = datetime.fromtimestamp(encoded_at - 60, zone).strftime("%d %b %Y")
    last = datetime.fromtimestamp(encoded_at + 60, zone).strftime("%d %b %Y")
    if first != last:
        raise ValueError("Historical training input date is ambiguous across midnight")
    return {
        "date_string": first,
        "timezone": "America/Chicago",
        "log_mtime": ended,
        "encoding_time_estimate": encoded_at,
        "margin_seconds": 60,
    }
