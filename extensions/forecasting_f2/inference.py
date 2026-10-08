"""Explicit artifact dispatch for frozen online/evidence CLIs."""

from contextlib import contextmanager
import json
from pathlib import Path

from third_eye.forecasting.training import Forecaster
from extensions.forecasting_f2.training import (
    ContrastForecaster,
    EXTENSION,
    extension_inventory,
)


@contextmanager
def registered_loader():
    """Extend loading without disguising new weights as the legacy GRU."""
    original = Forecaster.__dict__["load"]
    legacy = Forecaster.load

    def load(cls, path, device="cpu"):
        meta = json.loads((Path(path) / "metadata.json").read_text())
        extension = meta.get("extension")
        if extension == EXTENSION:
            return ContrastForecaster.load(path, device)
        if extension:
            raise ValueError("Unknown forecaster extension")
        return legacy(path, device)

    Forecaster.load = classmethod(load)
    try:
        yield extension_inventory()
    finally:
        Forecaster.load = original
