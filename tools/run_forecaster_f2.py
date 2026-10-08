"""Run a frozen experiment with explicit versioned F2 inference provenance."""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from extensions.forecasting_f2.inference import registered_loader
from extensions.forecasting_f2.training import extension_inventory
from third_eye import provenance
from third_eye.io import file_digest


def main():
    # This wrapper exposes the same CLI. The recorded architecture/extension
    # identifies the method; policy 'direct'/'matched_h1' is the selector role.
    original = provenance.source_inventory

    def inventory(root=None):
        return {
            **original(root),
            **extension_inventory(),
            "extension_runner_sha256": file_digest(Path(__file__)),
        }

    provenance.source_inventory = inventory
    try:
        with registered_loader():
            runpy.run_path(str(ROOT / "experiments/run.py"), run_name="__main__")
    finally:
        provenance.source_inventory = original


if __name__ == "__main__":
    main()
