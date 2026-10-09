# ruff: noqa: E402
"""Fix only E2 generated launcher paths; frozen scientific implementation intact."""

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools")]
import empirical_checkpoint_v3 as core
from third_eye.io import file_digest, write_json

OUT = core.OUT / "path_repair_v1"
ORIGINAL = core.arguments


def arguments(mode, index=None, dependency=None):
    return [
        a.replace(
            "empirical_checkpoint_gpu_v2.slurm", "empirical_checkpoint_gpu_v3.slurm"
        ).replace(
            "empirical_checkpoint_cpu_v2.slurm", "empirical_checkpoint_cpu_v3.slurm"
        )
        for a in ORIGINAL(mode, index, dependency)
    ]


def checked():
    core.frozen()
    release = json.loads((OUT / "release.json").read_text())
    if release["core_release_sha256"] != file_digest(core.OUT / "release.json"):
        raise RuntimeError("Path repair core release differs")
    if any(file_digest(ROOT / p) != h for p, h in release["files"].items()):
        raise RuntimeError("Path repair source differs")


def main():
    checked()
    mode = sys.argv[1]
    if mode == "validate":
        assert "_gpu_v2.slurm" in ORIGINAL("noise", 0)[-3]
        for m in ("trajectory", "noise", "confirm", "review"):
            args = arguments(m, 0 if m in ("trajectory", "noise") else None)
            assert not any("_v2.slurm" in a for a in args)
        write_json(
            OUT / "cpu_passed.json",
            dict(status="passed", release_sha256=file_digest(OUT / "release.json")),
        )
        return
    passed = json.loads((OUT / "cpu_passed.json").read_text())
    if passed["status"] != "passed" or passed["release_sha256"] != file_digest(
        OUT / "release.json"
    ):
        raise RuntimeError("Validated path repair required")
    core.passed("cpu")
    core.passed("cuda")
    core.arguments = arguments
    write_json(
        OUT / "jobs" / f"{os.environ['SLURM_JOB_ID']}.json",
        dict(
            mode=mode,
            argv=sys.argv[1:],
            path_repair_release_sha256=file_digest(OUT / "release.json"),
            core_release_sha256=file_digest(core.OUT / "release.json"),
            scientific_parameters_changed=False,
        ),
    )
    core.main()


if __name__ == "__main__":
    main()
