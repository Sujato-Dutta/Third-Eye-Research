"""CPU-only type repair for frozen E2 native-record diagnostic regret."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools")]
from empirical_noise_review_v1 import regret as original_regret


def list_safe_regret(one, two):
    # Preserve the declared expression, winner/tie rule and outcomes. Native
    # saved labels arrive as lists, whereas the helper uses array indexing.
    return original_regret(np.asarray(one, dtype=float), np.asarray(two, dtype=float))


def checked():
    from empirical_checkpoint_v3 import OUT, frozen
    from third_eye.io import file_digest
    frozen()  # Requires compute allocation and unchanged original sources/methods.
    out = OUT / "review_array_repair_v1"
    release_path = out / "release.json"
    release = json.loads(release_path.read_text())
    if release["core_release_sha256"] != file_digest(OUT / "release.json") or any(file_digest(ROOT / p) != h for p,h in release["files"].items()):
        raise RuntimeError("Review repair source or original prerequisite changed")
    return out, file_digest(release_path)


def main():
    from third_eye.io import file_digest, write_json
    out, release_sha = checked()
    if sys.argv[1:] == ["validate"]:
        write_json(out / "cpu_passed.json", dict(status="passed", release_sha256=release_sha))
        return
    if sys.argv[1:] != ["review"]:
        raise ValueError("Specify validate or review")
    receipt = json.loads((out / "cpu_passed.json").read_text())
    if receipt != dict(status="passed", release_sha256=release_sha):
        raise RuntimeError("Matching CPU validation required")
    import empirical_noise_review_v1 as helpers
    import empirical_checkpoint_v3 as core
    helpers.regret = list_safe_regret
    write_json(out / "started.json", dict(release_sha256=release_sha, original_review_job="1063239", scientific_parameters_changed=False, change="Convert native diagnostic utility lists to float arrays before the unchanged regret expression"))
    core.review()
    write_json(out / "success.json", dict(status="complete", release_sha256=release_sha, report_sha256=file_digest(core.OUT / "noise_review.json")))


if __name__ == "__main__":
    main()
