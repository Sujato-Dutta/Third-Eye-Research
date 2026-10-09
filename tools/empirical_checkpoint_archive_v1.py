"""Archive E2 failed validation, frozen sources/methods and initial valid queue."""
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools")]
from empirical_checkpoint_v3 import BASE, OUT, frozen  # noqa: E402
from third_eye.io import file_digest, write_json  # noqa: E402


def main():
    frozen()
    old = BASE / "checkpoint_fallback_v2"
    old_release = json.loads((old / "release.json").read_text())
    current = json.loads((OUT / "release.json").read_text())
    methods = json.loads((BASE / "methods/manifest.json").read_text())
    inputs = {ROOT / name for name in [*old_release["files"], *current["files"]]}
    inputs |= {Path(p) for p in methods["artifact_sha256"]}
    inputs |= set(old.rglob("*"))
    inputs |= {OUT / name for name in ["release.json", "decision.json", "cpu_passed.json", "validation_chain.json", "trajectory_submissions.json"]}
    inputs |= {BASE / "methods/manifest.json", ROOT / "third_eye_e2_cpu_1057708.out", ROOT / "third_eye_e2_cpu_1057726.out", ROOT / "third_eye_e2_cpu_1057729.out", Path(__file__)}
    inputs = sorted(p for p in inputs if p.is_file())
    if any(not p.resolve().is_relative_to(ROOT) for p in inputs):
        raise RuntimeError("Archive input outside deployment root")
    target = Path("/work/11617/sujato_ts/vista/third_eye_results/empirical_checkpoint_initial_v3_20261008.tar.gz")
    if target.exists():
        raise RuntimeError("Initial archive exists; preserve it")
    snapshot = dict(source_files_sha256={str(p.relative_to(ROOT)): file_digest(p) for p in inputs},
                    purpose="Frozen methods and failed-v2 evidence; initial v3 queue, not final new outcomes")
    receipt = OUT / "initial_archive_manifest.json"
    write_json(receipt, snapshot)
    partial = target.with_suffix(target.suffix + ".partial")
    with tarfile.open(partial, "w:gz", compresslevel=1) as tar:
        for path in [*inputs, receipt]:
            tar.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    partial.replace(target)
    write_json(OUT / "initial_archive_receipt.json", dict(status="complete", path=str(target), sha256=file_digest(target), bytes=target.stat().st_size))


if __name__ == "__main__":
    main()
