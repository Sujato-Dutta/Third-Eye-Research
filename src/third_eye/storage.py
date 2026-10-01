"""Storage budgets and controlled pruning of completed branch checkpoints."""

from pathlib import Path
import os
import shutil

from third_eye.io import write_json


def project_size(root):
    """Count actual regular files once; cache snapshot symlinks are excluded."""
    total = 0
    pending = [Path(root)]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    stat = entry.stat(follow_symlinks=False)
                    total += (
                        stat.st_blocks * 512
                        if hasattr(stat, "st_blocks")
                        else stat.st_size
                    )
    return total


def check_storage(root, quota_gb=50, headroom_gb=2):
    used = project_size(root)
    free = shutil.disk_usage(root).free
    if (
        used + headroom_gb * 1_000_000_000 > quota_gb * 1_000_000_000
        or free < headroom_gb * 1_000_000_000
    ):
        raise RuntimeError(
            "Insufficient project storage headroom; archive completed outputs or rotate the dedicated model cache"
        )
    return {
        "project_bytes": used,
        "filesystem_free_bytes": free,
        "quota_gb": quota_gb,
        "headroom_gb": headroom_gb,
    }


def prune_branches(output, state_id):
    """Keep verified labels, hashes, batches and logs; retain accepted adapters.

    Only known checkpoint subdirectories inside this completed state may be
    removed. Symlinks and paths escaping the run are rejected.
    """
    output = Path(output).resolve()
    state = output / "states" / state_id
    if state.is_symlink() or not state.resolve().is_relative_to(output):
        raise ValueError("Branch pruning path escapes its run directory")
    if not (state / "summary.json").is_file() or not (state / "labels.json").is_file():
        raise ValueError("Only complete published states may have checkpoints pruned")
    candidates = [state / "parent"]
    for directory in state.iterdir():
        if directory.is_dir() and directory.name.startswith("k"):
            candidates.extend((directory / "t1", directory / "t2"))
    removed = []
    for path in candidates:
        if path.is_symlink() or not path.resolve().is_relative_to(state.resolve()):
            raise ValueError("Checkpoint path escapes the completed state")
        if path.exists():
            shutil.rmtree(path)
            removed.append(str(path.relative_to(output)))
    write_json(
        state / "checkpoint_retention.json",
        {
            "policy": "accepted_only",
            "removed_checkpoints": removed,
            "preserved": "labels, adapter hashes, candidate data, probe and training logs; accepted adapters",
        },
    )
    return removed


def prune_accepted(output, generation, keep=1):
    """Prune older checkpoints only after the new accepted checkpoint exists."""
    if keep < 1:
        raise ValueError("Keep at least the latest accepted checkpoint")
    output = Path(output).resolve()
    accepted = output / "accepted"
    if not (accepted / f"generation_{generation}" / "resume.json").is_file():
        raise ValueError("Latest accepted checkpoint is not complete")
    removed = []
    for old in range(1, generation - keep + 1):
        checkpoint = accepted / f"generation_{old}"
        if checkpoint.is_symlink() or not checkpoint.resolve().is_relative_to(output):
            raise ValueError("Accepted checkpoint path escapes its run")
        if checkpoint.exists():
            shutil.rmtree(checkpoint)
            removed.append(str(checkpoint.relative_to(output)))
    write_json(
        accepted / "retention.json",
        {"keep_latest": keep, "latest_generation": generation, "removed": removed},
    )


def prune_online_branches(output, generation):
    output = Path(output).resolve()
    latest = output / "accepted" / f"generation_{generation + 1}"
    if not (latest / "resume.json").exists():
        raise ValueError("Selected adapter has not been safely committed")
    removed = []
    for index in range(3):
        path = output / "online" / f"generation_{generation}" / f"branch_{index}"
        if path.is_symlink() or not path.resolve().is_relative_to(output):
            raise ValueError("Online branch path escapes its run")
        if path.exists():
            shutil.rmtree(path)
            removed.append(str(path.relative_to(output)))
    write_json(
        output / "online" / f"generation_{generation}" / "checkpoint_retention.json",
        {"policy": "accepted_only", "removed": removed},
    )
