"""Inspect the dedicated model cache or evict one explicitly named model."""

import argparse
from pathlib import Path
import os


def main():
    from huggingface_hub import scan_cache_dir

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cache",
        default=str(Path(os.environ.get("HF_HOME", "models/hf_cache")) / "hub"),
    )
    p.add_argument(
        "--evict-model",
        help="Exact model repository ID; run only after its compute jobs have stopped",
    )
    a = p.parse_args()
    cache = scan_cache_dir(a.cache)
    repos = [r for r in cache.repos if r.repo_type == "model"]
    if a.evict_model:
        selected = [r for r in repos if r.repo_id == a.evict_model]
        if not selected:
            p.error("Requested model is absent from the dedicated cache")
        revisions = [v.commit_hash for r in selected for v in r.revisions]
        strategy = cache.delete_revisions(*revisions)
        print(f"Evicting {a.evict_model}: {strategy.expected_freed_size_str}")
        strategy.execute()
    else:
        for repo in sorted(repos, key=lambda r: r.repo_id):
            print(f"{repo.repo_id}: {repo.size_on_disk_str}")
        print(f"Total cache: {cache.size_on_disk_str}")


if __name__ == "__main__":
    main()
