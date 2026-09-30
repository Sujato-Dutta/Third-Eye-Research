"""Create an immutable manifest from evaluation-team JSONL splits."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from third_eye.data.schema import ROLES
from third_eye.data.splits import create_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ROLES:
        parser.add_argument("--" + role.replace("_", "-"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    create_manifest({role: getattr(args, role) for role in ROLES}, args.output)
    print(f"Manifest written: {args.output}")


if __name__ == "__main__":
    main()
