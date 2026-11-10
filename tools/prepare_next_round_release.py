"""Stage a private frozen source release; no scientific execution."""
import hashlib
import argparse
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from third_eye.provenance import source_inventory  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="runs/deployment/next_round_20261106")
args = parser.parse_args()
DEST=ROOT/args.output
if not DEST.resolve().is_relative_to(ROOT.resolve()):
    raise ValueError("Source package must stay within workspace")
DEST.mkdir(parents=True,exist_ok=True)
inventory=source_inventory(ROOT)
paths=[]
for directory in ("src","experiments","cluster","tests"):
    paths.extend(p for p in (ROOT/directory).rglob("*")
                 if p.is_file() and p.suffix in {".py",".slurm",".sh"})
paths.extend(ROOT.glob("requirements*.txt"))
paths.extend((ROOT/"experiments/configs").glob("*.json"))
paths += [ROOT/"pytest.ini",ROOT/"ruff.toml"]
archive=DEST/"source.tar.gz"
with tarfile.open(archive,"w:gz") as bundle:
    for path in sorted(set(paths)):
        bundle.add(path,arcname=path.relative_to(ROOT).as_posix())
result={**inventory,
        "archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
        "archive":str(archive),"bytes":archive.stat().st_size}
(DEST/"release.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:result[k] for k in ("source_tree_sha256","archive_sha256","bytes")}))
