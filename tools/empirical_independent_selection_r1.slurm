#!/bin/bash
#SBATCH --job-name=third-eye-independent-r1
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --output=third_eye_independent_r1_%j.out
source "${SLURM_SUBMIT_DIR:?}/tools/a2_environment.sh"
export CUDA_VISIBLE_DEVICES=""
if [[ "${1:?}" == validate ]]; then
    python -m pytest -q tools/test_independent_selection_r1.py tools/test_review_dispatch_r1.py --basetemp "$TMPDIR/r1_tests"
    python - <<'PY'
from pathlib import Path
import json, sys
root=Path.cwd()
sys.path[:0]=[str(root/'tools'),str(root/'src')]
from empirical_checkpoint_v3 import frozen, OUT
from third_eye.io import file_digest, write_json
frozen()
out=OUT/'independent_selection_r1';release=json.loads((out/'release.json').read_text())
assert all(file_digest(root/p)==h for p,h in release['files'].items())
assert file_digest(OUT/'release.json')==release['core_release_sha256']
write_json(out/'cpu_passed.json',dict(status='passed',release_sha256=file_digest(out/'release.json')))
PY
else
    [[ "$1" == review ]] || exit 1
    python tools/empirical_independent_selection_r1.py
fi
