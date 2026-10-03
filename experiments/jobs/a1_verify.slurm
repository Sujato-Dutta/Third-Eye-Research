#!/bin/bash
#SBATCH --job-name=third-eye-a1-verify
#SBATCH --partition=gh-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=third_eye_a1_verify_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/experiments/jobs/vista_environment.sh"
export THIRD_EYE_A1_VERIFY_DEVICE=cuda
python -m pytest -q tests/test_a1.py::test_terminal_continuation_preserves_real_adapter --basetemp "$TMPDIR/a1"
python - <<'PY'
import json,os,sys
sys.path.insert(0,'src')
from pathlib import Path
from third_eye.provenance import source_inventory,execution_environment
from third_eye.io import write_json
proof=json.loads(Path('runs/a1_cpu_passed.json').read_text())
inventory=source_inventory()
assert proof['source_tree_sha256']==inventory['source_tree_sha256']
write_json('runs/a1_gpu_passed.json',dict(status='passed',job_id=os.environ['SLURM_JOB_ID'],execution=execution_environment(),**inventory))
PY
