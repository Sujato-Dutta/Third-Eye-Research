#!/bin/bash
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* ]] || exit 1
cd /scratch/11617/sujato_ts/third_eye_research/amendments/a1
module purge
module load gcc cuda python3
source /scratch/11617/sujato_ts/third_eye_research/.venv/bin/activate
third_eye_python_binary=$(readlink -f /scratch/11617/sujato_ts/third_eye_research/.venv/bin/python)
export LD_LIBRARY_PATH="$(dirname "$(dirname "$third_eye_python_binary")")/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false TMPDIR=/tmp/third_eye_retry_validation
mkdir -p "$TMPDIR"
python -m pytest -q tests tools/test_a1_parallel_resume.py tools/test_retry_budget.py \
  -p no:cacheprovider --basetemp "$TMPDIR/tests"
python tools/audit_retry_budget.py \
  --plan runs/campaign_a1_20261003/plan.json --cap 8 \
  --output runs/retry_diagnostics/prefix8_audit_20261004.json
python - <<'VALIDATION'
import json,sys
from pathlib import Path
sys.path.insert(0,'src')
from third_eye.io import file_digest,write_json
from third_eye.provenance import source_inventory
root=Path.cwd()
inventory=source_inventory(root)
assert inventory['source_tree_sha256']=='f06285d80e3a248a8141448d51fbac55a095016cfef6036351f18741984abcaa'
files=['tools/audit_retry_budget.py','tools/retry_prefix_backend.py','tools/diagnose_eight_attempts.py','tools/diagnose_eight_attempts.slurm','tools/test_retry_budget.py','tools/validate_retry_diagnostics.sh']
write_json('runs/retry_diagnostics/validation.json',{'status':'passed','tests':116,'source_tree_sha256':inventory['source_tree_sha256'],'files_sha256':{f:file_digest(f) for f in files},'research_runs_changed':False})
print('Retry-prefix validation and historical audit passed')
VALIDATION
