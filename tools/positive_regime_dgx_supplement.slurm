#!/bin/bash
#SBATCH --job-name=te-positive-artifacts
#SBATCH --account=student
#SBATCH --partition=cpu_student
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --chdir=/dgxa_home/se23uari167/third_eye_dgx_20261026
#SBATCH --output=/dgxa_home/se23uari167/third_eye_dgx_20261026/runs/positive_regime_20261109/logs/artifacts-%j.out
#SBATCH --error=/dgxa_home/se23uari167/third_eye_dgx_20261026/runs/positive_regime_20261109/logs/artifacts-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* ]] || exit 1
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/dgxa_home/se23uari167/third_eye_dgx_20261026/releases/positive_regime_d5c3bc9fb65a/src:/dgxa_home/se23uari167/third_eye_dgx_20261026/releases/positive_regime_d5c3bc9fb65a/experiments
campaign=/dgxa_home/se23uari167/third_eye_dgx_20261026/runs/positive_regime_20261109
python=/dgxa_home/se23uari167/third_eye_dgx_20261026/.venv/bin/python
"$python" -m pytest -q "$campaign/validation/test_positive_regime_supplement.py" -k complete_artifact_flow --basetemp "$campaign/validation/artifact_fixtures"
"$python" - "$campaign" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
from third_eye.provenance import source_inventory
campaign = Path(sys.argv[1])
test = campaign/"validation/test_positive_regime_supplement.py"
record = {"passed":True,"scenarios":["nonterminal","initial scarcity","continuation scarcity"],
          "job_id":os.environ["SLURM_JOB_ID"],"test_sha256":hashlib.sha256(test.read_bytes()).hexdigest(),
          "source_tree_sha256":source_inventory()["source_tree_sha256"]}
(campaign/"validation/artifacts_completed.json").write_text(json.dumps(record,indent=2)+"\n")
PY
