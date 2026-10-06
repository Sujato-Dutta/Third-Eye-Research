#!/bin/bash
#SBATCH --job-name=third-eye-a2-cuda-dev-v2
#SBATCH --partition=gh-dev
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=third_eye_a2_cuda_dev_v2_%j.out
source "${SLURM_SUBMIT_DIR:?}/tools/a2_environment.sh"
mkdir -p runs/a2/validation/v2
srun --mpi=none --kill-on-bad-exit=1 --ntasks=2 --ntasks-per-node=1 bash -c '
  set -euo pipefail
  export SLURM_ARRAY_TASK_ID="${SLURM_PROCID:?}"
  source tools/a2_environment.sh
  export CUDA_VISIBLE_DEVICES=0
  python -c '\''import json, os, socket; from pathlib import Path; rank=int(os.environ["SLURM_PROCID"]); Path(f"runs/a2/validation/v2/step_host_{rank}.json").write_text(json.dumps({"host":socket.gethostname(),"slurm_job_id":os.environ["SLURM_JOB_ID"],"rank":rank,"step_node_list":os.environ.get("SLURM_STEP_NODELIST")}))'\''
  python tools/verify_a2_cuda_v2.py
  export THIRD_EYE_A1_VERIFY_DEVICE=cuda
  python -m pytest -q tests/test_a1.py -k terminal_continuation -p no:cacheprovider --basetemp "$TMPDIR/terminal_tests"
'
