#!/bin/bash
#SBATCH --job-name=third-eye-a2-cuda-dev
#SBATCH --partition=gh-dev
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=third_eye_a2_cuda_dev_%j.out
source "${SLURM_SUBMIT_DIR:?}/tools/a2_environment.sh"
srun --mpi=none --kill-on-bad-exit=1 --ntasks=2 --ntasks-per-node=1 bash -c '
  set -euo pipefail
  export SLURM_ARRAY_TASK_ID="${SLURM_PROCID:?}"
  python -c '\''import json, os, socket; from pathlib import Path; rank=int(os.environ["SLURM_PROCID"]); Path(f"runs/a2/validation/step_host_{rank}.json").write_text(json.dumps({"host":socket.gethostname(),"slurm_job_id":os.environ["SLURM_JOB_ID"],"rank":rank,"step_node_list":os.environ.get("SLURM_STEP_NODELIST")}))'\''
  bash tools/a2_cuda_verify.slurm
'
