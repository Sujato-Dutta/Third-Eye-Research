#!/bin/bash
#SBATCH --job-name=third-eye-relay-check
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=third_eye_vista_relay_check_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* ]] || exit 1
module load gcc cuda python3
source "${THIRD_EYE_VENV:-.venv}/bin/activate"
export PATH="$PWD/cluster/bin:$PATH"
python -m pytest -q tests/test_vista_submission.py
sbatch --test-only --parsable --partition=gh --account=IRI23021 experiments/jobs/vista_pilot.slurm
printf '%s\n' 'PASS: compute client relayed test-only sbatch validation to login node'
