#!/bin/bash
#SBATCH --job-name=third-eye-smol16-results
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --account=IRI23021
source "${THIRD_EYE_SOURCE_ROOT:?}/experiments/jobs/bounded_evidence_environment.sh"
export CUDA_VISIBLE_DEVICES=""
"$venv_root/bin/python" experiments/smollm_confirmation.py review --request "${THIRD_EYE_SMOL_REQUEST:?}"
