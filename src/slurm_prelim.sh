#!/bin/bash
#SBATCH --job-name=grounded-prelim
#SBATCH --output=src/results/prelim_%j.out
#SBATCH --error=src/results/prelim_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

: "${HYPERBOLIC_API_KEY:?Set HYPERBOLIC_API_KEY in the environment}"

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"
mkdir -p src/results
source src/.venv/bin/activate
timeout 1200 python3 src/run_experiment.py --prelim-only
