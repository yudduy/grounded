#!/bin/bash
#SBATCH --job-name=grounded-prelim
#SBATCH --output=results/prelim_%j.out
#SBATCH --error=results/prelim_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

export HYPERBOLIC_API_KEY="sk_live_nX5aYMk0DhQSs1q7aSgPHAp-x4BDLLuluCmiZQFojWicMCuicMh_Ulztxy52_2u_Q"

cd /home/users/duynguy/proj/grounded
source src/.venv/bin/activate
timeout 1200 python3 src/run_experiment.py --prelim-only
