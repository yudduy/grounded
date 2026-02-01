#!/bin/bash
#SBATCH --job-name=grounded-prelim
#SBATCH --output=results/prelim_%j.out
#SBATCH --error=results/prelim_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

cd /home/users/duynguy/proj/grounded
source src/.venv/bin/activate
timeout 1200 python3 src/run_experiment.py --prelim-only
