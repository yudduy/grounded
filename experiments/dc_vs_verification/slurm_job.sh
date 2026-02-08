#!/bin/bash
#SBATCH --job-name=dc-vs-verify
#SBATCH --output=experiments/dc_vs_verification/results/slurm_%j.out
#SBATCH --error=experiments/dc_vs_verification/results/slurm_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint=GPU_MEM:48GB

# DC vs Verification on AIME 2024
# Runs all 4 conditions x 3 seeds on a single L40S (48GB)

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"

echo "=== DC vs Verification Experiment ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start:  $(date)"
echo ""

mkdir -p experiments/dc_vs_verification/results

# Activate environment
if [ -d "src/.venv" ]; then
    source src/.venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run with timeout (100 min for experiment + buffer)
timeout 6000 python3 experiments/dc_vs_verification/run.py --seeds 42 123 7

EXIT_CODE=$?

echo ""
echo "Exit code: $EXIT_CODE"
echo "End:       $(date)"

# Auto-run analysis if experiment succeeded
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== Running Analysis ==="
    python3 experiments/dc_vs_verification/analysis.py
fi
