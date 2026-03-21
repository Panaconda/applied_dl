#!/bin/bash
# Run from project root (applied_dl/)

#SBATCH --qos=mcml
#SBATCH -p mcml-hgx-a100-80x4-mig
#SBATCH --job-name=baseline_classifier
#SBATCH --output=classifier/runs/slurm_%j.out
#SBATCH --error=classifier/runs/slurm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1

# Exit on error
set -e

# --- 1. Load Environment Variables ---
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# Fallbacks if .env not present
DATA_DIR="${DATA_DIR:-$MCMLSCRATCH/data}"
CKPT_DIR="${CKPT_DIR:-$MCMLSCRATCH/checkpoints}"
ENV_NAME="adl_env"

# --- 2. Environment Setup ---
source ~/.bashrc
mamba activate $ENV_NAME

# --- 3. Execution ---
RUN_NAME="baseline_slurm"

echo "Starting Baseline Training Job: ${SLURM_JOB_NAME} (ID: ${SLURM_JOB_ID})"
echo "Run Name: ${RUN_NAME}"
LOG_TARGET="classifier/runs/${RUN_NAME}/"
CKPT_TARGET="$CKPT_DIR/${RUN_NAME}/"

rm -rf "$LOG_TARGET" # Empty the folder first
mkdir -p "$LOG_TARGET"

rm -rf "$CKPT_TARGET" # Empty the folder first
mkdir -p "$CKPT_TARGET"

if [[ ! -d "classifier" ]]; then
    echo "Error: Must be run from the project root directory (applied_dl/). Current directory is $PWD" >&2
    exit 1
fi

python -m classifier.train \
    --run-name "${RUN_NAME}" \
    --data-dir "$DATA_DIR" \
    --ckpt-dir "$CKPT_DIR"

echo "Training complete. Starting Evaluation..."
python -m classifier.evaluate \
    --checkpoint "$CKPT_DIR/${RUN_NAME}/best.ckpt" \
    --data-dir "$DATA_DIR" \
    --output "$LOG_TARGET/metrics.json"

mv "classifier/runs/slurm_${SLURM_JOB_ID}.out" "$LOG_TARGET" 2>/dev/null || true
mv "classifier/runs/slurm_${SLURM_JOB_ID}.err" "$LOG_TARGET" 2>/dev/null || true

echo "Job completed successfully."
