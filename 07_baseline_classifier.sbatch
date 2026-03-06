#!/bin/bash
# Run from project root (applied_dl/)

#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
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

# --- 1. Variables ---
ENV_NAME="adl_env"

# --- 2. Environment Setup ---
source ~/.bashrc
mamba activate $ENV_NAME

# --- 3. Execution ---
RUN_NAME="baseline_slurm_${SLURM_JOB_ID}"

echo "Starting Baseline Training Job: ${SLURM_JOB_NAME} (ID: ${SLURM_JOB_ID})"
echo "Run Name: ${RUN_NAME}"

if [[ ! -d "classifier" ]]; then
    echo "Error: Must be run from the project root directory (applied_dl/). Current directory is $PWD" >&2
    exit 1
fi

python -m classifier.baseline.train \
    --train-image-dir "$MCMLSCRATCH/data/pcxr_png/train" \
    --test-image-dir "$MCMLSCRATCH/data/pcxr_png/test" \
    --train-labels-csv "$MCMLSCRATCH/data/pcxr_png/train/image_labels_train.csv" \
    --test-labels-csv "$MCMLSCRATCH/data/pcxr_png/test/image_labels_test.csv" \
    --train-index-json "$MCMLSCRATCH/data/pcxr_png/train/index.json" \
    --test-index-json "$MCMLSCRATCH/data/pcxr_png/test/index.json" \
    --run-name "${RUN_NAME}" \

LOG_TARGET="classifier/runs/${RUN_NAME}/"
mkdir -p "$LOG_TARGET"
mv "classifier/runs/slurm_${SLURM_JOB_ID}.out" "$LOG_TARGET" 2>/dev/null || true
mv "classifier/runs/slurm_${SLURM_JOB_ID}.err" "$LOG_TARGET" 2>/dev/null || true

echo "Job completed successfully."

echo "Job completed successfully."
