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

# --- 1. Environment Setup ---
source ~/miniforge3/bin/activate .venv

# --- 2. Data Paths ---
MCML_ROOT="/dss/mcmlscratch/04/ra58cib2"
DATASET_BASE="${MCML_ROOT}/data/pcxr_png"
TRAIN_DIR="${DATASET_BASE}/train"
TEST_DIR="${DATASET_BASE}/test"

# --- 3. Execution ---
RUN_NAME="baseline_slurm_${SLURM_JOB_ID}"

echo "Starting Baseline Training Job: ${SLURM_JOB_NAME} (ID: ${SLURM_JOB_ID})"
echo "Run Name: ${RUN_NAME}"

if [[ ! -d "classifier" ]]; then
    echo "Error: Must be run from the project root directory (applied_dl/). Current directory is $PWD" >&2
    exit 1
fi

# Launch training using module path
python -m classifier.baseline.train \
    --train-image-dir "${TRAIN_DIR}" \
    --test-image-dir "${TEST_DIR}" \
    --train-labels-csv "${TRAIN_DIR}/image_labels_train.csv" \
    --test-labels-csv "${TEST_DIR}/image_labels_test.csv" \
    --train-index-json "${TRAIN_DIR}/index.json" \
    --test-index-json "${TEST_DIR}/index.json" \
    --run-name "${RUN_NAME}" \
    --batch-size 32 \
    --num-workers 8 \
    --max-epochs 50 \
    --accelerator gpu \
    --devices 1 \
    --precision 16-mixed

# Move SLURM logs into the run directory.
LOG_TARGET="classifier/runs/${RUN_NAME}/"
mkdir -p "$LOG_TARGET"
mv "classifier/runs/slurm_${SLURM_JOB_ID}.out" "$LOG_TARGET" 2>/dev/null || true
mv "classifier/runs/slurm_${SLURM_JOB_ID}.err" "$LOG_TARGET" 2>/dev/null || true

echo "Job completed successfully."

echo "Job completed successfully."
