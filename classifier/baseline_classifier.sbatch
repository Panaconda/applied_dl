#!/bin/bash
# Run from the project root

#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --job-name=baseline_classifier
#SBATCH --output=/classifier/runs/slurm_%j.out
#SBATCH --error=/classifier/runs/slurm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1

# Exit on error
set -e

# --- 1. Environment Setup ---
# Use the mamba/conda instructions from the MCML guide
source ~/miniforge3/bin/activate adl_cheff

# --- 2. Data Paths ---
MCML_ROOT="/dss/mcmlscratch/04/ra58cib2"
DATASET_BASE="${MCML_ROOT}/data/machex_dataset/vindr-pcxr"
TRAIN_DIR="${DATASET_BASE}/train"
TEST_DIR="${DATASET_BASE}/test"

# --- 3. Execution ---
RUN_NAME="baseline_slurm_${SLURM_JOB_ID}"

echo "Starting Baseline Training Job: ${SLURM_JOB_NAME} (ID: ${SLURM_JOB_ID})"
echo "Run Name: ${RUN_NAME}"

# Ensure we are in the 'classifier' folder so that 'baseline' and 'core' modules are found.
if [ -d "classifier" ]; then
    echo "Changing directory to classifier/"
    cd classifier
elif [[ "$PWD" != *"/classifier" ]]; then
    echo "Warning: Not in 'classifier' folder and 'classifier' subfolder not found. Imports might fail."
fi

# Launch training
python -m baseline.train \
    --train-image-dir "${TRAIN_DIR}" \
    --test-image-dir "${TEST_DIR}" \
    --train-labels-csv "${TRAIN_DIR}/00/image_labels_train.csv" \
    --test-labels-csv "${TEST_DIR}/00/image_labels_test.csv" \
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
LOG_TARGET="runs/${RUN_NAME}/"
mkdir -p "$LOG_TARGET"
mv "runs/slurm_${SLURM_JOB_ID}.out" "$LOG_TARGET" 2>/dev/null || true
mv "runs/slurm_${SLURM_JOB_ID}.err" "$LOG_TARGET" 2>/dev/null || true

echo "Job completed successfully."
