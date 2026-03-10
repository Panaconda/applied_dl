#!/bin/bash

#SBATCH -p mcml-hgx-a100-80x4-mig
#SBATCH --qos=mcml
#SBATCH --job-name=synthetic_filtered_train
#SBATCH --output=classifier/runs/slurm_%j.out
#SBATCH --error=classifier/runs/slurm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

# Exit on error
set -e

# --- 1. Environment Setup ---
source ~/.bashrc
mamba activate adl_env 

# Ensure the classifier package is in the python path
export PYTHONPATH=$PYTHONPATH:.

# --- 2. Configuration via Environment Variables ---
RUN_NAME="synthetic_filtered_slurm"
DATA_DIR="$MCMLSCRATCH/data"
CKPT_DIR="$MCMLSCRATCH/checkpoints"
LOG_TARGET="classifier/runs/${RUN_NAME}/"
CKPT_TARGET="$CKPT_DIR/${RUN_NAME}/"

rm -rf "$LOG_TARGET" # Empty the folder first
mkdir -p "$LOG_TARGET"

rm -rf "$CKPT_TARGET" # Empty the folder first
mkdir -p "$CKPT_TARGET"

# --- 3. Execution ---
echo "Starting Synthetic Filtered Training: ${SLURM_JOB_NAME} (ID: ${SLURM_JOB_ID})"
echo "Run Name: ${RUN_NAME}"

# Execute using the module flag to avoid import errors
python -m classifier.train \
    --run-name "${RUN_NAME}" \
    --data-dir "$DATA_DIR" \
    --ckpt-dir "$CKPT_DIR" \
    --synthetic-classes Pneumonia Bronchiolitis Bronchitis Brocho-pneumonia \
    --filtered

echo "Training complete. Starting Evaluation..."
python -m classifier.evaluate \
    --checkpoint "$CKPT_DIR/${RUN_NAME}/best.ckpt" \
    --data-dir "$DATA_DIR" \
    --output "$LOG_TARGET/metrics.json"

# Move SLURM logs into the specific run directory for better organization
LOG_TARGET="classifier/runs/${RUN_NAME}/"
mkdir -p "$LOG_TARGET"
mv "classifier/runs/slurm_${SLURM_JOB_ID}.out" "$LOG_TARGET" 2>/dev/null || true [cite: 2]
mv "classifier/runs/slurm_${SLURM_JOB_ID}.err" "$LOG_TARGET" 2>/dev/null || true [cite: 2]

echo "Job completed successfully."