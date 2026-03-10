#!/bin/bash

#SBATCH -p lrz-v100x2
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
export RUN_NAME="synthetic_filtered_slurm"
export DATA_DIR="/dss/mcmlscratch/04/ra58cib2/data"
export CKPT_DIR="/dss/mcmlscratch/04/ra58cib2/checkpoints"

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

# Move SLURM logs into the specific run directory for better organization
LOG_TARGET="classifier/runs/${RUN_NAME}/"
mkdir -p "$LOG_TARGET"
mv "classifier/runs/slurm_${SLURM_JOB_ID}.out" "$LOG_TARGET" 2>/dev/null || true [cite: 2]
mv "classifier/runs/slurm_${SLURM_JOB_ID}.err" "$LOG_TARGET" 2>/dev/null || true [cite: 2]

echo "Job completed successfully."