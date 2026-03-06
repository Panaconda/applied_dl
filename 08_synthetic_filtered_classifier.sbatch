#!/bin/bash
# Run from project root (applied_dl/)

#SBATCH -p lrz-v100x2
#SBATCH --job-name=synthetic_filtered_classifier
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

# --- 3. Configuration & Overrides ---
RUN_NAME="synthetic_filtered_slurm"
export DATA_DIR="$MCMLSCRATCH/data"
export CKPT_DIR="$MCMLSCRATCH/checkpoints"

# SET THIS: Path to your best baseline classifier checkpoint
# Example: "$CKPT_DIR/baseline_slurm_12345/best.ckpt"
BASELINE_CKPT=""

if [[ -z "$BASELINE_CKPT" ]]; then
    echo "Error: BASELINE_CKPT is not set. Please provide the path to the baseline .ckpt file in the script."
    exit 1
fi

echo "Starting Synthetic Filtered Workflow: ${SLURM_JOB_NAME} (ID: ${SLURM_JOB_ID})"
echo "Run Name: ${RUN_NAME}"
echo "Using Baseline Oracle: $BASELINE_CKPT"

# Ensure we are in the project root
if [[ ! -d "classifier" ]]; then
    echo "Error: Must be run from the project root directory (applied_dl/). Current directory is $PWD" >&2
    exit 1
fi

# --- 4. Filtering Synthetic Data ---
# We loop through each class, index the raw images, and then filter them with the oracle.
CLASSES=("Pneumonia" "Bronchiolitis" "Bronchitis" "Brocho-pneumonia")
SYNTH_BASE="$DATA_DIR/synthetic"

for CLS in "${CLASSES[@]}"; do
    echo "----------------------------------------------------"
    echo "Processing class: $CLS"
    
    # Step A: Build the initial synthetic index (synthetic_labels.csv / synthetic_paths.json)
    # This script scans the folder for .png files and prepares the metadata.
    python -m classifier.synthetic.build_synthetic_index \
        --synthetic-dir "$SYNTH_BASE/$CLS"
        
    # Step B: Filter with the Oracle (produces filtered_labels.csv / filtered_paths.json)
    # This script keeps only images where the baseline model is confident.
    python -m classifier.synthetic.filter_synthetic \
        --ckpt "$BASELINE_CKPT" \
        --index "$SYNTH_BASE/$CLS/synthetic_paths.json" \
        --target "$CLS" \
        --threshold 0.70
done

# --- 5. Training ---
echo "----------------------------------------------------"
echo "Starting Synthetic Filtered Training Job..."

# Launch training with real + filtered synthetic data
# The --filtered flag tells the script to use 'filtered_labels.csv' instead of 'synthetic_labels.csv'
python -m classifier.synthetic.train \
    --run-name "${RUN_NAME}" \
    --filtered

# Move SLURM logs into the run directory.
LOG_TARGET="classifier/runs/${RUN_NAME}/"
mkdir -p "$LOG_TARGET"
mv "classifier/runs/slurm_${SLURM_JOB_ID}.out" "$LOG_TARGET" 2>/dev/null || true
mv "classifier/runs/slurm_${SLURM_JOB_ID}.err" "$LOG_TARGET" 2>/dev/null || true

echo "Job completed successfully."
