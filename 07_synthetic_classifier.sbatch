#!/bin/bash
# Run from project root (applied_dl/)

#SBATCH -p lrz-v100x2
#SBATCH --job-name=synthetic_classifier
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
RUN_NAME="synthetic_slurm"
DATA_DIR="$MCMLSCRATCH/data"
CKPT_DIR="$MCMLSCRATCH/checkpoints"

echo "Step 1: Building synthetic indices..."
CLASSES=("Pneumonia" "Bronchiolitis" "Bronchitis" "Brocho-pneumonia")
SYNTH_BASE="$DATA_DIR/synthetic"

for CLS in "${CLASSES[@]}"; do
    echo "  Indexing $CLS..."
    python -m classifier.synthetic.build_synthetic_index \
        --synthetic-dir "$SYNTH_BASE/$CLS"
done

echo "Step 2: Starting Synthetic Training Job: ${SLURM_JOB_NAME} (ID: ${SLURM_JOB_ID})"
echo "Run Name: ${RUN_NAME}"

# Ensure we are in the project root
if [[ ! -d "classifier" ]]; then
    echo "Error: Must be run from the project root directory (applied_dl/). Current directory is $PWD" >&2
    exit 1
fi

# Step 3: Train
# Step 3: Train
python -m classifier.train \
    --run-name "${RUN_NAME}" \
    --data-dir "$DATA_DIR" \
    --ckpt-dir "$CKPT_DIR" \
    --synthetic-classes Pneumonia Bronchiolitis Bronchitis Brocho-pneumonia


# Move SLURM logs into the run directory.
LOG_TARGET="classifier/runs/${RUN_NAME}/"
mkdir -p "$LOG_TARGET"
mv "classifier/runs/slurm_${SLURM_JOB_ID}.out" "$LOG_TARGET" 2>/dev/null || true
mv "classifier/runs/slurm_${SLURM_JOB_ID}.err" "$LOG_TARGET" 2>/dev/null || true

echo "Job completed successfully."
