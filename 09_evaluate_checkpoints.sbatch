#!/bin/bash
#SBATCH -p lrz-v100x2
#SBATCH --job-name=evaluate_checkpoints
#SBATCH --output=classifier/runs/eval_slurm_%j.out
#SBATCH --error=classifier/runs/eval_slurm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1

# Exit on error
set -e

# --- 1. Environment Setup ---
DATA_DIR="$MCMLSCRATCH/data"
OUTPUT_BASE="classifier/runs/eval_results"
ENV_NAME="adl_env"
CKPT_DIR="$MCMLSCRATCH/checkpoints"

mkdir -p "$OUTPUT_BASE"
source ~/.bashrc
mamba activate $ENV_NAME

echo "Starting Evaluation Job: ${SLURM_JOB_NAME} (ID: ${SLURM_JOB_ID})"

# --- 3. Execution (Manual Calls) ---

# 1. Baseline
echo "Evaluating Baseline..."
python -m classifier.evaluate \
    --checkpoint "$CKPT_DIR/baseline_slurm/best.ckpt" \
    --data-dir "$DATA_DIR" \
    --output "$OUTPUT_BASE/baseline_metrics.json"

# 2. Synthetic Filtered
echo "Evaluating Synthetic Filtered..."
python -m classifier.evaluate \
    --checkpoint "$CKPT_DIR/synthetic_filtered_slurm/best.ckpt" \
    --data-dir "$DATA_DIR" \
    --output "$OUTPUT_BASE/synthetic_filtered_metrics.json"

# 3. Synthetic
echo "Evaluating Synthetic..."
python -m classifier.evaluate \
    --checkpoint "$CKPT_DIR/synthetic_slurm/best.ckpt" \
    --data-dir "$DATA_DIR" \
    --output "$OUTPUT_BASE/synthetic_metrics.json"

echo "--------------------------------------------------------------------------------"
echo "All evaluations completed. Results are in $OUTPUT_BASE"

# Move slurm logs to the results directory
mv "classifier/runs/eval_slurm_${SLURM_JOB_ID}.out" "$OUTPUT_BASE/" 2>/dev/null || true
mv "classifier/runs/eval_slurm_${SLURM_JOB_ID}.err" "$OUTPUT_BASE/" 2>/dev/null || true

echo "Job completed successfully."