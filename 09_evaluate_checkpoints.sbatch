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
ENV_NAME="adl_env"
source ~/.bashrc
mamba activate $ENV_NAME

export PYTHONPATH=$PYTHONPATH:.

# --- 2. Variables ---
DATA_DIR="$MCMLSCRATCH/data"
OUTPUT_BASE="classifier/runs/eval_results"
mkdir -p "$OUTPUT_BASE"

# Checkpoints to evaluate
declare -A checkpoints
checkpoints["baseline"]="checkpoints/baseline_slurm/baseline_slurm/best.ckpt"
checkpoints["synthetic_filtered"]="checkpoints/synthetic_filtered_slurm/synthetic_filtered_slurm/best.ckpt"
checkpoints["synthetic"]="checkpoints/synthetic_slurm/best.ckpt"

echo "Starting Evaluation Job: ${SLURM_JOB_NAME} (ID: ${SLURM_JOB_ID})"

# --- 3. Execution ---
for name in "${!checkpoints[@]}"; do
    ckpt="${checkpoints[$name]}"
    echo "--------------------------------------------------------------------------------"
    echo "Evaluating model: $name"
    echo "Checkpoint: $ckpt"
    
    if [ ! -f "$ckpt" ]; then
        echo "Warning: Checkpoint $ckpt not found, skipping."
        continue
    fi

    # Run evaluation
    python classifier/evaluate.py \
        --checkpoint "$ckpt" \
        --data-dir "$DATA_DIR" \
        --output "$OUTPUT_BASE/${name}_metrics.json"
        
    echo "Evaluation for $name completed."
done

echo "--------------------------------------------------------------------------------"
echo "All evaluations completed. Results are in $OUTPUT_BASE"

# Move slurm logs to the results directory
mv "classifier/runs/eval_slurm_${SLURM_JOB_ID}.out" "$OUTPUT_BASE/" 2>/dev/null || true
mv "classifier/runs/eval_slurm_${SLURM_JOB_ID}.err" "$OUTPUT_BASE/" 2>/dev/null || true

echo "Job completed successfully."
