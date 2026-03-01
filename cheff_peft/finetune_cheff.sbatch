#!/bin/bash
# Run from ./cheff_peft folder

#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --job-name=finetune_cheff
#SBATCH --output=cheff_peft/runs/slurm_%j.out
#SBATCH --error=cheff_peft/runs/slurm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

# Exit on error
set -e

# --- 1. Environment Setup ---
# Use the mamba/conda instructions from the MCML guide
source ~/miniforge3/bin/activate adl_cheff

# --- 2. Configuration via Environment Variables ---
# Pydantic (ftcfg) will pick these up automatically.
# Field names must match the class fields exactly.

export RUN_NAME="finetune_cheff_slurm_${SLURM_JOB_ID}"

# Data & Model Paths (Adjust these to your actual paths on the cluster)
export MCML_ROOT="/dss/mcmlscratch/04/ra58cib2"
export DATASET_BASE="${MCML_ROOT}/data/machex_dataset/vindr-pcxr"
export MODEL_BASE="${MCML_ROOT}/pretrained_models"
export MACHEX_OUT="${MCML_ROOT}/data/machex_dataset/vindr-pcxr/train/"

# Pydantic Config Overrides
export VINDR_PCXR_TRAIN_INDEX="${DATASET_BASE}/train/index.json"
export TRAIN_IMAGE_DIR="${DATASET_BASE}/train"
export MACHEX_OUTPUT_DIR="${MACHEX_OUT}"
export CHEFF_T2I_CKPT="${MODEL_BASE}/cheff_diff_t2i.pt"
export CHEFF_AE_CKPT="${MODEL_BASE}/cheff_autoencoder.pt"

# Hyperparameters
export CHEFF_BATCH_SIZE=8
export CHEFF_LEARNING_RATE=5e-5
export CHEFF_MAX_EPOCHS=15
export LORA_RANK=16

# --- 3. Execution ---
echo "Starting CheFF LoRA Fine-Tuning: ${SLURM_JOB_NAME} (ID: ${SLURM_JOB_ID})"
echo "Run Name: ${RUN_NAME}"

# Ensure we are in the 'cheff_peft' folder for correct module discovery
if [ -d "cheff_peft" ]; then
    echo "Changing directory to cheff_peft/"
    cd cheff_peft
elif [[ "$PWD" != *"/cheff_peft" ]]; then
    echo "Warning: Not in 'cheff_peft' folder and 'cheff_peft' subfolder not found. Imports might fail."
fi

# Step 3a: Prepare Data (if not already done)
echo "Verifying/Preparing MaCheX data..."
python -m finetune_cheff.prepare_data

# Step 3b: Launch Training
# No CLI arguments needed; config is loaded from the 'export' statements above.
python -m finetune_cheff.train

# Move SLURM logs into the run directory
LOG_TARGET="runs/${RUN_NAME}/"
mkdir -p "$LOG_TARGET"
mv "runs/slurm_${SLURM_JOB_ID}.out" "$LOG_TARGET" 2>/dev/null || true
mv "runs/slurm_${SLURM_JOB_ID}.err" "$LOG_TARGET" 2>/dev/null || true

echo "Job completed successfully."
