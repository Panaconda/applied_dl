#!/bin/bash
# Run from ./cheff_peft folder

#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --job-name=sample_cheff
#SBATCH --output=cheff_peft/runs/slurm_%j.out
#SBATCH --error=cheff_peft/runs/slurm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

# Exit on error
set -e

# --- 1. Environment Setup ---
source ~/.bashrc
mamba activate adl_env

# --- 2. Configuration ---
export MCML_ROOT="/dss/mcmlscratch/04/ra58cib2"
export MODEL_BASE="${MCML_ROOT}/pretrained_models"
export OUTPUT_BASE="${MCML_ROOT}/data/synthetic"
export LORA_ADAPTER="runs/finetune_cheff_slurm_123456/lora_adapter"

# --- 3. Execution ---
echo "Starting CheFF Synthetic Generation: ${SLURM_JOB_NAME} (ID: ${SLURM_JOB_ID})"

# Ensure we are in the 'cheff_peft' folder
if [ -d "cheff_peft" ]; then
    echo "Changing directory to cheff_peft/"
    cd cheff_peft
elif [[ "$PWD" != *"/cheff_peft" ]]; then
    echo "Warning: Not in 'cheff_peft' folder and 'cheff_peft' subfolder not found. Imports might fail."
fi

# Determine if we use LoRA or base model
EXTRA_ARGS=""
if [ ! -z "$LORA_ADAPTER" ]; then
    echo "Using LoRA adapter: ${LORA_ADAPTER}"
    EXTRA_ARGS="--lora-adapter ${LORA_ADAPTER}"
else
    echo "Using base model (no LoRA)"
fi

python -m sample_cheff.generate_synthetic \
    --model-path "${MODEL_BASE}/cheff_diff_t2i.pt" \
    --ae-path "${MODEL_BASE}/cheff_autoencoder.pt" \
    --output-dir "${OUTPUT_BASE}" \
    --n 1000 \
    --steps 100 \
    ${EXTRA_ARGS}

echo "Job completed successfully. Output saved to: ${OUTPUT_BASE}"
