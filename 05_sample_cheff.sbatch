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
eval "$(mamba shell hook --shell bash)"
mamba activate adl_env

# --- 2. Configuration ---
export MCML_ROOT="/dss/mcmlscratch/02/ra68jip2"
export MODEL_BASE="${MCML_ROOT}/checkpoints"
export OUTPUT_BASE="${MCML_ROOT}/data/synthetic"
# Set FINETUNE_JOB_ID to the SLURM job ID from 04_finetune_cheff before submitting
# e.g.: export FINETUNE_JOB_ID=5515561; sbatch 05_sample_cheff.sbatch
export LORA_ADAPTER="${HOME}/applied_dl/cheff_peft/finetune_cheff/runs/finetune_cheff_slurm_${FINETUNE_JOB_ID}/lora_adapter"

# --- 3. Execution ---
echo "Starting CheFF Synthetic Generation: ${SLURM_JOB_NAME} (ID: ${SLURM_JOB_ID})"

cd ~/applied_dl/cheff_peft
mkdir -p runs

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
