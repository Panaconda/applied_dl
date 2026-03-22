#!/bin/bash
# Run from the project root (~/applied_dl)

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

# Load environment variables from .env if it exists
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# --- 1. Environment Setup ---
source ~/.bashrc
eval "$(mamba shell hook --shell bash)"
mamba activate adl_env

# --- 2. Configuration ---
export DATA_DIR="${DATA_DIR:-$MCMLSCRATCH/data}"
export CKPT_DIR="${CKPT_DIR:-$MCMLSCRATCH/checkpoints}"
export MODEL_BASE="${CKPT_DIR}"
export OUTPUT_BASE="${DATA_DIR}/synthetic"
# Set FINETUNE_JOB_ID to use a trained adapter, or leave unset to use the default repo adapter
if [ -n "${FINETUNE_JOB_ID:-}" ]; then
    export LORA_ADAPTER="${HOME}/applied_dl/cheff_peft/runs/finetune_slurm_${FINETUNE_JOB_ID}/lora_adapter"
else
    export LORA_ADAPTER="${HOME}/applied_dl/cheff_peft/runs/finetune_ref/lora_adapter"
fi

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

python -m inference.generate \
    --model-path "${MODEL_BASE}/cheff_diff_t2i.pt" \
    --ae-path "${MODEL_BASE}/cheff_autoencoder.pt" \
    --output-dir "${OUTPUT_BASE}" \
    --n 1000 \
    --steps 100 \
    ${EXTRA_ARGS}

echo "Job completed successfully. Output saved to: ${OUTPUT_BASE}"
