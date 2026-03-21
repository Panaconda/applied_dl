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

# Load environment variables from .env if it exists
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# --- 1. Environment Setup ---
source ~/.bashrc
eval "$(mamba shell hook --shell bash)"
mamba activate adl_env

# --- 2. Configuration via Environment Variables ---

export RUN_NAME="finetune_cheff_slurm_${SLURM_JOB_ID}"

# Data & Model Paths
export DATA_DIR="${DATA_DIR:-$MCMLSCRATCH/data/pcxr_png}"
export CKPT_DIR="${CKPT_DIR:-$MCMLSCRATCH/checkpoints}"
export TRAIN_IMAGE_DIR="${DATA_DIR}/train"
export CHEFF_T2I_CKPT="${CKPT_DIR}/cheff_diff_t2i.pt"
export CHEFF_AE_CKPT="${CKPT_DIR}/cheff_autoencoder.pt"

# --- 3. Execution ---
echo "Starting CheFF LoRA Fine-Tuning: ${SLURM_JOB_NAME} (ID: ${SLURM_JOB_ID})"
echo "Run Name: ${RUN_NAME}"

cd ~/applied_dl/cheff_peft
mkdir -p runs
python -m finetune_cheff.train

# Move SLURM logs into the run directory
LOG_TARGET="runs/${RUN_NAME}/"
mkdir -p "$LOG_TARGET"
mv "runs/slurm_${SLURM_JOB_ID}.out" "$LOG_TARGET" 2>/dev/null || true
mv "runs/slurm_${SLURM_JOB_ID}.err" "$LOG_TARGET" 2>/dev/null || true

echo "Job completed successfully." 
