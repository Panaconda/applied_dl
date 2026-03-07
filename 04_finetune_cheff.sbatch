#!/bin/bash
# Submit from the project root:  sbatch 04_finetune_cheff.sbatch

#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --job-name=finetune_cheff
#SBATCH --output=cheff_peft/finetune_cheff/runs/slurm_%j.out
#SBATCH --error=cheff_peft/finetune_cheff/runs/slurm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- 1. Environment Setup ---
source ~/.bashrc
mamba activate adl_env

# --- 2. Run name (config reads everything else from .env) ---
export RUN_NAME="finetune_cheff_slurm_${SLURM_JOB_ID}"

# --- 3. Ensure log dir exists before SLURM writes to it ---
mkdir -p "$PROJECT_ROOT/cheff_peft/finetune_cheff/runs"

# --- 4. Train ---
echo "Job ${SLURM_JOB_ID} — ${SLURM_JOB_NAME} — run: ${RUN_NAME}"
cd "$PROJECT_ROOT/cheff_peft"
python -m finetune_cheff.train

# --- 5. Move SLURM logs into the run directory ---
LOG_TARGET="finetune_cheff/runs/${RUN_NAME}/"
mkdir -p "$LOG_TARGET"
mv "finetune_cheff/runs/slurm_${SLURM_JOB_ID}.out" "$LOG_TARGET" 2>/dev/null || true
mv "finetune_cheff/runs/slurm_${SLURM_JOB_ID}.err" "$LOG_TARGET" 2>/dev/null || true

echo "Job completed successfully."
