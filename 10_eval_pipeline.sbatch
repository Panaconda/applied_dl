#!/bin/bash
# Evaluation pipeline: sample base + LoRA, then compute UMAP and FDD.
# Outputs: ~/applied_dl/results/finetune_viz/umap.png
#          ~/applied_dl/results/finetune_viz/fdd.csv
#
# Optional env overrides (set before submitting):
#   FINETUNE_JOB_ID  – use a trained LoRA adapter from a specific finetune run
#   N_PER_CLASS      – images per class per model (default: 100)

#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --job-name=eval_pipeline
#SBATCH --output=%x_slurm_%j.out
#SBATCH --error=%x_slurm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

set -e

# --- 1. Load Environment Variables ---
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# Fallbacks if .env not present
DATA_DIR="${DATA_DIR:-$MCMLSCRATCH/data/pcxr_png}"
CKPT_DIR="${CKPT_DIR:-$MCMLSCRATCH/checkpoints}"

# --- 2. Environment Setup ----------------------------------------------------------
source ~/.bashrc
eval "$(mamba shell hook --shell bash)"
mamba activate adl_env

# --- 3. Configuration --------------------------------------------------------
# Set SKIP_SAMPLING=1 to skip generation and jump straight to UMAP + FDD
# (useful when synthetic dirs already exist from a previous run)
SKIP_SAMPLING=${SKIP_SAMPLING:-0}

N=${N_PER_CLASS:-100}

# Synthetic output dirs (on scratch for speed)
SYNTH_BASE="$MCMLSCRATCH/data/finetune_viz/synthetic_base"
SYNTH_LORA="$MCMLSCRATCH/data/finetune_viz/synthetic_lora"

# Visualisation outputs go to the home project so they survive scratch purges
OUTPUT_DIR="${HOME}/applied_dl/results/finetune_viz"
mkdir -p "${OUTPUT_DIR}"

# LoRA adapter: use a trained run if FINETUNE_JOB_ID is set, else repo default
if [ -n "${FINETUNE_JOB_ID:-}" ]; then
    LORA_ADAPTER="${HOME}/applied_dl/cheff_peft/runs/finetune_slurm_${FINETUNE_JOB_ID}/lora_adapter"
else
    LORA_ADAPTER="${HOME}/applied_dl/cheff_peft/runs/finetune_ref/lora_adapter"
fi

echo "============================================================"
echo "  eval_pipeline  (job ${SLURM_JOB_ID})"
echo "  n per class : ${N}"
echo "  LoRA adapter: ${LORA_ADAPTER}"
echo "  outputs     : ${OUTPUT_DIR}"
echo "============================================================"

# --- 4. Sample base model (no LoRA) ------------------------------------------
if [ "${SKIP_SAMPLING}" = "1" ]; then
    echo "SKIP_SAMPLING=1 — skipping generation steps"
else
    echo ""
    echo "=== Step 1/4  Sample base model ==="
    cd "${HOME}/applied_dl/cheff_peft"
    python -m inference.generate \
        --model-path "${CKPT_DIR}/cheff_diff_t2i.pt" \
        --ae-path    "${CKPT_DIR}/cheff_autoencoder.pt" \
        --output-dir "${SYNTH_BASE}" \
        --n          "${N}" \
        --steps      100

    # --- 5. Sample LoRA model ------------------------------------------------
    echo ""
    echo "=== Step 2/4  Sample LoRA model ==="
    python -m inference.generate \
        --model-path   "${CKPT_DIR}/cheff_diff_t2i.pt" \
        --ae-path      "${CKPT_DIR}/cheff_autoencoder.pt" \
        --lora-adapter "${LORA_ADAPTER}" \
        --output-dir   "${SYNTH_LORA}" \
        --n            "${N}" \
        --steps        100
fi

# --- 6. UMAP -----------------------------------------------------------------
echo ""
echo "=== Step 3/4  UMAP ==="
cd "${HOME}/applied_dl/cheff_peft"
python -m eval.umap \
    --lora-dir  "${SYNTH_LORA}" \
    --base-dir  "${SYNTH_BASE}" \
    --per-class "${N}" \
    --output    "${OUTPUT_DIR}/umap.png"

# --- 7. FDD ------------------------------------------------------------------
echo ""
echo "=== Step 4/4  FDD ==="
python -m eval.fdd \
    --lora-dir  "${SYNTH_LORA}" \
    --base-dir  "${SYNTH_BASE}" \
    --per-class "${N}" \
    --output    "${OUTPUT_DIR}/fdd.csv"

echo ""
echo "Done. Results in ${OUTPUT_DIR}/"
echo "  umap.png"
echo "  fdd.csv"
