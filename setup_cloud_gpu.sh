#!/usr/bin/env bash
# Mirror of the env-setup portion of 03_migrate_to_cluster.sh,
# for a cloud GPU that already has data + checkpoints.
# Usage:  bash setup_cloud_gpu.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="adl_env"

echo "=== Cloud GPU Setup: $ROOT ==="

# --- 1. Create mamba env ---
source ~/.bashrc

if ! mamba info --envs | grep -q "$ENV_NAME"; then
    echo "Creating Mamba environment $ENV_NAME..."
    mamba create --name "$ENV_NAME" python=3.10 -y
fi

mamba activate "$ENV_NAME"

# --- 2. Install GPU deps ---
python -m pip install --upgrade pip -q
pip install -r "$ROOT/requirements/gpu.txt" -q

# --- 3. .env defaults ---
if [ ! -f "$ROOT/.env" ]; then
    cat > "$ROOT/.env" <<DOT
MACHEX_OUTPUT_DIR=$ROOT/machex_dataset/vindr-pcxr
ACCELERATOR=gpu
DOT
    echo "Created default .env"
fi

echo ""
echo "=== Done! ==="
echo "mamba activate $ENV_NAME"
