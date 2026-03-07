#!/usr/bin/env bash
# Mirror of the env-setup portion of 03_migrate_to_cluster.sh,
# for a cloud GPU that already has data + checkpoints.
# Usage:  bash setup_cloud_gpu.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="adl_env"

echo "=== Cloud GPU Setup: $ROOT ==="

if ! command -v mamba &>/dev/null; then
    echo "ERROR: mamba not found. Install Miniforge first." >&2
    exit 1
fi

# --- 1. Create mamba env ---
if ! mamba info --envs | grep -q "$ENV_NAME"; then
    echo "Creating Mamba environment $ENV_NAME..."
    mamba create --name "$ENV_NAME" python=3.10 -y
fi

# --- 2. Install GPU deps (use mamba run to avoid activate issues in scripts) ---
echo "Installing dependencies..."
mamba run -n "$ENV_NAME" pip install --upgrade pip -q
mamba run -n "$ENV_NAME" pip install -r "$ROOT/requirements/gpu.txt" -q

# --- 3. .env defaults ---
if [ ! -f "$ROOT/.env" ]; then
    cp "$ROOT/.env.example" "$ROOT/.env"
    echo "Created .env from .env.example — edit it to set your paths."
fi

echo ""
echo "=== Done! ==="
echo "Run:  mamba activate $ENV_NAME"
