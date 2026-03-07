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
echo "Creating Mamba environment $ENV_NAME (skipped if already exists)..."
mamba create --name "$ENV_NAME" python=3.10 -y --no-default-packages 2>/dev/null || \
    echo "Environment '$ENV_NAME' already exists, continuing..."

# --- 2. Install GPU deps (use mamba run to avoid activate issues in scripts) ---
echo "Installing dependencies..."
mamba run -n "$ENV_NAME" pip install --upgrade pip -q
mamba run -n "$ENV_NAME" pip install -r "$ROOT/requirements/gpu.txt" -q

# --- 3. Reinstall taming-transformers from local clone with absolute path ---
# The -e git+ clone lands in ./src/ but the .pth path can be unreliable.
# Reinstalling from the absolute local path guarantees Python finds it.
echo "Fixing taming-transformers install..."
TAMING_SRC="$ROOT/src/taming-transformers"
if [ -d "$TAMING_SRC" ]; then
    mamba run -n "$ENV_NAME" pip install -e "$TAMING_SRC" -q
else
    echo "WARNING: $TAMING_SRC not found — taming may not import correctly."
fi

# --- 4. Override torch for this GPU's CUDA version (cu126) ---
echo "Installing torch cu126..."
mamba run -n "$ENV_NAME" pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu126 -q

# --- 5. .env defaults ---
if [ ! -f "$ROOT/.env" ]; then
    cp "$ROOT/.env.example" "$ROOT/.env"
    echo "Created .env from .env.example — edit it to set your paths."
fi

echo ""
echo "=== Done! ==="
echo "Run:  mamba activate $ENV_NAME"
