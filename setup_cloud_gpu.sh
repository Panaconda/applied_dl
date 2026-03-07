#!/usr/bin/env bash
# Full from-scratch cloud GPU setup. Tears everything down first.
# Usage:  bash setup_cloud_gpu.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="adl_env"

echo "=== Cloud GPU Setup (clean): $ROOT ==="

if ! command -v mamba &>/dev/null; then
    echo "ERROR: mamba not found. Install Miniforge first." >&2
    exit 1
fi

# --- 1. Nuke existing env ---
echo "Removing existing '$ENV_NAME' env..."
mamba env remove -n "$ENV_NAME" -y 2>/dev/null || true

# --- 2. Nuke cloned deps so base.txt re-clones them fresh ---
echo "Removing src/..."
rm -rf "$ROOT/src"

# --- 3. Nuke stale .env ---
echo "Removing .env..."
rm -f "$ROOT/.env"

# --- 4. Remove stale PYTHONPATH lines from ~/.bashrc ---
sed -i '/taming-transformers/d' ~/.bashrc

# --- 5. Create env ---
echo "Creating '$ENV_NAME' (Python 3.10)..."
mamba create --name "$ENV_NAME" python=3.10 -y
mamba run -n "$ENV_NAME" pip install --upgrade pip -q

# --- 6. Install base deps (clones taming + CLIP into src/) ---
echo "Installing requirements/base.txt..."
mamba run -n "$ENV_NAME" pip install -r "$ROOT/requirements/base.txt" -q

# --- 7. Reinstall taming from absolute local path so .pth is correct ---
echo "Reinstalling taming-transformers from local clone..."
mamba run -n "$ENV_NAME" pip install -e "$ROOT/src/taming-transformers" -q

# --- 8. Install torch from PyPI (has aarch64 CUDA builds; whl server does not) ---
echo "Installing torch from PyPI..."
mamba run -n "$ENV_NAME" pip install torch torchvision -q

# --- 9. Verify CUDA ---
echo "Verifying torch CUDA..."
mamba run -n "$ENV_NAME" python -c "
import torch
print(f'  torch {torch.__version__}  |  cuda={torch.cuda.is_available()}')
if not torch.cuda.is_available():
    print('  WARNING: CUDA not available — check nvidia-smi')
"

# --- 10. PYTHONPATH for taming (belt-and-suspenders over the .pth) ---
echo "export PYTHONPATH=$ROOT/src/taming-transformers" >> ~/.bashrc
echo "Added PYTHONPATH to ~/.bashrc"

# --- 11. Write .env ---
echo "ACCELERATOR=gpu" > "$ROOT/.env"
echo "Created .env"

echo ""
echo "=== Done! ==="
echo "Now run:"
echo "  source ~/.bashrc && mamba activate $ENV_NAME"
echo "  cd $ROOT/cheff_peft && python -m finetune_cheff.train"
