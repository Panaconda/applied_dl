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

# --- 8. Install torch: use system CUDA torch via .pth, not PyPI (PyPI aarch64 = CPU-only) ---
# Lambda/cloud aarch64 machines have a CUDA-enabled torch at /usr/lib/python3/dist-packages.
# We point the conda env at it via a .pth file rather than installing a CPU wheel from PyPI.
echo "Linking system CUDA torch into env..."
SITE_PACKAGES="$ROOT/.local/share/mamba/envs/$ENV_NAME/lib/python3.10/site-packages"
SITE_PACKAGES="/home/$(whoami)/.local/share/mamba/envs/$ENV_NAME/lib/python3.10/site-packages"
SYSTEM_TORCH_DIR=""
for candidate in /usr/lib/python3/dist-packages /usr/local/lib/python3.10/dist-packages; do
    if python3 -c "import sys; sys.path.insert(0,'$candidate'); import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True; then
        SYSTEM_TORCH_DIR="$candidate"
        break
    fi
done

if [ -n "$SYSTEM_TORCH_DIR" ]; then
    # Remove any CPU torch that base.txt may have pulled in transitively
    mamba run -n "$ENV_NAME" pip uninstall torch torchvision -y -q 2>/dev/null || true
    echo "$SYSTEM_TORCH_DIR" > "$SITE_PACKAGES/system_torch.pth"
    echo "  Linked system torch from $SYSTEM_TORCH_DIR"
else
    echo "  WARNING: Could not find system CUDA torch. Falling back to PyPI (may be CPU-only)."
    mamba run -n "$ENV_NAME" pip install torch torchvision -q
fi

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
