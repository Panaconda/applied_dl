#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Setup: $ROOT ($(uname -m)) ==="

if [ "$(uname -m)" = "aarch64" ]; then
    python3 -m venv "$ROOT/adl_env" --system-site-packages
else
    python3 -m venv "$ROOT/adl_env"
fi
source "$ROOT/adl_env/bin/activate"
pip install --upgrade pip -q

# Install torch on x86_64 (already available via system on aarch64)
if [ "$(uname -m)" != "aarch64" ]; then
    pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
        --index-url https://download.pytorch.org/whl/cu121 -q
fi

# All other deps
pip install -r "$ROOT/requirements/base.txt" -q
pip install "numpy==1.26.4" -q

# taming-transformers: pip does not copy the module, do it manually
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
if [ ! -d "$SITE/taming" ]; then
    TMP=$(mktemp -d)
    git clone --depth 1 https://github.com/CompVis/taming-transformers.git "$TMP" -q
    cp -r "$TMP/taming" "$SITE/taming"
    rm -rf "$TMP"
fi

# .env defaults
if [ ! -f "$ROOT/.env" ]; then
    printf "MACHEX_OUTPUT_DIR=$ROOT/machex_dataset/vindr-pcxr\nACCELERATOR=gpu\n" > "$ROOT/.env"
fi

echo ""
echo "=== Done! ==="
echo "source $ROOT/adl_env/bin/activate"
echo "cd $ROOT/cheff_peft && python -m finetune_cheff.train"
