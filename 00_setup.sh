#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Setup: $ROOT ==="

# Venv inheriting system torch (Lambda pre-installs a CUDA-capable torch)
python3 -m venv "$ROOT/adl_env" --system-site-packages
source "$ROOT/adl_env/bin/activate"
pip install --upgrade pip -q

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
