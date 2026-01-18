#!/bin/bash
set -e

echo "🚀 Cheff Setup"

# Install Python 3.9 if needed
command -v python3.9 &> /dev/null || (sudo apt-get update -qq && sudo apt-get install -y python3.9 python3.9-venv python3.9-dev)

# Create venv
[ -d "venv" ] && rm -rf venv
python3.9 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q

# Install PyTorch with CUDA
if nvidia-smi &> /dev/null; then
    CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1)
    [ "$CUDA_VER" = "12" ] && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    pip install torch torchvision torchaudio
fi

# Install remaining dependencies
pip install -r requirements.txt

# Create model directory
mkdir -p cheff/trained_models

# Test
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import cheff; print('Cheff: OK')"

echo ""
echo "✅ Setup complete!"