#!/bin/bash
set -e

echo "🚀 Cheff Setup (H100 Optimized)"

# Install Python 3.9 if needed
if ! command -v python3.9 &> /dev/null; then
    echo "Installing Python 3.9..."
    sudo apt-get update -qq
    sudo apt-get install -y python3.9 python3.9-venv python3.9-dev
fi

echo "Python 3.9: $(python3.9 --version)"

# Create venv
echo "Creating virtual environment..."
[ -d "venv" ] && rm -rf venv
python3.9 -m venv venv
source venv/bin/activate

# Downgrade pip for compatibility with pytorch-lightning==1.6
# echo "Installing pip<24.1..."
# pip install -q "pip<24.1"

# Install PyTorch 2.x for H100 support (CUDA 12.1)
echo "Installing PyTorch 2.x with H100 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies (requirements.txt now excludes torch)
echo "Installing remaining dependencies..."
pip install -r requirements.txt

# Test
echo ""
echo "Testing installation..."
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import cheff; print('Cheff: OK')"

echo ""
echo "✅ Setup complete!"
echo "Models location: ../cheff-models/"