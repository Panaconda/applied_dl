#!/bin/bash
set -euo pipefail

# Load config from .env (same file Python uses)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a; source "$SCRIPT_DIR/.env"; set +a
else
    echo "ERROR: .env not found. Copy .env.example to .env and fill in your values." >&2
    exit 1
fi

# Configuration
REMOTE_USER="${LRZ_USER:?Set LRZ_USER in .env}"
REMOTE_HOST="${LRZ_HOST:-login.ai.lrz.de}"
MCMLSCRATCH_ROOT="${LRZ_SCRATCH:?Set LRZ_SCRATCH in .env}"
REPO_URL="https://github.com/Panaconda/applied_dl.git"
ENV_NAME="adl_env"

#1. Set up environment and download pretrained models on cluster

echo "Connecting to $REMOTE_HOST to set up Mamba and environment..."

ssh -t "$REMOTE_USER@$REMOTE_HOST" \
    ENV_NAME="$ENV_NAME" \
    REPO_URL="$REPO_URL" \
    MCMLSCRATCH_ROOT="$MCMLSCRATCH_ROOT" \
    bash << 'EOF'

    echo "Setting up repository in $HOME..."
    cd "$HOME" || exit

    if [ ! -d "applied_dl" ]; then
        git clone "$REPO_URL" applied_dl
    fi
    cd applied_dl || exit
    git pull
    git checkout experiment-01

    echo "Requesting CPU node for environment installation..."
    
    srun -p lrz-cpu --ntasks=1 --qos cpu --time=00:45:00 bash -c "

        source ~/.bashrc
        
        if ! mamba info --envs | grep -q '$ENV_NAME'; then
            echo 'Creating Mamba environment $ENV_NAME...'
            mamba create --name '$ENV_NAME' python=3.10 -y
        fi

        mamba activate '$ENV_NAME'

        cd ~/applied_dl

        echo 'Installing dependencies...'
        python -m pip install --upgrade pip
        pip install -r requirements/gpu.txt

        # Reinstall taming from local clone (editable .pth can be unreliable)
        pip install -e src/taming-transformers

        # PYTHONPATH for taming (belt-and-suspenders)
        PYTHONPATH_LINE='export PYTHONPATH=\$HOME/applied_dl/src/taming-transformers'
        grep -qF 'taming-transformers' ~/.bashrc || echo \"\$PYTHONPATH_LINE\" >> ~/.bashrc

        echo 'Downloading Pretrained CheFF Models to cluster...'
        python -m prepare_pcxr.download_cheff \
            --checkpoint_dir $MCMLSCRATCH_ROOT/checkpoints

    echo 'Download complete!'
    "
EOF

# 2. Sync data to cluster

# Create target directories on the cluster
echo "Creating root folder for png-data in $MCMLSCRATCH_ROOT/data/pcxr_png/"
ssh "$REMOTE_USER@$REMOTE_HOST" "mkdir -p $MCMLSCRATCH_ROOT/data $MCMLSCRATCH_ROOT/data/pcxr_png/" 

# Sync processed PNG data
echo "Syncing processed PNG data (pcxr_png)..."
scp -r ./data/pcxr_png "$REMOTE_USER@$REMOTE_HOST:$MCMLSCRATCH_ROOT/data"

echo "Sync complete! You can now run the .sbatch scripts on the cluster."

