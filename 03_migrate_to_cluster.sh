#!/bin/bash

# Configuration
REMOTE_USER="ra58cib2"
REMOTE_HOST="login.ai.lrz.de"
REPO_URL="https://github.com/Panaconda/applied_dl.git"
ENV_NAME="adl_env"
MCMLSCRATCH_ROOT="/dss/mcmlscratch/04/ra58cib2"

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

    echo "Requesting CPU node for environment installation..."
    
    srun -p lrz-cpu --ntasks=1 --qos cpu --time=00:45:00 bash -c "

        source ~/.bashrc
        
        if ! mamba info --envs | grep -q '$ENV_NAME'; then
            echo 'Creating Mamba environment $ENV_NAME...'
            mamba create --name '$ENV_NAME' python=3.9 -y
        fi

        mamba activate '$ENV_NAME'

        cd ~/applied_dl
        if [ -f 'requirements/gpu.txt' ]; then
            echo 'Installing dependencies...'
            python -m pip install --upgrade pip
            pip install -r requirements/gpu.txt
        else
            echo 'Warning: requirements/gpu.txt not found.'
        fi

        echo 'Downloading Pretrained CheFF Models to cluster...'
        
        echo 'Downloading CheFF pretrained models...'
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

