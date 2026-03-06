#!/bin/bash

# Configuration
REMOTE_USER="ra58cib2"
REMOTE_HOST="login.ai.lrz.de"
REMOTE_PATH="/dss/mcmlscratch/04/ra58cib2"

echo "Downloading results from LRZ cluster ($REMOTE_HOST)..."

# 1. Download Classifier Logs & Slurm Outputs (from Home)
echo "Downloading Classifier logs..."
mkdir -p classifier/runs
rsync -avz --progress "$REMOTE_USER@$REMOTE_HOST:~/applied_dl/classifier/runs/" ./classifier/runs/

# 2. Download Classifier Checkpoints (.ckpt weights from Scratch)
# Excluding large base model checkpoints to save space/time
echo "Downloading Classifier checkpoints..."
mkdir -p checkpoints
rsync -avz --progress \
    --exclude="cheff_diff_t2i.pt" \
    --exclude="cheff_autoencoder.pt" \
    --exclude="cheff_sr_fine.pt" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/checkpoints/" ./checkpoints/

# 3. Download CheFF PEFT runs (LoRA adapters, logs)
echo "Downloading CheFF PEFT runs..."
mkdir -p cheff_peft/runs
rsync -avz --progress "$REMOTE_USER@$REMOTE_HOST:~/applied_dl/cheff_peft/runs/" ./cheff_peft/runs/

# 4. Download Synthetic Data (from Scratch)
echo "Downloading Synthetic Data..."
mkdir -p data/synthetic
rsync -avz --progress "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/data/synthetic/" ./data/synthetic/

echo "Download complete! Your results are now available locally."
