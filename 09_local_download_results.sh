#!/bin/bash

# Configuration
REMOTE_USER="ra58cib2"
REMOTE_HOST="login.ai.lrz.de"
REMOTE_PATH="/dss/mcmlscratch/04/ra58cib2"

echo "Downloading results from LRZ cluster ($REMOTE_HOST)..."

# 1. Download Classifier runs (logs, checkpoints)
echo "Downloading Classifier runs..."
mkdir -p classifier/runs
rsync -avz --progress "$REMOTE_USER@$REMOTE_HOST:~/applied_dl/classifier/runs/" ./classifier/runs/

# 2. Download CheFF PEFT runs (LoRA adapters, logs)
echo "Downloading CheFF PEFT runs..."
mkdir -p cheff_peft/runs
rsync -avz --progress "$REMOTE_USER@$REMOTE_HOST:~/applied_dl/cheff_peft/runs/" ./cheff_peft/runs/

# 3. Download Synthetic Data (if generated on cluster)
echo "Downloading Synthetic Data..."
mkdir -p data/synthetic
rsync -avz --progress "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/data/synthetic/" ./data/synthetic/

echo "Download complete! Your results are now available locally."
