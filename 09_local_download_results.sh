#!/bin/bash

# Configuration
REMOTE_USER="ra58cib2"
REMOTE_HOST="login.ai.lrz.de"
REMOTE_PATH="/dss/mcmlscratch/04/ra58cib2"

echo "Downloading results from LRZ cluster ($REMOTE_HOST) using SCP..."

# 1. Download Classifier Logs & Slurm Outputs (from Home)
echo "Downloading Classifier logs..."
mkdir -p ./classifier/runs/
scp -r "$REMOTE_USER@$REMOTE_HOST:~/applied_dl/classifier/runs/*" ./classifier/runs/

# 2. Download Classifier Checkpoints (Excluding large files)
echo "Downloading Classifier checkpoints (skipping large base models)..."
mkdir -p ./checkpoints/baseline_slurm
mkdir -p ./checkpoints/synthetic_slurm
mkdir -p ./checkpoints/synthetic_filtered_slurm

scp -r "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/checkpoints/baseline_slurm" ./checkpoints/baseline_slurm
scp -r "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/checkpoints/synthetic_slurm" ./checkpoints/synthetic_slurm
scp -r "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/checkpoints/synthetic_filtered_slurm" ./checkpoints/synthetic_filtered_slurm

# 3. Download CheFF PEFT runs (LoRA adapters, logs)
echo "Downloading CheFF PEFT runs..."
mkdir -p ./cheff_peft/runs/
scp -r "$REMOTE_USER@$REMOTE_HOST:~/applied_dl/cheff_peft/runs/*" ./cheff_peft/runs/

# 4. Download Synthetic Data (from Scratch)
echo "Downloading Synthetic Data..."
mkdir -p ./data/synthetic/
scp -r "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/data/synthetic/*" ./data/synthetic/

echo "Download complete! Your results are now available locally."

baseline_slurm
synthetic_slurm
synthetic_filtered_slurm