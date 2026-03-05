#!/bin/bash

# --- Configuration ---
LOCAL_PATH="./data/pcxr_png/"
REMOTE_USER="ra58cib2"
REMOTE_HOST="login.ai.lrz.de"
# We use the $USER variable trick we discussed earlier for the destination
REMOTE_DEST="/dss/mcmlscratch/04/${REMOTE_USER}/data/pcxr_png/"

echo "Syncing data to MCML Scratch..."

# rsync is the gold standard for this:
# -a: archive mode (preserves permissions)
# -v: verbose (shows what's happening)
# -z: compress (faster transfer)
# -P: shows progress bar and allows resuming
rsync -avzP "$LOCAL_PATH" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DEST}"

echo "Upload complete!"