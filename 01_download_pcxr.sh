#!/bin/bash

ENV_NAME="adl_env"
PHYSIO_USERNAME=
PHYSIO_PASSWORD=
NUM_WORKERS=4

echo "Starting local PCXR data download..."

# Download 'test' split
python -m prepare_pcxr.download_pcxr \
    --split test \
    --pcxr_dicom_root ./data/pcxr_dicom \
    --username "$PHYSIO_USERNAME" \
    --password "$PHYSIO_PASSWORD" \
    --workers $NUM_WORKERS

# Download 'train' split
python -m prepare_pcxr.download_pcxr \
    --split train \
    --pcxr_dicom_root ./data/pcxr_dicom \
    --username "$PHYSIO_USERNAME" \
    --password "$PHYSIO_PASSWORD" \
    --workers $NUM_WORKERS

echo "Download complete!"
