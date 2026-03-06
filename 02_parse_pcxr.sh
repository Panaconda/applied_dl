#!/bin/bash

ENV_NAME="adl_env"
NUM_WORKERS=4

echo "Validating downloaded local data..."

if python -m prepare_pcxr.validate_download \
    --dicom_dir ./data/pcxr_dicom; then
    
    echo "Validation successful. Continuing with parsing the PCXR data (DICOM -> PNG)..."
    
    mkdir -p data/pcxr_png
    
    python -m prepare_pcxr.parse_pcxr \
        --pcxr_dicom_root ./data/pcxr_dicom \
        --pcxr_png_root ./data/pcxr_png \
        --num_workers $NUM_WORKERS
    
    echo "Data preparation complete! Processed images are in ./data/pcxr_png/"
else
    echo "ERROR: Validation failed. Some files are missing or corrupt."
    echo "Please repeat step 02_local_download.sh to redownload the missing files."
    exit 1
fi
