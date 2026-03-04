# VinDr-PCXR Dataset Preparation and Modeling

## Overview

This repository contains the code for preparing the VinDr-PCXR dataset and performing downstream tasks like finetuning and classification.

The repository structure:

- `cheff_peft/`: Logic for finetuning the Cheff model using Parameter-Efficient Fine-Tuning (PEFT).
- `classifier/`: Implementation of baseline and synthetic-enhanced classifiers.
- `data/`: Directory for storing DICOM, PNG, and synthetic data (should be created).
- `prepare_pcxr/`: Scripts for downloading and parsing the VinDr-PCXR dataset.
- `visuals/`: Utility scripts for generating visualizations and plots.

## Setup Environment

```bash
# Clone and setup
git clone https://github.com/Panaconda/applied_dl.git
cd applied_dl

# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Download Pediatric Chest X-Rays (PCXR)

"The LICENSEE will not share access to PhysioNet restricted data with anyone else."

Download the dataset by running the following script. You will need a PhysioNet account and must have signed the Data Use Agreement (DUA) for the [VinDr-PCXR dataset](https://physionet.org/content/vindr-pcxr/1.0.0/).

You can provide your credentials via command line or by creating a `prepare_pcxr/.env` file (see `prepare_pcxr/.env_example`).

```bash
# Download test set (~5GB)
python prepare_pcxr/download_pcxr.py --split test --username "your_email@domain.com" --password "your_password"

# Download train set (~28GB)
python prepare_pcxr/download_pcxr.py --split train --username "your_email@domain.com" --password "your_password"
```

## Parse PCXR to Prepare Dataset

The `parse_pcxr.py` script converts the downloaded DICOM files into PNG images, resizes them to 1024x1024, and organizes them into the `data/pcxr_png` directory. Moreover, creates the synthetic reports metadata and copies annotation and label CSV files to the target folders.

```bash
python prepare_pcxr/parse_pcxr.py
```
