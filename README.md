# Setup

## 1. Setup Environment (H100 GPU)

Optimized for H100 GPUs with Python 3.9 and PyTorch 2.x:

```bash
# Create persistent directory structure
mkdir -p ~/cheff-starter/cheff-models
cd ~/cheff-starter

# Clone and setup
git clone https://github.com/Panaconda/applied_dl.git
cd applied_dl
bash setup_h100.sh
```

This will:

- Install Python 3.9
- Create virtual environment
- Install PyTorch 2.x with CUDA 12.1 (H100 compatible)
- Install all dependencies (pytorch-lightning 1.6, transformers, etc.)

### 2. Upload Pre-trained Models

Upload model files to `~/cheff-starter/cheff-models/`:

```bash
# From your local machine
scp cheff_*.pt ubuntu@<gpu-ip>:~/cheff-starter/cheff-models/
```

Required models (~2.4GB total):

- `cheff_autoencoder.pt` (~200MB) - VAE encoder/decoder
- `cheff_diff_t2i.pt` (~900MB) - Text-to-image diffusion
- `cheff_diff_uncond.pt` (~900MB) - Unconditional diffusion
- `cheff_sr_fine.pt` (~400MB) - Super-resolution (256×256 → 1024×1024)

Directory structure:

```
~/cheff-starter/
├── applied_dl/          # Code repository
│   ├── venv/            # Virtual environment
│   └── cheff/
└── cheff-models/        # Model weights (persistent)
    ├── cheff_autoencoder.pt
    ├── cheff_diff_t2i.pt
    ├── cheff_diff_uncond.pt
    └── cheff_sr_fine.pt
```

### 3. Generate Images

The script automatically detects models from `../cheff-models/`:

```bash
source venv/bin/activate
cd cheff
python test_generation.py --prompt "Chest X-ray showing normal lungs"
```

**Example prompts:**

- "Chest X-ray showing bilateral pneumonia"
- "Normal chest radiograph with clear lung fields"
- "Chest X-ray demonstrating cardiomegaly"
- "Frontal chest radiograph showing right lower lobe consolidation"

### Python API

```python
from cheff import CheffLDMT2I, CheffSRModel
import torch
from torchvision.utils import save_image

# Load models
ldm = CheffLDMT2I(
    model_path='trained_models/cheff_diff_t2i.pt',
    ae_path='trained_models/cheff_autoencoder.pt',
    device='cuda'
)

sr = CheffSRModel(
    model_path='trained_models/cheff_sr_fine.pt',
    device='cuda'
)

# Generate 256×256 image
image_256 = ldm.sample(
    sampling_steps=100,
    eta=1.0,
    conditioning="Chest X-ray showing normal lungs"
)

# Upscale to 1024×1024
image_1024 = sr.sample(image_256, method='ddim', sampling_steps=100)
save_image(image_1024, 'output_1024.png')
```

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
