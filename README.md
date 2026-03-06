# Diffusion-Based Data Augmentation:Parameter-Efficient Fine-Tuning of Cheff for Rare Pathologies

## Overview

This repository provides a pipeline for adapting Cheff to pediatric chest x-rays using the VinDr-PCXR dataset. The project evaluates how Parameter-Efficient Fine-Tuning (PEFT) via LoRA can alleviate data scarcity in the PCXR domain for pathology classification.

Repository Structure:

- `cheff_peft/`: Logic for finetuning the Cheff model using Parameter-Efficient Fine-Tuning (PEFT).
- `classifier/`: Fine-tuning logic for the torchxrayvision pathology classifiers.
- `data/`: Directory for storing DICOM, PNG, and synthetic samples.
- `prepare_pcxr/`: Scripts for downloading and parsing the VinDr-PCXR dataset.
- `visuals/`: Utility scripts for generating visualizations and plots.

## 01_Local Setup

The initial data preprocessing is recommended to be done locally. The reason is to not occupy cluster resources for the lengthy dataset download.

```bash
git clone "https://github.com/Panaconda/applied_dl.git" applied_dl
cd applied_dl

python -m venv "adl_env"
source "adl_env/Scripts/activate"

python -m pip install --upgrade pip
pip install -r requirements/local.txt
```

### A. Download PCXR

The VinDr-PCXR download is rate-limited. Setting NUM_WORKERS above 4 in 01_download_pcxr.sh may result in a temporary IP ban from the hosting server. Given these safety limits, the 32GB transfer is time-intensive.

```bash
bash 01_download_pcxr.sh
```

### B. Validate and parse DICOMs to PNGs

```bash
bash 02_parse_pcxr.sh
```

### C. Migrate processed PNGs and repo to LRZ cluster

```bash
bash 03_migrate_to_cluster.sh
```

## 02_Remote Pipeline (LRZ Cluster)

The rest of the pipeline can be performed on the cluster.

### A. Finetune Cheff

```bash
sbatch 04_finetune_cheff.sbatch
```

### B. Sample Synthetic Data

```bash
sbatch 04_sample_cheff.sbatch
```

### C. Train Classifiers

```bash
# Baseline (Real data only)
sbatch 05_baseline_classifier.sbatch

# Synthetic (Real + Synthetic data)
sbatch 06_synthetic_classifier.sbatch

# Filtered (Real + Filtered Synthetic data)
sbatch 07_synthetic_filtered_classifier.sbatch
```

## END
