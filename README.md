# Diffusion-Based Data Augmentation:Parameter-Efficient Fine-Tuning of Cheff for Rare Pathologies

## Overview

This repository provides a pipeline for adapting the diffusion model Cheff, predominantly trained on adult Chest x-ray scans to the pediatric domain. The project evaluates how Parameter-Efficient Fine-Tuning (PEFT) via LoRA can alleviate data scarcity in the PCXR domain for pathology classification.
More information on the context of the research can be found in results/proposal.md.
The main results of the project are shown in results/results_summary.ipynb

## Repository Structure:

- `cheff_peft/`: Logic for finetuning the Cheff model using Parameter-Efficient Fine-Tuning (PEFT).
- `classifier/`: Fine-tuning logic for the torchxrayvision pathology classifiers.
- `data/`: Directory for storing DICOM, PNG, and synthetic samples.
- `prepare_pcxr/`: Scripts for downloading and parsing the VinDr-PCXR dataset.
- `visuals/`: Utility scripts for generating visualizations and plots.
- `results/`: Summarizes the main results of our research.

## Execution Guide

### 1. Local Setup

To conserve cluster resources, initial processing is performed locally.

```bash
git clone "https://github.com/Panaconda/applied_dl.git" applied_dl
cd applied_dl

python -m venv "adl_env"
source "adl_env/Scripts/activate"

python -m pip install --upgrade pip
pip install -r requirements/local.txt
```

Then run these script in sequence via bash to prepare the pediatric data:

- 01_download_pcxr.sh: Downloads the VinDr-PCXR dataset (keep NUM_WORKERS $\le$ 4 to avoid rate-limiting IP bans)
- 02_parse_pcxr.sh: Validates and converts DICOM files to PNG format
- 03_migrate_to_cluster.sh: Transfers processed images and the repository to the LRZ Cluster

### 2. Remote Pipeline (LRZ Cluster)

The rest of the pipeline can be performed on the cluster.

Cheff LoRA:

- 04_finetune_cheff.sbatch: Performs LoRA fine-tuning on the Cheff semantic diffusion model
- 05_sample_cheff.sbatch: Generates high-fidelity synthetic pediatric radiographs.

Pathology Classifier:

- 06_baseline_classifier.sbatch: Training on real pediatric data only
- 07_synthetic_classifier.sbatch: Training on real + synthetic data
- 08_synthetic_filtered_classifier.sbatch: Training on real + filtered synthetic data

### 3. Download Results

- 09_local_download_results.sh: Download the trained models and results to the local machine

## Credits

This projects builds upon the work **Cascaded Latent Diffusion Models for High-Resolution Chest X-ray Synthesis**. Our code and the model weights of Cheff are based on its conneceted repository:
https://github.com/saiboxx/machex
https://github.com/saiboxx/chexray-diffusion
Feel free to check them out!

This project was conducted in the context of an Applied Deep Learning course offered by Prof. Dr. David Rügamer at LMU Munich.
