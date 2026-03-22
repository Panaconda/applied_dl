# Project Proposal: Diffusion-Based Data Augmentation for Pediatric Pathologies

## Motivation

Medical imaging suffers from inherent data scarcity, which is especially pronounced in the pediatric domain. Publicly available chest X-ray (CXR) datasets predominantly feature adult patient scans, this limits the development of model for the pedriatic patients.

This project explores the adaptation of Cheff, a cascaded latent diffusion model primarily trained on adult CXRs. By applying Parameter-Efficient Fine-Tuning via Low-Rank Adaptation (LoRA), we aim to generate high-fidelity synthetic pediatric radiographs. The objective is to determine whether augmenting real datasets with these synthetic samples can improve pathology classifier performance in data-constrained scenarios.

## Data

The project utilizes the **VinDr-PCXR** dataset, a large-scale collection of pediatric chest X-rays.

- **Content**: DICOM images annotated for various pathologies including Bronchitis, Bronchiolitis, Pleural Effusion, and Pneumonia.
- **Challenges**: The distinct anatomical features of pediatric patients compared to adults make this a challenging domain for standard generative models and hence requires adaptation.

## Methodology

The pipeline consists of the following key stages:

1. **Generative Model Adaptation**: Fine-tuning the Cheff diffusion model on the VinDr-PCXR training set using LoRA. This allows for domain adaptation with minimal trainable parameters.
2. **Synthetic Sample Generation**: Using the fine-tuned model to generate synthetic images for each of the pathologies.
3. **Downstream Classification**: Training a ResNet-based pathology classifier (from `torchxrayvision`) in three distinct experimental setups:
   - **Baseline**: Trained only on real VinDr-PCXR data.
   - **Synthetic**: Trained on real data plus all generated synthetic samples.
   - **Synthetic Filtered**: Trained on real data plus synthetic samples that passed a filtering criteria. For this the baseline model is utilized to evaluate synthetic data, retaining only images where the predicted probability of the target pathology exceeds 0.6.

## Results

Preliminary results indicate the following:

- **Baseline Performance**: The model achieved a mean AUROC of approximately **0.709** on the test set across major pathologies (No finding, Bronchitis, Broncho-pneumonia, etc.).
- **Impact of Synthetic Data**: Training with unfiltered synthetic data led to a performance drop (Mean AUROC ~**0.683**), suggesting that low-quality generative samples can introduce noise that confuses the classifier.
- **Impact of Filtering**: Introducing a filtering stage improved results compared to the unfiltered synthetic set, bringing the mean AUROC back to **0.704**, nearly matching the baseline.
- **Conclusion**: While the current diffusion-based augmentation did not significantly outperform the baseline, the filtering mechanism proved crucial in maintaining model stability. This shows that the adapted diffusion model was able to replicate useful patterns in at least some of the images.

## References

- **Cheff Model**: Weber, T., Ingrisch, M., Bischl, B., & Rügamer, D. (2023). Cascaded Latent Diffusion Models for High-Resolution Chest X-ray Synthesis. Advances in Knowledge Discovery and Data Mining: 27th Pacific-Asia Conference, PAKDD 2023
- **VinDr-PCXR Dataset**: Pham, H. H., Tran, T. T., & Nguyen, H. Q. (2022). VinDr-PCXR: An open, large-scale pediatric chest X-ray dataset for interpretation of common thoracic diseases. PhysioNet
- **TorchXRayVision**: Cohen, J. P., Viviano, J. D., Bertin, P., Morrison, P., Torabian, P., Guarrera, M., ... & Bertrand, H. (2021). TorchXRayVision: A library of chest X-ray datasets and models. arXiv preprint arXiv:2111.00595
