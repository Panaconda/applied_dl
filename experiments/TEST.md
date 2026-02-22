# Data: VinDr-PCXR

**Description**: VinDr-PCXR is a pediatric chest X‑ray dataset containing 9,125 images collected from three major hospitals in Vietnam. It is annotated for 36 critical findings and 15 diseases, and is split into 7,728 training and 1,397 test images. The training set images are each labeled by one of three experienced radiologists, while the test set is more thoroughly curated for benchmarking.

Each image carries one or more positive labels:
- Train: $223$ images ($2.9\%$) have ≥2 positive labels
- Test: $41$ images ($2.9\%$) have ≥2 positive labels

> **Note**: While the original dataset was still downloading I used this preprocessed dataset from [Kaggle](https://www.kaggle.com/datasets/nhantran712/vindr-pcxr-32-256) which contains 7,728 training and 1,397 test images at 256 × 256 resolution. For the results of the same experiments on the original dataset preprocessed with `machex` see the README markdown file. 

## Label Distribution

| Pathology                  | Train | Test  |
|----------------------------|------:|------:|
| **Total images**           | $7,728$ | $1,397$ |
|                            |       |       |
| No finding                 | $5,143$ | $907$ |
| Bronchitis                 | $842$ | $174$ |
| Brocho-pneumonia           | $545$ | $84$ |
| Bronchiolitis              | $497$ | $90$ |
| Other disease              | $412$ | $77$ |
| Pneumonia                  | $392$ | $89$ |
| *Hyaline membrane disease* | $19$ | $3$ |
| *Tuberculosis*             | $14$ | $1$ |
| *Situs inversus*           | $11$ | $2$ |
| *Mediastinal tumor*        | $8$ | $1$ |
| *Pleuro-pneumonia*         | $6$ | $0$ |
| *CPAM*                     | $5$ | $1$ |
| *Lung tumor*               | $5$ | $0$ |
| *Diagphramatic hernia*     | $3$ | $0$ |
| *Congenital emphysema*     | $2$ | $0$ |

## Label Transformation

9 classes, marked in italic above, have ≤19 training examples and are **not viable** for per-pathology experiments. To preserve the information they carry, we collapse them into the existing `Other disease` label via logical OR, then drop the original 9 rare columns. The target vector goes from 15 to 6 classes:

```python
rare_cols = [c for c in all_label_cols if c not in VIABLE_CLASSES]
labels['Other disease'] = labels[['Other disease'] + rare_cols].max(axis=1)
labels = labels[VIABLE_CLASSES]  # drop rare cols
```

> "Other disease" train count increases from 412 (raw) to 463 after absorbing 9 rare classes.

| Index | Class            | Train                 | Test |
|-------|------------------|----------------------:|-----:|
| 0     | No finding       |                 5,143 |  907 |
| 1     | Bronchitis       |                   842 |  174 |
| 2     | Brocho-pneumonia |                   545 |   84 |
| 3     | Bronchiolitis    |                   497 |   90 |
| 4     | Pneumonia        |                   392 |   89 |
| 5     | Other disease    |                   463 |   77 |

# Experiments

## Setup 

**Model**: TorchXRayVision DenseNet121 (`densenet121-res224-all`), 6-class head, `BCEWithLogitsLoss` with per-class `pos_weight`; AdamW; 10 % iterative-stratified val split; 224 × 224 grayscale input normalised to [−1024, 1024] (TXRV convention).

**Two-phase schedule**: backbone frozen for 3 warm-up epochs (head lr=1e-4), then unfrozen for full fine-tune (backbone lr=1e-5, head lr=1e-4). Early stopping on val AUC-ROC, patience=5.

---

## Classifier – Baseline

**Objective**: Train on original 7,728 training images only; no augmentation and no synthetic data. Baseline for all other conditions.

### Results

| Class            | AUC-ROC | F1     | Sens   | Spec   |
|------------------|--------:|-------:|-------:|-------:|
| No finding       | 0.6773  | 0.7374 | 0.7321 | 0.5306 |
| Bronchitis       | 0.6749  | 0.2944 | 0.5000 | 0.7302 |
| Brocho-pneumonia | 0.7508  | 0.1959 | 0.6786 | 0.6641 |
| Bronchiolitis    | 0.6775  | 0.1747 | 0.7222 | 0.5493 |
| Pneumonia        | 0.7366  | 0.2319 | 0.5393 | 0.7882 |
| Other disease    | 0.5846  | 0.1320 | 0.4198 | 0.6960 |
| **Mean**         | **0.6836** | **0.2944** | **0.5987** | **0.6597** |

---

## Classifier – Standard Augmentation

**Objective**: Same as baseline but with standard computer vision augmentations applied to the training set only. Tests whether geometric and photometric augmentation alone improves generalisation without any additional data.

**Augmentations (train only — val/test always use plain transform):**

| Transform | Params | Rationale |
|---|---|---|
| `RandomRotation` | ±10° | Simulates patient tilt |
| `RandomAffine` | translate=3%, scale=0.95–1.05 | Patient positioning variation (tighter than typical CV) |

### Results

| Class            | AUC-ROC | F1     | Sens   | Spec   |
|------------------|--------:|-------:|-------:|-------:|
| No finding       | 0.6828  | 0.7375 | 0.7387 | 0.5102 |
| Bronchitis       | 0.6717  | 0.2829 | 0.5000 | 0.7105 |
| Brocho-pneumonia | 0.7538  | 0.2035 | 0.6310 | 0.7075 |
| Bronchiolitis    | 0.6899  | 0.1860 | 0.7667 | 0.5539 |
| Pneumonia        | 0.7311  | 0.2414 | 0.5506 | 0.7951 |
| Other disease    | 0.5870  | 0.1303 | 0.4198 | 0.6907 |
| **Mean**         | **0.6861** | **0.2969** | **0.6011** | **0.6613** |

---

## CheFF LoRA Fine-Tuning

**Objective**: Fine-tune the pre-trained CheFF Text-to-Image diffusion model on VinDr-PCXR using LoRA, so it can generate realistic *pediatric* chest X-rays conditioned on radiology report text. The resulting adapter is used by later conditions (C, D, E) to generate synthetic training data.

**Data preparation (temporary solution for testing)** (`python -m finetune_cheff.prepare_data`):
1. Read the 7,728 training PNGs from `Pre-256/Train/` (the mini dataset).
2. Join with `image_labels_train.csv` (disease labels) and `annotations_train.csv` (finding annotations).
3. Collapse the 9 rare classes into "Other disease" (same as downstream experiments).
4. **Discard images whose only active label is "Other disease"** — they contribute nothing to the 5 target pathologies.
5. Convert each kept PNG to RGB JPG and generate a templated report:
   > *"Findings: Frontal radiograph of a child. Evaluation reveals {annotations}. Impressions: {pathologies}."*
6. Write a MaCheX-formatted `index.json` consumed by CheFF's `MimicT2IDataset`.

### Results

#### Visual Evaluation

#### Latent Space Evaluation (UMAP) 

Using the **pretrained** (out-of-the-box) XRV DenseNet121 as a feature extractor, we embed 600 images into a shared 1024-d feature space and project to 2-D with UMAP:
- 200 real images (100 from train, 100 from test — 25 per pathology class each)
- 200 synthetic images generated by the LoRA-adapted CheFF model (50 per class, sampled from existing 8 000)
- 200 synthetic images generated by the original CheFF model (50 per class, generated on-the-fly)

```shell
python -m inference.umap_latent \
    --lora-dir ../samples/lora \
    --output runs/umap_latent.png
```

![UMAP Latent Space](../docs/umap_latent_test.png)

#### Frechet Densenet Distance (FDD)

Same setup as for the Latent Space Evaluation, but we compute the FDD between the real and synthetic feature distributions to get a quantitative measure of domain shift.

---

## CheFF Class-Conditional Generation

We use the template report from the previous condition, but now we explicitly manipulate the `{pathologies}` slot to generate synthetic images with specific diseases. For each of the $4$ target pathologies (`Bronchitis`, `Brocho-pneumonia`, `Bronchiolitis`, `Pneumonia`, excl. `No finding`), we generate $2000$ synthetic images conditioned on a report that mentions only that pathology from the fine-tuned CheFF model.

```shell
python -m inference.generate_synthetic \
    --lora-ckpt runs/finetune_cheff/logs/version_3/checkpoints/last.ckpt \
    --output-dir ../samples/lora \
    --n 2000
```

## Classifier – Original + CheFF Synthetic Data

We augment the original training set with the $8000$ synthetic images generated in the previous step (2,000 per pathology) and train the same XRV classifier on this combined dataset.

Build the synthetic index files consumed by the classifier training script:
```shell
for cls in Pneumonia Bronchitis Bronchiolitis Brocho-pneumonia; do
    python -m inference.build_synthetic_index --synthetic-dir ../samples/lora/$cls
done
```

Run the training:
```shell
python -m synthetic.train \
    --synthetic-dirs ../samples/lora/Pneumonia \
                     ../samples/lora/Bronchitis \
                     ../samples/lora/Bronchiolitis \
                     "../samples/lora/Brocho-pneumonia" \
    --run-name synthetic_all_unfiltered
```

Evaluate:
```shell
python -m baseline.evaluate \
    --checkpoint runs/synthetic_all_unfiltered/version_0/checkpoints/best-epoch*.ckpt
```

### Results

| Class            | AUC-ROC | F1     | Sens   | Spec   |
|------------------|--------:|-------:|-------:|-------:|
| No finding       | 0.6673  | 0.7939 | 0.9515 | 0.1755 |
| Bronchitis       | 0.6603  | 0.2268 | 0.1897 | 0.9313 |
| Brocho-pneumonia | 0.7658  | 0.2094 | 0.2381 | 0.9337 |
| Bronchiolitis    | 0.6280  | 0.0360 | 0.0222 | 0.9855 |
| Pneumonia        | 0.7167  | 0.1750 | 0.1573 | 0.9564 |
| Other disease    | 0.5964  | 0.1244 | 0.7654 | 0.3511 |
| **Mean**         | **0.6724** | **0.2609** | **0.3874** | **0.7222** |

## Synthetic Data Filtering with Baseline Oracle

Since the fine-tuned CheFF model is not perfect, we expect some of the generated images to be of low quality or incorrectly labelled. To get an upper bound on the potential of synthetic data, we use our baseline classifier from the first step as an **oracle** to filter the generated images. We set a filter threshold at $0.7$ (probability of the target pathology) and only keep synthetic images that exceed this threshold.

```shell
for cls in Pneumonia Bronchitis Bronchiolitis Brocho-pneumonia; do
    python -m inference.filter_synthetic \
        --ckpt     runs/baseline/version_0/checkpoints/best-epoch*.ckpt \
        --index    ../samples/lora/$cls/synthetic_paths.json \
        --target   $cls \
        --threshold 0.70 \
        --output-dir ../samples/lora/$cls
done
```

### Oracle Filter Results (threshold = 0.70)

| Class            | Generated | Accepted | Discarded | Yield | Score mean / median |
|------------------|----------:|---------:|----------:|------:|--------------------:|
| Pneumonia        | 2,000     | 253      | 1,747     | 12.7% | 0.398 / 0.391       |
| Bronchitis       | 2,000     | 27       | 1,973     |  1.4% | 0.401 / 0.393       |
| Bronchiolitis    | 2,000     | 146      | 1,854     |  7.3% | 0.413 / 0.427       |
| Brocho-pneumonia | 2,000     | 285      | 1,715     | 14.2% | 0.458 / 0.474       |
| **Total**        | **8,000** | **711**  | **7,289** | **8.9%** |                  |

## Classifier – Original + Filtered CheFF Synthetic Data

We train the same XRV classifier on the original training set augmented with the **filtered** synthetic images from the previous step, and compare its performance to the unfiltered version and the baseline.

```shell
python -m synthetic.train \
    --synthetic-dirs ../samples/lora/Pneumonia \
                     ../samples/lora/Bronchitis \
                     ../samples/lora/Bronchiolitis \
                     "../samples/lora/Brocho-pneumonia" \
    --run-name synthetic_all_filtered
```

```shell
python -m baseline.evaluate \
    --checkpoint runs/synthetic_all_filtered/version_0/checkpoints/best-epochepoch=29.ckpt
```

### Results

#### Original + All Filtered Synthetic

| Class            | AUC-ROC | F1     | Sens   | Spec   |
|------------------|--------:|-------:|-------:|-------:|
| No finding       | 0.7074  | 0.7555 | 0.7563 | 0.5449 |
| Bronchitis       | 0.6826  | 0.3013 | 0.6034 | 0.6582 |
| Brocho-pneumonia | 0.7783  | 0.2571 | 0.5952 | 0.8058 |
| Bronchiolitis    | 0.7004  | 0.1905 | 0.5556 | 0.7054 |
| Pneumonia        | 0.7731  | 0.2662 | 0.4607 | 0.8639 |
| Other disease    | 0.6009  | 0.1366 | 0.4444 | 0.6884 |
| **Mean**         | **0.7071** | **0.3179** | **0.5693** | **0.7111** |


#### Original + Pneumonia Filtered Synthetic

| Class            | AUC-ROC | F1     | Sens   | Spec   |
|------------------|--------:|-------:|-------:|-------:|
| No finding       | 0.7051  | 0.7114 | 0.6604 | 0.6367 |
| Bronchitis       | 0.6768  | 0.3101 | 0.6379 | 0.6476 |
| Brocho-pneumonia | 0.7562  | 0.2045 | 0.8095 | 0.6093 |
| Bronchiolitis    | 0.6697  | 0.1851 | 0.6333 | 0.6412 |
| Pneumonia        | 0.7851  | 0.3123 | 0.5843 | 0.8532 |
| Other disease    | 0.5851  | 0.1172 | 0.3704 | 0.6953 |
| **Mean**         | **0.6963** | **0.3068** | **0.6160** | **0.6805** |

**vs Baseline** (Pneumonia synthetic only)

| Class            | Baseline AUC | AUC    | Baseline F1 | F1     |
|------------------|-------------:|-------:|------------:|-------:|
| No finding       | 0.6773       | 0.7051 | 0.7374      | 0.7114 |
| Bronchitis       | 0.6749       | 0.6768 | 0.2944      | 0.3101 |
| Brocho-pneumonia | 0.7508       | 0.7562 | 0.1959      | 0.2045 |
| Bronchiolitis    | 0.6775       | 0.6697 | 0.1747      | 0.1851 |
| Pneumonia        | *0.7366*     | **0.7851** | *0.2319*  | **0.3123** |
| Other disease    | 0.5846       | 0.5851 | 0.1320      | 0.1172 |
| **Mean**         | **0.6836**   | **0.6963** | **0.2944** | **0.3068** |


#### Original + Bronchitis Filtered Synthetic


#### Original + Bronchiolitis Filtered Synthetic


#### Original + Brocho-pneumonia Filtered Synthetic


## Classifier – Original + Filtered CheFF Synthetic Data + Augmentation

Finally, we train the same XRV classifier on the original training set augmented with the **filtered** synthetic images from the previous step, **plus** standard augmentations applied to the real training images. This tests whether the combination of synthetic data and augmentation can further improve generalisation.

```shell
python -m synthetic.train \
    --synthetic-dirs ../samples/lora/Pneumonia \
                     ../samples/lora/Bronchitis \
                     ../samples/lora/Bronchiolitis \
                     "../samples/lora/Brocho-pneumonia" \
    --augment \
    --run-name synthetic_all_filtered_aug
```

```shell
python -m baseline.evaluate \
    --checkpoint runs/synthetic_all_filtered_aug/version_0/checkpoints/best-epoch*.ckpt
```

### Results

| Class            | AUC-ROC | F1     | Sens   | Spec   |
|------------------|--------:|-------:|-------:|-------:|
| No finding       | 0.7155  | 0.7658 | 0.7806 | 0.5224 |
| Bronchitis       | 0.6785  | 0.2952 | 0.5345 | 0.7032 |
| Brocho-pneumonia | 0.7859  | 0.2519 | 0.5833 | 0.8050 |
| Bronchiolitis    | 0.6927  | 0.1796 | 0.5000 | 0.7200 |
| Pneumonia        | 0.7699  | 0.2717 | 0.5281 | 0.8394 |
| Other disease    | 0.6031  | 0.1315 | 0.4691 | 0.6512 |
| **Mean**         | **0.7076** | **0.3160** | **0.5659** | **0.7069** |

## Conclusion

**AUC-ROC**

| Class            | Baseline | +Augmentation | +Synthetic | +Filtered Synthetic | +Filtered Synthetic + Aug |
|------------------|--------:|-------------:|-----------:|--------------------:|--------------------------:|
| No finding       | 0.6773  | 0.6828        | 0.6673     | 0.7074              | 0.7155                    |
| Bronchitis       | 0.6749  | 0.6717        | 0.6603     | 0.6826              | 0.6785                    |
| Brocho-pneumonia | 0.7508  | 0.7538        | 0.7658     | 0.7783              | 0.7859                    |
| Bronchiolitis    | 0.6775  | 0.6899        | 0.6280     | 0.7004              | 0.6927                    |
| Pneumonia        | 0.7366  | 0.7311        | 0.7167     | 0.7731              | 0.7699                    |
| Other disease    | 0.5846  | 0.5870        | 0.5964     | 0.6009              | 0.6031                    |
| **Mean**         | **0.6836** | **0.6861** | **0.6724** | **0.7071**          | **0.7076**                |

**F1**

| Class            | Baseline | +Augmentation | +Synthetic | +Filtered Synthetic | +Filtered Synthetic + Aug |
|------------------|--------:|-------------:|-----------:|--------------------:|--------------------------:|
| No finding       | 0.7374  | 0.7375        | 0.7939     | 0.7555              | 0.7658                    |
| Bronchitis       | 0.2944  | 0.2829        | 0.2268     | 0.3013              | 0.2952                    |
| Brocho-pneumonia | 0.1959  | 0.2035        | 0.2094     | 0.2571              | 0.2519                    |
| Bronchiolitis    | 0.1747  | 0.1860        | 0.0360     | 0.1905              | 0.1796                    |
| Pneumonia        | 0.2319  | 0.2414        | 0.1750     | 0.2662              | 0.2717                    |
| Other disease    | 0.1320  | 0.1303        | 0.1244     | 0.1366              | 0.1315                    |
| **Mean**         | **0.2944** | **0.2969** | **0.2609** | **0.3179**          | **0.3160**                |
