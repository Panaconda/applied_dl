# Data: VinDr-PCXR

**Description**: VinDr-PCXR is a pediatric chest X‑ray dataset containing 9,125 images collected from three major hospitals in Vietnam. It is annotated for 36 critical findings and 15 diseases, and is split into 7,728 training and 1,397 test images. The training set images are each labeled by one of three experienced radiologists, while the test set is more thoroughly curated for benchmarking.

Each image carries one or more positive labels:
- Train: $223$ images ($2.9\%$) have ≥2 positive labels
- Test: $41$ images ($2.9\%$) have ≥2 positive labels

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

## Baseline Scenario

**Goal**: Fine-tune TorchXRayVision DenseNet121 on the original 7,728 training images; evaluate on the fixed 1,397 test images. This is the "original only" condition and the floor all other conditions are measured against.

### Model

```python
import torchxrayvision as xrv, torch

model = xrv.models.DenseNet(weights="densenet121-res224-all")
model.classifier = torch.nn.Linear(model.classifier.in_features, 6)
```

### Training

| Setting         | Value                                              |
|-----------------|----------------------------------------------------|
| Loss            | `BCEWithLogitsLoss(pos_weight=pos_weight_tensor)` — weights: [0.50, 8.18, 13.18, 14.55, 18.71, 15.69] |
| Optimizer       | AdamW — see two-phase schedule below               |
| Validation      | 10 % iterative stratified hold-out (≈ 700 images) via `skmultilearn.model_selection.IterativeStratification` — required for multilabel arrays; standard `stratify=` will fail |
| Early stopping  | Monitor val AUC-ROC, patience = 5 epochs           |
| Input           | 224 × 224 grayscale; scale pixel values from [0, 255] to **[-1024, 1024]** via `xrv.datasets.normalize(img, maxval=255)` — TXRV pre-trained weights expect this range |

#### Two-phase fine-tuning

The new 6-class head is randomly initialised; training it end-to-end from epoch 0 risks large gradients destroying the pre-trained backbone features (catastrophic forgetting).

| Phase | Backbone | Head | Optimizer | Duration |
|-------|----------|------|-----------|----------|
| 1 — warm-up | **Frozen** (`model.features.requires_grad_(False)`) | Trained | AdamW lr = 1e-4 | 3 epochs |
| 2 — full fine-tune | **Unfrozen** | Trained | AdamW lr = 1e-5 | until early stop |

```python
# Phase 1
model.features.requires_grad_(False)
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-4)

# Phase 2  (after 3 epochs)
model.features.requires_grad_(True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
```

### Evaluation

Apply `torch.sigmoid` to raw logits before computing metrics. Report per class for all 6 classes: **No finding, Bronchitis, Brocho-pneumonia, Bronchiolitis, Pneumonia, Other disease**. "Other disease" is now a well-defined catch-all (original label + 9 collapsed rare classes) and is included in all headline numbers.

| Metric       | Notes                                      |
|--------------|--------------------------------------------|
| AUC-ROC      | Primary; threshold-free                    |
| F1-Score     | Threshold 0.5 (tune per class if needed)   |
| Sensitivity  | True positive rate                         |
| Specificity  | True negative rate                         |

