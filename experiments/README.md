# Data: VinDr-PCXR

**Description**: VinDr-PCXR is a pediatric chest X‑ray dataset containing 9,125 images collected from three major hospitals in Vietnam. It is annotated for 36 critical findings and 15 diseases, and is split into 7,728 training and 1,397 test images. The training set images are each labeled by one of three experienced radiologists, while the test set is more thoroughly curated for benchmarking.

Each image carries one or more positive labels:
- Train: $223$ images ($2.9\%$) have ≥2 positive labels
- Test: $41$ images ($2.9\%$) have ≥2 positive labels

## Data Format

The pipeline expects the MaCheX-processed VinDr-PCXR dataset located at
data/vindr-pcxr/ (relative to the project root).

  data/vindr-pcxr/
  ├── Train/                  # 7 728 × 1024×1024 RGB JPEG images
  │    000000.jpg
  │    000001.jpg
  │    …
  ├── Test/                   # 1 397 × 1024×1024 RGB JPEG images
  │    000000.jpg
  │    …
  ├── image_labels_train.csv  # 15-class multi-label annotations (train)
  ├── image_labels_test.csv   # 15-class multi-label annotations (test)
  ├── annotations_train.csv   # bounding-box annotations (train)
  ├── annotations_test.csv    # bounding-box annotations (test)
  ├── index.json              # MaCheX train index (seq-key → image_id mapping)
  └── index_test.json         # MaCheX test index

MaCheX assigns sequential zero-padded filenames (000000.jpg, 000001.jpg, …).
The original image_id (hex string matching the label CSVs) is stored in the
"key" field of each index.json entry as "<image_id>.dicom".  The datamodule
calls load_image_id_map() at setup time to resolve image paths automatically.

## Environment Variables

Copy .env.example → .env (already pre-filled; no edits required for the
default MaCheX dataset):

  cp experiments/.env.example experiments/.env

Key variables:
  TRAIN_IMAGE_DIR            path to Train/ directory
  TEST_IMAGE_DIR             path to Test/ directory
  TRAIN_LABELS_CSV           path to image_labels_train.csv
  TEST_LABELS_CSV            path to image_labels_test.csv
  VINDR_PCXR_TRAIN_INDEX     path to index.json  (train)
  VINDR_PCXR_TEST_INDEX      path to index_test.json  (test)

The two index vars are optional: when empty, the dataset falls back to
looking for <image_id>.png files directly in image_dir (legacy behaviour
for the old Pre-256 PNG dataset).

# Experiments

## Classifier – Baseline

```shell
python -m baseline.train
```

```shell
python -m baseline.evaluate \
  --checkpoint runs/baseline/{version_*}/checkpoints/best.ckpt
```

### Result

| Class               | AUC-ROC | F1     | Sens   | Spec   |
|---------------------|---------|--------|--------|--------|
| No finding          | 0.7031  | 0.7161 | 0.6604 | 0.6592 |
| Bronchitis          | 0.6884  | 0.3193 | 0.6322 | 0.6688 |
| Brocho-pneumonia    | 0.7531  | 0.1933 | 0.7262 | 0.6299 |
| Bronchiolitis       | 0.7024  | 0.2028 | 0.7222 | 0.6282 |
| Pneumonia           | 0.7640  | 0.2414 | 0.6292 | 0.7561 |
| Other disease       | 0.5924  | 0.1181 | 0.3580 | 0.7105 |
| **Mean**            | **0.7005** | **0.2985** | **0.6214** | **0.6754** |

---

## Classifier – Standard Data Augmentation

**Objective**: Same as baseline but with standard computer vision augmentations applied to the training set only. Tests whether geometric and photometric augmentation alone improves generalisation without any additional data.

**Augmentations (train only — val/test always use plain transform):**

| Transform | Params | Rationale |
|---|---|---|
| `RandomAffine` | degrees=±5°, translate=2%, scale=0.97–1.03, fill=0 | Mild rotation, shift, and zoom; fill=0 maps to ~−1 after xrv normalisation (black border) |

```shell
python -m augmented.train
```

```shell
python -m baseline.evaluate \
  --checkpoint runs/augmented/{version_*}/checkpoints/best.ckpt
```

### Result

| Class               | AUC-ROC | F1     | Sens   | Spec   |
|---------------------|---------|--------|--------|--------|
| No finding          | 0.7117  | 0.7133 | 0.6637 | 0.6347 |
| Bronchitis          | 0.6856  | 0.3058 | 0.5632 | 0.6983 |
| Brocho-pneumonia    | 0.7501  | 0.2000 | 0.6667 | 0.6801 |
| Bronchiolitis       | 0.6927  | 0.1935 | 0.7000 | 0.6190 |
| Pneumonia           | 0.7846  | 0.2593 | 0.6629 | 0.7653 |
| Other disease       | 0.6038  | 0.1181 | 0.3951 | 0.6740 |
| **Mean**            | **0.7047** | **0.2983** | **0.6086** | **0.6786** |

## CheFF LoRA Fine-Tuning

**Objective**: Fine-tune the pre-trained CheFF T2I diffusion model on VinDr-PCXR using LoRA
(rank=16, α=32, all attention layers, 0.64% trainable params) so it can generate realistic
pediatric chest X-rays conditioned on radiology report text.

**LoRA hyper-parameters**: rank=16, alpha=32, dropout=0.0, scope=all attention (self + cross),
batch size=8, lr=5e-5, max epochs=15, fp32.

**Prepare fine-tuning data** (writes `{MACHEX_OUTPUT_DIR}/mimic/index.json`):

```shell
python -m finetune_cheff.prepare_data
```

**Fine-tune**:

```shell
python -m finetune_cheff.train
```

LoRA adapter is exported to `runs/finetune_cheff/lora_adapter/` after training.

### Visual Evaluation — 4×4 sample grids

```shell
# Fine-tuned model (4 pathologies × 4 samples)
python -m inference.generate_grid \
  --lora-adapter runs/finetune_cheff/lora_adapter \
  --output runs/grid_finetuned.png \
  --steps 50

# Base model (no LoRA) — for comparison
python -m inference.generate_grid \
  --output runs/grid_base.png \
  --steps 50
```

### Latent Space Evaluation — UMAP

Generate LoRA synthetic images first (step required before UMAP):

```shell
python -m inference.generate_synthetic \
  --lora-adapter runs/finetune_cheff/lora_adapter \
  --n 50 \
  --steps 50 \
  --output-dir ../samples/lora
```

Run UMAP (base model images generated on-the-fly):

```shell
python -m inference.umap_latent \
  --lora-dir ../samples/lora \
  --output runs/umap_latent.png \
  --steps 50
```

### Fréchet DenseNet Distance (FDD)

```shell
python -m inference.fdd \
  --lora-dir ../samples/lora \
  --per-class 25 \
  --n-real-train 50 \
  --n-real-test 50 \
  --base-on-the-fly \
  --model-path ../models/cheff_diff_t2i.pt \
  --ae-path ../models/cheff_autoencoder.pt \
  --steps 50 \
  --output runs/fdd.txt
```

---

## Classifier – Original + Unfiltered Synthetic

**Objective**: Augment the original 7,728 training images with 1,000 unfiltered synthetic images
per class (4,000 total) generated by the fine-tuned CheFF model.

**Step 1 — Generate synthetic images** (if not already done):

```shell
python -m inference.generate_synthetic \
  --lora-adapter runs/finetune_cheff/lora_adapter \
  --n 1000 \
  --steps 100 \
  --output-dir ../samples/lora
```

**Step 2 — Build index files**:

```shell
for cls in Pneumonia Bronchitis Bronchiolitis Brocho-pneumonia; do
    python -m inference.build_synthetic_index --synthetic-dir ../samples/lora/$cls
done
```

**Step 3 — Train**:

```shell
python -m synthetic.train \
  --synthetic-dirs ../samples/lora/Pneumonia \
                   ../samples/lora/Bronchitis \
                   ../samples/lora/Bronchiolitis \
                   "../samples/lora/Brocho-pneumonia" \
  --run-name synthetic_all_unfiltered
```

**Step 4 — Evaluate**:

```shell
python -m baseline.evaluate \
  --checkpoint runs/synthetic_all_unfiltered/version_0/checkpoints/best.ckpt
```

### Result

---

## Classifier – Original + Oracle-Filtered Synthetic

**Objective**: Use the baseline classifier as an oracle to discard synthetic images where
P(target class) < 0.70, then retrain on the filtered set.

**Step 1 — Oracle filter** (requires baseline checkpoint):

```shell
for cls in Pneumonia Bronchitis Bronchiolitis Brocho-pneumonia; do
    python -m inference.filter_synthetic \
        --ckpt runs/baseline/version_0/checkpoints/best.ckpt \
        --index ../samples/lora/$cls/synthetic_paths.json \
        --target $cls \
        --threshold 0.70 \
        --output-dir ../samples/lora/$cls
done
```

**Step 2 — Train**:

```shell
python -m synthetic.train \
  --synthetic-dirs ../samples/lora/Pneumonia \
                   ../samples/lora/Bronchitis \
                   ../samples/lora/Bronchiolitis \
                   "../samples/lora/Brocho-pneumonia" \
  --run-name synthetic_all_filtered
```

**Step 3 — Evaluate**:

```shell
python -m baseline.evaluate \
  --checkpoint runs/synthetic_all_filtered/version_0/checkpoints/best.ckpt
```

### Result

---

## Classifier – Original + Oracle-Filtered Synthetic + Augmentation

**Objective**: Combine oracle-filtered synthetic data with standard augmentation on real images.

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
  --checkpoint runs/synthetic_all_filtered_aug/version_0/checkpoints/best.ckpt
```

### Result

---

## Ablation – Per-Class Filtered Synthetic (no augmentation)

**Objective**: Train one model per pathology using only that class's filtered synthetic images,
to isolate the per-class contribution of the synthetic data.

```shell
for cls in Pneumonia Bronchitis Bronchiolitis Brocho-pneumonia; do
    python -m synthetic.train \
      --synthetic-dirs "../samples/lora/$cls" \
      --run-name "synthetic_${cls}_filtered"
done
```

```shell
for cls in Pneumonia Bronchitis Bronchiolitis Brocho-pneumonia; do
    python -m baseline.evaluate \
      --checkpoint "runs/synthetic_${cls}_filtered/version_0/checkpoints/best.ckpt"
done
```

### Result