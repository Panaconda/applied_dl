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
