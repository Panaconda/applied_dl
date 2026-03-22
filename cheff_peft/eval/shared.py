from __future__ import annotations

import json
import os
import random
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchxrayvision as xrv
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.functional import resize as tv_resize
from tqdm import tqdm

from config import ftcfg
from inference.generate import CLASS_PROMPTS, load_model

PATHOLOGY_CLASSES = ["Pneumonia", "Bronchitis", "Bronchiolitis", "Brocho-pneumonia"]


def _load_labels(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    label_cols = [c for c in df.columns if c not in ("image_id", "rad_ID")]
    df = df.groupby("image_id")[label_cols].max().astype(int)
    return df[[c for c in PATHOLOGY_CLASSES if c in df.columns]]


def _load_image_id_map(index_json_path: str, image_dir: str) -> dict[str, str]:
    with open(index_json_path) as f:
        index = json.load(f)
    return {
        entry["key"].replace(".dicom", ""): os.path.join(image_dir, f"{seq_key}.jpg")
        for seq_key, entry in index.items()
    }


def build_transform(size: int = 224):
    """Return an XRV-normalised image transform (PIL → [1, size, size] tensor)."""
    def transform(img: Image.Image) -> torch.Tensor:
        arr = np.array(img.convert("L")).astype(np.float32)
        arr = xrv.datasets.normalize(arr, maxval=255)
        tensor = torch.from_numpy(arr).unsqueeze(0)
        tensor = tv_resize(
            tensor, [size, size],
            interpolation=InterpolationMode.BICUBIC, antialias=True,
        )
        return torch.clamp(tensor, -1024, 1024)
    return transform


def sample_real_paths(n_train: int = 100, n_test: int = 100, seed: int = 42) -> list[str]:
    """Return paths for real images balanced across 4 pathology classes."""
    rng = random.Random(seed)
    per_cls_train = n_train // len(PATHOLOGY_CLASSES)
    per_cls_test = n_test // len(PATHOLOGY_CLASSES)
    paths: list[str] = []

    train_image_dir = os.path.join(ftcfg.data_dir, "train")
    test_image_dir = os.path.join(ftcfg.data_dir, "test")
    train_labels_csv = os.path.join(train_image_dir, "image_labels_train.csv")
    test_labels_csv = os.path.join(test_image_dir, "image_labels_test.csv")
    train_index_json = os.path.join(train_image_dir, "index.json")
    test_index_json = os.path.join(test_image_dir, "index.json")

    for image_dir, csv_path, index_json, per_cls in [
        (train_image_dir, train_labels_csv, train_index_json, per_cls_train),
        (test_image_dir, test_labels_csv, test_index_json, per_cls_test),
    ]:
        labels = _load_labels(csv_path)
        id_map = _load_image_id_map(index_json, image_dir) if index_json else {}
        for cls in PATHOLOGY_CLASSES:
            positive_ids = labels[labels[cls] == 1].index.tolist()
            sampled = rng.sample(positive_ids, min(per_cls, len(positive_ids)))
            for img_id in sampled:
                path = id_map.get(img_id, os.path.join(image_dir, f"{img_id}.png"))
                paths.append(path)

    return paths


def sample_lora_paths(lora_dir: str, per_class: int = 50, seed: int = 42) -> list[str]:
    """Return paths for LoRA-synthetic images (per_class per pathology class)."""
    rng = random.Random(seed)
    paths: list[str] = []
    for cls in PATHOLOGY_CLASSES:
        cls_dir = os.path.join(lora_dir, cls)
        all_pngs = sorted(glob(os.path.join(cls_dir, "*.png")))
        if len(all_pngs) == 0:
            raise FileNotFoundError(f"No PNGs in {cls_dir}")
        paths.extend(rng.sample(all_pngs, min(per_class, len(all_pngs))))
    return paths


@torch.no_grad()
def generate_base_images(
    wrapper, per_class: int = 50, steps: int = 100, eta: float = 1.0
) -> list[Image.Image]:
    """Generate base CheFF images on-the-fly and return as PIL images."""
    images: list[Image.Image] = []
    for cls in PATHOLOGY_CLASSES:
        prompt = CLASS_PROMPTS[cls]
        for _ in tqdm(range(per_class), desc=f"Base {cls}", leave=False):
            tensor = wrapper.sample(
                sampling_steps=steps, eta=eta, decode=True, conditioning=prompt,
            )
            arr = (tensor.squeeze(0).permute(1, 2, 0).clamp(-1, 1).cpu().numpy() + 1) / 2
            arr = (arr * 255).astype(np.uint8)
            if arr.shape[2] == 1:
                images.append(Image.fromarray(arr.squeeze(-1), mode="L"))
            else:
                images.append(Image.fromarray(arr))
    return images


def load_paths_as_tensors(paths: list[str], transform) -> list[torch.Tensor]:
    """Load image files and apply XRV transform."""
    return [transform(Image.open(p)) for p in tqdm(paths, desc="Loading images")]


def pils_to_tensors(pil_images: list[Image.Image], transform) -> list[torch.Tensor]:
    """Apply XRV transform to in-memory PIL images."""
    return [transform(img) for img in pil_images]


@torch.no_grad()
def extract_features(
    tensors: list[torch.Tensor], device: str, batch_size: int = 32
) -> np.ndarray:
    """Extract 1024-d feature vectors using the pretrained XRV DenseNet121.

    Args:
        tensors:    list of [1, H, W] XRV-normalised image tensors.
        device:     "cuda" or "cpu".
        batch_size: inference batch size.

    Returns:
        (N, 1024) float32 numpy array.
    """
    model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device).eval()
    features: list[np.ndarray] = []
    for i in tqdm(range(0, len(tensors), batch_size), desc="Extracting features", leave=False):
        batch = torch.stack(tensors[i : i + batch_size]).to(device)
        feats = model.features(batch) # (B, 1024, 7, 7)
        feats = F.relu(feats, inplace=False)
        feats = F.adaptive_avg_pool2d(feats, (1, 1)) # (B, 1024, 1, 1)
        feats = feats.view(feats.size(0), -1).cpu().numpy() # (B, 1024)
        features.append(feats)
    del model
    torch.cuda.empty_cache()
    return np.vstack(features)
