"""VinDr-PCXR dataset: label loading, image transform, and Dataset class."""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torchxrayvision as xrv
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as tv_transforms
from torchvision.transforms.functional import resize as tv_resize

from core.config import cfg


def compute_pos_weights(labels: pd.DataFrame) -> torch.Tensor:
    """Compute per-class positive weights for BCEWithLogitsLoss.

    Formula: pos_weight[i] = (N - pos_i) / pos_i

    Args:
        labels: DataFrame indexed by image_id with int columns for each class,
                as returned by :func:`load_labels`.

    Returns:
        Float32 tensor of shape [num_classes].
    """
    N = len(labels)
    pos = labels.sum(axis=0).clip(lower=1)  # avoid division by zero
    weights = (N - pos) / pos
    return torch.tensor(weights.values, dtype=torch.float32)


def load_labels(csv_path: str) -> pd.DataFrame:
    """Load a VinDr-PCXR label CSV and return a 6-class DataFrame.

    Steps:
    - Group by image_id (handles any residual multi-row images via max).
    - Collapse the 9 non-viable classes into "Other disease" via logical OR.
    - Slice down to the 6 VIABLE_CLASSES.

    Returns a DataFrame indexed by image_id with int columns for each class.
    """
    df = pd.read_csv(csv_path)
    label_cols = [c for c in df.columns if c not in ("image_id", "rad_ID")]

    # One annotation per image in practice; groupby max is safe and explicit.
    df = df.groupby("image_id")[label_cols].max().astype(int)

    rare_cols = [c for c in label_cols if c not in cfg.viable_classes]
    df["Other disease"] = df[["Other disease"] + rare_cols].max(axis=1)

    return df[cfg.viable_classes]


class XRVTransform:
    """Picklable transform: PIL image → TXRV-normalised [1, size, size] tensor.

    Pipeline:
      PIL → grayscale uint8 numpy [H, W]
        → xrv.datasets.normalize → float32 [-1024, 1024]
        → unsqueeze → [1, H, W] tensor
        → bilinear resize → [1, size, size]
    """

    def __init__(self, size: int = cfg.image_size) -> None:
        self.size = size

    def __call__(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img.convert("L")).astype(np.float32)           # [H, W]
        arr = xrv.datasets.normalize(arr, maxval=255)                  # [-1024, 1024]
        tensor = torch.from_numpy(arr).unsqueeze(0)                    # [1, H, W]
        tensor = tv_resize(tensor, [self.size, self.size], antialias=True)  # [1, size, size]
        return tensor


def build_transform(size: int = cfg.image_size) -> XRVTransform:
    """Return a picklable XRVTransform for the given output size."""
    return XRVTransform(size)


class AugmentedXRVTransform:
    """Picklable augmented transform for training only."""

    def __init__(self, size: int = 224) -> None:
        self.size = size
        self.pil_aug = tv_transforms.Compose([
            tv_transforms.RandomRotation(degrees=10), # Reduced from 15
            tv_transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05)), # Tighter boundaries
        ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = self.pil_aug(img)
        arr = np.array(img.convert("L")).astype(np.float32)           
        arr = xrv.datasets.normalize(arr, maxval=255)                  
        tensor = torch.from_numpy(arr).unsqueeze(0)                    
        tensor = tv_transforms.functional.resize(tensor, [self.size, self.size], antialias=True)
        
        return tensor


def build_augmented_transform(size: int = cfg.image_size) -> AugmentedXRVTransform:
    """Return a picklable AugmentedXRVTransform for training."""
    return AugmentedXRVTransform(size)


class VinDrPCXRDataset(Dataset):
    """VinDr-PCXR dataset backed by pre-256 PNG files.

    Args:
        image_ids:  Ordered list of image_id strings to include.
        labels:     DataFrame indexed by image_id, columns = VIABLE_CLASSES.
        image_dir:  Directory containing ``<image_id>.png`` files.
        transform:  Callable applied to a PIL Image; defaults to build_transform().
    """

    def __init__(
        self,
        image_ids: List[str],
        labels: pd.DataFrame,
        image_dir: str,
        transform=None,
    ) -> None:
        self.image_ids = image_ids
        self.labels = labels
        self.image_dir = image_dir
        self.transform = transform if transform is not None else build_transform()

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{image_id}.png")

        img = Image.open(img_path)
        img = self.transform(img)

        label = torch.tensor(
            self.labels.loc[image_id].values, dtype=torch.float32
        )
        return img, label
