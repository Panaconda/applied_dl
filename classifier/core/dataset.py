"""VinDr-PCXR dataset: label loading, image transform, and Dataset class."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchxrayvision as xrv
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as tv_transforms
from torchvision.transforms.functional import InterpolationMode, resize as tv_resize

from classifier.core.config import cfg


def compute_pos_weights(labels: pd.DataFrame) -> torch.Tensor:
    N = len(labels)
    pos = labels.sum(axis=0).clip(lower=1)  # avoid division by zero
    weights = (N - pos) / pos
    return torch.tensor(weights.values, dtype=torch.float32)


def load_image_id_map(index_json_path: str, image_dir: str) -> Dict[str, str]:
    """Build an {image_id: abs_path} mapping from a MaCheX index.json.

    MaCheX assigns sequential zero-padded filenames (e.g. ``000000.jpg``) and
    stores the original ``<image_id>.dicom`` name in the ``key`` field of each
    index entry.  This function inverts that mapping so the rest of the pipeline
    can look up images by their original ``image_id``.

    Args:
        index_json_path: Path to the MaCheX ``index.json`` file.
        image_dir:       Directory where the sequential ``.jpg`` files live.

    Returns:
        Dict mapping ``image_id`` (hex string, no extension) to the absolute
        path of the corresponding ``.jpg`` file inside ``image_dir``.
    """
    with open(index_json_path) as f:
        index = json.load(f)
    return {
        entry["key"].replace(".dicom", ""): os.path.join(
            image_dir, f"{seq_key}.jpg"
        )
        for seq_key, entry in index.items()
    }


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
    def __init__(self, size: int = cfg.image_size) -> None:
        self.size = size

    def __call__(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img.convert("L")).astype(np.float32)           # [H, W]
        arr = xrv.datasets.normalize(arr, maxval=255)                  # [-1024, 1024]
        tensor = torch.from_numpy(arr).unsqueeze(0)                    # [1, H, W]
        tensor = tv_resize(
                    tensor, 
                    [self.size, self.size], 
                    interpolation=InterpolationMode.BICUBIC, 
                    antialias=True
                )
        return tensor

def build_transform(size: int = cfg.image_size) -> XRVTransform:
    """Return a picklable XRVTransform for the given output size."""
    return XRVTransform(size)

class VinDrPCXRDataset(Dataset):
    """VinDr-PCXR dataset backed by pre-256 PNG files.

    Args:
        image_ids:           Ordered list of image_id strings to include.
        labels:              DataFrame indexed by image_id, columns = VIABLE_CLASSES.
        image_dir:           Directory containing ``<image_id>.png`` files.
        transform:           Callable applied to a PIL Image; defaults to build_transform().
        image_path_overrides: Optional dict mapping image_id → absolute file path.
                              When present for a given id, the stored path is used
                              instead of ``{image_dir}/{image_id}.png``.  Useful for
                              synthetic images stored outside ``image_dir``.
    """

    def __init__(
        self,
        image_ids: List[str],
        labels: pd.DataFrame,
        image_dir: str,
        transform=None,
        image_path_overrides: Optional[dict] = None,
    ) -> None:
        self.image_ids = image_ids
        self.labels = labels
        self.image_dir = image_dir
        self.transform = transform if transform is not None else build_transform()
        self.image_path_overrides = image_path_overrides or {}

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_id = self.image_ids[idx]
        img_path = self.image_path_overrides.get(
            image_id, os.path.join(self.image_dir, f"{image_id}.png")
        )

        img = Image.open(img_path)
        img = self.transform(img)

        label = torch.tensor(
            self.labels.loc[image_id].values, dtype=torch.float32
        )
        return img, label
