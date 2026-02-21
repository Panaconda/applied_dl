"""Lightning DataModule for VinDr-PCXR — shared across all experiment conditions."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from skmultilearn.model_selection import iterative_train_test_split
from torch.utils.data import DataLoader

from core.dataset import VinDrPCXRDataset, build_transform, load_labels


class VinDrPCXRDataModule(pl.LightningDataModule):
    """Manages train / val / test splits for VinDr-PCXR.

    The test split is always derived from ``test_labels_csv``; it is never
    touched during training or hyperparameter selection.

    The train CSV is split into train / val using multilabel iterative
    stratification (``skmultilearn.IterativeStratification``) to preserve
    per-class ratios across minority classes.

    Args:
        train_image_dir:  Directory with ``<image_id>.png`` training images.
        test_image_dir:   Directory with ``<image_id>.png`` test images.
        train_labels_csv: Path to ``image_labels_train.csv``.
        test_labels_csv:  Path to ``image_labels_test.csv``.
        val_fraction:     Fraction of training data held out for validation.
        batch_size:       DataLoader batch size.
        num_workers:      DataLoader worker processes.
        train_transform:  Transform applied to training images only.
                          Defaults to the plain ``XRVTransform``.
        eval_transform:   Transform applied to val and test images.
                          Defaults to the plain ``XRVTransform``.
        extra_train_ids:  Optional list of additional image_ids (e.g. synthetic
                          samples) appended to the training set after the split.
                          Files must be present in ``train_image_dir``.
        extra_labels:     Corresponding 6-column label DataFrame for
                          ``extra_train_ids`` (same format as load_labels).
    """

    def __init__(
        self,
        train_image_dir: str,
        test_image_dir: str,
        train_labels_csv: str,
        test_labels_csv: str,
        val_fraction: float = 0.10,
        batch_size: int = 32,
        num_workers: int = 4,
        train_transform=None,
        eval_transform=None,
        extra_train_ids: Optional[List[str]] = None,
        extra_labels: Optional[pd.DataFrame] = None,
    ) -> None:
        super().__init__()
        self.train_image_dir = train_image_dir
        self.test_image_dir = test_image_dir
        self.train_labels_csv = train_labels_csv
        self.test_labels_csv = test_labels_csv
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform or build_transform()
        self.eval_transform = eval_transform or build_transform()
        self.extra_train_ids = extra_train_ids or []
        self.extra_labels = extra_labels

        self._train_ds: Optional[VinDrPCXRDataset] = None
        self._val_ds: Optional[VinDrPCXRDataset] = None
        self._test_ds: Optional[VinDrPCXRDataset] = None


    def setup(self, stage: Optional[str] = None) -> None:
        test_labels = load_labels(self.test_labels_csv)
        self._test_ds = VinDrPCXRDataset(
            image_ids=list(test_labels.index),
            labels=test_labels,
            image_dir=self.test_image_dir,
            transform=self.eval_transform,
        )

        if stage == "test":
            return

        train_labels = load_labels(self.train_labels_csv)
        all_ids = list(train_labels.index)
        ids_arr = np.array(all_ids).reshape(-1, 1)  # [N, 1] used as X
        y = train_labels.values.astype(int)          # [N, 6]

        train_ids_arr, _, val_ids_arr, _ = iterative_train_test_split(
            ids_arr, y, test_size=self.val_fraction
        )
        train_ids = train_ids_arr.flatten().tolist()
        val_ids = val_ids_arr.flatten().tolist()

        # Append any extra training samples (synthetic data, augmentations…)
        all_train_labels = train_labels
        if self.extra_train_ids:
            if self.extra_labels is None:
                raise ValueError("extra_labels must be provided when extra_train_ids is set")
            all_train_labels = pd.concat([train_labels, self.extra_labels])
            train_ids = train_ids + self.extra_train_ids

        self._train_ds = VinDrPCXRDataset(
            image_ids=train_ids,
            labels=all_train_labels,
            image_dir=self.train_image_dir,
            transform=self.train_transform,
        )
        self._val_ds = VinDrPCXRDataset(
            image_ids=val_ids,
            labels=train_labels,
            image_dir=self.train_image_dir,
            transform=self.eval_transform,
        )


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
