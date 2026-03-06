"""Lightning DataModule for VinDr-PCXR — shared across all experiment conditions."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from skmultilearn.model_selection import iterative_train_test_split
from torch.utils.data import DataLoader

from classifier.core.dataset import VinDrPCXRDataset, build_transform, load_image_id_map, load_labels


class VinDrPCXRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        val_fraction: float = 0.10,
        batch_size: int = 32,
        num_workers: int = 4,
        train_transform=None,
        eval_transform=None,
        synthetic_classes: Optional[List[str]] = None,
        use_filtered: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.train_image_dir = os.path.join(data_dir, "pcxr_png", "train")
        self.test_image_dir = os.path.join(data_dir, "pcxr_png", "test")
        self.synthetic_base_dir = os.path.join(data_dir, "synthetic")

        self.train_labels_csv = os.path.join(self.train_image_dir, "image_labels_train.csv")
        self.test_labels_csv = os.path.join(self.test_image_dir, "image_labels_test.csv")
        self.train_index_json = os.path.join(self.train_image_dir, "index.json")
        self.test_index_json = os.path.join(self.test_image_dir, "index.json")

        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform or build_transform()
        self.eval_transform = eval_transform or build_transform()

        self.synthetic_classes = synthetic_classes or []
        self.use_filtered = use_filtered

        self._train_ds: Optional[VinDrPCXRDataset] = None
        self._val_ds: Optional[VinDrPCXRDataset] = None
        self._test_ds: Optional[VinDrPCXRDataset] = None


    def _load_synthetic_data(self) -> Tuple[List[str], pd.DataFrame, Dict[str, str]]:
        """Discover and load synthetic indices from the data_dir/synthetic/ subfolders."""
        import json
        all_ids, all_labels, all_paths = [], [], {}

        labels_file = "filtered_labels.csv" if self.use_filtered else "synthetic_labels.csv"
        paths_file = "filtered_paths.json" if self.use_filtered else "synthetic_paths.json"

        for cls_name in self.synthetic_classes:
            cls_dir = os.path.join(self.synthetic_base_dir, cls_name)
            csv_path = os.path.join(cls_dir, labels_file)
            json_path = os.path.join(cls_dir, paths_file)

            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Missing labels file: {csv_path}")
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"Missing paths file: {json_path}")
            labels = pd.read_csv(csv_path, index_col="image_id")
            with open(json_path) as f:
                paths = json.load(f)

            all_ids.extend(list(labels.index))
            all_labels.append(labels)
            # Resolve paths relative to the class directory
            all_paths.update({
                img_id: os.path.join(cls_dir, p) for img_id, p in paths.items()
            })

        combined_labels = pd.concat(all_labels) if all_labels else None
        return all_ids, combined_labels, all_paths


    def setup(self, stage: Optional[str] = None) -> None:

        # Build image-path overrides from MaCheX index files when provided.
        train_overrides = (
            load_image_id_map(self.train_index_json, self.train_image_dir)
            if os.path.exists(self.train_index_json)
            else {}
        )
        test_overrides = (
            load_image_id_map(self.test_index_json, self.test_image_dir)
            if os.path.exists(self.test_index_json)
            else {}
        )

        test_labels = load_labels(self.test_labels_csv)
        if test_overrides:
            test_ids = [i for i in list(test_labels.index) if i in test_overrides]
            test_labels = test_labels.loc[test_ids]

        self._test_ds = VinDrPCXRDataset(
            image_ids=list(test_labels.index),
            labels=test_labels,
            image_dir=self.test_image_dir,
            transform=self.eval_transform,
            image_path_overrides=test_overrides,
        )

        if stage == "test":
            return

        train_labels = load_labels(self.train_labels_csv)
        all_ids = list(train_labels.index)

        # If using MaCheX index, filter IDs to those present in the index.
        if train_overrides:
            all_ids = [i for i in all_ids if i in train_overrides]
            train_labels = train_labels.loc[all_ids]

        ids_arr = np.array(all_ids).reshape(-1, 1)  # [N, 1] used as X
        y = train_labels.values.astype(int)          # [N, 6]

        train_ids_arr, _, val_ids_arr, _ = iterative_train_test_split(
            ids_arr, y, test_size=self.val_fraction
        )
        train_ids = train_ids_arr.flatten().tolist()
        val_ids = val_ids_arr.flatten().tolist()

        # Handle synthetic data internally
        extra_ids, extra_labels, extra_paths = self._load_synthetic_data()
        
        all_train_labels = train_labels
        if extra_ids:
            all_train_labels = pd.concat([train_labels, extra_labels])
            train_ids = train_ids + extra_ids

        # Merge base image overrides (MaCheX) with synthetic-image paths.
        train_path_overrides = {**train_overrides, **extra_paths}

        self._train_ds = VinDrPCXRDataset(
            image_ids=train_ids,
            labels=all_train_labels,
            image_dir=self.train_image_dir,
            transform=self.train_transform,
            image_path_overrides=train_path_overrides,
        )
        self._val_ds = VinDrPCXRDataset(
            image_ids=val_ids,
            labels=train_labels,
            image_dir=self.train_image_dir,
            transform=self.eval_transform,
            image_path_overrides=train_overrides,
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

    def get_pos_weights(self) -> torch.Tensor:
        """Compute positive class weights from the training set labels.
        
        Must be called AFTER setup().
        """
        if self._train_ds is None:
            raise RuntimeError("DataModule.setup() must be called before get_pos_weights()")
        from classifier.core.dataset import compute_pos_weights
        return compute_pos_weights(self._train_ds.labels)

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
