"""Project-wide configuration for all VinDr-PCXR experiments."""
from __future__ import annotations

import os
from typing import List

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

_CLASSIFIER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_CLASSIFIER_DIR)
_RUNS_DIR = os.path.join(_CLASSIFIER_DIR, "runs")
_ENV_FILE = os.path.join(_PROJECT_ROOT, ".env")

class CoreConfig(BaseSettings):
    """Experiment configuration."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: str = os.path.join(_PROJECT_ROOT, "data")
    runs_dir: str = _RUNS_DIR
    CKPT_DIR: str = os.path.join(_CLASSIFIER_DIR, "checkpoints")
    
    @computed_field
    @property
    def train_image_dir(self) -> str:
        return os.path.join(self.data_dir, "pcxr_png", "train")

    @computed_field
    @property
    def test_image_dir(self) -> str:
        return os.path.join(self.data_dir, "pcxr_png", "test")

    @computed_field
    @property
    def train_labels_csv(self) -> str:
        return os.path.join(self.train_image_dir, "image_labels_train.csv")

    @computed_field
    @property
    def test_labels_csv(self) -> str:
        return os.path.join(self.test_image_dir, "image_labels_test.csv")

    @computed_field
    @property
    def train_index(self) -> str:
        return os.path.join(self.train_image_dir, "index.json")

    @computed_field
    @property
    def test_index(self) -> str:
        return os.path.join(self.test_image_dir, "index.json")
    
    @computed_field
    @property
    def synthetic_data_dir(self) -> str:
        return os.path.join(self.data_dir, "synthetic")

    pretrain_setup: str = "densenet121-res224-chex"

    image_size: int = 224

    @computed_field
    @property
    def classifier_dir(self) -> str:
        return _CLASSIFIER_DIR

    @computed_field
    @property
    def project_root(self) -> str:
        return _PROJECT_ROOT

    @computed_field
    @property
    def runs_dir(self) -> str:
        return _RUNS_DIR

    @computed_field
    @property
    def viable_classes(self) -> List[str]:
        return [
            "No finding",
            "Bronchitis",
            "Brocho-pneumonia",
            "Bronchiolitis",
            "Pneumonia",
            "Other disease",
        ]

    @computed_field
    @property
    def num_classes(self) -> int:
        return len(self.viable_classes)

cfg = CoreConfig()
