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

    train_image_dir: str = os.path.join(_PROJECT_ROOT, "data", "pcxr_png", "train")
    test_image_dir: str = os.path.join(_PROJECT_ROOT, "data", "pcxr_png", "test")
    train_labels_csv: str = os.path.join(_PROJECT_ROOT, "data", "pcxr_png", "train", "image_labels_train.csv")
    test_labels_csv: str = os.path.join(_PROJECT_ROOT, "data", "pcxr_png", "test", "image_labels_test.csv")
    train_index: str = os.path.join(_PROJECT_ROOT, "data", "pcxr_png", "train", "index.json")
    test_index: str = os.path.join(_PROJECT_ROOT, "data", "pcxr_png", "test", "index.json")
    synthetic_data_dir: str = os.path.join(_PROJECT_ROOT, "data", "synthetic")
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
