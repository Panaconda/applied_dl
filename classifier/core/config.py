"""Project-wide configuration for all VinDr-PCXR experiments."""
from __future__ import annotations

import os
from typing import List

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")


class CoreConfig(BaseSettings):
    """Experiment configuration."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    train_image_dir: str = ""
    test_image_dir: str = ""
    train_labels_csv: str = ""
    test_labels_csv: str = ""
    synthetic_data_dir: str = ""
    pretrain_setup: str = "densenet121-res224-chex"

    # Optional: MaCheX index files mapping sequential JPG filenames to image_ids.
    # When set, the datamodule will resolve image paths via the index instead of
    # looking for <image_id>.png files directly in image_dir.
    vindr_pcxr_train_index: str = ""
    vindr_pcxr_test_index: str = ""

    image_size: int = 224

    @computed_field
    @property
    def experiments_dir(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @computed_field
    @property
    def project_root(self) -> str:
        return os.path.dirname(self.experiments_dir)

    @computed_field
    @property
    def runs_dir(self) -> str:
        return os.path.join(self.experiments_dir, "runs")

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
