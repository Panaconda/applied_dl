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
    """General project configuration."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: str = os.path.join(_PROJECT_ROOT, "data")
    
    @computed_field
    @property
    def runs_dir(self) -> str:
        return _RUNS_DIR

    @computed_field
    @property
    def ckpt_dir(self) -> str:
        return os.path.join(_PROJECT_ROOT, "checkpoints")
    
    @computed_field
    @property
    def synthetic_data_dir(self) -> str:
        return os.path.join(self.data_dir, "synthetic")

    pretrain_setup: str = "densenet121-res224-chex"
    image_size: int = 224

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

class TrainConfig(BaseSettings):
    """Default training hyperparameters."""
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Run identity
    run_name: str = "baseline"

    # Data split
    val_fraction: float = 0.10

    # DataLoader
    batch_size: int = 32
    num_workers: int = 8

    # Training schedule
    max_epochs: int = 50
    warmup_epochs: int = 3    # Phase 1: head only, backbone frozen
    lr_head: float = 1e-4
    lr_backbone: float = 1e-5  # Phase 2: full fine-tuning

    # Early stopping
    patience: int = 10
    monitor_metric: str = "val/auroc"

    # Hardware
    accelerator: str = "auto"
    devices: str = "auto"
    precision: str = "16-mixed"

cfg = CoreConfig()
tcfg = TrainConfig()
