"""Configuration for the augmented experiment."""
from __future__ import annotations

import os

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")


class AugmentedConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    run_name: str = "augmented"

    val_fraction: float = 0.10
    batch_size: int = 32
    num_workers: int = 4

    max_epochs: int = 50
    warmup_epochs: int = 3
    lr_head: float = 1e-4
    lr_backbone: float = 1e-5

    patience: int = 5
    monitor_metric: str = "val/auroc"

    accelerator: str = "auto"
    devices: str = "auto"
    precision: str = "16-mixed"


ac = AugmentedConfig()
