"""Configuration for the baseline experiment."""
from __future__ import annotations

import os
from pydantic_settings import BaseSettings, SettingsConfigDict

_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_ENV_FILE = os.path.join(_PROJECT_ROOT, ".env")

class BaselineConfig(BaseSettings):
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
    num_workers: int = 4

    # Training schedule
    max_epochs: int = 50
    warmup_epochs: int = 3    # Phase 1: head only, backbone frozen
    lr_head: float = 1e-4
    lr_backbone: float = 1e-5  # Phase 2: full fine-tuning

    # Early stopping
    patience: int = 5
    monitor_metric: str = "val/auroc"

    # Hardware
    accelerator: str = "auto"
    devices: str = "auto"
    precision: str = "16"

bcfg = BaselineConfig()
