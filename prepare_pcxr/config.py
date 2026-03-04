from __future__ import annotations

import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_ENV_FILE = os.path.join(_PROJECT_ROOT, ".env")

class ParseConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Data paths (relative to the prepare_pcxr/ directory)
    pcxr_png_root: str = os.path.join(_PROJECT_ROOT, "data", "pcxr_png")
    pcxr_dicom_root: str = os.path.join(_PROJECT_ROOT, "data", "pcxr_dicom")

    # Worker settings
    num_workers: int = 4
    frontal_only: bool = True

    # Credentials for downloading raw PCXR dicoms
    physio_username: str = "your_username_here"
    physio_password: str = "your_password_here"

cfg = ParseConfig()
