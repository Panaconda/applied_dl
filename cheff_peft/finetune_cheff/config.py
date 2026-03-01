from __future__ import annotations

import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


_ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")


class FinetuneCheffConfig(BaseSettings):
    """Configuration for CheFF fine-tuning, independent of the classifier core.

    All fields can be set via the shared ``.env`` file or environment variables.
    """
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Shared VinDr-PCXR paths (from .env) -----------------------------
    train_image_dir: str = ""
    train_labels_csv: str = ""
    vindr_pcxr_train_index: str = ""  # MaCheX index for training

    # --- CheFF-specific input paths --------------------------------------
    train_annotations_csv: str = ""   # annotations_train.csv (finding bboxes)

    # --- MaCheX output ---------------------------------------------------
    machex_output_dir: str = ""       # prepare_data.py writes here

    # --- CheFF model checkpoints -----------------------------------------
    cheff_t2i_ckpt: str = ""          # pre-trained CheFF T2I weights
    cheff_ae_ckpt: str = ""           # pre-trained CheFF autoencoder weights

    # --- LoRA hyper-parameters -------------------------------------------
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_scope: str = "attn"          # all self + cross attention

    # --- training ---------------------------------------------------------
    cheff_batch_size: int = 8
    cheff_learning_rate: float = 5e-5
    cheff_num_workers: int = 4
    cheff_test_size: int = 500
    cheff_max_epochs: int = 15
    seed: int = 42

    run_name: str = "finetune_cheff"

    @property
    def runs_dir(self) -> str:
        """Save runs into the cheff_peft/runs directory."""
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        d = os.path.join(base, "runs")
        os.makedirs(d, exist_ok=True)
        return d


ftcfg = FinetuneCheffConfig()
