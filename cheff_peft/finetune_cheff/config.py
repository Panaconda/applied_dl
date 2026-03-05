from __future__ import annotations

import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

_FINETUNE_DIR = os.path.dirname(os.path.abspath(__file__))
_CHEFF_PEFT_ROOT = os.path.dirname(_FINETUNE_DIR)
_PROJECT_ROOT = os.path.dirname(_CHEFF_PEFT_ROOT)
_ENV_FILE = os.path.join(_PROJECT_ROOT, ".env")

class FinetuneCheffConfig(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Shared VinDr-PCXR paths (from .env) -----------------------------
    train_image_dir: str = os.path.join(_PROJECT_ROOT, "data", "pcxr_png", "train")
    train_labels_csv: str = os.path.join(_PROJECT_ROOT, "data", "pcxr_png", "train", "image_labels_train.csv")
    train_index: str = os.path.join(_PROJECT_ROOT, "data", "pcxr_png", "train", "index.json")
    train_annotations_csv: str = os.path.join(_PROJECT_ROOT, "data", "pcxr_png", "train", "annotations.csv")

    # Path to prepared MaCheX dataset (where index.json is located)
    machex_output_dir: str = os.path.join(_PROJECT_ROOT, "data", "pcxr_png", "train")

    # --- CheFF model checkpoints -----------------------------------------
    cheff_t2i_ckpt: str = os.path.join(_CHEFF_PEFT_ROOT, "checkpoints", "cheff_t2i_ckpt.pt")          # pre-trained CheFF T2I weights
    cheff_ae_ckpt: str = os.path.join(_CHEFF_PEFT_ROOT, "checkpoints", "cheff_ae_ckpt.pt")           # pre-trained CheFF autoencoder weights

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
        d = os.path.join(_FINETUNE_DIR, "runs")
        os.makedirs(d, exist_ok=True)
        return d

ftcfg = FinetuneCheffConfig()
