from __future__ import annotations

import os
import sys
from pydantic_settings import BaseSettings, SettingsConfigDict

_CHEFF_PEFT_ROOT = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CHEFF_PEFT_ROOT)

# Make the bundled cheff source importable
sys.path.insert(0, os.path.join(_CHEFF_PEFT_ROOT, "cheff"))

class FinetuneConfig(BaseSettings):

    model_config = SettingsConfigDict(
        extra="ignore",
    )

    # VinDr-PCXR paths
    data_dir: str = os.path.join(_PROJECT_ROOT, "data", "pcxr_png")
    train_image_dir: str = os.path.join(data_dir, "train")
    train_labels_csv: str = os.path.join(data_dir, "train", "image_labels_train.csv")
    train_index: str = os.path.join(data_dir, "train", "index.json")
    train_annotations_csv: str = os.path.join(data_dir, "train", "annotations.csv")

    # CheFF model checkpoints
    checkpoint_dir: str = os.path.join(_CHEFF_PEFT_ROOT, "checkpoints")
    cheff_t2i_ckpt: str = os.path.join(checkpoint_dir, "cheff_t2i_ckpt.pt")
    cheff_ae_ckpt: str = os.path.join(checkpoint_dir, "cheff_ae_ckpt.pt")

    # LoRA hyper-parameters
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_scope: str = "attn" # all self + cross attention

    # Training
    cheff_batch_size: int = 8
    cheff_learning_rate: float = 5e-5
    cheff_num_workers: int = 4
    cheff_test_size: int = 500
    cheff_max_epochs: int = 15
    seed: int = 42

    run_name: str = "finetune"

    @property
    def runs_dir(self) -> str:
        d = os.path.join(_CHEFF_PEFT_ROOT, "runs")
        os.makedirs(d, exist_ok=True)
        return d

ftcfg = FinetuneConfig()
