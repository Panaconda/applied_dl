"""Configuration for CheFF LoRA fine-tuning on VinDr-PCXR."""
from __future__ import annotations

from core.config import CoreConfig


class FinetuneCheffConfig(CoreConfig):
    """Extends CoreConfig with CheFF fine-tuning paths and LoRA hyper-parameters.

    All fields can be set via the shared ``.env`` file or environment variables.
    """

    # --- extra input path ------------------------------------------------
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


ftcfg = FinetuneCheffConfig()
