"""Launch CheFF LoRA fine-tuning on prepared VinDr-PCXR data.

Directly instantiates CheffLDMT2I, applies LoRA, builds a DataLoader from
the prepared MaCheX index.json, and trains with a minimal PL Trainer.
No dependency on the legacy 01_train_ldm.py script.

Usage:
    cd cheff_peft
    python -m finetune_cheff.train
"""
from __future__ import annotations

import os
import sys

# Add cheff source to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CHEFF_PEFT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.join(_CHEFF_PEFT_ROOT, "cheff"))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode

from finetune_cheff.config import ftcfg


# ---------------------------------------------------------------------------
# Gradient-checkpointing patch (must run BEFORE CheFF model imports)
# ---------------------------------------------------------------------------
def _patch_gradient_checkpointing() -> None:
    """Replace OpenAI's legacy checkpoint with use_reentrant=False.

    Without this, backprop on frozen layers with LoRA crashes.
    """
    def _wrapper(func, inputs, params, flag):
        return torch.utils.checkpoint.checkpoint(func, *inputs, use_reentrant=False)

    import cheff.ldm.modules.diffusionmodules.util as cheff_util
    import cheff.ldm.modules.attention as cheff_attn
    import cheff.ldm.modules.diffusionmodules.openaimodel as cheff_openai

    cheff_util.checkpoint = _wrapper
    cheff_attn.checkpoint = _wrapper
    if hasattr(cheff_openai, "checkpoint"):
        cheff_openai.checkpoint = _wrapper
    print("Gradient-checkpointing patched (use_reentrant=False)")


_patch_gradient_checkpointing()

from cheff.ldm.inference import CheffLDMT2I                     # noqa: E402
from cheff.machex import MimicT2IDataset                        # noqa: E402
from cheff.peft_modules.inject_lora import export_lora_weights   # noqa: E402


# ---------------------------------------------------------------------------
# LoRA helpers
# ---------------------------------------------------------------------------
def apply_lora(model: pl.LightningModule) -> pl.LightningModule:
    """Wrap UNet attention layers with LoRA adapters."""
    target_modules = ".*attn.*(to_q|to_k|to_v|to_out.0)"

    peft_config = LoraConfig(
        r=ftcfg.lora_rank,
        lora_alpha=ftcfg.lora_alpha,
        target_modules=target_modules,
        lora_dropout=ftcfg.lora_dropout,
        bias="none",
    )

    unet = model.model.diffusion_model
    if not hasattr(unet, "config"):
        class _MockConfig:
            def to_dict(self):
                return {}
        unet.config = _MockConfig()

    model.model.diffusion_model = get_peft_model(unet, peft_config)
    model.model.diffusion_model.print_trainable_parameters()

    return model


def freeze_non_lora(model: pl.LightningModule) -> None:
    """Freeze everything except LoRA parameters."""
    # Freeze first-stage (autoencoder) — already frozen by CheFF, but be safe
    for p in model.first_stage_model.parameters():
        p.requires_grad = False

    # Freeze BERT conditioner
    if hasattr(model, "cond_stage_model") and model.cond_stage_model is not None:
        for p in model.cond_stage_model.parameters():
            p.requires_grad = False

    # Freeze all UNet params, then unfreeze LoRA
    for name, p in model.model.diffusion_model.named_parameters():
        p.requires_grad = "lora_" in name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    for name, path in [
        ("cheff_t2i_ckpt", ftcfg.cheff_t2i_ckpt),
        ("cheff_ae_ckpt", ftcfg.cheff_ae_ckpt),
    ]:
        if not path or not os.path.exists(path):
            print(f"Error: '{name}' not set or file not found ({path!r}).")
            sys.exit(1)

    # ---- seed --------------------------------------------------------------
    pl.seed_everything(ftcfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    # ---- load model --------------------------------------------------------
    print("Loading CheFF T2I model …")
    wrapper = CheffLDMT2I(
        model_path=ftcfg.cheff_t2i_ckpt,
        ae_path=ftcfg.cheff_ae_ckpt,
        device="cpu",  # load on CPU first, Trainer moves to GPU
    )
    model = wrapper.model  # LatentDiffusion (LightningModule)

    # ---- apply LoRA --------------------------------------------------------
    print("Applying LoRA …")
    model = apply_lora(model)
    freeze_non_lora(model)

    # Disable EMA — it was built before LoRA and lacks adapter keys.
    # Not needed for adapter fine-tuning anyway.
    model.use_ema = False

    # Don't include BERT params in the optimizer (we freeze them).
    model.cond_stage_trainable = False

    # PL 1.9.5 does not pass dataloader_idx for single-dataloader setups,
    # but LatentDiffusion.on_train_batch_start requires it.  Make it optional.
    _orig_otbs = model.on_train_batch_start.__func__
    model.on_train_batch_start = lambda batch, batch_idx, dataloader_idx=0: \
        _orig_otbs(model, batch, batch_idx, dataloader_idx)

    # Set learning rate (used by LatentDiffusion.configure_optimizers)
    model.learning_rate = ftcfg.cheff_learning_rate

    # ---- data --------------------------------------------------------------
    transforms = transforms.Compose([
        Resize(256, interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    dataset = MimicT2IDataset(ftcfg.machex_output_dir, transforms)
    print(f"Dataset: {len(dataset)} images")

    train_size = len(dataset) - ftcfg.cheff_test_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, ftcfg.cheff_test_size],
        generator=torch.Generator().manual_seed(ftcfg.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=ftcfg.cheff_batch_size,
        shuffle=True,
        num_workers=ftcfg.cheff_num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=ftcfg.cheff_batch_size,
        shuffle=False,
        num_workers=ftcfg.cheff_num_workers,
        pin_memory=True,
    )

    # ---- trainer -----------------------------------------------------------
    log_dir = os.path.join(ftcfg.runs_dir, ftcfg.run_name)
    logger = CSVLogger(save_dir=log_dir, name="logs")

    checkpoint_cb = ModelCheckpoint(
        monitor="val/loss_simple_ema",
        mode="min",
        save_top_k=1,
        save_last=False,
        filename="best",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=ftcfg.cheff_max_epochs,
        logger=logger,
        callbacks=[checkpoint_cb],
        precision=32,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
    )

    # ---- banner ------------------------------------------------------------
    print("=" * 60)
    print("CheFF LoRA Fine-Tuning")
    print("=" * 60)
    print(f"  Rank:       {ftcfg.lora_rank}")
    print(f"  Scope:      {ftcfg.lora_scope}")
    print(f"  LR:         {ftcfg.cheff_learning_rate}")
    print(f"  Batch size: {ftcfg.cheff_batch_size}")
    print(f"  Max epochs: {ftcfg.cheff_max_epochs}")
    print(f"  Train:      {len(train_ds)} | Val: {len(val_ds)}")
    print(f"  Data:       {ftcfg.machex_output_dir}")
    print(f"  Log dir:    {log_dir}")
    print("=" * 60)

    # ---- train -------------------------------------------------------------
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # ---- export LoRA adapter -----------------------------------------------
    export_lora_weights(model, log_dir)
    adapter_dir = os.path.join(log_dir, "lora_adapter")
    print(f"\nLoRA adapter saved to: {adapter_dir}")


if __name__ == "__main__":
    main()
