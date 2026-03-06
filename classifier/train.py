"""Unified training entry point for all VinDr-PCXR experiments."""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from classifier.core.config import cfg, tcfg
from classifier.core.datamodule import VinDrPCXRDataModule
from classifier.core.model import VinDrClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train VinDr-PCXR classifier (Baseline or Real + Synthetic)"
    )
    # Core paths
    p.add_argument("--data-dir", default=cfg.data_dir, help="Base directory for data")
    p.add_argument("--ckpt-dir", default=cfg.ckpt_dir, help="Directory to save checkpoints")
    p.add_argument("--run-name", default=tcfg.run_name)
    p.add_argument("--pretrain-setup", default=cfg.pretrain_setup)

    # Synthetic data options
    p.add_argument(
        "--synthetic-classes",
        nargs="+",
        default=[],
        metavar="CLASS",
        help="Optional: per-class synthetic folders to include from data_dir/synthetic/",
    )
    p.add_argument(
        "--filtered", 
        action="store_true",
        help="If using synthetic data, prefer filtered_ labels/paths"
    )

    # Hyperparameters
    p.add_argument("--val-fraction", type=float, default=tcfg.val_fraction)
    p.add_argument("--batch-size", type=int, default=tcfg.batch_size)
    p.add_argument("--num-workers", type=int, default=tcfg.num_workers)
    p.add_argument("--max-epochs", type=int, default=tcfg.max_epochs)
    p.add_argument("--warmup-epochs", type=int, default=tcfg.warmup_epochs)
    p.add_argument("--lr-head", type=float, default=tcfg.lr_head)
    p.add_argument("--lr-backbone", type=float, default=tcfg.lr_backbone)
    p.add_argument("--patience", type=int, default=tcfg.patience)

    # Hardware
    p.add_argument("--accelerator", default=tcfg.accelerator)
    p.add_argument("--devices", default=tcfg.devices)
    p.add_argument("--precision", default=tcfg.precision)

    return p.parse_args()


def train(
    data_dir: str,
    ckpt_dir: str = cfg.ckpt_dir,
    run_name: str = tcfg.run_name,
    synthetic_classes: list[str] | None = None,
    filtered: bool = False,
    val_fraction: float = tcfg.val_fraction,
    batch_size: int = tcfg.batch_size,
    num_workers: int = tcfg.num_workers,
    max_epochs: int = tcfg.max_epochs,
    warmup_epochs: int = tcfg.warmup_epochs,
    lr_head: float = tcfg.lr_head,
    lr_backbone: float = tcfg.lr_backbone,
    patience: int = tcfg.patience,
    accelerator: str = tcfg.accelerator,
    devices: str = tcfg.devices,
    precision: str = tcfg.precision,
    pretrain_setup: str = cfg.pretrain_setup,
) -> None:
    """Unified training pipeline."""
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(42, workers=True)

    # 1. DataModule handles everything: real data split, synthetic loading, MaCheX overrides
    dm = VinDrPCXRDataModule(
        data_dir=data_dir,
        val_fraction=val_fraction,
        batch_size=batch_size,
        num_workers=num_workers,
        synthetic_classes=synthetic_classes,
        use_filtered=filtered,
    )
    dm.setup()

    # 2. Model (using weights computed by DM)
    model = VinDrClassifier(
        warmup_epochs=warmup_epochs,
        lr_head=lr_head,
        lr_backbone=lr_backbone,
        pos_weights=dm.get_pos_weights(),
        pretrain_setup=pretrain_setup,
    )

    # 3. Trainer setup
    logger = CSVLogger(save_dir=cfg.runs_dir, name=run_name, version="")
    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(ckpt_dir, run_name),
        filename="best",
        monitor=tcfg.monitor_metric,
        mode="max",
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor=tcfg.monitor_metric,
        mode="max",
        patience=patience,
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger,
        log_every_n_steps=10,
    )

    # 4. Run
    trainer.fit(model, dm)


def main() -> None:
    args = parse_args()
    train(
        data_dir=args.data_dir,
        ckpt_dir=args.ckpt_dir,
        run_name=args.run_name,
        synthetic_classes=args.synthetic_classes,
        filtered=args.filtered,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        warmup_epochs=args.warmup_epochs,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        patience=args.patience,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        pretrain_setup=args.pretrain_setup,
    )


if __name__ == "__main__":
    main()
