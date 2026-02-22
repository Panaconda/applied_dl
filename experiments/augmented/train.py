"""Training entry point for the augmented experiment."""
from __future__ import annotations

import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from augmented.config import ac
from baseline.model import VinDrClassifier
from core.config import cfg
from core.datamodule import VinDrPCXRDataModule
from core.dataset import build_augmented_transform, compute_pos_weights, load_labels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the VinDr-PCXR augmented classifier")

    p.add_argument("--train-image-dir", default=cfg.train_image_dir)
    p.add_argument("--test-image-dir", default=cfg.test_image_dir)
    p.add_argument("--train-labels-csv", default=cfg.train_labels_csv)
    p.add_argument("--test-labels-csv", default=cfg.test_labels_csv)
    p.add_argument("--train-index-json", default=cfg.vindr_pcxr_train_index)
    p.add_argument("--test-index-json", default=cfg.vindr_pcxr_test_index)

    p.add_argument("--run-name", default=ac.run_name)

    p.add_argument("--val-fraction", type=float, default=ac.val_fraction)
    p.add_argument("--batch-size", type=int, default=ac.batch_size)
    p.add_argument("--num-workers", type=int, default=ac.num_workers)

    p.add_argument("--max-epochs", type=int, default=ac.max_epochs)
    p.add_argument("--warmup-epochs", type=int, default=ac.warmup_epochs)
    p.add_argument("--lr-head", type=float, default=ac.lr_head)
    p.add_argument("--lr-backbone", type=float, default=ac.lr_backbone)
    p.add_argument("--patience", type=int, default=ac.patience)

    p.add_argument("--accelerator", default=ac.accelerator)
    p.add_argument("--devices", default=ac.devices)
    p.add_argument("--precision", default=ac.precision)

    return p.parse_args()


def main() -> None:
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(42, workers=True)
    args = parse_args()

    logger = CSVLogger(save_dir=cfg.runs_dir, name=args.run_name)

    dm = VinDrPCXRDataModule(
        train_image_dir=args.train_image_dir,
        test_image_dir=args.test_image_dir,
        train_labels_csv=args.train_labels_csv,
        test_labels_csv=args.test_labels_csv,
        train_index_json=args.train_index_json,
        test_index_json=args.test_index_json,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transform=build_augmented_transform(),
        # eval_transform defaults to plain XRVTransform
    )

    pos_weights = compute_pos_weights(load_labels(args.train_labels_csv))

    model = VinDrClassifier(
        warmup_epochs=args.warmup_epochs,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        pos_weights=pos_weights,
    )

    checkpoint_cb = ModelCheckpoint(
        filename="best-epoch{epoch:02d}",
        monitor=ac.monitor_metric,
        mode="max",
        save_top_k=1,
        verbose=True,
    )
    early_stop_cb = EarlyStopping(
        monitor=ac.monitor_metric,
        mode="max",
        patience=args.patience,
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger,
        log_every_n_steps=10,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
