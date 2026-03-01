"""Training entry point for the baseline experiment."""
from __future__ import annotations

import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from baseline.config import bc
from baseline.model import VinDrClassifier
from core.config import cfg
from core.datamodule import VinDrPCXRDataModule
from core.dataset import compute_pos_weights, load_labels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the VinDr-PCXR baseline classifier")
    p.add_argument("--train-image-dir", default=cfg.train_image_dir)
    p.add_argument("--test-image-dir", default=cfg.test_image_dir)
    p.add_argument("--train-labels-csv", default=cfg.train_labels_csv)
    p.add_argument("--test-labels-csv", default=cfg.test_labels_csv)
    p.add_argument("--train-index-json", default=cfg.vindr_pcxr_train_index)
    p.add_argument("--test-index-json", default=cfg.vindr_pcxr_test_index)
    p.add_argument("--pretrain-setup", default=cfg.pretrain_setup)
    p.add_argument("--run-name", default=bc.run_name)
    p.add_argument("--val-fraction", type=float, default=bc.val_fraction)
    p.add_argument("--batch-size", type=int, default=bc.batch_size)
    p.add_argument("--num-workers", type=int, default=bc.num_workers)
    p.add_argument("--max-epochs", type=int, default=bc.max_epochs)
    p.add_argument("--warmup-epochs", type=int, default=bc.warmup_epochs)
    p.add_argument("--lr-head", type=float, default=bc.lr_head)
    p.add_argument("--lr-backbone", type=float, default=bc.lr_backbone)
    p.add_argument("--patience", type=int, default=bc.patience)
    p.add_argument("--accelerator", default=bc.accelerator)
    p.add_argument("--devices", default=bc.devices)
    p.add_argument("--precision", default=bc.precision)
    return p.parse_args()


def main() -> None:
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(42, workers=True)
    args = parse_args()

    logger = CSVLogger(save_dir=cfg.runs_dir, name=args.run_name, version="")

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
    )

    pos_weights = compute_pos_weights(load_labels(args.train_labels_csv))

    model = VinDrClassifier(
        warmup_epochs=args.warmup_epochs,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        pos_weights=pos_weights,
        pretrain_setup=args.pretrain_setup,
    )

    checkpoint_cb = ModelCheckpoint(
        filename="best",
        monitor=bc.monitor_metric,
        mode="max",
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor=bc.monitor_metric,
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
