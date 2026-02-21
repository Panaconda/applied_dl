"""Training entry point for Condition C: real data + synthetic Pneumonia."""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from baseline.config import bc
from baseline.model import VinDrClassifier
from core.config import cfg
from core.datamodule import VinDrPCXRDataModule
from core.dataset import compute_pos_weights, load_labels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train VinDr-PCXR classifier with real + synthetic Pneumonia data"
    )
    # Paths
    p.add_argument("--train-image-dir", default=cfg.train_image_dir)
    p.add_argument("--test-image-dir", default=cfg.test_image_dir)
    p.add_argument("--train-labels-csv", default=cfg.train_labels_csv)
    p.add_argument("--test-labels-csv", default=cfg.test_labels_csv)
    p.add_argument(
        "--synthetic-dir",
        default="../samples/lora/pneumonia",
        help="Directory containing the synthetic index files",
    )
    p.add_argument(
        "--labels-csv", default=None,
        help="Override labels CSV filename inside --synthetic-dir "
             "(default: filtered_labels.csv, fallback: synthetic_labels.csv)",
    )
    p.add_argument(
        "--paths-json", default=None,
        help="Override paths JSON filename inside --synthetic-dir "
             "(default: filtered_paths.json, fallback: synthetic_paths.json)",
    )

    # Run identity
    p.add_argument("--run-name", default="synthetic_pneumonia_filtered")

    # Data
    p.add_argument("--val-fraction", type=float, default=bc.val_fraction)
    p.add_argument("--batch-size", type=int, default=bc.batch_size)
    p.add_argument("--num-workers", type=int, default=bc.num_workers)

    # Training schedule
    p.add_argument("--max-epochs", type=int, default=bc.max_epochs)
    p.add_argument("--warmup-epochs", type=int, default=bc.warmup_epochs)
    p.add_argument("--lr-head", type=float, default=bc.lr_head)
    p.add_argument("--lr-backbone", type=float, default=bc.lr_backbone)
    p.add_argument("--patience", type=int, default=bc.patience)

    # Hardware
    p.add_argument("--accelerator", default=bc.accelerator)
    p.add_argument("--devices", default=bc.devices)
    p.add_argument("--precision", default=bc.precision)

    return p.parse_args()


def _resolve(synth_dir: str, override: str | None, *candidates: str) -> str:
    """Return the first existing candidate path, or raise FileNotFoundError."""
    names = [override] if override else list(candidates)
    for name in names:
        path = os.path.join(synth_dir, name)
        if os.path.exists(path):
            return path
    tried = ", ".join(names)
    raise FileNotFoundError(
        f"None of [{tried}] found in {synth_dir}. "
        "Run `python -m inference.build_synthetic_index` (and optionally "
        "`python -m inference.filter_synthetic`) first."
    )


def load_synthetic_index(
    synthetic_dir: str,
    labels_csv_override: str | None = None,
    paths_json_override: str | None = None,
):
    """Load the pre-built labels CSV and paths JSON from *synthetic_dir*.

    Prefers filtered_* files produced by filter_synthetic.py; falls back to
    the raw synthetic_* files from build_synthetic_index.py.
    """
    synth_dir = os.path.abspath(synthetic_dir)

    csv_path = _resolve(synth_dir, labels_csv_override,
                        "filtered_labels.csv", "synthetic_labels.csv")
    json_path = _resolve(synth_dir, paths_json_override,
                         "filtered_paths.json", "synthetic_paths.json")

    labels = pd.read_csv(csv_path, index_col="image_id")
    with open(json_path) as f:
        paths = json.load(f)

    image_ids = list(labels.index)
    print(f"Loaded {len(image_ids)} synthetic images")
    print(f"  labels : {os.path.basename(csv_path)}")
    print(f"  paths  : {os.path.basename(json_path)}")
    return image_ids, labels, paths


def main() -> None:
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(42, workers=True)
    args = parse_args()

    extra_ids, extra_labels, extra_paths = load_synthetic_index(
        args.synthetic_dir,
        labels_csv_override=args.labels_csv,
        paths_json_override=args.paths_json,
    )
    print(f"  Pneumonia positives: {extra_labels['Pneumonia'].sum()}")

    logger = CSVLogger(save_dir=cfg.runs_dir, name=args.run_name)

    dm = VinDrPCXRDataModule(
        train_image_dir=args.train_image_dir,
        test_image_dir=args.test_image_dir,
        train_labels_csv=args.train_labels_csv,
        test_labels_csv=args.test_labels_csv,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        extra_train_ids=extra_ids,
        extra_labels=extra_labels,
        extra_image_paths=extra_paths,
    )

    # Recalculate pos-weights over the combined real + synthetic training labels
    real_labels = load_labels(args.train_labels_csv)
    combined_labels = pd.concat([real_labels, extra_labels])
    pos_weights = compute_pos_weights(combined_labels)

    model = VinDrClassifier(
        warmup_epochs=args.warmup_epochs,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        pos_weights=pos_weights,
    )

    checkpoint_cb = ModelCheckpoint(
        filename="best-epoch{epoch:02d}",
        monitor=bc.monitor_metric,
        mode="max",
        save_top_k=1,
        verbose=True,
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
