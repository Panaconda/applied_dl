"""Standalone evaluation of a saved checkpoint on the VinDr-PCXR test set."""
from __future__ import annotations

import argparse
import json
import os

import torch

from baseline.config import bc
from baseline.model import VinDrClassifier
from core.config import cfg
from core.datamodule import VinDrPCXRDataModule
from core.metrics import compute_metrics, format_metrics_table, mean_auroc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a VinDr-PCXR classifier checkpoint")
    p.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    p.add_argument("--test-image-dir", default=cfg.test_image_dir)
    p.add_argument("--train-labels-csv", default=cfg.train_labels_csv)
    p.add_argument("--test-labels-csv", default=cfg.test_labels_csv)
    p.add_argument("--test-index-json", default=cfg.vindr_pcxr_test_index)
    p.add_argument("--batch-size", type=int, default=bc.batch_size)
    p.add_argument("--num-workers", type=int, default=bc.num_workers)
    p.add_argument("--output", default=None, help="Optional path to save metrics as JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model = VinDrClassifier.load_from_checkpoint(args.checkpoint, strict=False)
    model.eval()

    # train_image_dir is unused when stage="test"; pass test dir as placeholder
    dm = VinDrPCXRDataModule(
        train_image_dir=args.test_image_dir,
        test_image_dir=args.test_image_dir,
        train_labels_csv=args.train_labels_csv,
        test_labels_csv=args.test_labels_csv,
        test_index_json=args.test_index_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup(stage="test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    all_probs, all_targets = [], []
    with torch.no_grad():
        for imgs, labels in dm.test_dataloader():
            probs = torch.sigmoid(model(imgs.to(device))).cpu()
            all_probs.append(probs)
            all_targets.append(labels)

    probs_np = torch.cat(all_probs).numpy()
    targets_np = torch.cat(all_targets).numpy()

    metrics = compute_metrics(targets_np, probs_np)
    print("\n" + format_metrics_table(metrics))
    print(f"\nMean AUC-ROC: {mean_auroc(metrics):.4f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {args.output}")


if __name__ == "__main__":
    main()
