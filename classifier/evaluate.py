"""Standalone evaluation of a saved checkpoint on the VinDr-PCXR test set."""
from __future__ import annotations

import argparse
import json
import os

import torch

from classifier.core.config import cfg, tcfg
from classifier.core.datamodule import VinDrPCXRDataModule
from classifier.core.metrics import compute_metrics, format_metrics_table, mean_auroc
from classifier.core.model import VinDrClassifier

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a VinDr-PCXR classifier checkpoint")
    p.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    p.add_argument("--data-dir", default=cfg.data_dir, help="Base directory for data")
    p.add_argument("--batch-size", type=int, default=tcfg.batch_size)
    p.add_argument("--num-workers", type=int, default=tcfg.num_workers)
    p.add_argument("--output", default=None, help="Optional path to save metrics as JSON")
    return p.parse_args()


def evaluate(
    checkpoint_path: str,
    data_dir: str,
    batch_size: int = tcfg.batch_size,
    num_workers: int = tcfg.num_workers,
    output_path: str | None = None,
) -> None:
    """Evaluate a saved model on the test set."""
    model = VinDrClassifier.load_from_checkpoint(checkpoint_path, strict=False)
    model.eval()

    dm = VinDrPCXRDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
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

    metrics = compute_metrics(targets_np, probs_np, class_names=model.class_names)
    print("\n" + format_metrics_table(metrics))
    print(f"\nMean AUC-ROC: {mean_auroc(metrics):.4f}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {output_path}")


def main() -> None:
    args = parse_args()
    evaluate(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
