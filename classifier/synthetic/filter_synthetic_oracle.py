"""Oracle Filter: discard synthetic images where the baseline model is not confident about the target pathology."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torchxrayvision as xrv
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from classifier.core.model import VinDrClassifier
from classifier.core.config import cfg
from classifier.core.dataset import build_transform


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Oracle-filter synthetic images using the baseline classifier"
    )
    p.add_argument(
        "--ckpt", required=True,
        help="Path to baseline VinDrClassifier .ckpt"
    )
    p.add_argument(
        "--index", required=True,
        help="synthetic_paths.json produced by build_synthetic_index"
    )
    p.add_argument(
        "--target", default="Pneumonia",
        choices=cfg.viable_classes,
        help="Class to filter on"
    )
    p.add_argument(
        "--threshold", type=float, default=0.70,
        help="Minimum P(target) to keep an image"
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Where to write filtered_paths.json / filtered_labels.csv "
             "(defaults to directory of --index)"
    )
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    target_idx = cfg.viable_classes.index(args.target)
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.index))
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------ model
    print(f"Loading baseline oracle from {args.ckpt} …")
    model = VinDrClassifier.load_from_checkpoint(args.ckpt, strict=False)
    model.eval()
    model = model.to(args.device)

    # ------------------------------------------------------------------ index
    with open(args.index) as f:
        path_index: dict[str, str] = json.load(f)   # {image_id: filename}
    
    index_dir = os.path.dirname(os.path.abspath(args.index))
    transform = build_transform()   # same XRVTransform used in training

    # ------------------------------------------------------------------ filter
    filtered_paths: dict[str, str] = {}
    scores: list[float] = []

    print(f"Filtering {len(path_index)} images  (target={args.target}, "
          f"threshold={args.threshold}) …")

    with torch.no_grad():
        for image_id, img_rel_path in tqdm(path_index.items()):
            img_path = os.path.join(index_dir, img_rel_path)
            img = Image.open(img_path)
            tensor = transform(img).unsqueeze(0).to(args.device)  # [1,1,224,224]

            logits = model(tensor)                         # [1, 6]
            prob = torch.sigmoid(logits)[0, target_idx].item()
            scores.append(prob)

            if prob >= args.threshold:
                filtered_paths[image_id] = img_rel_path  # Store relative path/filename

    # ------------------------------------------------------------------ report
    accepted = len(filtered_paths)
    total = len(path_index)
    yield_rate = accepted / total if total > 0 else 0
    stats = {
        "target_pathology": args.target,
        "threshold": args.threshold,
        "total_evaluated": total,
        "accepted": accepted,
        "discarded": total - accepted,
        "yield_rate": yield_rate,
        "score_mean": float(np.mean(scores)) if scores else 0,
        "score_median": float(np.median(scores)) if scores else 0,
    }

    print("-" * 50)
    print(f"Total evaluated : {total}")
    print(f"Accepted (≥{args.threshold:.0%}): {accepted}")
    print(f"Discarded       : {total - accepted}")
    print(f"Yield rate      : {yield_rate * 100:.1f}%")
    print(f"Score  mean/med : {stats['score_mean']:.3f} / {stats['score_median']:.3f}")
    print("-" * 50)

    # ------------------------------------------------------------------ save
    # Metadata
    meta_path = os.path.join(out_dir, "filter_oracle_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Written: {meta_path}")

    filtered_json = os.path.join(out_dir, "filtered_paths_orcale.json")
    with open(filtered_json, "w") as f:
        json.dump(filtered_paths, f, indent=2)
    print(f"Written: {filtered_json}")

    # Build matching labels DataFrame — target class = 1, all others = 0
    label_row = {cls: 0 for cls in cfg.viable_classes}
    label_row[args.target] = 1
    labels_df = pd.DataFrame(
        [label_row] * accepted,
        index=pd.Index(list(filtered_paths.keys()), name="image_id"),
        columns=cfg.viable_classes,
    )

    filtered_csv = os.path.join(out_dir, "filtered_labels_orcale.csv")
    labels_df.to_csv(filtered_csv)
    print(f"Written: {filtered_csv}")

    # Plot
    if scores:
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=50, alpha=0.75, color='lightgreen', edgecolor='black', label='Oracle Scores')
        plt.axvline(args.threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold ({args.threshold})')
        
        plt.title(f'Oracle Confidence Distribution - {args.target}')
        plt.xlabel(f'P({args.target})')
        plt.ylabel('Frequency')
        plt.xlim(0, 1)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plot_path = os.path.join(out_dir, f'oracle_distribution_{args.target}.png')
        plt.savefig(plot_path)
        print(f"Oracle distribution plot saved to: {plot_path}")

    if not filtered_paths:
        print("WARNING: no images passed the threshold — filtered_labels.csv is empty.")



if __name__ == "__main__":
    main()
