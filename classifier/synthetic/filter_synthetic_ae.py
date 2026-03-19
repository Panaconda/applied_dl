
"""AE Filter: discard synthetic images with high reconstruction error from an XRV AutoEncoder."""
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
from classifier.core.config import cfg
from classifier.core.dataset import build_transform


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filter synthetic images using an XRV AutoEncoder reconstruction error"
    )
    p.add_argument(
        "--synthetic-dir", required=True,
        help="Directory containing actual synthetic images"
    )
    p.add_argument(
        "--input-index", default=None,
        help="Path to index JSON file (e.g. synthetic_paths.json or filtered_paths.json). "
             "If None, looks for synthetic_paths.json in --synthetic-dir"
    )
    p.add_argument(
        "--input-labels", default=None,
        help="Path to labels CSV file (e.g. synthetic_labels.csv or filtered_labels.csv). "
             "If None, looks for synthetic_labels.csv in --synthetic-dir"
    )
    p.add_argument(
        "--threshold", type=float, default=4500.0,
        help="Maximum MSE reconstruction error to keep an image. "
             "Note: XRV uses [-1024, 1024] range, so MSE can be large."
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Where to write filtered_paths.json / filtered_labels.csv "
             "(defaults to --synthetic-dir)"
    )
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = args.output_dir or args.synthetic_dir
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------ model
    print("Loading XRV AE model: ResNetAE (weights=101-elastic) …")
    model = xrv.autoencoders.ResNetAE(weights="101-elastic")
    
    model.eval()
    model = model.to(args.device)

    # ------------------------------------------------------------------ index
    index_path = args.input_index or os.path.join(args.synthetic_dir, "synthetic_paths.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            path_index: dict[str, str] = json.load(f)
    else:
        print(f"ERROR: No index found at {index_path}")
        return

    transform = build_transform()   # uses cfg.image_size=224 and XRV normalization

    # ------------------------------------------------------------------ filter
    filtered_paths: dict[str, str] = {}
    errors: list[float] = []

    print(f"Filtering {len(path_index)} images  (threshold={args.threshold}) …")

    with torch.no_grad():
        for image_id, img_rel_path in tqdm(path_index.items()):
            img_path = os.path.join(args.synthetic_dir, img_rel_path)
                
            img = Image.open(img_path)
            # transform converts to [1, 224, 224] in range [-1024, 1024]
            tensor = transform(img).unsqueeze(0).to(args.device)

            # AE forward pass
            reconstruction = model(tensor)  # [1, 1, 224, 224]
            
            # Reconstruction is often a dict for some XRV models or a direct tensor
            if isinstance(reconstruction, dict):
                reconstruction = reconstruction["out"]

            # Compute MSE on the [-1024, 1024] scale
            mse = torch.mean((tensor - reconstruction)**2).item()
            errors.append(mse)

            if mse <= args.threshold:
                filtered_paths[image_id] = img_rel_path

    # ------------------------------------------------------------------ report
    accepted = len(filtered_paths)
    total = len(path_index)
    yield_rate = accepted / total if total > 0 else 0
    stats = {
        "total_evaluated": total,
        "accepted": accepted,
        "discarded": total - accepted,
        "yield_rate": yield_rate,
        "mse_threshold": args.threshold,
        "mse_mean": float(np.mean(errors)) if errors else 0,
        "mse_median": float(np.median(errors)) if errors else 0,
        "mse_min": float(np.min(errors)) if errors else 0,
        "mse_max": float(np.max(errors)) if errors else 0,
    }

    print("-" * 50)
    print(f"Total evaluated : {total}")
    print(f"Accepted (≤{args.threshold}): {accepted}")
    print(f"Discarded       : {total - accepted}")
    print(f"Yield rate      : {yield_rate * 100:.1f}%")
    print(f"MSE  mean/med   : {stats['mse_mean']:.1f} / {stats['mse_median']:.1f}")
    print("-" * 50)

    # ------------------------------------------------------------------ save
    # Metadata
    meta_path = os.path.join(out_dir, "filter_ae_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Written: {meta_path}")

    # Paths JSON
    filtered_json = os.path.join(out_dir, "filtered_paths_combined.json")
    with open(filtered_json, "w") as f:
        json.dump(filtered_paths, f, indent=2)
    print(f"Written: {filtered_json}")

    # Labels CSV
    labels_csv = args.input_labels or os.path.join(args.synthetic_dir, "synthetic_labels.csv")
    if os.path.exists(labels_csv):
        labels_df = pd.read_csv(labels_csv, index_col="image_id")
        # Filter to only include accepted image_ids
        # Some image_ids from path_index might not be in labels_df if it was partially filtered manually
        valid_ids = [iid for iid in filtered_paths.keys() if iid in labels_df.index]
        filtered_labels = labels_df.loc[valid_ids]
        
        filtered_csv_out = os.path.join(out_dir, "filtered_labels_combined.csv")
        filtered_labels.to_csv(filtered_csv_out)
        print(f"Written: {filtered_csv_out}")
    else:
        print(f"WARNING: No labels file found at {labels_csv}, skipping label filtering.")

    if not filtered_paths:
        print("WARNING: no images passed the threshold.")

    # Plot
    if errors:
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, alpha=0.75, color='skyblue', edgecolor='black', label='MSE Distribution')
        plt.axvline(args.threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold ({args.threshold})')
        
        pathol_name = os.path.basename(args.synthetic_dir)
        plt.title(f'MSE Reconstruction Error Distribution - {pathol_name}')
        plt.xlabel('Mean Squared Error')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plot_path = os.path.join(out_dir, f'mse_distribution_{pathol_name}.png')
        plt.savefig(plot_path)
        print(f"MSE distribution plot saved to: {plot_path}")

if __name__ == "__main__":
    main()
