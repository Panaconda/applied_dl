"""Build a label CSV and path-lookup JSON for a synthetic class directory.

The target class is inferred from the directory name (case-insensitive match
against viable_classes).  All *.png files in the directory are indexed.

Usage (from experiments/):
    # Single class
    python -m inference.build_synthetic_index --synthetic-dir ../samples/lora/Pneumonia

    # All 4 classes at once
    for cls in Pneumonia Bronchitis Bronchiolitis Brocho-pneumonia; do
        python -m inference.build_synthetic_index --synthetic-dir ../samples/lora/$cls
    done
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

from classifier.core.config import cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index synthetic images for use in training"
    )
    parser.add_argument(
        "--synthetic-dir",
        required=True,
        help="Directory containing *.png synthetic images. "
             "Directory name must match a viable class name (case-insensitive).",
    )
    args = parser.parse_args()

    synth_dir = os.path.abspath(args.synthetic_dir)
    dirname = os.path.basename(synth_dir)

    # Infer target class from directory name
    class_map = {c.lower(): c for c in cfg.viable_classes}
    target_class = class_map.get(dirname.lower())
    if target_class is None:
        print(f"Error: directory name '{dirname}' does not match any viable class.")
        print(f"  Viable classes: {cfg.viable_classes}")
        return

    pattern = os.path.join(synth_dir, "*.png")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No *.png files found in {synth_dir}")
        return

    print(f"Found {len(files)} synthetic images in {synth_dir}")
    print(f"  Target class: {target_class}")

    # Build ID list and path lookup
    image_ids = []
    paths = {}
    for path in files:
        filename = os.path.basename(path)
        stem = os.path.splitext(filename)[0]  # e.g. 000001
        image_id = f"synth_{dirname.lower()}_{stem}"         # e.g. synth_pneumonia_000001
        image_ids.append(image_id)
        paths[image_id] = filename  # Store only the filename

    # Build labels DataFrame — target class=1, all others=0
    label_data = {cls: 0 for cls in cfg.viable_classes}
    label_data[target_class] = 1

    labels_df = pd.DataFrame(
        [label_data] * len(image_ids),
        index=pd.Index(image_ids, name="image_id"),
    )[cfg.viable_classes]

    # Write outputs
    csv_out = os.path.join(synth_dir, "synthetic_labels.csv")
    json_out = os.path.join(synth_dir, "synthetic_paths.json")

    labels_df.to_csv(csv_out)
    with open(json_out, "w") as f:
        json.dump(paths, f, indent=2)

    print(f"Written: {csv_out}")
    print(f"Written: {json_out}")
    print(f"  {len(image_ids)} entries, {target_class}=1")


if __name__ == "__main__":
    main()
