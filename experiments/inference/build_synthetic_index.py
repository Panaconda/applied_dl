"""Build a label CSV and path-lookup JSON for the synthetic Pneumonia images."""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

from core.config import cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index synthetic Pneumonia images for use in training"
    )
    parser.add_argument(
        "--synthetic-dir",
        default="../samples/lora/pneumonia",
        help="Directory containing pneumonia_*.png files",
    )
    args = parser.parse_args()

    synth_dir = os.path.abspath(args.synthetic_dir)
    pattern = os.path.join(synth_dir, "pneumonia_*.png")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files matched {pattern}")
        return

    print(f"Found {len(files)} synthetic images in {synth_dir}")

    # Build ID list and path lookup
    image_ids = []
    paths = {}
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]  # e.g. pneumonia_012
        image_id = f"synth_{stem}"                           # e.g. synth_pneumonia_012
        image_ids.append(image_id)
        paths[image_id] = path

    # Build labels DataFrame — Pneumonia=1, all others=0
    label_data = {cls: 0 for cls in cfg.viable_classes}
    label_data["Pneumonia"] = 1

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
    print(f"  {len(image_ids)} entries, class Pneumonia=1")


if __name__ == "__main__":
    main()
