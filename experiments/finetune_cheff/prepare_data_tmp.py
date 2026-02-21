"""Prepare VinDr-PCXR PNGs into MaCheX format for CheFF T2I fine-tuning.

Reads Pre-256 PNGs + label/annotation CSVs, filters out "Other disease"-only
images, generates templated radiology reports, and writes a MaCheX-compatible
index.json consumed by CheFF's ``MimicT2IDataset``.

Usage:
    python -m finetune_cheff.prepare_data
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from finetune_cheff.config import ftcfg


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_report(findings: List[str], active_pathologies: List[str]) -> str:
    """Build a templated radiology report.

    Template follows the VinDrPCXRParser convention from machex:
      "Findings: Frontal radiograph of a child.
       {Evaluation reveals <findings>.} | {No acute radiographic abnormalities …}
       Impressions: <pathologies>."

    Args:
        findings: Fine-grained annotation names from ``annotations_train.csv``.
        active_pathologies: Disease column names where value == 1 from
            ``image_labels_train.csv`` (original 15 classes, *not* collapsed).
    """
    prefix = "Findings: Frontal radiograph of a child."

    real_findings = [f for f in findings if f != "No finding"]
    if real_findings:
        middle = f" Evaluation reveals {', '.join(real_findings)}."
    else:
        middle = " No acute radiographic abnormalities are observed."

    real_pathologies = [p for p in active_pathologies if p != "No finding"]
    pathology_text = ", ".join(real_pathologies) if real_pathologies else "no finding"
    suffix = f" Impressions: {pathology_text}."

    return prefix + middle + suffix


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- validate paths ---------------------------------------------------
    for name, path in [
        ("train_image_dir", ftcfg.train_image_dir),
        ("train_labels_csv", ftcfg.train_labels_csv),
        ("train_annotations_csv", ftcfg.train_annotations_csv),
        ("machex_output_dir", ftcfg.machex_output_dir),
    ]:
        if not path:
            print(f"Error: '{name}' is not set. Add it to .env or set the env var.")
            sys.exit(1)

    # ---- load CSVs --------------------------------------------------------
    print("Loading labels and annotations …")
    labels_df = pd.read_csv(ftcfg.train_labels_csv)
    annots_df = pd.read_csv(ftcfg.train_annotations_csv)

    label_cols = [c for c in labels_df.columns if c not in ("image_id", "rad_ID")]

    # One row per image (handles multi-radiologist annotation via max).
    raw_labels = labels_df.groupby("image_id")[label_cols].max().astype(int)
    print(f"  {len(raw_labels)} unique image-ids in label CSV")

    # ---- collapse rare classes (same logic as core/dataset.py) -------------
    viable = ftcfg.viable_classes  # 6 classes
    rare_cols = [c for c in label_cols if c not in viable]

    collapsed = raw_labels.copy()
    collapsed["Other disease"] = collapsed[["Other disease"] + rare_cols].max(axis=1)
    collapsed = collapsed[viable]

    # ---- filter: discard "Other disease"-only images -----------------------
    non_other = [c for c in viable if c != "Other disease"]
    mask = collapsed[non_other].max(axis=1) > 0
    keep_ids = collapsed[mask].index.tolist()

    n_discarded = (~mask).sum()
    print(f"  Keeping {len(keep_ids)} images, discarding {n_discarded} (only 'Other disease')")

    # ---- pre-aggregate findings from annotations CSV -----------------------
    findings_map: Dict[str, List[str]] = (
        annots_df
        .groupby("image_id")["class_name"]
        .apply(lambda x: sorted(x.unique().tolist()))
        .to_dict()
    )

    # ---- prepare output directory (MaCheX convention) ----------------------
    # MimicT2IDataset expects root/mimic/index.json
    mimic_dir = os.path.join(ftcfg.machex_output_dir, "mimic")
    os.makedirs(mimic_dir, exist_ok=True)

    # ---- process images + build index --------------------------------------
    index: Dict[str, dict] = {}
    skipped = 0

    for counter, image_id in enumerate(tqdm(keep_ids, desc="Preparing images")):
        png_path = os.path.join(ftcfg.train_image_dir, f"{image_id}.png")
        if not os.path.exists(png_path):
            skipped += 1
            continue

        # Six-digit MaCheX key → grouped into 10 k-file subdirectories.
        file_id = str(counter).zfill(6)
        file_dir = file_id[:2]
        out_subdir = os.path.join(mimic_dir, file_dir)
        os.makedirs(out_subdir, exist_ok=True)

        jpg_path = os.path.join(out_subdir, f"{file_id}.jpg")

        # Convert grayscale PNG → RGB JPG (CheFF expects 3-channel input).
        img = Image.open(png_path).convert("RGB")
        img.save(jpg_path, quality=95)

        # --- report ----------------------------------------------------------
        findings = findings_map.get(image_id, [])
        active_pathologies = [c for c in label_cols if raw_labels.loc[image_id, c] == 1]
        report = build_report(findings, active_pathologies)

        # --- 6-class collapsed label vector ----------------------------------
        label_vec = collapsed.loc[image_id].values.tolist()

        index[file_id] = {
            "path": os.path.abspath(jpg_path),
            "key": image_id,
            "report": report,
            "label_vec": label_vec,
        }

    # ---- write index.json --------------------------------------------------
    index_path = os.path.join(mimic_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nDone — {len(index)} images indexed, {skipped} PNGs not found")
    print(f"Index written to {index_path}")

    # ---- sanity sample -----------------------------------------------------
    if index:
        sample_key = next(iter(index))
        sample = index[sample_key]
        print(f"\nSample entry [{sample_key}]:")
        print(f"  path:   {sample['path']}")
        print(f"  report: {sample['report']}")
        print(f"  labels: {sample['label_vec']}")


if __name__ == "__main__":
    main()
