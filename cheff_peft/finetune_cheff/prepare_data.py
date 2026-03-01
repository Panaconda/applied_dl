"""Prepare VinDr-PCXR data for CheFF LoRA fine-tuning."""
from __future__ import annotations

import json
import os
from pathlib import Path

from config import ftcfg


def main() -> None:
    # ------------------------------------------------------------------ load
    src_path = ftcfg.vindr_pcxr_train_index
    if not src_path or not os.path.exists(src_path):
        print(
            f"Error: VINDR_PCXR_TRAIN_INDEX not set or file not found: {src_path!r}\n"
            "Check cheff_peft/.env."
        )
        return

    with open(src_path) as f:
        src_index: dict = json.load(f)

    print(f"Loaded {len(src_index):,} entries from {src_path}")

    img_dir = ftcfg.train_image_dir
    if not img_dir or not os.path.isdir(img_dir):
        print(
            f"Error: TRAIN_IMAGE_DIR not set or directory not found: {img_dir!r}\n"
            "Check cheff_peft/.env."
        )
        return

    # --------------------------------------------------------------- convert
    target_index: dict = {}
    skipped = 0

    for key, meta in src_index.items():
        report: str = meta.get("report", "")

        # Resolve path from the local Train directory, not from the stored absolute path.
        img_path = os.path.join(img_dir, f"{key}.jpg")
        if not os.path.exists(img_path):
            skipped += 1
            continue
        if not report:
            skipped += 1
            continue

        target_index[key] = {"path": img_path, "report": report}

    print(f"  Kept:    {len(target_index):,}")
    print(f"  Skipped: {skipped}  (missing path or empty report)")

    # ----------------------------------------------------------------- write
    machex_output_dir = ftcfg.machex_output_dir
    if not machex_output_dir:
        print(
            "Error: MACHEX_OUTPUT_DIR not set.\n"
            "Check cheff_peft/.env."
        )
        return

    out_dir = Path(machex_output_dir) / "mimic"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "index.json"

    with open(out_path, "w") as f:
        json.dump(target_index, f)

    print(f"\nWritten: {out_path}")
    print("Done — run `python -m finetune_cheff.train` to start fine-tuning.")


if __name__ == "__main__":
    main()
