"""Launch CheFF LoRA fine-tuning on prepared VinDr-PCXR data.

Constructs the CLI command for ``cheff/scripts/01_train_ldm.py`` with the
LoRA config YAML and path overrides, then runs it as a subprocess from the
CheFF repo directory.

Usage:
    python -m finetune_cheff.train
"""
from __future__ import annotations

import os
import subprocess
import sys

from finetune_cheff.config import ftcfg


def main() -> None:
    # ---- validate -----------------------------------------------------------
    index_path = os.path.join(ftcfg.machex_output_dir, "mimic", "index.json")
    if not os.path.exists(index_path):
        print(f"Error: Prepared data not found at {index_path}")
        print("Run `python -m finetune_cheff.prepare_data` first.")
        sys.exit(1)

    for name, path in [
        ("cheff_t2i_ckpt", ftcfg.cheff_t2i_ckpt),
        ("cheff_ae_ckpt", ftcfg.cheff_ae_ckpt),
    ]:
        if not path or not os.path.exists(path):
            print(f"Error: '{name}' not set or file not found ({path!r}).")
            print("Add it to .env or set the env var.")
            sys.exit(1)

    # ---- resolve paths ------------------------------------------------------
    cheff_dir = os.path.join(ftcfg.project_root, "cheff")
    train_script = os.path.join(cheff_dir, "scripts", "01_train_ldm.py")
    config_yml = os.path.join(
        ftcfg.experiments_dir, "finetune_cheff", "cheff_lora_vindr.yml"
    )
    log_dir = os.path.join(ftcfg.runs_dir, ftcfg.run_name)

    # ---- build command ------------------------------------------------------
    cmd = [
        sys.executable,
        train_script,
        "-b", config_yml,
        "-t", "True",
        "-s", str(ftcfg.seed),
        "-l", log_dir,
        # --- data paths (override YAML placeholders) -------------------------
        f"data.params.machex_path={ftcfg.machex_output_dir}",
        f"data.params.batch_size={ftcfg.cheff_batch_size}",
        f"data.params.test_size={ftcfg.cheff_test_size}",
        f"data.params.num_workers={ftcfg.cheff_num_workers}",
        # --- model checkpoints -----------------------------------------------
        f"model.params.ckpt_path={ftcfg.cheff_t2i_ckpt}",
        f"model.params.first_stage_config.params.ckpt_path={ftcfg.cheff_ae_ckpt}",
        # --- learning rate ----------------------------------------------------
        f"model.base_learning_rate={ftcfg.cheff_learning_rate}",
        # --- epoch limit ------------------------------------------------------
        f"lightning.trainer.max_epochs={ftcfg.cheff_max_epochs}",
    ]

    # ---- launch -------------------------------------------------------------
    print("=" * 60)
    print("CheFF LoRA Fine-Tuning")
    print("=" * 60)
    print(f"  Rank:       {ftcfg.lora_rank}")
    print(f"  Scope:      {ftcfg.lora_scope}")
    print(f"  LR:         {ftcfg.cheff_learning_rate}")
    print(f"  Batch size: {ftcfg.cheff_batch_size}")
    print(f"  Max epochs: {ftcfg.cheff_max_epochs}")
    print(f"  Data:       {ftcfg.machex_output_dir}")
    print(f"  T2I ckpt:   {ftcfg.cheff_t2i_ckpt}")
    print(f"  AE ckpt:    {ftcfg.cheff_ae_ckpt}")
    print(f"  Log dir:    {log_dir}")
    print("=" * 60)

    # Run from the cheff repo root so that `scripts.*` imports resolve.
    subprocess.run(cmd, cwd=cheff_dir, check=True)

    # The training script exports LoRA weights automatically via
    # export_lora_weights() when lora_config.enabled is true.
    adapter_dir = os.path.join(log_dir, "*", "lora_adapter")
    print(f"\nDone. LoRA adapter should be at: {adapter_dir}")


if __name__ == "__main__":
    main()
