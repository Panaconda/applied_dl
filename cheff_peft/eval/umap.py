from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap

from config import _CHEFF_PEFT_ROOT
from inference.generate import load_model
from eval.shared import (
    PATHOLOGY_CLASSES,
    build_transform,
    extract_features,
    generate_base_images,
    load_paths_as_tensors,
    pils_to_tensors,
    sample_lora_paths,
    sample_real_paths,
)

SET_LABELS = {0: "Real", 1: "LoRA Synthetic", 2: "Base Synthetic"}
SET_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}


def fit_and_plot(
    X_real: np.ndarray,
    X_lora: np.ndarray,
    X_base: np.ndarray,
    output_path: str,
    seed: int = 42,
) -> None:
    """Fit UMAP on all 600 feature vectors and save scatter plot."""
    X = np.vstack([X_real, X_lora, X_base])
    y = np.array(
        [0] * len(X_real) + [1] * len(X_lora) + [2] * len(X_base)
    )

    print(f"Fitting UMAP on {len(X)} samples ({X.shape[1]}-d) …")
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2, random_state=seed
    )
    emb = reducer.fit_transform(X)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 7))
    for label_id, label_name in SET_LABELS.items():
        mask = y == label_id
        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            c=SET_COLORS[label_id],
            label=f"{label_name} (n={mask.sum()})",
            s=12,
            alpha=0.7,
            edgecolors="none",
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.set_title("UMAP — XRV DenseNet121 Feature Space", fontsize=13)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="UMAP latent-space evaluation")
    parser.add_argument(
        "--lora-dir", required=True,
        help="Root dir with LoRA synthetic images (one subdir per class)",
    )
    parser.add_argument(
        "--base-dir", default=None,
        help="Root dir with base-model synthetic images.  When provided, images are "
             "loaded from disk instead of being generated on-the-fly.",
    )
    parser.add_argument("--per-class", type=int, default=100,
                        help="Synthetic images per class to sample (default 100)")
    parser.add_argument("--model-path",
                        default=os.path.join(_CHEFF_PEFT_ROOT, "checkpoints", "cheff_diff_t2i.pt"))
    parser.add_argument("--ae-path",
                        default=os.path.join(_CHEFF_PEFT_ROOT, "checkpoints", "cheff_autoencoder.pt"))
    parser.add_argument("--steps", type=int, default=100,
                        help="DDIM sampling steps (only used when --base-dir is not set)")
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--output", default="eval/runs/umap.png")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    transform = build_transform()

    # ---- 1. Real images ----
    print("\n=== 1/5  Sampling real images ===")
    real_paths = sample_real_paths(n_train=100, n_test=100, seed=args.seed)
    real_tensors = load_paths_as_tensors(real_paths, transform)
    print(f"  {len(real_tensors)} real images loaded")

    # ---- 2. LoRA synthetic images ----
    print("\n=== 2/5  Sampling LoRA-synthetic images ===")
    lora_paths = sample_lora_paths(args.lora_dir, per_class=args.per_class, seed=args.seed)
    lora_tensors = load_paths_as_tensors(lora_paths, transform)
    print(f"  {len(lora_tensors)} LoRA-synthetic images loaded")

    # ---- 3. Base CheFF images (from disk or on-the-fly) ----
    if args.base_dir:
        print("\n=== 3/5  Loading base-synthetic images from disk ===")
        base_paths = sample_lora_paths(args.base_dir, per_class=args.per_class, seed=args.seed)
        base_tensors = load_paths_as_tensors(base_paths, transform)
        print(f"  {len(base_tensors)} base-synthetic images loaded")
    else:
        print("\n=== 3/5  Generating base CheFF images on-the-fly (no LoRA) ===")
        wrapper = load_model(
            model_path=args.model_path,
            ae_path=args.ae_path,
            lora_adapter=None,
            device=args.device,
        )
        base_pils = generate_base_images(
            wrapper, per_class=args.per_class, steps=args.steps, eta=args.eta
        )
        del wrapper
        torch.cuda.empty_cache()
        base_tensors = pils_to_tensors(base_pils, transform)
        del base_pils
        print(f"  {len(base_tensors)} base-synthetic images generated")

    # ---- 4. Extract features ----
    print("\n=== 4/5  Extracting XRV features ===")
    X_real = extract_features(real_tensors, args.device, args.batch_size)
    X_lora = extract_features(lora_tensors, args.device, args.batch_size)
    X_base = extract_features(base_tensors, args.device, args.batch_size)
    print(f"  Feature shapes: real {X_real.shape}, lora {X_lora.shape}, base {X_base.shape}")

    # ---- 5. UMAP ----
    print("\n=== 5/5  UMAP ===")
    fit_and_plot(X_real, X_lora, X_base, args.output, seed=args.seed)
    print("\nDone.")


if __name__ == "__main__":
    main()
