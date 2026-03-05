"""UMAP latent-space visualization: Real vs LoRA-synthetic vs Base-synthetic CXR.

Self-contained script that:
  1. Samples 200 real VinDr-PCXR images (100 train + 100 test, 25 per pathology)
  2. Samples 200 LoRA-adapted CheFF synthetic images (50 per class from disk)
  3. Generates 200 base CheFF synthetic images on-the-fly (50 per class, no LoRA)
  4. Extracts 1024-d features with the *pretrained* XRV DenseNet121
  5. Fits UMAP on all 600 vectors and saves a scatter plot colored by source

Usage (from experiments/):
    python -m inference.umap_latent \
        --lora-dir ../samples/lora \
        --output runs/umap_latent.png
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchxrayvision as xrv
import umap
from PIL import Image
from tqdm import tqdm

from classifier.core.config import cfg
from classifier.core.dataset import build_transform, load_image_id_map, load_labels

# Reuse class prompts & model loader from generation script
from inference.generate_synthetic import CLASS_PROMPTS, load_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PATHOLOGY_CLASSES = ["Pneumonia", "Bronchitis", "Bronchiolitis", "Brocho-pneumonia"]

SET_LABELS = {0: "Real", 1: "LoRA Synthetic", 2: "Base Synthetic"}
SET_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}  # blue, orange, green


# ---------------------------------------------------------------------------
# 1. Sample real images
# ---------------------------------------------------------------------------
def sample_real_paths(n_train: int = 100, n_test: int = 100, seed: int = 42) -> list[str]:
    """Return paths for 200 real images balanced across 4 pathology classes."""
    rng = random.Random(seed)
    per_cls_train = n_train // len(PATHOLOGY_CLASSES)  # 25
    per_cls_test = n_test // len(PATHOLOGY_CLASSES)    # 25
    paths: list[str] = []

    for image_dir, csv_path, index_json, per_cls in [
        (cfg.train_image_dir, cfg.train_labels_csv, cfg.vindr_pcxr_train_index, per_cls_train),
        (cfg.test_image_dir, cfg.test_labels_csv, cfg.vindr_pcxr_test_index, per_cls_test),
    ]:
        labels = load_labels(csv_path)
        # Build image_id → file path mapping (handles sequential JPG names)
        id_map = load_image_id_map(index_json, image_dir) if index_json else {}
        for cls in PATHOLOGY_CLASSES:
            positive_ids = labels[labels[cls] == 1].index.tolist()
            sampled = rng.sample(positive_ids, min(per_cls, len(positive_ids)))
            for img_id in sampled:
                path = id_map.get(img_id, os.path.join(image_dir, f"{img_id}.png"))
                paths.append(path)

    return paths


# ---------------------------------------------------------------------------
# 2. Sample existing LoRA synthetic images
# ---------------------------------------------------------------------------
def sample_lora_paths(lora_dir: str, per_class: int = 50, seed: int = 42) -> list[str]:
    """Return paths for 200 LoRA-synthetic images (50 per pathology class)."""
    rng = random.Random(seed)
    paths: list[str] = []

    for cls in PATHOLOGY_CLASSES:
        cls_dir = os.path.join(lora_dir, cls)
        all_pngs = sorted(glob(os.path.join(cls_dir, "*.png")))
        if len(all_pngs) == 0:
            raise FileNotFoundError(f"No PNGs in {cls_dir}")
        sampled = rng.sample(all_pngs, min(per_class, len(all_pngs)))
        paths.extend(sampled)

    return paths


# ---------------------------------------------------------------------------
# 3. Generate base CheFF images (no LoRA)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_base_images(
    wrapper, per_class: int = 50, steps: int = 100, eta: float = 1.0
) -> list[Image.Image]:
    """Generate base CheFF images on-the-fly and return as PIL images."""
    images: list[Image.Image] = []

    for cls in PATHOLOGY_CLASSES:
        prompt = CLASS_PROMPTS[cls]
        for _ in tqdm(range(per_class), desc=f"Base {cls}", leave=False):
            tensor = wrapper.sample(
                sampling_steps=steps, eta=eta, decode=True, conditioning=prompt,
            )
            # [1, 3, 256, 256] in [-1, 1] → uint8 RGB PIL
            arr = (tensor.squeeze(0).permute(1, 2, 0).clamp(-1, 1).cpu().numpy() + 1) / 2
            arr = (arr * 255).astype(np.uint8)
            if arr.shape[2] == 1:
                images.append(Image.fromarray(arr.squeeze(-1), mode="L"))
            else:
                images.append(Image.fromarray(arr))

    return images


# ---------------------------------------------------------------------------
# 4. Feature extraction with pretrained XRV
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_features(
    tensors: list[torch.Tensor], device: str, batch_size: int = 32
) -> np.ndarray:
    """Extract 1024-d feature vectors using the pretrained XRV DenseNet121.

    Args:
        tensors: list of [1, 224, 224] image tensors (XRV-normalised).
        device:  "cuda" or "cpu".
        batch_size: inference batch size.

    Returns:
        (N, 1024) float32 numpy array.
    """
    model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device).eval()

    features: list[np.ndarray] = []
    for i in tqdm(range(0, len(tensors), batch_size), desc="Extracting features"):
        batch = torch.stack(tensors[i : i + batch_size]).to(device)
        feats = model.features(batch)                          # (B, 1024, 7, 7)
        feats = F.relu(feats, inplace=False)
        feats = F.adaptive_avg_pool2d(feats, (1, 1))           # (B, 1024, 1, 1)
        feats = feats.view(feats.size(0), -1)                  # (B, 1024)
        features.append(feats.cpu().numpy())

    del model
    torch.cuda.empty_cache()
    return np.vstack(features)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_paths_as_tensors(paths: list[str], transform) -> list[torch.Tensor]:
    """Load image files and apply XRV transform."""
    return [transform(Image.open(p)) for p in tqdm(paths, desc="Loading images")]


def pils_to_tensors(pil_images: list[Image.Image], transform) -> list[torch.Tensor]:
    """Apply XRV transform to in-memory PIL images."""
    return [transform(img) for img in pil_images]


# ---------------------------------------------------------------------------
# 5. UMAP + plot
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="UMAP latent-space evaluation")
    parser.add_argument(
        "--lora-dir", default="../samples/lora",
        help="Root dir with LoRA synthetic images (subdirs per class)",
    )
    parser.add_argument("--model-path", default="../models/cheff_diff_t2i.pt")
    parser.add_argument("--ae-path", default="../models/cheff_autoencoder.pt")
    parser.add_argument("--steps", type=int, default=100,
                        help="DDIM sampling steps for base CheFF generation")
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--output", default="runs/umap_latent.png")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    # Ensure cheff is importable
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root / "cheff"))

    transform = build_transform()

    # ---- 1. Real images ----
    print("\n=== 1/5  Sampling real images ===")
    real_paths = sample_real_paths(n_train=100, n_test=100, seed=args.seed)
    real_tensors = load_paths_as_tensors(real_paths, transform)
    print(f"  {len(real_tensors)} real images loaded")

    # ---- 2. LoRA synthetic images ----
    print("\n=== 2/5  Sampling LoRA-synthetic images ===")
    lora_paths = sample_lora_paths(args.lora_dir, per_class=50, seed=args.seed)
    lora_tensors = load_paths_as_tensors(lora_paths, transform)
    print(f"  {len(lora_tensors)} LoRA-synthetic images loaded")

    # ---- 3. Generate base CheFF images (no LoRA) ----
    print("\n=== 3/5  Generating base CheFF images (no LoRA) ===")
    wrapper = load_model(
        model_path=args.model_path,
        ae_path=args.ae_path,
        lora_adapter=None,
        device=args.device,
    )
    base_pils = generate_base_images(wrapper, per_class=50, steps=args.steps, eta=args.eta)
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
