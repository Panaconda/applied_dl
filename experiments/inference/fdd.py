"""Fréchet DenseNet Distance (FDD): Real vs LoRA-synthetic vs Base-synthetic CXR."""
from __future__ import annotations

import argparse
import os
import random
import sys
from glob import glob
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F
import torchxrayvision as xrv
from PIL import Image
from tqdm import tqdm

from core.config import cfg
from core.dataset import build_transform, load_image_id_map, load_labels
from inference.generate_synthetic import CLASS_PROMPTS, load_model
from inference.umap_latent import (
    PATHOLOGY_CLASSES,
    generate_base_images,
    sample_lora_paths,
    sample_real_paths,
    load_paths_as_tensors,
    pils_to_tensors,
)

# ---------------------------------------------------------------------------
# Fréchet distance
# ---------------------------------------------------------------------------

def compute_frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray,
    mu2: np.ndarray, sigma2: np.ndarray,
    eps: float = 1e-4,
) -> float:
    """Compute FDD between two Gaussians N(μ₁,Σ₁) and N(μ₂,Σ₂).

    Numerically stabilised: adds eps·I to both covariance matrices before
    computing the matrix square root to handle rank-deficient cases (n << d).
    """
    diff = mu1 - mu2
    I = np.eye(sigma1.shape[0], dtype=sigma1.dtype)
    s1 = sigma1 + eps * I
    s2 = sigma2 + eps * I

    covmean, _ = scipy.linalg.sqrtm(s1 @ s2, disp=False)

    # Discard tiny imaginary residuals from floating-point arithmetic
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(s1 + s2 - 2 * covmean))


def gaussian_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, covariance) of feature matrix (N, D)."""
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    tensors: list[torch.Tensor], device: str, batch_size: int = 32
) -> np.ndarray:
    """Extract 1024-d XRV DenseNet121 features from a list of image tensors."""
    model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device).eval()
    features: list[np.ndarray] = []
    for i in tqdm(range(0, len(tensors), batch_size), desc="Extracting features", leave=False):
        batch = torch.stack(tensors[i : i + batch_size]).to(device)
        feats = model.features(batch)                        # (B, 1024, 7, 7)
        feats = F.relu(feats, inplace=False)
        feats = F.adaptive_avg_pool2d(feats, (1, 1))         # (B, 1024, 1, 1)
        feats = feats.view(feats.size(0), -1).cpu().numpy()  # (B, 1024)
        features.append(feats)
    del model
    torch.cuda.empty_cache()
    return np.vstack(features)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute FDD: real vs synthetic CXRs")
    parser.add_argument("--lora-dir", default="../samples/lora",
                        help="Root dir of LoRA synthetic images (subdirs per class)")
    parser.add_argument("--per-class", type=int, default=200,
                        help="Synthetic images per class to sample (default 200)")
    parser.add_argument("--n-real-train", type=int, default=200,
                        help="Real train images sampled (balanced across classes)")
    parser.add_argument("--n-real-test", type=int, default=200,
                        help="Real test images sampled (balanced across classes)")
    # Base model on-the-fly generation
    parser.add_argument("--base-on-the-fly", action="store_true",
                        help="Generate base model images on-the-fly instead of "
                             "loading from --base-dir")
    parser.add_argument("--base-dir", default=None,
                        help="Pre-generated base-model images dir (same layout as "
                             "--lora-dir).  Used when --base-on-the-fly is not set.")
    parser.add_argument("--model-path", default="../models/cheff_diff_t2i.pt")
    parser.add_argument("--ae-path", default="../models/cheff_autoencoder.pt")
    parser.add_argument("--steps", type=int, default=50,
                        help="DDIM steps for on-the-fly base generation")
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--output", default="runs/fdd.txt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root / "cheff"))
    transform = build_transform()

    # ---- 1. Real images --------------------------------------------------------
    print("\n=== 1/4  Real images ===")
    real_paths = sample_real_paths(args.n_real_train, args.n_real_test, seed=args.seed)
    real_tensors = load_paths_as_tensors(real_paths, transform)
    print(f"  {len(real_tensors)} real images loaded")

    # ---- 2. LoRA synthetic images ----------------------------------------------
    print("\n=== 2/4  LoRA-synthetic images ===")
    lora_paths = sample_lora_paths(args.lora_dir, per_class=args.per_class, seed=args.seed)
    lora_tensors = load_paths_as_tensors(lora_paths, transform)
    print(f"  {len(lora_tensors)} LoRA-synthetic images loaded")

    # ---- 3. Base synthetic images ----------------------------------------------
    print("\n=== 3/4  Base-synthetic images ===")
    if args.base_on_the_fly:
        for name, path in [("model_path", args.model_path), ("ae_path", args.ae_path)]:
            if not os.path.exists(path):
                print(f"Error: {name}={path!r} not found.")
                sys.exit(1)
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
    elif args.base_dir:
        base_paths = sample_lora_paths(args.base_dir, per_class=args.per_class, seed=args.seed)
        base_tensors = load_paths_as_tensors(base_paths, transform)
    else:
        print("  Skipping base model (pass --base-on-the-fly or --base-dir to include).")
        base_tensors = []

    print(f"  {len(base_tensors)} base-synthetic images ready")

    # ---- 4. Extract features + compute FDD ------------------------------------
    print("\n=== 4/4  Feature extraction + FDD ===")
    X_real  = extract_features(real_tensors,  args.device, args.batch_size)
    X_lora  = extract_features(lora_tensors,  args.device, args.batch_size)

    mu_real,  sig_real  = gaussian_stats(X_real)
    mu_lora,  sig_lora  = gaussian_stats(X_lora)

    fdd_lora = compute_frechet_distance(mu_real, sig_real, mu_lora, sig_lora)

    lines = [
        "Fréchet DenseNet Distance (FDD)",
        "=" * 40,
        f"Feature space : XRV DenseNet121 (1024-d)",
        f"Real images   : {len(X_real)}",
        f"LoRA synthetic: {len(X_lora)}",
        "",
        f"FDD(real, LoRA)  = {fdd_lora:.2f}",
    ]

    if base_tensors:
        X_base = extract_features(base_tensors, args.device, args.batch_size)
        mu_base, sig_base = gaussian_stats(X_base)
        fdd_base = compute_frechet_distance(mu_real, sig_real, mu_base, sig_base)
        lines += [
            f"Base synthetic: {len(X_base)}",
            f"FDD(real, Base)  = {fdd_base:.2f}",
            "",
            f"Δ FDD (LoRA − Base) = {fdd_lora - fdd_base:.2f}  "
            f"({'LoRA closer to real' if fdd_lora < fdd_base else 'Base closer to real'})",
        ]

    # Self-FDD sanity check (should be ~0)
    half = len(X_real) // 2
    fdd_self = compute_frechet_distance(
        *gaussian_stats(X_real[:half]), *gaussian_stats(X_real[half:])
    )
    lines += ["", f"FDD(real, real) self-check = {fdd_self:.2f}  (should be ~0)"]

    report = "\n".join(lines)
    print("\n" + report)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
