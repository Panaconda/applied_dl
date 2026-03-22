from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import scipy.linalg
import torch

from config import _CHEFF_PEFT_ROOT
from eval.shared import (
    build_transform,
    extract_features,
    generate_base_images,
    load_paths_as_tensors,
    pils_to_tensors,
    sample_lora_paths,
    sample_real_paths,
)
from inference.generate import load_model


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute FDD: real vs synthetic CXRs")
    parser.add_argument("--lora-dir", required=True,
                        help="Root dir of LoRA synthetic images (subdirs per class)")
    parser.add_argument("--per-class", type=int, default=100,
                        help="Synthetic images per class to sample (default 100)")
    parser.add_argument("--n-real-train", type=int, default=200,
                        help="Real train images sampled (balanced across classes)")
    parser.add_argument("--n-real-test", type=int, default=200,
                        help="Real test images sampled (balanced across classes)")
    parser.add_argument("--base-dir", default=None,
                        help="Pre-generated base-model images dir (same layout as --lora-dir).")
    parser.add_argument("--model-path",
                        default=os.path.join(_CHEFF_PEFT_ROOT, "checkpoints", "cheff_diff_t2i.pt"),
                        help="CheFF T2I weights (only needed when --base-dir is not set)")
    parser.add_argument("--ae-path",
                        default=os.path.join(_CHEFF_PEFT_ROOT, "checkpoints", "cheff_autoencoder.pt"),
                        help="CheFF autoencoder weights (only needed when --base-dir is not set)")
    parser.add_argument("--steps", type=int, default=50,
                        help="DDIM steps for on-the-fly base generation (ignored when --base-dir is set)")
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--output", default="eval/runs/fdd.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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
    if args.base_dir:
        base_paths = sample_lora_paths(args.base_dir, per_class=args.per_class, seed=args.seed)
        base_tensors = load_paths_as_tensors(base_paths, transform)
    else:
        print("  --base-dir not set; generating base images on-the-fly …")
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

    fdd_base: float | None = None
    X_base: np.ndarray | None = None
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

    print("\n" + "\n".join(lines))

    csv_path = Path(args.output)
    os.makedirs(csv_path.parent, exist_ok=True)

    rows = [
        {"comparison": "real_vs_lora", "n_real": len(X_real), "n_synthetic": len(X_lora), "fdd": round(fdd_lora, 4)},
    ]
    if fdd_base is not None and X_base is not None:
        rows.append({"comparison": "real_vs_base", "n_real": len(X_real), "n_synthetic": len(X_base), "fdd": round(fdd_base, 4)})
    rows.append({"comparison": "real_self_check", "n_real": len(X_real), "n_synthetic": len(X_real), "fdd": round(fdd_self, 4)})

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["comparison", "n_real", "n_synthetic", "fdd"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved → {csv_path}")


if __name__ == "__main__":
    main()
