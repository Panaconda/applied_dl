"""Generate a 4×4 comparison grid: 4 pathologies × 4 samples."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image
from PIL import Image, ImageDraw, ImageFont

# Re-use shared helpers from the main generate script
from inference.generate_synthetic import (
    CLASS_PROMPTS,
    _patch_gradient_checkpointing,
    load_model,
)

CLASSES = ["Pneumonia", "Bronchiolitis", "Bronchitis", "Brocho-pneumonia"]
COLS = 4  # samples per class


@torch.no_grad()
def sample_images(wrapper, prompt: str, n: int, steps: int, eta: float):
    """Return a list of n tensors in [0, 1] range."""
    images = []
    for _ in range(n):
        img = wrapper.sample(
            sampling_steps=steps,
            eta=eta,
            decode=True,
            conditioning=prompt,
        )
        img = (img.clamp(-1, 1) + 1) / 2  # [-1,1] → [0,1]
        # img shape: (1, C, H, W) or (C, H, W)
        if img.dim() == 4:
            img = img.squeeze(0)
        images.append(img.cpu())
    return images


def add_row_labels(grid_path: str, labels: list[str], cell_h: int) -> None:
    """Overlay row labels on the left side of a saved grid image."""
    img = Image.open(grid_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Try to use a reasonable font size
    font_size = max(14, cell_h // 12)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except OSError:
            font = ImageFont.load_default()

    pad = 2  # torchvision make_grid default padding
    for i, label in enumerate(labels):
        y = pad + i * (cell_h + pad) + cell_h // 2
        # White text with black outline for visibility
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((6 + dx, y + dy), label, fill="black", font=font, anchor="lm")
        draw.text((6, y), label, fill="white", font=font, anchor="lm")

    img.save(grid_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 4×4 sample grid")
    parser.add_argument("--model-path", default="../models/cheff_diff_t2i.pt")
    parser.add_argument("--ae-path", default="../models/cheff_autoencoder.pt")
    parser.add_argument("--lora-adapter", default=None,
                        help="PEFT adapter dir (omit for base model)")
    parser.add_argument("--output", default="grid.png",
                        help="Output image path")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root / "cheff"))

    for name, path in [("model_path", args.model_path), ("ae_path", args.ae_path)]:
        if not os.path.exists(path):
            print(f"Error: {name}={path!r} not found.")
            sys.exit(1)

    wrapper = load_model(
        model_path=args.model_path,
        ae_path=args.ae_path,
        lora_adapter=args.lora_adapter,
        device=args.device,
    )

    tag = "fine-tuned" if args.lora_adapter else "base"
    print(f"\nGenerating 4×4 grid ({tag}) …")

    all_images: list[torch.Tensor] = []
    for cls in CLASSES:
        prompt = CLASS_PROMPTS[cls]
        print(f"  [{cls}] sampling {COLS} images …")
        imgs = sample_images(wrapper, prompt, COLS, args.steps, args.eta)
        all_images.extend(imgs)

    # make_grid arranges in row-major: nrow=COLS gives 4 rows × 4 cols
    grid = make_grid(all_images, nrow=COLS, padding=2, normalize=False)
    save_image(grid, args.output)

    # Compute cell height for label overlay
    cell_h = all_images[0].shape[-2]
    add_row_labels(args.output, CLASSES, cell_h)

    print(f"\nSaved → {args.output}  ({tag})")


if __name__ == "__main__":
    main()
