"""Generate a synthetic chest X-ray dataset using CheFF T2I."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import torch
from peft import LoraConfig, get_peft_model
from torchvision.utils import save_image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Class prompts
# ---------------------------------------------------------------------------
CLASS_PROMPTS: Dict[str, str] = {
    "Pneumonia": (
        "Findings: Frontal radiograph of a child. "
        "Evaluation reveals opacities. "
        "Impressions: pneumonia."
    ),
    "Bronchiolitis": (
        "Findings: Frontal radiograph of a child. "
        "Evaluation reveals reticulonodular opacities. "
        "Impressions: bronchiolitis."
    ),
    "Bronchitis": (
        "Findings: Frontal radiograph of a child. "
        "Evaluation reveals bronchial wall thickening. "
        "Impressions: bronchitis."
    ),
    "Brocho-pneumonia": (
        "Findings: Frontal radiograph of a child. "
        "Evaluation reveals patchy opacities. "
        "Impressions: brocho-pneumonia."
    ),
}

# ---------------------------------------------------------------------------
# Gradient-checkpointing patch (needed only if cheff uses use_checkpoint=True
# in the UNet — safe to apply unconditionally)
# ---------------------------------------------------------------------------
def _patch_gradient_checkpointing() -> None:
    def _wrapper(func, inputs, params, flag):
        return torch.utils.checkpoint.checkpoint(func, *inputs, use_reentrant=False)

    import cheff.ldm.modules.diffusionmodules.util as cheff_util
    import cheff.ldm.modules.attention as cheff_attn
    import cheff.ldm.modules.diffusionmodules.openaimodel as cheff_openai

    cheff_util.checkpoint = _wrapper
    cheff_attn.checkpoint = _wrapper
    if hasattr(cheff_openai, "checkpoint"):
        cheff_openai.checkpoint = _wrapper


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_path: str, ae_path: str, lora_ckpt: str | None,
               lora_rank: int, lora_alpha: int, device: str):
    """Load CheFF T2I, optionally patching in LoRA weights from a PL checkpoint."""
    _patch_gradient_checkpointing()

    from cheff.ldm.inference import CheffLDMT2I  # noqa: E402

    print(f"Loading CheFF T2I …  ({device})")
    wrapper = CheffLDMT2I(model_path=model_path, ae_path=ae_path, device=device)

    if lora_ckpt:
        print(f"Applying LoRA from {lora_ckpt} …")
        unet = wrapper.model.model.diffusion_model
        if not hasattr(unet, "config"):
            class _MockConfig:
                def to_dict(self): return {}
            unet.config = _MockConfig()

        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=".*attn.*(to_q|to_k|to_v|to_out.0)",
            lora_dropout=0.0,
            bias="none",
        )
        wrapper.model.model.diffusion_model = get_peft_model(unet, peft_config)

        ckpt = torch.load(lora_ckpt, map_location="cpu")
        missing, _ = wrapper.model.load_state_dict(ckpt["state_dict"], strict=False)
        lora_missing = [k for k in missing if "lora_" in k]
        if lora_missing:
            print(f"  WARNING: {len(lora_missing)} LoRA keys not loaded — check rank/ckpt")
        else:
            print(f"  LoRA loaded ({sum('lora_' in k for k in ckpt['state_dict'])} keys)")

        wrapper.model.to(device)

    wrapper.model.eval()
    return wrapper


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_class(wrapper, prompt: str, n: int, out_dir: str,
                   steps: int, eta: float, device: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for i in tqdm(range(1, n + 1), desc=os.path.basename(out_dir), leave=False):
        image = wrapper.sample(
            sampling_steps=steps,
            eta=eta,
            decode=True,
            conditioning=prompt,
        )
        image = (image.clamp(-1, 1) + 1) / 2  # [-1,1] → [0,1]
        save_image(image, os.path.join(out_dir, f"{i:06d}.png"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic VinDr-PCXR images with CheFF T2I"
    )
    parser.add_argument("--model-path", default="../models/cheff_diff_t2i.pt")
    parser.add_argument("--ae-path", default="../models/cheff_autoencoder.pt")
    parser.add_argument(
        "--lora-ckpt", default=None,
        help="PL .ckpt from finetune_cheff training (recommended)"
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--output-dir", default="../data/synthetic",
        help="Root directory for generated images"
    )
    parser.add_argument(
        "--n", type=int, default=1000,
        help="Images to generate per class"
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--classes", nargs="+", default=list(CLASS_PROMPTS.keys()),
        choices=list(CLASS_PROMPTS.keys()),
        help="Subset of classes to generate (default: all 4)"
    )
    args = parser.parse_args()

    # Resolve paths relative to project root (script is run from experiments/)
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root / "cheff"))

    # Validate model files
    for name, path in [("model_path", args.model_path), ("ae_path", args.ae_path)]:
        if not os.path.exists(path):
            print(f"Error: {name}={path!r} not found.")
            sys.exit(1)

    wrapper = load_model(
        model_path=args.model_path,
        ae_path=args.ae_path,
        lora_ckpt=args.lora_ckpt,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        device=args.device,
    )

    total = len(args.classes) * args.n
    print(f"\n{'='*60}")
    print(f"Synthetic generation")
    print(f"{'='*60}")
    print(f"  Classes:    {', '.join(args.classes)}")
    print(f"  Per class:  {args.n}")
    print(f"  Total:      {total}")
    print(f"  Steps/eta:  {args.steps} / {args.eta}")
    print(f"  LoRA:       {args.lora_ckpt or 'none (base model)'}")
    print(f"  Output:     {args.output_dir}")
    print(f"{'='*60}\n")

    for cls in args.classes:
        prompt = CLASS_PROMPTS[cls]
        out_dir = os.path.join(args.output_dir, cls)
        print(f"[{cls}]  →  {out_dir}")
        print(f"  Prompt: {prompt}")
        generate_class(wrapper, prompt, args.n, out_dir, args.steps, args.eta, args.device)
        print(f"  ✓ {args.n} images saved\n")

    print(f"Done. {total} images written to {args.output_dir}")


if __name__ == "__main__":
    main()
