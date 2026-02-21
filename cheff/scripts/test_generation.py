#!/usr/bin/env python3
"""
Generate chest X-rays from text prompts with optional super-resolution.
Usage: python test_generation.py --prompt "your prompt" [--full-res]
"""

import argparse
import torch
from cheff import CheffLDMT2I, CheffSRModel
from torchvision.utils import save_image
import os
from pathlib import Path
from peft import LoraConfig, get_peft_model, PeftModel

def main():
    parser = argparse.ArgumentParser(description='Generate chest X-ray from text prompt')
    parser.add_argument(
        '--prompt', 
        type=str, 
        default='Chest X-ray showing normal lungs',
        help='Text prompt for image generation'
    )
    parser.add_argument(
        '--steps', 
        type=int, 
        default=100,
        help='Number of diffusion sampling steps (50-200)'
    )
    parser.add_argument(
        '--full-res',
        action='store_true',
        help='Generate full resolution 1024×1024 image (requires SR model)'
    )
    parser.add_argument(
        '--sr-steps',
        type=int,
        default=100,
        help='Number of SR sampling steps (only with --full-res)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='generated_xray.png',
        help='Output file path'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to diffusion model checkpoint (auto-detected from persistent volume)'
    )
    parser.add_argument(
        '--ae-path',
        type=str,
        default=None,
        help='Path to autoencoder checkpoint (auto-detected from persistent volume)'
    )
    parser.add_argument(
        '--sr-path',
        type=str,
        default=None,
        help='Path to super-resolution model checkpoint (auto-detected from persistent volume)'
    )
    parser.add_argument(
        '--eta',
        type=float,
        default=1.0,
        help='Stochasticity parameter (0=deterministic, 1=stochastic)'
    )
    parser.add_argument(
        '--lora-ckpt',
        type=str,
        default=None,
        help='Path to a PL .ckpt file from LoRA fine-tuning'
    )
    parser.add_argument(
        '--lora-rank',
        type=int,
        default=16,
        help='LoRA rank used during fine-tuning (must match checkpoint)'
    )
    parser.add_argument(
        '--lora-alpha',
        type=int,
        default=32,
        help='LoRA alpha used during fine-tuning (must match checkpoint)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1,
        help='Number of images to generate (saved as output_001.png, output_002.png, …)'
    )
    
    args = parser.parse_args()
    
    # Set default model paths if not specified
    if args.model_path is None:
        args.model_path = 'models/cheff_diff_t2i.pt'
    if args.ae_path is None:
        args.ae_path = 'models/cheff_autoencoder.pt'
    if args.sr_path is None:
        args.sr_path = 'models/cheff_sr_fine.pt'
    
    # Check if model files exist
    if args.model_path is None or not os.path.exists(args.model_path):
        print(f"Error: Diffusion model not found.")
        print("Expected location: models/")
        return
    
    if args.ae_path is None or not os.path.exists(args.ae_path):
        print(f"Error: Autoencoder model not found.")
        print("Expected location: models/")
        return
    
    if args.full_res and (args.sr_path is None or not os.path.exists(args.sr_path)):
        print(f"Error: SR model not found.")
        print("Expected location: models/")
        return
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cpu':
        print("Warning: Running on CPU. This will be slow. Consider using a GPU.")
    
    # Load diffusion model
    print(f"Loading models...")
    print(f"  Diffusion: {args.model_path}")
    print(f"  Autoencoder: {args.ae_path}")
    
    model = CheffLDMT2I(
        model_path=args.model_path,
        ae_path=args.ae_path,
        device=device
    )
    
    # Load LoRA fine-tuned checkpoint if provided
    if args.lora_ckpt:
        print(f"  LoRA checkpoint: {args.lora_ckpt}")
        # Re-create LoRA architecture on the UNet
        unet = model.model.model.diffusion_model
        if not hasattr(unet, "config"):
            class _MockConfig:
                def to_dict(self): return {}
            unet.config = _MockConfig()
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=".*attn.*(to_q|to_k|to_v|to_out.0)",
            lora_dropout=0.0,
            bias="none",
        )
        model.model.model.diffusion_model = get_peft_model(unet, peft_config)
        # Load trained weights from PL checkpoint
        ckpt = torch.load(args.lora_ckpt, map_location="cpu")
        missing, unexpected = model.model.load_state_dict(ckpt["state_dict"], strict=False)
        # Move LoRA layers (initialized on CPU) to the target device
        model.model.to(device)
        model.model.eval()
        lora_keys_in_ckpt = [k for k in ckpt["state_dict"] if "lora_" in k]
        lora_keys_missing = [k for k in missing if "lora_" in k]
        print(f"  LoRA keys in checkpoint : {len(lora_keys_in_ckpt)}")
        print(f"  LoRA keys not loaded    : {len(lora_keys_missing)}")
        if lora_keys_missing:
            print(f"  WARNING: some LoRA keys were not loaded: {lora_keys_missing[:5]}")
        else:
            print("  LoRA checkpoint loaded successfully.")
    
    print(f"\nGenerating {args.num_samples} × 256×256 image(s) from prompt:")
    print(f"  '{args.prompt}'")
    print(f"  Diffusion steps: {args.steps}")
    print(f"  Eta: {args.eta}")
    print("\nThis may take 20-60 seconds per image...")

    # Determine output path template
    base, ext = os.path.splitext(args.output)
    if not ext:
        ext = ".png"

    for i in range(1, args.num_samples + 1):
        out_path = f"{base}_{i:03d}{ext}" if args.num_samples > 1 else args.output
        print(f"\n  [{i}/{args.num_samples}] Sampling → {out_path}")

        # Generate 256×256 image
        image = model.sample(
            sampling_steps=args.steps,
            eta=args.eta,
            decode=True,
            conditioning=args.prompt
        )

        # Post-process
        image = image.clamp(-1, 1)
        image = (image + 1) / 2  # Convert from [-1, 1] to [0, 1]

        # Super-resolution if requested
        if args.full_res:
            from torchvision.transforms.functional import rgb_to_grayscale
            image_gray = rgb_to_grayscale(image)
            sr_model = CheffSRModel(model_path=args.sr_path, device=device)
            image = sr_model.sample(image_gray, method='ddim', sampling_steps=args.sr_steps, eta=0.0)
            resolution = "1024×1024"
        else:
            resolution = "256×256"

        save_image(image, out_path)
        print(f"  ✓ Saved ({resolution})")

    print(f"\nDone. {args.num_samples} image(s) saved.")

if __name__ == '__main__':
    main()
