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
    
    args = parser.parse_args()
    
    # Set default model paths if not specified
    if args.model_path is None:
        args.model_path = 'cheff/trained_models/cheff_diff_t2i.pt'
    if args.ae_path is None:
        args.ae_path = 'cheff/trained_models/cheff_autoencoder.pt'
    if args.sr_path is None:
        args.sr_path = 'cheff/trained_models/cheff_sr_fine.pt'
    
    # Check if model files exist
    if args.model_path is None or not os.path.exists(args.model_path):
        print(f"Error: Diffusion model not found.")
        print("Expected location: cheff/trained_models/")
        return
    
    if args.ae_path is None or not os.path.exists(args.ae_path):
        print(f"Error: Autoencoder model not found.")
        print("Expected location: cheff/trained_models/")
        return
    
    if args.full_res and (args.sr_path is None or not os.path.exists(args.sr_path)):
        print(f"Error: SR model not found.")
        print("Expected location: cheff/trained_models/")
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
    
    print(f"\nGenerating 256×256 image from prompt:")
    print(f"  '{args.prompt}'")
    print(f"  Diffusion steps: {args.steps}")
    print(f"  Eta: {args.eta}")
    print("\nThis may take 20-60 seconds...")
    
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
        print(f"\nUpscaling to 1024×1024 with SR model...")
        print(f"  SR model: {args.sr_path}")
        print(f"  SR steps: {args.sr_steps}")
        print("\nThis may take another 60-120 seconds...")
        
        # Convert RGB to grayscale for SR model (expects 1 channel)
        from torchvision.transforms.functional import rgb_to_grayscale
        image_gray = rgb_to_grayscale(image)
        
        sr_model = CheffSRModel(
            model_path=args.sr_path,
            device=device
        )
        
        # SR expects grayscale image in [0, 1] range
        image = sr_model.sample(
            image_gray,
            method='ddim',
            sampling_steps=args.sr_steps,
            eta=0.0
        )
        
        resolution = "1024×1024"
    else:
        resolution = "256×256"
    
    save_image(image, args.output)
    print(f"\n✓ Image saved to: {args.output}")
    print(f"  Resolution: {resolution}")
    
    # Display image info
    print(f"\nImage statistics:")
    print(f"  Shape: {image.shape}")
    print(f"  Min: {image.min().item():.3f}, Max: {image.max().item():.3f}, Mean: {image.mean().item():.3f}")

if __name__ == '__main__':
    main()
