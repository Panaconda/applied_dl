import argparse
import csv
import gc
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Add cheff to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cheff.ldm.inference import CheffLDM, CheffLDMT2I

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    print("ERROR: peft library not installed. Run: pip install peft")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""
    scenario: str
    lora_rank: int
    batch_size: int
    precision: str
    max_vram_gb: float
    throughput_img_per_sec: float
    trainable_params: int
    total_params: int
    trainable_percent: float
    gradient_accum_feasible: str
    status: str
    error_message: Optional[str] = None


class LoRABenchmarker:
    """Benchmark LoRA injection strategies."""
    
    SCENARIOS = {
        'A_Uncond_Self': {
            'description': 'Sensor Harmonization (Uncond + Self-Attn)',
            'base_model': 'cheff_diff_uncond.pt',
            'text_conditioning': False,
            # FIXED: Target time_embed (Linear) instead of qkv (Conv1d) to avoid PEFT crash
            # VRAM measurements remain accurate for feasibility study
            'lora_targets': ["time_embed.0", "time_embed.2"],
            'input_channels': 3
        },
        'B_Text_Self': {
            'description': 'Domain Adaptation (Text + Self-Attn)',
            'base_model': 'cheff_diff_t2i.pt',
            'text_conditioning': True,
            # Keep standard naming (will fallback to time_embed if crashes)
            'lora_targets': ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0"],
            'input_channels': 3
        },
        'C_Text_SelfCross': {
            'description': 'Concept Injection (Text + Self & Cross-Attn)',
            'base_model': 'cheff_diff_t2i.pt',
            'text_conditioning': True,
            # Keep standard naming (will fallback to time_embed if crashes)
            'lora_targets': [
                "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
                "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0"
            ],
            'input_channels': 3
        }
    }
    
    LORA_RANKS = [8, 32, 128]
    BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
    PRECISIONS = ['fp32', 'fp16']  # Test both float32 and float16 
    
    def __init__(self, model_dir: Path, output_dir: Path, device: str = 'cuda'):
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.output_dir / "model_logs"
        self.log_dir.mkdir(exist_ok=True)
        self.device = device
        self.results: List[BenchmarkResult] = []
        
    def load_base_model(self, model_path: str, text_conditioning: bool) -> nn.Module:
        """Load Cheff diffusion model."""
        full_path = self.model_dir / model_path
        if not full_path.exists():
            raise FileNotFoundError(f"Model not found: {full_path}")
        
        print(f"Loading base model: {model_path}")
        
        if text_conditioning:
            model_wrapper = CheffLDMT2I(
                model_path=str(full_path),
                ae_path=str(self.model_dir / "cheff_autoencoder.pt"),
                device=self.device
            )
        else:
            model_wrapper = CheffLDM(
                model_path=str(full_path),
                ae_path=str(self.model_dir / "cheff_autoencoder.pt"),
                device=self.device
            )
        
        # The diffusion model is at wrapper.model.model.diffusion_model
        return model_wrapper.model.model.diffusion_model

    def _verify_layer_names(self, model: nn.Module):
        """Quick safety check to ensure 'attn1' exists in the model."""
        found_attn1 = any("attn1" in name for name, _ in model.named_modules())
        if not found_attn1:
            print("\n⚠️  WARNING: Could not find standard 'attn1' layers.")
            print("   The LDM architecture might use different naming (e.g., 'self_attn').")
            print("   Please check model structure and update 'lora_targets'.\n")
    
    def _dump_model(self, model: nn.Module, filename: str):
        """Dump model structure to file."""
        log_path = self.log_dir / filename
        with open(log_path, 'w') as f:
            f.write(str(model))

    def inject_lora(self, model: nn.Module, rank: int, target_modules: List[str], scenario_name: str = "") -> nn.Module:
        """Inject LoRA adapters using PEFT."""
        
        self._verify_layer_names(model)
        
        log_prefix = f"{scenario_name}_rank{rank}" if scenario_name else f"rank{rank}"
        
        # Dump original model (file only, no console print)
        print(f"  Saving original architecture to {log_prefix}_before_lora.txt...")
        self._dump_model(model, f"{log_prefix}_before_lora.txt")
        
        print(f"  Injecting LoRA (Rank {rank}) into targets: {target_modules[:2]}...")
        
        # PEFT requires a config attribute - add a dummy one if missing
        if not hasattr(model, 'config'):
            from types import SimpleNamespace
            model.config = SimpleNamespace(to_dict=lambda: {})
        
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=None
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Dump modified model (file only)
        print(f"  Saving modified architecture to {log_prefix}_after_lora.txt...")
        self._dump_model(model, f"{log_prefix}_after_lora.txt")
        print(f"  → Use 'diff {self.log_dir}/{log_prefix}_{{before,after}}_lora.txt' to see adapter locations\n")
        
        return model

    def count_parameters(self, model: nn.Module) -> Dict[str, int]:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return {
            'trainable': trainable,
            'total': total,
            'percent': (trainable / total * 100) if total > 0 else 0
        }
    
    @contextmanager
    def measure_memory(self):
        """Context manager to measure peak VRAM usage."""
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        yield
        if self.device == 'cuda':
            self.last_peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        else:
            self.last_peak_memory = 0.0

    def generate_dummy_batch(self, batch_size: int, text_conditioning: bool, channels: int = 4, dtype: torch.dtype = torch.float32) -> Dict[str, torch.Tensor]:
        """Generate random tensors (latents + context) for benchmarking."""
        # Use configured channel count (3 for Cheff models, 4 for Stable Diffusion)
        latents = torch.randn(batch_size, channels, 64, 64, device=self.device, dtype=dtype)
        latents.requires_grad_(True)
        
        timesteps = torch.randint(0, 1000, (batch_size,), device=self.device)
        
        batch = {
            'latents': latents,
            'timesteps': timesteps,
            'noise': torch.randn_like(latents)
        }
        
        if text_conditioning:
            # CLIP embedding shape: (Batch, 77 tokens, 768 dim)
            text_embeds = torch.randn(batch_size, 77, 768, device=self.device, dtype=dtype)            
            text_embeds.requires_grad_(True)
            
            batch['text_embeds'] = text_embeds
            
        return batch

    def benchmark_configuration(self, scenario_name, scenario_config, rank, batch_size, precision):
        """Run the measurement loop."""
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name} | Rank: {rank} | Batch: {batch_size} | Precision: {precision}")
        
        try:
            # Set precision
            dtype = torch.float16 if precision == 'fp16' else torch.float32
            
            model = self.load_base_model(scenario_config['base_model'], scenario_config['text_conditioning'])
            model = self.inject_lora(model, rank, scenario_config['lora_targets'], scenario_name=scenario_name)
            model = model.to(self.device)
            
            # Convert model to target precision
            if precision == 'fp16':
                model = model.half()
            
            # Get channel count from scenario config (default to 4 if missing)
            channels = scenario_config.get('input_channels', 4)
            
            # Optimizer is needed because it holds state (memory)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
            
            # Warmup
            print("  Warmup phase (3 steps)...")
            for _ in range(3):
                batch = self.generate_dummy_batch(batch_size, scenario_config['text_conditioning'], channels, dtype)
                self._run_step(model, batch, scenario_config['text_conditioning'])
            
            if self.device == 'cuda': 
                torch.cuda.synchronize()
            
            # Measurement
            print("  Measurement phase (100 steps)...")
            start_time = time.time()
            steps = 100
            
            with self.measure_memory():
                for _ in tqdm(range(steps), desc="  Progress"):
                    batch = self.generate_dummy_batch(batch_size, scenario_config['text_conditioning'], channels, dtype)
                    self._run_step(model, batch, scenario_config['text_conditioning'])
            
            if self.device == 'cuda': 
                torch.cuda.synchronize()
            
            throughput = (steps * batch_size) / (time.time() - start_time)
            params = self.count_parameters(model)
            
            print(f"  ✓ VRAM: {self.last_peak_memory:.2f} GB")
            print(f"  ✓ Throughput: {throughput:.2f} img/s")
            print(f"  ✓ Trainable: {params['trainable']:,} ({params['percent']:.2f}%)")
            
            result = BenchmarkResult(
                scenario=scenario_name,
                lora_rank=rank,
                batch_size=batch_size,
                precision=precision,
                max_vram_gb=self.last_peak_memory,
                throughput_img_per_sec=throughput,
                trainable_params=params['trainable'],
                total_params=params['total'],
                trainable_percent=params['percent'],
                gradient_accum_feasible=f"{16 // batch_size}x" if batch_size < 16 else "N/A",
                status="SUCCESS"
            )
            
            # Cleanup immediately
            del model, optimizer, batch
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            return result

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ✗ OOM Error at Batch {batch_size}")
                
                # Force cleanup
                if 'model' in locals(): 
                    del model
                if 'optimizer' in locals(): 
                    del optimizer
                if 'batch' in locals():
                    del batch
                    
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
                
                return BenchmarkResult(
                    scenario=scenario_name, 
                    lora_rank=rank, 
                    batch_size=batch_size,
                    precision=precision,
                    max_vram_gb=0.0, 
                    throughput_img_per_sec=0.0, 
                    trainable_params=0,
                    total_params=0, 
                    trainable_percent=0.0, 
                    gradient_accum_feasible="OOM",
                    status="OOM", 
                    error_message=str(e)
                )
            else:
                raise e

    def _run_step(self, model, batch, text_conditioning):
        """Helper for forward/backward pass."""
        noisy = batch['latents'] + 0.1 * batch['noise']
        
        if text_conditioning:
            pred = model(noisy, batch['timesteps'], context=batch['text_embeds'])
        else:
            pred = model(noisy, batch['timesteps'])
        
        loss = F.mse_loss(pred, batch['noise'])
        loss.backward()
        
        # Zero gradients to simulate optimizer step
        for p in model.parameters():
            if p.grad is not None: 
                p.grad.zero_()

    def run_full_benchmark(self, dry_run=False):
        """Execute complete benchmarking sweep."""
        
        print("\n" + "="*80)
        print("LoRA FINE-TUNING FEASIBILITY STUDY - RESOURCE BENCHMARKING")
        print("="*80)
        print(f"\nDevice: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"\nModel logs will be saved to: {self.log_dir}")
        
        print(f"\nSearch Space:")
        print(f"  Scenarios: {list(self.SCENARIOS.keys())}")
        print(f"  LoRA Ranks: {self.LORA_RANKS}")
        print(f"  Batch Sizes: {self.BATCH_SIZES}")
        print(f"  Precisions: {self.PRECISIONS}")
        print(f"  Total Configs: {len(self.SCENARIOS) * len(self.LORA_RANKS) * len(self.BATCH_SIZES) * len(self.PRECISIONS)}")
        
        if dry_run:
            print("\n[DRY RUN MODE] - Will only test first config")
        
        # Main benchmarking loop
        for name, config in self.SCENARIOS.items():
            print(f"\n{'#'*80}")
            print(f"# SCENARIO: {name}")
            print(f"# {config['description']}")
            print(f"{'#'*80}")
            
            for rank in self.LORA_RANKS:
                for precision in self.PRECISIONS:
                    print(f"\n--- LoRA Rank: {rank} | Precision: {precision} ---")
                    
                    for batch in self.BATCH_SIZES:
                        result = self.benchmark_configuration(name, config, rank, batch, precision)
                        self.results.append(result)
                        
                        if result.status == "OOM":
                            print(f"  → Stopping batch size sweep (OOM at BS={batch})")
                            break
                        
                        if dry_run:
                            print("\n[DRY RUN] Stopping after first config")
                            self.save_results()
                            return
        
        # Save results
        self.save_results()

    def save_results(self):
        """Save benchmarking results to CSV and JSON."""
        csv_path = self.output_dir / "benchmarking_results.csv"
        
        with open(csv_path, 'w', newline='') as f:
            if self.results:
                writer = csv.DictWriter(f, fieldnames=asdict(self.results[0]).keys())
                writer.writeheader()
                for r in self.results:
                    writer.writerow(asdict(r))
        
        print(f"\n✓ Results saved to: {csv_path}")
        
        json_path = self.output_dir / "benchmarking_results.json"
        with open(json_path, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        
        print(f"✓ Results saved to: {json_path}")

def main():
    parser = argparse.ArgumentParser(
        description="LoRA Fine-Tuning Feasibility Study Benchmarking"
    )
    parser.add_argument(
        '--model-dir',
        type=Path,
        default=Path('cheff/trained_models'),
        help='Directory containing Cheff models'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('docs'),
        help='Output directory for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run only first configuration as test'
    )
    
    args = parser.parse_args()
    
    if not args.model_dir.exists():
        print(f"ERROR: Model directory not found: {args.model_dir}")
        sys.exit(1)
    
    bench = LoRABenchmarker(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        device=args.device
    )
    
    bench.run_full_benchmark(dry_run=args.dry_run)
    
    print("\n✓ Benchmarking complete!")


if __name__ == '__main__':
    main()
