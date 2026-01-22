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
import torch.utils.checkpoint
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from cheff.ldm.inference import CheffLDM
import cheff.ldm.modules.diffusionmodules.util as cheff_util

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    print("ERROR: peft library not installed. Run: pip install peft")
    sys.exit(1)

# Monkey-patch for gradient checkpointing stability
def robust_checkpoint_wrapper(func, inputs, params, flag):
    return torch.utils.checkpoint.checkpoint(func, *inputs, use_reentrant=False)

print(">> Applying monkey-patch to gradient checkpointing...")
cheff_util.checkpoint = robust_checkpoint_wrapper

@dataclass
class BenchmarkResult:
    scenario: str
    lora_rank: int
    batch_size: int
    precision: str
    max_vram_gb: float
    throughput_img_per_sec: float
    trainable_params: int
    total_params: int
    trainable_percent: float
    status: str
    error_message: Optional[str] = None

class LoRABenchmarkerUncond:
    """Benchmark for Unconditional Model (Scenario A)."""
    
    # SCENARIO A ONLY
    SCENARIOS = {
        'A_Uncond_Self': {
            'description': 'Sensor Harmonization (Uncond + Self-Attn)',
            'base_model': 'cheff_diff_uncond.pt',
            'input_channels': 3,
            # STRATEGY: ResNet-Heavy. 
            # We target Time Embeds (Linear) and ResNet Features (Conv2d).
            # We SKIP 'qkv' (Attention) because it uses Conv1d which causes crashes.
            'lora_targets': [
                "time_embed.0", "time_embed.2", 
                "in_layers.2", "out_layers.3", "out.2"
            ]
        }
    }
    
    LORA_RANKS = [8, 32, 128]
    BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
    PRECISIONS = ['fp32'] 
    
    def __init__(self, model_dir: Path, output_dir: Path, device: str = 'cuda'):
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.output_dir / "model_logs_uncond"
        self.log_dir.mkdir(exist_ok=True)
        self.device = device
        self.results: List[BenchmarkResult] = []
        
    def load_base_model(self, model_path: str) -> nn.Module:
        full_path = self.model_dir / model_path
        if not full_path.exists():
            raise FileNotFoundError(f"Model not found: {full_path}")
        
        print(f"Loading base model: {model_path}")
        # Always use Unconditional Wrapper
        model_wrapper = CheffLDM(
            model_path=str(full_path),
            ae_path=str(self.model_dir / "cheff_autoencoder.pt"),
            device=self.device
        )
        return model_wrapper.model.model.diffusion_model

    def inject_lora(self, model: nn.Module, rank: int, target_modules: List[str]) -> nn.Module:
        print(f"  Injecting LoRA (Rank {rank}) into targets: {target_modules[:2]}...")
        
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
        return get_peft_model(model, lora_config)

    def count_parameters(self, model: nn.Module) -> Dict[str, int]:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return {'trainable': trainable, 'total': total, 'percent': (trainable / total * 100) if total > 0 else 0}
    
    @contextmanager
    def measure_memory(self):
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        yield
        if self.device == 'cuda':
            self.last_peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        else:
            self.last_peak_memory = 0.0

    def generate_dummy_batch(self, batch_size: int, channels: int = 4, dtype: torch.dtype = torch.float32):
        # NO TEXT EMBEDDINGS needed here
        latents = torch.randn(batch_size, channels, 64, 64, device=self.device, dtype=dtype)
        timesteps = torch.randint(0, 1000, (batch_size,), device=self.device)
        return {
            'latents': latents,
            'timesteps': timesteps,
            'noise': torch.randn_like(latents, dtype=dtype)
        }

    def benchmark_configuration(self, name, config, rank, batch_size, precision):
        print(f"\nScenario: {name} | Rank: {rank} | Batch: {batch_size}")
        try:
            dtype = torch.float16 if precision == 'fp16' else torch.float32
            model = self.load_base_model(config['base_model'])
            model = self.inject_lora(model, rank, config['lora_targets'])
            model = model.to(self.device).train()
            if precision == 'fp16': model = model.half()
            
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
            
            # Warmup
            for _ in range(3):
                batch = self.generate_dummy_batch(batch_size, config['input_channels'], dtype)
                self._run_step(model, batch)
            
            # Measure
            start_time = time.time()
            steps = 50 
            with self.measure_memory():
                for _ in range(steps):
                    batch = self.generate_dummy_batch(batch_size, config['input_channels'], dtype)
                    self._run_step(model, batch)
            
            throughput = (steps * batch_size) / (time.time() - start_time)
            params = self.count_parameters(model)
            
            print(f"  ✓ VRAM: {self.last_peak_memory:.2f} GB | Throughput: {throughput:.2f} img/s")
            
            return BenchmarkResult(
                scenario=name, lora_rank=rank, batch_size=batch_size, precision=precision,
                max_vram_gb=self.last_peak_memory, throughput_img_per_sec=throughput,
                trainable_params=params['trainable'], total_params=params['total'], 
                trainable_percent=params['percent'], status="SUCCESS"
            )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ✗ OOM Error at Batch {batch_size}")
                torch.cuda.empty_cache()
                return BenchmarkResult(name, rank, batch_size, precision, 0, 0, 0, 0, 0, "OOM", str(e))
            raise e

    def _run_step(self, model, batch):
        noisy = batch['latents'] + 0.1 * batch['noise']
        pred = model(noisy, batch['timesteps']) # No context arg
        loss = F.mse_loss(pred, batch['noise'])
        loss.backward()
        for p in model.parameters(): p.grad = None

    def run(self):
        for name, config in self.SCENARIOS.items():
            for rank in self.LORA_RANKS:
                for batch in self.BATCH_SIZES:
                    self.results.append(self.benchmark_configuration(name, config, rank, batch, 'fp32'))
        
        csv_path = self.output_dir / "benchmarking_results_uncond.csv"
        with open(csv_path, 'w', newline='') as f:
            if self.results:
                writer = csv.DictWriter(f, fieldnames=asdict(self.results[0]).keys())
                writer.writeheader()
                for r in self.results: writer.writerow(asdict(r))
        print(f"\n✓ Saved to {csv_path}")

if __name__ == '__main__':
    LoRABenchmarkerUncond(Path('cheff/trained_models'), Path('docs')).run()
