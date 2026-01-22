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
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from cheff.peft_modules.inject_lora import apply_lora_peft
except ImportError:
    sys.path.append(str(Path(__file__).parent / '..' / 'cheff' / 'peft_modules'))
    from inject_lora import apply_lora_peft

def robust_checkpoint_wrapper(func, inputs, params, flag):
    return torch.utils.checkpoint.checkpoint(func, *inputs, use_reentrant=False)

try:
    import cheff.ldm.modules.diffusionmodules.util as cheff_util
    print(">> Applying monkey-patch to gradient checkpointing (Pre-Import)...")
    cheff_util.checkpoint = robust_checkpoint_wrapper
except ImportError:
    print(">> WARNING: Could not apply monkey-patch. Dependencies might be missing.")

from cheff.ldm.inference import CheffLDMT2I

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

class LoRABenchmarkerT2I:
    
    SCENARIOS = {
        'B_Text_Self': {
            'description': 'Domain Adaptation (Text + Self-Attn Only)',
            'base_model': 'cheff_diff_t2i.pt',
            'input_channels': 3,
            'lora_config': {
                'adaptation_scope': 'attn',
                'target_modules': ['to_q', 'to_k', 'to_v', 'to_out.0'],
                'ff_modules': [],
                'dropout': 0.05
            }
        },
        'C_Cross_Only': {
            'description': 'Pure Concept Injection (Cross-Attn Only)',
            'base_model': 'cheff_diff_t2i.pt',
            'input_channels': 3,
            'lora_config': {
                'adaptation_scope': 'cross',
                'target_modules': ['to_q', 'to_k', 'to_v', 'to_out.0'],
                'ff_modules': [],
                'dropout': 0.05
            }
        }
    }
    
    LORA_RANKS = [8, 32, 64, 128]
    BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
    PRECISIONS = ['fp32']
    
    def __init__(self, model_dir: Path, output_dir: Path, device: str = 'cuda'):
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.results: List[BenchmarkResult] = []
        
    def load_base_model(self, model_path: str):
        full_path = self.model_dir / model_path
        if not full_path.exists():
            raise FileNotFoundError(f"Model not found: {full_path}")
        
        print(f"Loading base model: {model_path}")
        model_wrapper = CheffLDMT2I(
            model_path=str(full_path),
            ae_path=str(self.model_dir / "cheff_autoencoder.pt"),
            device=self.device
        )
        return model_wrapper

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
        latents = torch.randn(batch_size, channels, 64, 64, device=self.device, dtype=dtype)
        latents.requires_grad_(True)
        
        timesteps = torch.randint(0, 1000, (batch_size,), device=self.device)
        
        text_embeds = torch.randn(batch_size, 77, 1280, device=self.device, dtype=dtype)
        text_embeds.requires_grad_(True)
        
        return {
            'latents': latents,
            'timesteps': timesteps,
            'noise': torch.randn_like(latents, dtype=dtype),
            'text_embeds': text_embeds
        }

    def benchmark_configuration(self, name, config, rank, batch_size, precision):
        print(f"\nScenario: {name} | Rank: {rank} | Batch: {batch_size}")
        try:
            dtype = torch.float16 if precision == 'fp16' else torch.float32
            
            model_wrapper = self.load_base_model(config['base_model'])
            
            conf_data = config['lora_config']
            lora_config_obj = SimpleNamespace(
                rank=rank,
                alpha=rank,
                adaptation_scope=conf_data['adaptation_scope'],
                target_modules=conf_data['target_modules'],
                ff_modules=conf_data['ff_modules'],
                dropout=conf_data['dropout']
            )

            model_wrapper = apply_lora_peft(model_wrapper, lora_config_obj)
            
            model = model_wrapper.model.model.diffusion_model
            model = model.to(self.device).train()
            if precision == 'fp16': model = model.half()
            
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
            
            print("  Warmup phase...")
            for _ in range(3):
                batch = self.generate_dummy_batch(batch_size, config['input_channels'], dtype)
                self._run_step(model, batch)
            
            print("  Measurement phase...")
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
        pred = model(noisy, batch['timesteps'], context=batch['text_embeds'])
        loss = F.mse_loss(pred, batch['noise'])
        loss.backward()
        for p in model.parameters(): p.grad = None

    def run(self):
        for name, config in self.SCENARIOS.items():
            for rank in self.LORA_RANKS:
                for batch in self.BATCH_SIZES:
                    self.results.append(self.benchmark_configuration(name, config, rank, batch, 'fp32'))
        
        csv_path = self.output_dir / "benchmarking_results_t2i.csv"
        with open(csv_path, 'w', newline='') as f:
            if self.results:
                writer = csv.DictWriter(f, fieldnames=asdict(self.results[0]).keys())
                writer.writeheader()
                for r in self.results: writer.writerow(asdict(r))
        print(f"\n✓ Saved to {csv_path}")

if __name__ == '__main__':
    LoRABenchmarkerT2I(Path('cheff/trained_models'), Path('docs')).run()
