import csv
import gc
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

# --- Configuration Constants ---
TEXT_EMBED_DIM = 1280   # OpenCLIP dimension (SD 2.x standard)
INPUT_CHANNELS = 3      # Latent channels for Cheff models
LORA_RANKS = [8, 32, 64, 128]
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
from cheff.peft_modules.inject_lora import apply_lora_peft

def patch_gradient_checkpointing():
    """
    Patches OpenAI's legacy checkpoint function to handle frozen layers.
    Without this, backprop crashes on tensors requiring no gradients.
    """
    def robust_checkpoint_wrapper(func, inputs, params, flag):
        return torch.utils.checkpoint.checkpoint(func, *inputs, use_reentrant=False)

    print(">> System: Patching legacy gradient checkpointing...")
    try:
        import cheff.ldm.modules.diffusionmodules.util as cheff_util
        import cheff.ldm.modules.attention as cheff_attn
        import cheff.ldm.modules.diffusionmodules.openaimodel as cheff_openai

        cheff_util.checkpoint = robust_checkpoint_wrapper
        cheff_attn.checkpoint = robust_checkpoint_wrapper
        if hasattr(cheff_openai, 'checkpoint'):
            cheff_openai.checkpoint = robust_checkpoint_wrapper
        print("   ✓ Patch applied successfully.")
    except ImportError as e:
        print(f"   ⚠️ WARNING: Patching failed: {e}")

# Apply patch before loading model classes
patch_gradient_checkpointing()

# Now safe to import model wrappers
from cheff.ldm.inference import CheffLDMT2I

# Filter noise
warnings.filterwarnings("ignore", category=FutureWarning)


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
    """Benchmark suite for Text-to-Image LoRA fine-tuning."""

    SCENARIOS = {
        'B_Text_Self': {
            'base_model': 'cheff_diff_t2i.pt',
            'lora_config': {
                'adaptation_scope': 'attn',
                'target_modules': ['to_q', 'to_k', 'to_v', 'to_out.0'],
                'ff_modules': [],
                'dropout': 0.05
            }
        },
        'C_Cross_Only': {
            'base_model': 'cheff_diff_t2i.pt',
            'lora_config': {
                'adaptation_scope': 'cross',
                'target_modules': ['to_q', 'to_k', 'to_v', 'to_out.0'],
                'ff_modules': [],
                'dropout': 0.05
            }
        }
    }

    def __init__(self, model_dir: Path, output_dir: Path, device: str = 'cuda'):
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.results: List[BenchmarkResult] = []

    def load_base_model(self, model_path: str):
        full_path = self.model_dir / model_path
        print(f"Loading base model: {model_path}")
        return CheffLDMT2I(
            model_path=str(full_path),
            ae_path=str(self.model_dir / "cheff_autoencoder.pt"),
            device=self.device
        )

    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return {
            'trainable': trainable, 
            'total': total, 
            'percent': (trainable / total * 100) if total > 0 else 0
        }

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

    def generate_batch(self, batch_size: int, dtype: torch.dtype):
        """Generates dummy inputs with correct dimensions for the T2I model."""
        latents = torch.randn(batch_size, INPUT_CHANNELS, 64, 64, device=self.device, dtype=dtype)
        latents.requires_grad_(True)
        
        timesteps = torch.randint(0, 1000, (batch_size,), device=self.device)
        
        text_embeds = torch.randn(batch_size, 77, TEXT_EMBED_DIM, device=self.device, dtype=dtype)
        text_embeds.requires_grad_(True)
        
        return {
            'latents': latents,
            'timesteps': timesteps,
            'noise': torch.randn_like(latents),
            'text_embeds': text_embeds
        }

    def _run_step(self, model, batch):
        noisy = batch['latents'] + 0.1 * batch['noise']
        pred = model(noisy, batch['timesteps'], context=batch['text_embeds'])
        loss = F.mse_loss(pred, batch['noise'])
        loss.backward()
        for p in model.parameters(): 
            p.grad = None

    def benchmark_config(self, name, config, rank, batch_size):
        print(f"\nScenario: {name} | Rank: {rank} | Batch: {batch_size}")
        try:
            dtype = torch.float32
            
            wrapper = self.load_base_model(config['base_model'])
            
            lora_conf = SimpleNamespace(
                rank=rank, alpha=rank, 
                **config['lora_config']
            )
            
            apply_lora_peft(wrapper.model, lora_conf)
            
            model = wrapper.model.model.diffusion_model.to(self.device).train()
            
            print("  • Warmup...")
            for _ in range(3):
                batch = self.generate_batch(batch_size, dtype)
                self._run_step(model, batch)
            
            print("  • Measuring...")
            start_time = time.time()
            steps = 100
            with self.measure_memory():
                for _ in range(steps):
                    batch = self.generate_batch(batch_size, dtype)
                    self._run_step(model, batch)
            
            throughput = (steps * batch_size) / (time.time() - start_time)
            params = self.count_parameters(model)
            
            print(f"  ✓ VRAM: {self.last_peak_memory:.2f} GB | Speed: {throughput:.2f} img/s")
            
            del model, wrapper
            gc.collect()
            torch.cuda.empty_cache()

            return BenchmarkResult(
                scenario=name, lora_rank=rank, batch_size=batch_size, precision="fp32",
                max_vram_gb=self.last_peak_memory, throughput_img_per_sec=throughput,
                trainable_params=params['trainable'], total_params=params['total'], 
                trainable_percent=params['percent'], status="SUCCESS"
            )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ✗ OOM Error")
                torch.cuda.empty_cache()
                return BenchmarkResult(
                    scenario=name, lora_rank=rank, batch_size=batch_size, precision="fp32",
                    max_vram_gb=0, throughput_img_per_sec=0, trainable_params=0, 
                    total_params=0, trainable_percent=0, status="OOM", error_message=str(e)
                )
            raise e

    def run(self):
        print("="*60)
        print("LoRA FEASIBILITY BENCHMARK (T2I)")
        print("="*60)
        
        for name, config in self.SCENARIOS.items():
            for rank in LORA_RANKS:
                for batch in BATCH_SIZES:
                    self.results.append(self.benchmark_config(name, config, rank, batch))
        
        csv_path = self.output_dir / "benchmarking_results_t2i.csv"
        with open(csv_path, 'w', newline='') as f:
            if self.results:
                writer = csv.DictWriter(f, fieldnames=asdict(self.results[0]).keys())
                writer.writeheader()
                for r in self.results: 
                    writer.writerow(asdict(r))
        print(f"\n✓ Results saved to {csv_path}")

if __name__ == '__main__':
    LoRABenchmarkerT2I(
        model_dir=Path('cheff/trained_models'), 
        output_dir=Path('docs')
    ).run()
