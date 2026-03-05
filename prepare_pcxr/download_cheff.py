import os
import requests
import argparse
from tqdm import tqdm
from pathlib import Path

# Configuration
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
DEFAULT_CHECKPOINT_DIR = _PROJECT_ROOT / "cheff_peft" / "checkpoints"

MODELS = {
    "cheff_diff_t2i.pt": "https://syncandshare.lrz.de/dl/fi4R87B3cEWgSx4Wivyizb/cheff_diff_t2i.pt",
    "cheff_autoencoder.pt": "https://syncandshare.lrz.de/dl/fiQ6wTe7K7otQzyifNh9av/cheff_autoencoder.pt",
    "cheff_sr_fine.pt": "https://syncandshare.lrz.de/dl/fiHM4uAfy7uxcfBXkefySJ/cheff_sr_fine.pt"
}

def download_file(url, dest):
    if dest.exists():
        print(f"File {dest.name} already exists. Skipping download.")
        return

    print(f"Downloading {dest.name}...")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest.name) as pbar:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        print(f"Successfully downloaded {dest.name}")
    except Exception as e:
        if dest.exists():
            os.remove(dest)
        print(f"Error downloading {dest.name}: {e}")

def main(args):
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving Pretrained Models to: {checkpoint_dir}")
    
    for filename, url in MODELS.items():
        dest = checkpoint_dir / filename
        download_file(url, dest)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CheFF pretrained models.")
    parser.add_argument('--checkpoint_dir', type=str, default=str(DEFAULT_CHECKPOINT_DIR))
    
    args = parser.parse_args()
    main(args)
