import argparse
import os
import pandas as pd
import requests
from tqdm.auto import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from prepare_pcxr.config import cfg

def physionet_login(session, username, password):
    print(f"Logging in to PhysioNet as {username}...")
    login_url = "https://physionet.org/login/"
    
    try:
        session.get(login_url, timeout=10)
        csrftoken = session.cookies.get('csrftoken')
        
        login_data = {
            'username': username,
            'password': password,
            'csrfmiddlewaretoken': csrftoken,
            'next': '/content/vindr-pcxr/1.0.0/'
        }
        
        response = session.post(login_url, data=login_data, headers={'Referer': login_url}, timeout=10)
        if response.status_code != 200:
            print(f"Warning: Login returned status code {response.status_code}. Double check credentials.")
        else:
            print("Login request sent successfully.")
    except Exception as e:
        print(f"Login failed: {e}")
    return session

def download_checksum(session, dicom_dir):
    checksum_url = "https://physionet.org/files/vindr-pcxr/1.0.0/SHA256SUMS.txt"
    dest = Path(dicom_dir) / "SHA256SUMS.txt"
    
    if dest.exists():
        print("Checksum file already exists, skipping download.")
        return
    
    print("Downloading checksum file...")
    try:
        res = session.get(checksum_url, timeout=20)
        if res.status_code == 200:
            dest.write_bytes(res.content)
            print("Checksum file downloaded successfully.")
        else:
            print(f"Failed to download checksum file (Status {res.status_code})")
    except Exception as e:
        print(f"Error downloading checksum file: {e}")

def download_metadata(session, split_dir, split):
    targets = [f"image_labels_{split}.csv", f"annotations_{split}.csv"]
    base_url = "https://physionet.org/files/vindr-pcxr/1.0.0/"
    
    split_dir.mkdir(parents=True, exist_ok=True)
    
    for filename in targets:
        dest = split_dir / filename
        if dest.exists():
            continue
        
        print(f"  Downloading {filename}...")
        res = session.get(f"{base_url}{filename}", timeout=60)
        if res.status_code == 200:
            dest.write_bytes(res.content)
        else:
            print(f"  ✖ Failed to download {filename} (Status {res.status_code})")

def download_image(session, url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return True
    
    try:
        response = session.get(url, stream=True, timeout=60)
        if response.status_code == 200:
            with open(dest, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
            return True
        return False
    except Exception:
        return False

def main(args):
    pcxr_dicom_root = Path(args.pcxr_dicom_root)
    split_dir = pcxr_dicom_root / args.split
    label_csv = split_dir / f"image_labels_{args.split}.csv"
    base_url = f"https://physionet.org/files/vindr-pcxr/1.0.0/{args.split}/"

    print(f"--- Mode: {args.split.upper()} | Target: {split_dir} ---")

    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=args.workers, pool_maxsize=args.workers)
    session.mount('https://', adapter)
    session = physionet_login(session, args.username, args.password)

    print("\nChecking metadata files...")
    download_metadata(session, split_dir, args.split)

    if not label_csv.exists():
        print(f"Error: {label_csv} not found. Metadata download failed.")
        return

    df = pd.read_csv(label_csv)
    image_ids = df['image_id'].unique()
    print(f"Found {len(image_ids)} images to download.")

    print(f"Downloading images using {args.workers} workers...")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_image, 
                session, 
                f"{base_url}{img_id}.dicom", 
                split_dir / f"{img_id}.dicom"
            ): img_id for img_id in image_ids
        }
        
        with tqdm(total=len(futures), desc=f"Downloading {args.split}") as pbar:
            for future in as_completed(futures):
                try:
                    if not future.result():
                        pass 
                except Exception:
                    pass
                pbar.update(1)

    download_checksum(session, pcxr_dicom_root)

    print("\nProcess complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streamlined VinDr-PCXR downloader.")
    parser.add_argument('--username', type=str, default=cfg.physio_username)
    parser.add_argument('--password', type=str, default=cfg.physio_password)
    parser.add_argument('--split', type=str, choices=['train', 'test'], default='test')
    parser.add_argument('--pcxr_dicom_root', type=str, default=cfg.pcxr_dicom_root)
    parser.add_argument('--workers', type=int, default=cfg.num_workers)

    args = parser.parse_args()

    if not args.username or not args.password:
        parser.error("Username and password must be provided.")

    main(args)
