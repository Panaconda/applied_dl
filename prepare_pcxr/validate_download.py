import hashlib
import os
import sys
import argparse

def calculate_sha256(filepath):
    """Calculates the SHA256 hash of a file in chunks."""
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except FileNotFoundError:
        return None

def verify_and_clean(base_dir):
    """Verifies files in base_dir against a SHA256SUMS.txt file."""
    checksum_file = os.path.join(base_dir, "SHA256SUMS.txt")
    
    if not os.path.exists(checksum_file):
        print(f"Error: {checksum_file} not found.")
        return False

    print(f"--- Starting Quality Check in {base_dir} ---")
    missing_or_corrupt = False
    files_checked = 0

    with open(checksum_file, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) < 2:
                continue

            expected_hash, rel_path = parts
            
            # --- NEW: Skip metadata, licenses, and non-DICOM files ---
            if not (rel_path.lower().endswith('.dicom') or rel_path.lower().endswith('.dcm')):
                continue

            full_path = os.path.normpath(os.path.join(base_dir, rel_path))

            if os.path.exists(full_path):
                print(f"Checking: {rel_path}...", end="\r")
                actual_hash = calculate_sha256(full_path)
                files_checked += 1

                if actual_hash != expected_hash:
                    # Clear the line before printing the error so it doesn't overlap
                    print(" " * 80, end="\r") 
                    print(f"[CORRUPT] {rel_path} - Hash mismatch! Deleting...")
                    os.remove(full_path)
                    missing_or_corrupt = True
            else:
                print(" " * 80, end="\r") 
                print(f"[MISSING] {rel_path}")
                missing_or_corrupt = True

    # Clear the "Checking..." line at the very end
    print(" " * 80, end="\r")

    if missing_or_corrupt:
        print(f"--- Scan Complete ({files_checked} files checked). Some DICOMs are MISSING or CORRUPT. ---")
        return False
    
    print(f"--- Scan Complete ({files_checked} files checked). All DICOM files verified! ---")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify DICOM file integrity via SHA256.")

    parser.add_argument(
        "--dicom_dir", 
        type=str, 
        required=True, 
        help="The path to the directory containing DICOM files and SHA256SUMS.txt"
    )

    args = parser.parse_args()
    target_dir = os.path.abspath(args.dicom_dir)

    if not os.path.isdir(target_dir):
        print(f"Error: {target_dir} is not a valid directory.")
        sys.exit(1)

    if not verify_and_clean(target_dir):
        sys.exit(1)