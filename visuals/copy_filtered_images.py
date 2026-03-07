import os
import json
import shutil
from tqdm import tqdm

def main():
    source_root = "data/synthetic"
    dest_root = "data/synthetic_filtered"
    
    # Pathologies to check
    pathologies = ["Brocho-pneumonia", "Bronchiolitis", "Bronchitis", "Pneumonia"]
    
    if not os.path.exists(dest_root):
        os.makedirs(dest_root)
        print(f"Created {dest_root}")
    
    for pathology in pathologies:
        pathology_dir = os.path.join(source_root, pathology)
        filtered_paths_json = os.path.join(pathology_dir, "filtered_paths.json")
        
        if not os.path.exists(filtered_paths_json):
            print(f"Skipping {pathology}: filtered_paths.json not found")
            continue
            
        print(f"Processing {pathology}...")
        
        with open(filtered_paths_json, "r") as f:
            filtered_paths = json.load(f)
            
        dest_pathology_dir = os.path.join(dest_root, pathology)
        if not os.path.exists(dest_pathology_dir):
            os.makedirs(dest_pathology_dir)
            
        # The values in filtered_paths are filenames like "000001.png"
        filenames = list(filtered_paths.values())
        
        count = 0
        for filename in tqdm(filenames, desc=f"Copying {pathology}"):
            src_file = os.path.join(pathology_dir, filename)
            dest_file = os.path.join(dest_pathology_dir, filename)
            
            if os.path.exists(src_file):
                shutil.copy2(src_file, dest_file)
                count += 1
            else:
                print(f"Warning: {src_file} does not exist")
                
        print(f"Copied {count} images for {pathology}")

if __name__ == "__main__":
    main()
