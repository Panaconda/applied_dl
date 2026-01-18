#!/usr/bin/env python3
"""
Generate 50 sample chest X-rays from clinical prompts with super-resolution.
Saves images to samples/ directory and tracks metadata in samples_metadata.json
"""

import os
import json
import sys
from datetime import datetime
from pathlib import Path

# Add cheff to path
sys.path.insert(0, str(Path(__file__).parent / "cheff"))

import torch
from cheff.ldm.inference import CheffLDMT2I
from cheff.sr.sampler import CheffSRModel
from torchvision.utils import save_image
from torchvision.transforms.functional import rgb_to_grayscale


# 50 clinical prompts
PROMPTS = [
    "Normal chest radiograph. Lungs are clear without focal consolidation, pleural effusion, or pneumothorax. Cardiomediastinal silhouette is within normal limits. No acute osseous abnormality identified.",
    "Mild bilateral basal atelectasis with low lung volumes. No focal consolidation. Heart size at the upper limits of normal. No pleural effusion or pneumothorax.",
    "Right lower lobe pneumonia with airspace consolidation and air bronchograms. Trace right-sided pleural effusion. No pneumothorax.",
    "Left upper lobe consolidation suspicious for lobar pneumonia. Cardiac silhouette normal. No pleural effusion or pneumothorax.",
    "Diffuse bilateral reticular opacities compatible with interstitial lung disease. No focal consolidation. Heart size normal. No significant pleural effusion.",
    "Bilateral perihilar ground-glass opacities with mild vascular redistribution, consistent with pulmonary edema. Small bilateral pleural effusions. Enlarged cardiac silhouette.",
    "Marked cardiomegaly with pulmonary vascular congestion and prominent upper lobe vessels. Small right and left pleural effusions. No focal consolidation.",
    "Large left pleural effusion causing left lower lobe atelectasis and mild mediastinal shift to the right. No visible pneumothorax.",
    "Moderate right hydropneumothorax with air–fluid level and partial collapse of the right lung. Left lung clear. Cardiomediastinal silhouette within normal limits.",
    "Large right tension pneumothorax with near-complete collapse of the right lung and leftward mediastinal shift. Left lung appears hyperinflated.",
    "Upper lobe–predominant centrilobular emphysema with hyperinflated lungs and flattened hemidiaphragms. No focal consolidation or pleural effusion.",
    "Chronic obstructive pulmonary disease with hyperlucent lungs, increased retrosternal airspace, and prominent pulmonary arteries. No acute infiltrate.",
    "Fine reticular and honeycombing pattern in the bilateral subpleural lower lobes, compatible with pulmonary fibrosis. Mildly decreased lung volumes.",
    "Multiple bilateral pulmonary nodules up to 1.5 cm, predominantly in the lower lobes, suspicious for metastatic disease. No pleural effusion.",
    "Solitary 2 cm spiculated nodule in the right upper lobe. No associated pleural effusion or lymphadenopathy evident on radiograph.",
    "Patchy bilateral peripheral ground-glass opacities, more pronounced in the lower zones, compatible with atypical or viral pneumonia pattern.",
    "Right upper lobe cavitary lesion with thick, irregular walls, suspicious for cavitary tuberculosis. Adjacent fibrotic changes and volume loss.",
    "Multiple small calcified nodules throughout both lungs, consistent with healed granulomatous disease. No acute infiltrate.",
    "Bilateral hilar and mediastinal prominence with reticulonodular opacities, compatible with sarcoidosis stage II. No pleural effusion.",
    "Diffuse bilateral miliary nodular pattern, suggesting miliary tuberculosis or disseminated fungal infection. Cardiac silhouette normal.",
    "Postoperative changes from median sternotomy with sternal wires and prosthetic heart valve. Mild cardiomegaly. Lungs clear without focal consolidation.",
    "Left-sided dual-lead pacemaker in appropriate position. Mild pulmonary venous congestion. No focal consolidation or effusion.",
    "Endotracheal tube tip 3 cm above the carina. Right internal jugular central venous catheter with tip at the cavoatrial junction. Mild bilateral perihilar opacities.",
    "Right chest tube in place with re-expanded right lung and small residual apical pneumothorax. Lungs otherwise clear.",
    "Right-sided Port-a-Cath with catheter tip in the superior vena cava. No pneumothorax. Mild left basal atelectasis.",
    "Kyphoscoliosis of the thoracic spine with associated volume loss of the left hemithorax and crowding of ribs. No focal consolidation or effusion.",
    "Multiple anterior rib fractures on the left with overlying chest wall soft tissue swelling. Small left pleural effusion without visible pneumothorax.",
    "Chronic healed rib fractures bilaterally with callus formation. Lungs clear. Heart size within normal limits.",
    "Hiatal hernia projecting behind the heart with retrocardiac air–fluid level. Lungs clear, no pleural effusion.",
    "Marked hyperinflation with flattened diaphragms and bullous changes in the upper lobes, consistent with advanced emphysema.",
    "Left lower lobe consolidation with air bronchograms and silhouetting of the left hemidiaphragm. Small adjacent parapneumonic effusion.",
    "Right middle lobe collapse with triangular opacity silhouetting the right heart border and volume loss in the right hemithorax.",
    "Bilateral lower lobe bronchiectasis with tram-track opacities and ring shadows. No significant pleural effusion.",
    "Right apical pleural thickening with associated volume loss and elevated minor fissure, compatible with prior granulomatous disease.",
    "Diffuse hazy opacities with air bronchograms and decreased lung volumes, suggesting acute respiratory distress syndrome pattern.",
    "Prominent pulmonary arteries with pruning of peripheral vessels, compatible with pulmonary arterial hypertension. Heart size mildly enlarged.",
    "Right main pulmonary artery enlargement with peripheral oligemia in the right lung, suspicious for chronic thromboembolic pulmonary hypertension.",
    "Cardiomegaly with left ventricular configuration and pulmonary venous congestion, consistent with chronic congestive heart failure.",
    "Normal cardiomediastinal silhouette with clear lungs in a pediatric patient; mild thymic sail sign noted. No consolidation or effusion.",
    "Chronic interstitial changes and biapical pleural thickening in an elderly patient, with no acute focal consolidation or effusion.",
    "Bilateral peribronchial cuffing and increased perihilar markings, consistent with viral bronchiolitis or reactive airway disease.",
    "Large right upper lobe mass with well-defined margins and associated right hilar enlargement, suspicious for primary bronchogenic carcinoma.",
    "Multiple lytic lesions in the ribs and clavicles with associated soft tissue masses, compatible with metastatic disease. Lungs otherwise clear.",
    "Extensive bilateral pleural calcifications with volume loss of both hemithoraces, consistent with prior asbestos exposure.",
    "Left-sided diaphragmatic elevation with bowel gas pattern projecting into the left hemithorax, suggestive of diaphragmatic hernia or eventration.",
    "Post-lobectomy changes in the right upper lobe with surgical clips and volume loss, compensatory hyperinflation of remaining right lung.",
    "Post-pleurodesis changes with diffuse pleural thickening and apical scarring on the left; no recurrent pneumothorax.",
    "Diffuse bilateral coarse reticulation with traction bronchiectasis and reduced lung volumes, compatible with advanced fibrotic lung disease.",
    "Bilateral patchy consolidations and ground-glass opacities in peripheral and basal distributions, compatible with organizing pneumonia pattern.",
    "Predominantly right lower lobe tree-in-bud nodularity and bronchial wall thickening, consistent with endobronchial infectious or inflammatory process.",
]


def main():
    print("=" * 80)
    print("CHEFF Batch Sample Generation with Super-Resolution")
    print("=" * 80)
    
    # Create samples directory
    samples_dir = Path("samples")
    samples_dir.mkdir(exist_ok=True)
    print(f"\n✓ Output directory: {samples_dir.absolute()}")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")
    
    # Set model paths
    print("\n" + "-" * 80)
    print("Loading models...")
    print("-" * 80)
    
    diffusion_path = "../trained_models/cheff_diff_t2i.pt"
    autoencoder_path = "../trained_models/cheff_autoencoder.pt"
    sr_path = "../trained_models/cheff_sr_fine.pt"
    
    # Check if models exist
    for model_path in [diffusion_path, autoencoder_path, sr_path]:
        if not Path(model_path).exists():
            print(f"\n✗ Error: Model not found: {model_path}")
            print("\nPlease ensure models are in: cheff/trained_models/")
            sys.exit(1)
    
    print(f"✓ Diffusion model: {diffusion_path}")
    print(f"✓ Autoencoder: {autoencoder_path}")
    print(f"✓ SR model: {sr_path}")
    
    # Load diffusion model
    print("\nLoading diffusion model...")
    ldm = CheffLDMT2I(
        checkpoint_path=diffusion_path,
        autoencoder_path=autoencoder_path,
        device=device
    )
    print("✓ Diffusion model loaded")
    
    # Load SR model
    print("Loading super-resolution model...")
    sr_model = CheffSRModel(
        checkpoint_path=sr_path,
        device=device
    )
    print("✓ SR model loaded")
    
    # Metadata for tracking
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "device": str(device),
        "models": {
            "diffusion": diffusion_path,
            "autoencoder": autoencoder_path,
            "sr": sr_path,
        },
        "samples": []
    }
    
    # Generate samples
    print("\n" + "=" * 80)
    print("Generating 50 samples...")
    print("=" * 80 + "\n")
    
    for idx, prompt in enumerate(PROMPTS, start=1):
        print(f"[{idx:2d}/50] Generating sample_{idx:03d}.png")
        print(f"         Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        
        try:
            # Generate 256x256 image
            print("         → Diffusion (256×256)...", end=" ", flush=True)
            image = ldm.sample(prompt, num_samples=1)  # Returns [1, 3, 256, 256]
            print("✓")
            
            # Convert RGB to grayscale for SR model (expects 1 channel)
            print("         → Converting to grayscale...", end=" ", flush=True)
            image_gray = rgb_to_grayscale(image)  # [1, 1, 256, 256]
            print("✓")
            
            # Super-resolution to 1024x1024
            print("         → Super-resolution (1024×1024)...", end=" ", flush=True)
            image_hr = sr_model.sample(image_gray)  # Returns [1, 1, 1024, 1024]
            print("✓")
            
            # Save image
            output_path = samples_dir / f"sample_{idx:03d}.png"
            save_image(image_hr, output_path, normalize=True, value_range=(-1, 1))
            print(f"         ✓ Saved to: {output_path}")
            
            # Track metadata
            metadata["samples"].append({
                "id": idx,
                "filename": f"sample_{idx:03d}.png",
                "prompt": prompt,
                "resolution": "1024x1024",
                "super_resolution": True,
            })
            
        except Exception as e:
            print(f"\n         ✗ Error generating sample {idx}: {e}")
            metadata["samples"].append({
                "id": idx,
                "filename": f"sample_{idx:03d}.png",
                "prompt": prompt,
                "resolution": "1024x1024",
                "super_resolution": True,
                "error": str(e),
            })
        
        print()  # Blank line between samples
    
    # Save metadata
    metadata_path = samples_dir / "samples_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, indent=2, fp=f)
    
    print("=" * 80)
    print("Generation Complete!")
    print("=" * 80)
    print(f"\n✓ Generated {len([s for s in metadata['samples'] if 'error' not in s])} / {len(PROMPTS)} samples")
    print(f"✓ Metadata saved to: {metadata_path}")
    print(f"✓ All samples in: {samples_dir.absolute()}")
    
    # Count errors
    errors = [s for s in metadata['samples'] if 'error' in s]
    if errors:
        print(f"\n⚠ {len(errors)} samples failed:")
        for err in errors:
            print(f"  - Sample {err['id']}: {err['error']}")
    
    print("\nYou can now create samples_original.md using the metadata JSON file.")


if __name__ == "__main__":
    main()
