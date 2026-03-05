import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_metrics(run_dir):
    csv_path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        # Check in version_0 if not in root (legacy)
        csv_path = os.path.join(run_dir, "version_0", "metrics.csv")
        if not os.path.exists(csv_path):
            print(f"Error: metrics.csv not found in {run_dir} or {run_dir}/version_0/")
            return

    df = pd.read_csv(csv_path)
    
    # Lightning CSVLogger logs train and val on separate rows for the same step/epoch
    # We aggregate by step/epoch to merge these rows.
    df = df.groupby(['epoch', 'step']).first().reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot Losses
    axes[0].plot(df['epoch'], df['train/loss'], 'b-o', label='Train Loss')
    axes[0].plot(df['epoch'], df['val/loss'], 'r-o', label='Val Loss')
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot AUROC
    auroc_cols = [c for c in df.columns if 'val/auroc' in c]
    for col in auroc_cols:
        label = col.replace('val/auroc_', '').replace('val/auroc', 'Overall')
        axes[1].plot(df['epoch'], df[col], '-o', label=label)
    
    axes[1].set_title('Validation AUROC (Overall & Per-Class)', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('AUROC', fontsize=12)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    output_path = os.path.join(run_dir, 'training_metrics_plot.png')
    plt.savefig(output_path, dpi=150)
    print(f"Plot successfully saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Path to the experiments/runs/RUN_NAME directory")
    args = parser.parse_args()
    plot_metrics(args.run_dir)
