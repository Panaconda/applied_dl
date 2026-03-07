import pandas as pd
import os
import glob

def compare_runs():
    # Define paths
    base_path = r"D:\01 - Dokumente\01 - Studium\01 - StaDS Experience\01 - Kursmaterial\ADL - Applied Deep Learning\applied_dl\classifier\runs"
    runs = {
        "Baseline": os.path.join(base_path, "baseline_slurm"),
        "Synthetic": os.path.join(base_path, "synthetic_slurm"),
        "Synthetic Filtered": os.path.join(base_path, "synthetic_filtered_slurm")
    }

    results = []

    for name, run_dir in runs.items():
        csv_path = os.path.join(run_dir, "metrics.csv")
        
        # Check if CSV exists, if not look for versions (e.g. version_0/metrics.csv)
        if not os.path.exists(csv_path):
            version_paths = glob.glob(os.path.join(run_dir, "version_*", "metrics.csv"))
            if version_paths:
                csv_path = version_paths[0] # Use the first version found
            else:
                print(f"Warning: No metrics.csv found for {name} in {run_dir}")
                continue
        
        try:
            df = pd.read_csv(csv_path)
            
            # Aggregate to remove NaN rows (Lightning logs train and val separately)
            df_clean = df.groupby(['epoch', 'step']).first().reset_index()
            
            # Extract best values
            best_metrics = {"Run": name}
            
            # For each column that starts with 'val/auroc', we want the maximum
            auroc_cols = [c for c in df_clean.columns if 'val/auroc' in c]
            for col in auroc_cols:
                best_metrics[col] = df_clean[col].max()
            
            # For losses, we want the minimum
            if 'val/loss' in df_clean.columns:
                best_metrics['val/loss'] = df_clean['val/loss'].min()
            if 'train/loss' in df_clean.columns:
                best_metrics['train/loss'] = df_clean['train/loss'].min()
            
            # Also record at which epoch the best overall AUROC was achieved
            if 'val/auroc' in df_clean.columns:
                best_epoch = df_clean.loc[df_clean['val/auroc'].idxmax(), 'epoch']
                best_metrics['best_epoch'] = int(best_epoch)

            results.append(best_metrics)
        except Exception as e:
            print(f"Error processing {name}: {e}")

    if not results:
        print("No results found to compare.")
        return

    # Create a summary DataFrame
    summary_df = pd.DataFrame(results).set_index("Run")
    
    # Sort by overall AUROC if available
    if 'val/auroc' in summary_df.columns:
        summary_df = summary_df.sort_values(by='val/auroc', ascending=False)

    print("\n" + "="*80)
    print("CLASSIFIER PERFORMANCE COMPARISON (BEST VALUES)")
    print("="*80)
    
    # Display the table
    pd.set_option('display.precision', 4)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print(summary_df)
    print("\n" + "="*80)
    
    # Save to CSV
    output_csv = os.path.join(os.path.dirname(__file__), "runs_comparison.csv")
    summary_df.to_csv(output_csv)
    print(f"Summary saved to: {output_csv}")

    # Optional: Highlight the best run per category
    print("\nBEST PER CATEGORY:")
    for col in summary_df.columns:
        if col in ['best_epoch', 'Run']: continue
        try:
            if 'loss' in col:
                best_run = summary_df[col].idxmin()
                best_val = summary_df.loc[best_run, col]
                print(f"  {col:30}: {best_val:.4f} ({best_run})")
            else:
                best_run = summary_df[col].idxmax()
                best_val = summary_df.loc[best_run, col]
                print(f"  {col:30}: {best_val:.4f} ({best_run})")
        except:
            pass
    print("="*80)

if __name__ == "__main__":
    compare_runs()
