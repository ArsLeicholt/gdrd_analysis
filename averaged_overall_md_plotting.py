import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

def plot_md_comparisons(prefixes, labels=None):
    """
    Plots averaged RMSD, RMSF, and Gyration Radius data from multiple simulations.
    
    Args:
        prefixes (list): List of prefixes for the different simulations
        labels (list, optional): List of labels for the legend. If None, prefixes are used.
    """
    if labels is None:
        labels = prefixes
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define a distinct color palette for comparisons (warm/diverse colors)
    comparison_colors = ['#FF7F0E', '#2CA02C', '#D62728', '#9467BD', 
                        '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', 
                        '#17BECF', '#FF9896']
    
    # Ensure we have enough colors
    if len(prefixes) > len(comparison_colors):
        comparison_colors = comparison_colors * (len(prefixes) // len(comparison_colors) + 1)
    
    # Take only the colors we need
    colors = comparison_colors[:len(prefixes)]
    
    # Plot RMSD
    rmsd_plotted = False
    for prefix, label, color in zip(prefixes, labels, colors):
        try:
            df = pd.read_csv(f'{prefix}_rmsd_averaged.csv')
            time_data = df['Time'].values
            avg_data = df['Average'].values
            axes[0].plot(time_data, avg_data, label=label, color=color, linewidth=2)
            rmsd_plotted = True
        except Exception as e:
            print(f"Warning: Could not load RMSD data for {prefix}: {e}")
    
    axes[0].set_xlabel('Time (ns)')
    axes[0].set_ylabel('RMSD (nm)')
    axes[0].set_title('RMSD Comparison')
    if rmsd_plotted:
        axes[0].legend()
    
    # Plot RMSF
    rmsf_plotted = False
    for prefix, label, color in zip(prefixes, labels, colors):
        try:
            df = pd.read_csv(f'{prefix}_rmsf_averaged.csv')
            residue_data = df['Residue'].values
            avg_data = df['Average'].values
            axes[1].plot(residue_data, avg_data, label=label, color=color, linewidth=2)
            rmsf_plotted = True
        except Exception as e:
            print(f"Warning: Could not load RMSF data for {prefix}: {e}")
    
    axes[1].set_xlabel('Residue Number')
    axes[1].set_ylabel('RMSF (nm)')
    axes[1].set_title('RMSF Comparison')
    if rmsf_plotted:
        axes[1].legend()
    
    # Plot Gyration Radius
    gyrate_plotted = False
    for prefix, label, color in zip(prefixes, labels, colors):
        try:
            df = pd.read_csv(f'{prefix}_gyrate_averaged.csv')
            time_data = df['Time'].values
            avg_data = df['Average'].values
            axes[2].plot(time_data, avg_data, label=label, color=color, linewidth=2)
            gyrate_plotted = True
        except Exception as e:
            print(f"Warning: Could not load gyration data for {prefix}: {e}")
    
    axes[2].set_xlabel('Time (ns)')
    axes[2].set_ylabel('Rg (nm)')
    axes[2].set_title('Radius of Gyration Comparison')
    if gyrate_plotted:
        axes[2].legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('md_simulations_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved plot as md_simulations_comparison.png")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_md_comparison.py prefix1 [prefix2 prefix3 ...]")
        print("Optional: Add -l 'label1' 'label2' ... after prefixes to specify custom labels")
        sys.exit(1)
        
    # Parse command line arguments
    try:
        label_index = sys.argv.index('-l')
        prefixes = sys.argv[1:label_index]
        labels = sys.argv[label_index + 1:]
        if len(labels) != len(prefixes):
            print("Error: Number of labels must match number of prefixes")
            sys.exit(1)
    except ValueError:
        prefixes = sys.argv[1:]
        labels = None
    
    print("\nProcessing data for prefixes:", ", ".join(prefixes))
    plot_md_comparisons(prefixes, labels)
