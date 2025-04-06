import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_md_analysis(prefix):
    """
    Plots RMSD, RMSF, and Gyration Radius for three simulation runs with averages.
    Saves averaged data to CSV files.
    
    Args:
        prefix (str): The common file prefix for the runs.
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    rmsd_data = []
    rmsf_data = []
    gyrate_data = []

    # Plot RMSD
    print("\nProcessing RMSD data...")
    for i in range(1, 4):
        try:
            data = np.loadtxt(f'rmsd_{prefix}{i}.dat')
            print(f"Loaded RMSD data for run {i}, shape: {data.shape}")
            time = data[:, 0]
            values = data[:, 1]
            axes[0].plot(time, values, label=f'Run {i}', alpha=0.7)
            rmsd_data.append(pd.DataFrame({
                'Time': time,
                f'Run_{i}': values
            }).astype({'Time': 'float64', f'Run_{i}': 'float64'}))
        except Exception as e:
            print(f"Warning: Could not load RMSD data for run {i}: {e}")
    
    if rmsd_data:
        rmsd_df = pd.concat([df.set_index('Time') for df in rmsd_data], axis=1)
        rmsd_df['Average'] = rmsd_df.mean(axis=1)
        axes[0].plot(rmsd_df.index.values, rmsd_df['Average'].values, 
                    'k-', label='Average', linewidth=2)
        rmsd_df.reset_index().to_csv(f'{prefix}_rmsd_averaged.csv', index=False)
    
    axes[0].set_xlabel('Time (ns)')
    axes[0].set_ylabel('RMSD (nm)')
    axes[0].set_title('RMSD over Time')
    if len(rmsd_data) > 0:
        axes[0].legend()

    # Plot RMSF
    print("\nProcessing RMSF data...")
    for i in range(1, 4):
        try:
            data = np.loadtxt(f'rmsf_{prefix}{i}.dat')
            print(f"Loaded RMSF data for run {i}, shape: {data.shape}")
            residue = data[:, 0]
            values = data[:, 1]
            axes[1].plot(residue, values, label=f'Run {i}', alpha=0.7)
            rmsf_data.append(pd.DataFrame({
                'Residue': residue,
                f'Run_{i}': values
            }).astype({'Residue': 'float64', f'Run_{i}': 'float64'}))
        except Exception as e:
            print(f"Warning: Could not load RMSF data for run {i}: {e}")
    
    if rmsf_data:
        rmsf_df = pd.concat([df.set_index('Residue') for df in rmsf_data], axis=1)
        rmsf_df['Average'] = rmsf_df.mean(axis=1)
        axes[1].plot(rmsf_df.index.values, rmsf_df['Average'].values, 
                    'k-', label='Average', linewidth=2)
        rmsf_df.reset_index().to_csv(f'{prefix}_rmsf_averaged.csv', index=False)
    
    axes[1].set_xlabel('Residue Number')
    axes[1].set_ylabel('RMSF (nm)')
    axes[1].set_title('RMSF per Residue')
    if len(rmsf_data) > 0:
        axes[1].legend()

    # Plot Gyration Radius
    print("\nProcessing Gyration Radius data...")
    for i in range(1, 4):
        try:
            data = np.loadtxt(f'gyrate_{prefix}{i}.dat')
            print(f"Loaded gyration data for run {i}, shape: {data.shape}")
            time = data[:, 0] / 1000.0  # Convert from ps to ns
            # Average columns 1-4 which contain the gyration values
            values = np.mean(data[:, 1:], axis=1)
            axes[2].plot(time, values, label=f'Run {i}', alpha=0.7)
            gyrate_data.append(pd.DataFrame({
                'Time': time,
                f'Run_{i}': values
            }).astype({'Time': 'float64', f'Run_{i}': 'float64'}))
        except Exception as e:
            print(f"Warning: Could not load gyration data for run {i}: {e}")
    
    if gyrate_data:
        gyrate_df = pd.concat([df.set_index('Time') for df in gyrate_data], axis=1)
        gyrate_df['Average'] = gyrate_df.mean(axis=1)
        axes[2].plot(gyrate_df.index.values, gyrate_df['Average'].values, 
                    'k-', label='Average', linewidth=2)
        gyrate_df.reset_index().to_csv(f'{prefix}_gyrate_averaged.csv', index=False)
    
    axes[2].set_xlabel('Time (ns)')
    axes[2].set_ylabel('Rg (nm)')
    axes[2].set_title('Radius of Gyration over Time')
    if len(gyrate_data) > 0:
        axes[2].legend()

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'{prefix}_md_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved plot as {prefix}_md_analysis.png")
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python plot_md.py <prefix>")
        sys.exit(1)
    plot_md_analysis(sys.argv[1])
