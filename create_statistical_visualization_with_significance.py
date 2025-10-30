#!/usr/bin/env python3
"""
Create statistical visualization with significance indicators.
Displays top 5 most significant pairwise comparisons with asterisks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from itertools import combinations

# Directories
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(script_dir), 'data_computational_gdrd_300325/orthologs/MD_data')
output_dir = script_dir

species_list = ['mel', 'sim', 'yak', 'ana', 'moj', 'vir', 'gri']

# Define protein region boundaries
protein_regions = {
    'mel': {'N': (1, 39), 'central_helix': (40, 78), 'C': (79, 113)},
    'sim': {'N': (1, 39), 'central_helix': (40, 78), 'C': (79, 113)},
    'yak': {'N': (1, 41), 'central_helix': (42, 80), 'C': (81, 115)},
    'ana': {'N': (1, 38), 'central_helix': (39, 77), 'C': (78, 111)},
    'moj': {'N': (1, 71), 'central_helix': (72, 110), 'C': (111, 164)},
    'vir': {'N': (1, 92), 'central_helix': (93, 131), 'C': (132, 190)},
    'gri': {'N': (1, 35), 'central_helix': (36, 74), 'C': (75, 125)}
}

# Color scheme
color_map = {
    'mel': '#32CD32', 'sim': '#90EE90', 'yak': '#7FFF00', 'ana': '#00FA9A',
    'moj': '#4169E1', 'vir': '#87CEEB', 'gri': '#1E90FF'
}


def add_significance_bar(ax, x1, x2, y, p_value, height_increment=0.05):
    """
    Add a significance bar with asterisks above two bars.

    Args:
        ax: matplotlib axis
        x1, x2: x positions of the two bars
        y: y position for the bar (top of current data + offset)
        p_value: p-value to determine number of asterisks
        height_increment: how much to offset bar above data
    """
    # Determine significance level
    if p_value < 0.001:
        sig_text = '***'
    elif p_value < 0.01:
        sig_text = '**'
    elif p_value < 0.05:
        sig_text = '*'
    else:
        return  # Don't draw if not significant

    # Draw the bar
    ax.plot([x1, x1, x2, x2], [y, y+height_increment, y+height_increment, y],
            'k-', linewidth=1.5)

    # Add asterisks
    ax.text((x1+x2)/2, y+height_increment, sig_text, ha='center', va='bottom',
            fontsize=10, fontweight='bold')


def get_significant_pairs(results_df, metric_name):
    """
    Extract significant pairwise comparisons for a given metric.

    Returns list of tuples: [(sp1, sp2, p_value), ...]
    """
    sig_pairs = []

    # Filter for this metric and Dunn's test
    metric_results = results_df[
        (results_df['Metric'] == metric_name) &
        (results_df['Test'] == "Dunn's test")
    ]

    for idx, row in metric_results.iterrows():
        comparison = row['Comparison']
        p_value = float(row['p-value'])
        significance = row['Significance']

        if significance != 'ns':  # Only significant ones
            # Parse "sp1 vs sp2"
            sp1, sp2 = comparison.split(' vs ')
            sig_pairs.append((sp1, sp2, p_value))

    return sig_pairs


def select_top_comparisons(sig_pairs, species_list, max_bars=5):
    """
    Select the most significant comparisons to display.
    Prioritizes lowest p-values and spacing.
    """
    # Sort by p-value
    sorted_pairs = sorted(sig_pairs, key=lambda x: x[2])

    # Take top max_bars comparisons
    selected = []
    for sp1, sp2, p_val in sorted_pairs[:max_bars]:
        selected.append((sp1, sp2, p_val))

    return selected


# Read statistical results
results_file = os.path.join(output_dir, 'statistical_results_FINAL.txt')
results_df = pd.read_csv(results_file, sep='\t')

print("Reading statistical results from:", results_file)
print(f"Total rows: {len(results_df)}")

# Collect data for each metric
rmsd_data = {}
gyrate_data = {}
rmsf_data = {'N-terminus': {}, 'Central helix': {}, 'C-terminus': {}}

print("\nLoading data for visualization...")

for species in species_list:
    try:
        # RMSD data (50-250 ns, subsampled)
        df = pd.read_csv(os.path.join(data_dir, f'{species}_rmsd_averaged.csv'))
        df_filtered = df[(df['Time'] >= 50) & (df['Time'] <= 250)]
        df_subsampled = df_filtered.iloc[::100]

        all_values = []
        for run_col in ['Run_1', 'Run_2', 'Run_3']:
            if run_col in df_subsampled.columns:
                all_values.extend(df_subsampled[run_col].dropna().values)
        rmsd_data[species] = all_values

        # Gyrate data
        df = pd.read_csv(os.path.join(data_dir, f'{species}_gyrate_averaged.csv'))
        df_filtered = df[(df['Time'] >= 50) & (df['Time'] <= 250)]
        df_subsampled = df_filtered.iloc[::100]

        all_values = []
        for run_col in ['Run_1', 'Run_2', 'Run_3']:
            if run_col in df_subsampled.columns:
                all_values.extend(df_subsampled[run_col].dropna().values)
        gyrate_data[species] = all_values

        # RMSF data by region
        df = pd.read_csv(os.path.join(data_dir, f'{species}_rmsf_averaged.csv'))

        if species in protein_regions:
            regions = protein_regions[species]

            # N-terminus
            n_bounds = regions['N']
            df_region = df[(df['Residue'] >= n_bounds[0]) & (df['Residue'] <= n_bounds[1])]
            all_values = []
            for run_col in ['Run_1', 'Run_2', 'Run_3']:
                if run_col in df_region.columns:
                    all_values.extend(df_region[run_col].dropna().values)
            rmsf_data['N-terminus'][species] = all_values

            # Central helix
            c_bounds = regions['central_helix']
            df_region = df[(df['Residue'] >= c_bounds[0]) & (df['Residue'] <= c_bounds[1])]
            all_values = []
            for run_col in ['Run_1', 'Run_2', 'Run_3']:
                if run_col in df_region.columns:
                    all_values.extend(df_region[run_col].dropna().values)
            rmsf_data['Central helix'][species] = all_values

            # C-terminus
            ct_bounds = regions['C']
            df_region = df[(df['Residue'] >= ct_bounds[0]) & (df['Residue'] <= ct_bounds[1])]
            all_values = []
            for run_col in ['Run_1', 'Run_2', 'Run_3']:
                if run_col in df_region.columns:
                    all_values.extend(df_region[run_col].dropna().values)
            rmsf_data['C-terminus'][species] = all_values

    except Exception as e:
        print(f"Error loading {species}: {e}")

print("Data loaded successfully.")

# Get significant comparisons (note: results file still has "Core" not "Central helix")
rmsd_sig_pairs = get_significant_pairs(results_df, 'RMSD')
rg_sig_pairs = get_significant_pairs(results_df, 'Rg')
rmsf_n_sig_pairs = get_significant_pairs(results_df, 'RMSF N-terminus')
rmsf_ch_sig_pairs = get_significant_pairs(results_df, 'RMSF Core')  # Maps to "Central helix"
rmsf_c_sig_pairs = get_significant_pairs(results_df, 'RMSF C-terminus')

print(f"\nSignificant comparisons found:")
print(f"  RMSD: {len(rmsd_sig_pairs)}")
print(f"  Rg: {len(rg_sig_pairs)}")
print(f"  RMSF N-terminus: {len(rmsf_n_sig_pairs)}")
print(f"  RMSF Central helix: {len(rmsf_ch_sig_pairs)}")
print(f"  RMSF C-terminus: {len(rmsf_c_sig_pairs)}")

# Select top comparisons to display
rmsd_top = select_top_comparisons(rmsd_sig_pairs, species_list, max_bars=5)
rg_top = select_top_comparisons(rg_sig_pairs, species_list, max_bars=5)

print("\nCreating visualization with significance indicators...")

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ============================================================================
# RMSD bar plot with significance
# ============================================================================
ax = axes[0, 0]
species = [sp for sp in species_list if sp in rmsd_data]
means = [np.mean(rmsd_data[sp]) for sp in species]
sems = [np.std(rmsd_data[sp]) / np.sqrt(len(rmsd_data[sp])) for sp in species]
colors = [color_map.get(sp, '#888888') for sp in species]

x_pos = np.arange(len(species))
ax.bar(x_pos, means, yerr=sems, capsize=5, color=colors, alpha=0.7, edgecolor='black')

# Calculate original y-axis range
max_y_data = max([means[i] + sems[i] for i in range(len(means))])
min_y_data = 0
data_range = max_y_data - min_y_data

# Add significance bars with 30% white space
bar_height_start = max_y_data + data_range * 0.05

for idx, (sp1, sp2, p_val) in enumerate(rmsd_top):
    if sp1 in species and sp2 in species:
        x1 = species.index(sp1)
        x2 = species.index(sp2)
        y_pos = bar_height_start + idx * data_range * 0.06
        add_significance_bar(ax, x1, x2, y_pos, p_val, height_increment=data_range*0.015)

# Set y-axis limits: tight to data + invisible white space above for sig bars
# The white space extends the plot but doesn't show on y-axis ticks
tight_y_max = max_y_data * 1.05  # Original tight scaling (5% margin)
plot_y_max = max_y_data + data_range * 0.40  # Extended for significance bars

ax.set_ylim(0, plot_y_max)
# Set y-ticks to only show the data range, not the white space
ax.set_yticks(np.linspace(0, tight_y_max, 6))
ax.set_xticks(x_pos)
ax.set_xticklabels(species, rotation=45, ha='right')
ax.set_ylabel('Mean RMSD (nm)', fontsize=12)
ax.set_title('RMSD (50-250 ns)', fontsize=13)

# ============================================================================
# Radius of Gyration bar plot with significance
# ============================================================================
ax = axes[0, 1]
species = [sp for sp in species_list if sp in gyrate_data]
means = [np.mean(gyrate_data[sp]) for sp in species]
sems = [np.std(gyrate_data[sp]) / np.sqrt(len(gyrate_data[sp])) for sp in species]
colors = [color_map.get(sp, '#888888') for sp in species]

x_pos = np.arange(len(species))
ax.bar(x_pos, means, yerr=sems, capsize=5, color=colors, alpha=0.7, edgecolor='black')

# Calculate original y-axis range
max_y_data = max([means[i] + sems[i] for i in range(len(means))])
min_y_data = 0
data_range = max_y_data - min_y_data

# Add significance bars with 30% white space
bar_height_start = max_y_data + data_range * 0.05

for idx, (sp1, sp2, p_val) in enumerate(rg_top):
    if sp1 in species and sp2 in species:
        x1 = species.index(sp1)
        x2 = species.index(sp2)
        y_pos = bar_height_start + idx * data_range * 0.06
        add_significance_bar(ax, x1, x2, y_pos, p_val, height_increment=data_range*0.015)

# Set y-axis limits: tight to data + invisible white space above for sig bars
tight_y_max = max_y_data * 1.05  # Original tight scaling (5% margin)
plot_y_max = max_y_data + data_range * 0.40  # Extended for significance bars

ax.set_ylim(0, plot_y_max)
# Set y-ticks to only show the data range, not the white space
ax.set_yticks(np.linspace(0, tight_y_max, 6))
ax.set_xticks(x_pos)
ax.set_xticklabels(species, rotation=45, ha='right')
ax.set_ylabel('Mean Rg (nm)', fontsize=12)
ax.set_title('Radius of Gyration (50-250 ns)', fontsize=13)

# ============================================================================
# RMSF by region - grouped bar plot (no significance bars due to complexity)
# ============================================================================
ax = axes[1, 0]
region_names = ['N-terminus', 'Central helix', 'C-terminus']
x_pos = np.arange(len(species_list))
bar_width = 0.25

max_height = 0
for idx, region in enumerate(region_names):
    region_data = rmsf_data[region]
    species = [sp for sp in species_list if sp in region_data]
    means = [np.mean(region_data[sp]) for sp in species]
    sems = [np.std(region_data[sp]) / np.sqrt(len(region_data[sp])) for sp in species]

    max_height = max(max_height, max([means[i] + sems[i] for i in range(len(means))]))

    x_positions = [i + idx * bar_width for i in range(len(species))]
    ax.bar(x_positions, means, bar_width, yerr=sems, capsize=3,
           label=region, alpha=0.7, edgecolor='black')

# Set y-axis limits: original scale + minimal white space (10%)
data_range = max_height
new_y_max = max_height + data_range * 0.10
ax.set_ylim(0, new_y_max)
ax.set_ylabel('Mean RMSF (nm)', fontsize=12)
ax.set_title('RMSF by Protein Region', fontsize=13)
ax.set_xticks([i + bar_width for i in range(len(species))])
ax.set_xticklabels(species_list, rotation=45, ha='right')
ax.legend(fontsize=10)

# ============================================================================
# Heatmap of RMSF values
# ============================================================================
ax = axes[1, 1]
heatmap_data = []
for region in region_names:
    region_means = []
    for sp in species_list:
        if sp in rmsf_data[region]:
            region_means.append(np.mean(rmsf_data[region][sp]))
        else:
            region_means.append(np.nan)
    heatmap_data.append(region_means)

# Use viridis colormap - white text readable across entire range
im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto')
ax.set_xticks(np.arange(len(species_list)))
ax.set_yticks(np.arange(len(region_names)))
ax.set_xticklabels(species_list, rotation=45, ha='right')
ax.set_yticklabels(region_names)
ax.set_title('RMSF Heatmap by Region', fontsize=13)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Mean RMSF (nm)', rotation=270, labelpad=15)

# Add text annotations to heatmap with grey color for visibility on all backgrounds
for i in range(len(region_names)):
    for j in range(len(species_list)):
        if not np.isnan(heatmap_data[i][j]):
            text = ax.text(j, i, f'{heatmap_data[i][j]:.3f}',
                          ha="center", va="center", color="#888888", fontsize=9, fontweight='bold')

plt.tight_layout()
output_path = os.path.join(output_dir, 'statistical_visualization_with_significance.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSaved visualization to: {output_path}")
plt.close()

print("\nVisualization complete!")
print("\nTop 5 most significant comparisons shown:")
print("\nRMSD:")
for sp1, sp2, p in rmsd_top:
    print(f"  {sp1} vs {sp2}: p = {p:.2e}")
print("\nRg:")
for sp1, sp2, p in rg_top:
    print(f"  {sp1} vs {sp2}: p = {p:.2e}")
