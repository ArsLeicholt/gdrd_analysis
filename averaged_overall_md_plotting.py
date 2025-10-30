import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from scipy import stats
from scikit_posthocs import posthoc_dunn
from matplotlib.backends.backend_pdf import PdfPages

def plot_md_comparisons(prefixes, labels=None, data_dir='.', output_dir='.'):
    """
    Plots averaged RMSD, RMSF, and Gyration Radius data from multiple simulations.

    Args:
        prefixes (list): List of prefixes for the different simulations
        labels (list, optional): List of labels for the legend. If None, prefixes are used.
        data_dir (str): Directory containing the data files
        output_dir (str): Directory to save output files
    """
    if labels is None:
        labels = prefixes

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

    # Define color scheme: greenish for mel/sim/yak/ana, bluish for moj/vir/gri
    color_map = {
        'mel': '#32CD32',  # Lime green
        'sim': '#90EE90',  # Light green
        'yak': '#7FFF00',  # Chartreuse
        'ana': '#00FA9A',  # Medium spring green
        'moj': '#4169E1',  # Royal blue
        'vir': '#87CEEB',  # Sky blue
        'gri': '#1E90FF'   # Dodger blue
    }

    # Map colors to prefixes
    colors = [color_map.get(prefix, '#888888') for prefix in prefixes]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Storage for statistical analysis
    rmsd_data_dict = {}
    gyrate_data_dict = {}
    rmsf_data_dict = {region: {} for region in ['N-terminus', 'Central helix', 'C-terminus']}
    
    # Plot RMSD
    rmsd_plotted = False
    for prefix, label, color in zip(prefixes, labels, colors):
        try:
            df = pd.read_csv(os.path.join(data_dir, f'{prefix}_rmsd_averaged.csv'))
            time_data = df['Time'].values
            avg_data = df['Average'].values
            axes[0].plot(time_data, avg_data, label=label, color=color, linewidth=2)
            rmsd_plotted = True

            # Store data for statistical analysis (50-250 ns)
            mask = (time_data >= 50) & (time_data <= 250)
            replicate_means = []
            for run_col in ['Run_1', 'Run_2', 'Run_3']:
                if run_col in df.columns:
                    replicate_means.append(df[run_col][mask].mean())
            rmsd_data_dict[prefix] = replicate_means

        except Exception as e:
            print(f"Warning: Could not load RMSD data for {prefix}: {e}")

    axes[0].set_xlabel('Time (ns)')
    axes[0].set_ylabel('RMSD (nm)')
    axes[0].set_title('RMSD Comparison')
    if rmsd_plotted:
        axes[0].legend()
    
    # Plot RMSF (normalized positions)
    rmsf_plotted = False
    rmsf_data_for_supplementary = []  # Store for supplementary non-normalized plot

    for prefix, label, color in zip(prefixes, labels, colors):
        try:
            df = pd.read_csv(os.path.join(data_dir, f'{prefix}_rmsf_averaged.csv'))
            residue_data = df['Residue'].values
            avg_data = df['Average'].values

            # Normalize positions to 0-1
            normalized_positions = (residue_data - residue_data.min()) / (residue_data.max() - residue_data.min())
            axes[1].plot(normalized_positions, avg_data, label=label, color=color, linewidth=2)
            rmsf_plotted = True

            # Store for non-normalized supplementary plot
            rmsf_data_for_supplementary.append((residue_data, avg_data, label, color))

            # Calculate mean RMSF for each protein region
            if prefix in protein_regions:
                regions = protein_regions[prefix]

                # N-terminus
                n_mask = (residue_data >= regions['N'][0]) & (residue_data <= regions['N'][1])
                n_rmsf = []
                for run_col in ['Run_1', 'Run_2', 'Run_3']:
                    if run_col in df.columns:
                        n_rmsf.append(df[run_col][n_mask].mean())
                rmsf_data_dict['N-terminus'][prefix] = n_rmsf

                # Central helix
                central_helix_mask = (residue_data >= regions['central_helix'][0]) & (residue_data <= regions['central_helix'][1])
                central_helix_rmsf = []
                for run_col in ['Run_1', 'Run_2', 'Run_3']:
                    if run_col in df.columns:
                        central_helix_rmsf.append(df[run_col][central_helix_mask].mean())
                rmsf_data_dict['Central helix'][prefix] = central_helix_rmsf

                # C-terminus
                c_mask = (residue_data >= regions['C'][0]) & (residue_data <= regions['C'][1])
                c_rmsf = []
                for run_col in ['Run_1', 'Run_2', 'Run_3']:
                    if run_col in df.columns:
                        c_rmsf.append(df[run_col][c_mask].mean())
                rmsf_data_dict['C-terminus'][prefix] = c_rmsf

        except Exception as e:
            print(f"Warning: Could not load RMSF data for {prefix}: {e}")

    axes[1].set_xlabel('Normalized Position')
    axes[1].set_ylabel('RMSF (nm)')
    axes[1].set_title('RMSF Comparison (Normalized)')
    if rmsf_plotted:
        axes[1].legend()
    
    # Plot Gyration Radius
    gyrate_plotted = False
    for prefix, label, color in zip(prefixes, labels, colors):
        try:
            df = pd.read_csv(os.path.join(data_dir, f'{prefix}_gyrate_averaged.csv'))
            time_data = df['Time'].values
            avg_data = df['Average'].values
            axes[2].plot(time_data, avg_data, label=label, color=color, linewidth=2)
            gyrate_plotted = True

            # Store data for statistical analysis (50-250 ns)
            mask = (time_data >= 50) & (time_data <= 250)
            replicate_means = []
            for run_col in ['Run_1', 'Run_2', 'Run_3']:
                if run_col in df.columns:
                    replicate_means.append(df[run_col][mask].mean())
            gyrate_data_dict[prefix] = replicate_means

        except Exception as e:
            print(f"Warning: Could not load gyration data for {prefix}: {e}")

    axes[2].set_xlabel('Time (ns)')
    axes[2].set_ylabel('Rg (nm)')
    axes[2].set_title('Radius of Gyration Comparison')
    if gyrate_plotted:
        axes[2].legend()

    # Adjust layout and save main figure
    plt.tight_layout()
    main_plot_path = os.path.join(output_dir, 'md_simulations_comparison.png')
    plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved main plot as {main_plot_path}")
    plt.close()

    # Create supplementary figure with non-normalized RMSF
    if rmsf_data_for_supplementary:
        fig_supp, ax_supp = plt.subplots(figsize=(10, 6))
        for residue_data, avg_data, label, color in rmsf_data_for_supplementary:
            ax_supp.plot(residue_data, avg_data, label=label, color=color, linewidth=2)
        ax_supp.set_xlabel('Residue Number')
        ax_supp.set_ylabel('RMSF (nm)')
        ax_supp.set_title('RMSF Comparison (Non-normalized)')
        ax_supp.legend()
        plt.tight_layout()
        supp_plot_path = os.path.join(output_dir, 'md_rmsf_supplementary.png')
        plt.savefig(supp_plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved supplementary RMSF plot as {supp_plot_path}")
        plt.close()

    # Perform statistical analyses
    perform_statistical_analyses(rmsd_data_dict, gyrate_data_dict, rmsf_data_dict, prefixes, output_dir)


def perform_statistical_analyses(rmsd_data, gyrate_data, rmsf_data, species_list, output_dir):
    """
    Perform Kruskal-Wallis and Dunn's post-hoc tests on MD simulation data.

    Args:
        rmsd_data (dict): Dictionary of RMSD means per species
        gyrate_data (dict): Dictionary of gyration radius means per species
        rmsf_data (dict): Nested dictionary of RMSF means per region per species
        species_list (list): List of species names
        output_dir (str): Directory to save results
    """
    results = []

    # 1. RMSD Analysis
    print("\n=== RMSD Statistical Analysis (50-250 ns) ===")
    if len(rmsd_data) > 2:
        rmsd_groups = [rmsd_data[sp] for sp in species_list if sp in rmsd_data]
        rmsd_names = [sp for sp in species_list if sp in rmsd_data]

        # Kruskal-Wallis test
        h_stat, p_val = stats.kruskal(*rmsd_groups)
        print(f"Kruskal-Wallis H-statistic: {h_stat:.4f}, p-value: {p_val:.4e}")
        results.append(['RMSD', 'Overall', 'Kruskal-Wallis', h_stat, p_val, ''])

        # Dunn's post-hoc test
        if p_val < 0.05:
            # Prepare data for Dunn's test
            rmsd_df = pd.DataFrame({
                'value': [item for sublist in rmsd_groups for item in sublist],
                'group': [name for name, group in zip(rmsd_names, rmsd_groups) for _ in group]
            })
            dunn_result = posthoc_dunn(rmsd_df, val_col='value', group_col='group', p_adjust='bonferroni')
            print("Dunn's post-hoc test results (Bonferroni-corrected):")
            print(dunn_result)

            # Add pairwise comparisons to results
            for i, sp1 in enumerate(dunn_result.index):
                for j, sp2 in enumerate(dunn_result.columns):
                    if i < j:
                        p_pairwise = dunn_result.loc[sp1, sp2]
                        results.append(['RMSD', f'{sp1} vs {sp2}', "Dunn's test", '', p_pairwise,
                                      '*' if p_pairwise < 0.05 else 'ns'])

    # 2. Radius of Gyration Analysis
    print("\n=== Radius of Gyration Statistical Analysis (50-250 ns) ===")
    if len(gyrate_data) > 2:
        gyrate_groups = [gyrate_data[sp] for sp in species_list if sp in gyrate_data]
        gyrate_names = [sp for sp in species_list if sp in gyrate_data]

        # Kruskal-Wallis test
        h_stat, p_val = stats.kruskal(*gyrate_groups)
        print(f"Kruskal-Wallis H-statistic: {h_stat:.4f}, p-value: {p_val:.4e}")
        results.append(['Radius of Gyration', 'Overall', 'Kruskal-Wallis', h_stat, p_val, ''])

        # Dunn's post-hoc test
        if p_val < 0.05:
            gyrate_df = pd.DataFrame({
                'value': [item for sublist in gyrate_groups for item in sublist],
                'group': [name for name, group in zip(gyrate_names, gyrate_groups) for _ in group]
            })
            dunn_result = posthoc_dunn(gyrate_df, val_col='value', group_col='group', p_adjust='bonferroni')
            print("Dunn's post-hoc test results (Bonferroni-corrected):")
            print(dunn_result)

            # Add pairwise comparisons to results
            for i, sp1 in enumerate(dunn_result.index):
                for j, sp2 in enumerate(dunn_result.columns):
                    if i < j:
                        p_pairwise = dunn_result.loc[sp1, sp2]
                        results.append(['Radius of Gyration', f'{sp1} vs {sp2}', "Dunn's test", '',
                                      p_pairwise, '*' if p_pairwise < 0.05 else 'ns'])

    # 3. RMSF Analysis by Region
    for region_name in ['N-terminus', 'Central helix', 'C-terminus']:
        print(f"\n=== RMSF Statistical Analysis - {region_name} ===")
        region_data = rmsf_data[region_name]

        if len(region_data) > 2:
            rmsf_groups = [region_data[sp] for sp in species_list if sp in region_data]
            rmsf_names = [sp for sp in species_list if sp in region_data]

            # Kruskal-Wallis test
            h_stat, p_val = stats.kruskal(*rmsf_groups)
            print(f"Kruskal-Wallis H-statistic: {h_stat:.4f}, p-value: {p_val:.4e}")
            results.append([f'RMSF {region_name}', 'Overall', 'Kruskal-Wallis', h_stat, p_val, ''])

            # Dunn's post-hoc test
            if p_val < 0.05:
                rmsf_df = pd.DataFrame({
                    'value': [item for sublist in rmsf_groups for item in sublist],
                    'group': [name for name, group in zip(rmsf_names, rmsf_groups) for _ in group]
                })
                dunn_result = posthoc_dunn(rmsf_df, val_col='value', group_col='group', p_adjust='bonferroni')
                print("Dunn's post-hoc test results (Bonferroni-corrected):")
                print(dunn_result)

                # Add pairwise comparisons to results
                for i, sp1 in enumerate(dunn_result.index):
                    for j, sp2 in enumerate(dunn_result.columns):
                        if i < j:
                            p_pairwise = dunn_result.loc[sp1, sp2]
                            results.append([f'RMSF {region_name}', f'{sp1} vs {sp2}', "Dunn's test", '',
                                          p_pairwise, '*' if p_pairwise < 0.05 else 'ns'])

    # Save results as tabular text file
    results_df = pd.DataFrame(results, columns=['Metric', 'Comparison', 'Test', 'Statistic', 'p-value', 'Significance'])
    txt_path = os.path.join(output_dir, 'statistical_results.txt')
    results_df.to_csv(txt_path, sep='\t', index=False)
    print(f"\n\nSaved statistical results as {txt_path}")

    # Save results as PDF table
    save_results_as_pdf(results_df, output_dir)

    # Create visualization suggestion
    create_statistical_visualization(rmsd_data, gyrate_data, rmsf_data, species_list, output_dir)


def save_results_as_pdf(results_df, output_dir):
    """
    Save statistical results as a formatted PDF table.

    Args:
        results_df (DataFrame): Results dataframe
        output_dir (str): Directory to save PDF
    """
    pdf_path = os.path.join(output_dir, 'statistical_results.pdf')

    fig, ax = plt.subplots(figsize=(14, len(results_df) * 0.3 + 2))
    ax.axis('tight')
    ax.axis('off')

    # Format p-values for display
    formatted_df = results_df.copy()
    for idx, row in formatted_df.iterrows():
        if pd.notna(row['p-value']) and row['p-value'] != '':
            p_val = float(row['p-value'])
            if p_val < 0.001:
                formatted_df.at[idx, 'p-value'] = '<0.001'
            else:
                formatted_df.at[idx, 'p-value'] = f'{p_val:.4f}'
        if pd.notna(row['Statistic']) and row['Statistic'] != '':
            formatted_df.at[idx, 'Statistic'] = f'{float(row["Statistic"]):.4f}'

    table = ax.table(cellText=formatted_df.values, colLabels=formatted_df.columns,
                    cellLoc='left', loc='center', bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Style header
    for i in range(len(formatted_df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(formatted_df) + 1):
        for j in range(len(formatted_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')

    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Saved statistical results as PDF: {pdf_path}")
    plt.close()


def create_statistical_visualization(rmsd_data, gyrate_data, rmsf_data, species_list, output_dir):
    """
    Create visualization of statistical results showing mean values with error bars.

    Args:
        rmsd_data (dict): Dictionary of RMSD means per species
        gyrate_data (dict): Dictionary of gyration radius means per species
        rmsf_data (dict): Nested dictionary of RMSF means per region per species
        species_list (list): List of species names
        output_dir (str): Directory to save visualization
    """
    # Define color scheme
    color_map = {
        'mel': '#32CD32', 'sim': '#90EE90', 'yak': '#7FFF00', 'ana': '#00FA9A',
        'moj': '#4169E1', 'vir': '#87CEEB', 'gri': '#1E90FF'
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # RMSD bar plot
    if rmsd_data:
        species = [sp for sp in species_list if sp in rmsd_data]
        means = [np.mean(rmsd_data[sp]) for sp in species]
        sems = [np.std(rmsd_data[sp]) / np.sqrt(len(rmsd_data[sp])) for sp in species]
        colors = [color_map.get(sp, '#888888') for sp in species]

        axes[0, 0].bar(species, means, yerr=sems, capsize=5, color=colors, alpha=0.7, edgecolor='black')
        axes[0, 0].set_ylabel('Mean RMSD (nm)')
        axes[0, 0].set_title('RMSD (50-250 ns)')
        axes[0, 0].tick_params(axis='x', rotation=45)

    # Radius of Gyration bar plot
    if gyrate_data:
        species = [sp for sp in species_list if sp in gyrate_data]
        means = [np.mean(gyrate_data[sp]) for sp in species]
        sems = [np.std(gyrate_data[sp]) / np.sqrt(len(gyrate_data[sp])) for sp in species]
        colors = [color_map.get(sp, '#888888') for sp in species]

        axes[0, 1].bar(species, means, yerr=sems, capsize=5, color=colors, alpha=0.7, edgecolor='black')
        axes[0, 1].set_ylabel('Mean Rg (nm)')
        axes[0, 1].set_title('Radius of Gyration (50-250 ns)')
        axes[0, 1].tick_params(axis='x', rotation=45)

    # RMSF by region - grouped bar plot
    region_names = ['N-terminus', 'Central helix', 'C-terminus']
    x_pos = np.arange(len(species_list))
    bar_width = 0.25

    for idx, region in enumerate(region_names):
        region_data = rmsf_data[region]
        species = [sp for sp in species_list if sp in region_data]
        means = [np.mean(region_data[sp]) for sp in species]
        sems = [np.std(region_data[sp]) / np.sqrt(len(region_data[sp])) for sp in species]

        x_positions = [i + idx * bar_width for i in range(len(species))]
        axes[1, 0].bar(x_positions, means, bar_width, yerr=sems, capsize=3,
                      label=region, alpha=0.7, edgecolor='black')

    axes[1, 0].set_ylabel('Mean RMSF (nm)')
    axes[1, 0].set_title('RMSF by Protein Region')
    axes[1, 0].set_xticks([i + bar_width for i in range(len(species))])
    axes[1, 0].set_xticklabels(species, rotation=45)
    axes[1, 0].legend()

    # Heatmap of RMSF values
    heatmap_data = []
    for region in region_names:
        region_means = []
        for sp in species_list:
            if sp in rmsf_data[region]:
                region_means.append(np.mean(rmsf_data[region][sp]))
            else:
                region_means.append(np.nan)
        heatmap_data.append(region_means)

    im = axes[1, 1].imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    axes[1, 1].set_xticks(np.arange(len(species_list)))
    axes[1, 1].set_yticks(np.arange(len(region_names)))
    axes[1, 1].set_xticklabels(species_list, rotation=45)
    axes[1, 1].set_yticklabels(region_names)
    axes[1, 1].set_title('RMSF Heatmap by Region')

    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1, 1])
    cbar.set_label('Mean RMSF (nm)', rotation=270, labelpad=15)

    # Add text annotations to heatmap
    for i in range(len(region_names)):
        for j in range(len(species_list)):
            if not np.isnan(heatmap_data[i][j]):
                text = axes[1, 1].text(j, i, f'{heatmap_data[i][j]:.3f}',
                                      ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    stat_viz_path = os.path.join(output_dir, 'statistical_visualization.png')
    plt.savefig(stat_viz_path, dpi=300, bbox_inches='tight')
    print(f"Saved statistical visualization as {stat_viz_path}")
    plt.close()

    print("\n=== Visualization Suggestion ===")
    print("The statistical_visualization.png shows:")
    print("1. Top left: Mean RMSD with error bars (SEM) for each species")
    print("2. Top right: Mean radius of gyration with error bars (SEM) for each species")
    print("3. Bottom left: Grouped bar chart showing mean RMSF for N-terminus, Central helix, and C-terminus")
    print("4. Bottom right: Heatmap showing RMSF patterns across regions and species")
    print("\nThese visualizations complement the statistical tests by showing:")
    print("- Effect sizes and variability (error bars)")
    print("- Regional differences in protein flexibility (RMSF by region)")
    print("- Overall patterns across all metrics (heatmap)")


def write_methods_text(output_dir):
    """
    Write a methods section describing the statistical analyses.

    Args:
        output_dir (str): Directory to save the methods text
    """
    methods_text = """
STATISTICAL ANALYSIS OF MOLECULAR DYNAMICS SIMULATIONS

Statistical analyses were performed on molecular dynamics (MD) simulation data to compare
structural properties across seven Drosophila species orthologues. All statistical tests
were conducted using Python with the scipy.stats and scikit-posthocs libraries.

Root Mean Square Deviation (RMSD):
For each replicate simulation, the mean RMSD was calculated from the equilibrated portion
of the trajectory (50-250 ns). The three replicate means per species were used as input
for statistical testing. Non-parametric Kruskal-Wallis H-test was applied to assess
overall differences among species, followed by Dunn's post-hoc test with Bonferroni
correction for multiple comparisons when the overall test was significant (α = 0.05).

Radius of Gyration (Rg):
Analysis was performed identically to RMSD, using mean Rg values calculated from the
50-250 ns time window for each replicate. Kruskal-Wallis test was used to test for
overall differences, with Dunn's post-hoc test applied for pairwise comparisons when
appropriate.

Root Mean Square Fluctuation (RMSF):
RMSF values were analyzed separately for three functionally distinct protein regions:
N-terminus, central helix, and C-terminus. Region boundaries were defined based on
structural alignment and domain annotation for each species. For each region and replicate,
mean RMSF was calculated across all residues within that region. Kruskal-Wallis H-test
was performed independently for each protein region to test for differences among species,
with Dunn's post-hoc test (Bonferroni-corrected) applied when significant differences
were detected.

All tests were two-tailed with significance threshold set at α = 0.05. P-values less than
0.001 are reported as p < 0.001. The Kruskal-Wallis test was selected over parametric
alternatives due to the small sample size (n = 3 replicates per species) and to avoid
assumptions about the underlying data distribution.
"""

    methods_path = os.path.join(output_dir, 'methods_section.txt')
    with open(methods_path, 'w') as f:
        f.write(methods_text.strip())

    print(f"\nSaved methods section text as {methods_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python averaged_overall_md_plotting.py prefix1 [prefix2 prefix3 ...]")
        print("Optional: Add -l 'label1' 'label2' ... after prefixes to specify custom labels")
        print("Optional: Add -d 'data_directory' to specify data directory")
        print("Optional: Add -o 'output_directory' to specify output directory")
        sys.exit(1)

    # Parse command line arguments
    data_dir = '.'
    output_dir = '.'

    # Check for -d flag
    if '-d' in sys.argv:
        d_index = sys.argv.index('-d')
        data_dir = sys.argv[d_index + 1]
        sys.argv.pop(d_index)  # Remove -d
        sys.argv.pop(d_index)  # Remove directory path

    # Check for -o flag
    if '-o' in sys.argv:
        o_index = sys.argv.index('-o')
        output_dir = sys.argv[o_index + 1]
        sys.argv.pop(o_index)  # Remove -o
        sys.argv.pop(o_index)  # Remove directory path

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
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    plot_md_comparisons(prefixes, labels, data_dir, output_dir)
    write_methods_text(output_dir)
