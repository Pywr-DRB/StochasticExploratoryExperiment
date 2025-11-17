"""
SI4: Plot NYC Demand vs Diversion Diagnostics

This script creates diagnostic figures comparing NYC water demand and actual
diversions (delivery) from the Pywr-DRB model outputs.

The plots include:
- Distribution comparison (demand vs diversion)
- 2D histogram with joint marginals showing demand-diversion relationship
- 1-1 scatter plot of demand vs diversion

These visualizations help understand:
- How well diversions meet demand
- When and how often shortages occur (diversion < demand)
- The joint distribution of demand and diversion

Usage:
    python SI4_plot_nyc_demand_vs_diversion.py <dataset_id>

Example:
    python SI4_plot_nyc_demand_vs_diversion.py stationary_ensemble
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from methods.config import *
from methods.load import load_shortage_data

# Matplotlib settings for publication-quality figures
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Output directory
FIG_DIR_DEMAND_DIVERSION = f"{FIG_DIR}/demand_vs_diversion"
os.makedirs(FIG_DIR_DEMAND_DIVERSION, exist_ok=True)


def extract_demand_diversion_data(data, dataset_id):
    """
    Extract demand and diversion data from pywrdrb.Data object.

    Parameters
    ----------
    data : pywrdrb.Data
        Data object with ibt_demands and ibt_diversions
    dataset_id : str
        Dataset identifier

    Returns
    -------
    demand : np.ndarray
        Flattened array of all demand values across realizations
    diversion : np.ndarray
        Flattened array of all diversion values across realizations
    demand_by_real : dict
        Dictionary mapping realization_id to demand Series
    diversion_by_real : dict
        Dictionary mapping realization_id to diversion Series
    """
    print("Extracting demand and diversion data...")

    # Get the data from the Data object
    demands_dict = data.ibt_demands[dataset_id]
    diversions_dict = data.ibt_diversions[dataset_id]

    # Collect all data
    all_demand = []
    all_diversion = []
    demand_by_real = {}
    diversion_by_real = {}

    for realization_id in demands_dict.keys():
        # Get demand and diversion DataFrames for this realization
        demand_df = demands_dict[realization_id]
        diversion_df = diversions_dict[realization_id]

        # Extract NYC columns
        # Demand columns: 'demand_nyc'
        # Diversion columns: 'delivery_nyc' (note: delivery, not diversion)
        if 'demand_nyc' in demand_df.columns:
            demand_vals = demand_df['demand_nyc'].values
        else:
            print(f"  Warning: 'demand_nyc' not found in demand columns: {demand_df.columns}")
            continue

        if 'delivery_nyc' in diversion_df.columns:
            diversion_vals = diversion_df['delivery_nyc'].values
        else:
            print(f"  Warning: 'delivery_nyc' not found in diversion columns: {diversion_df.columns}")
            continue

        # Store by realization
        demand_by_real[realization_id] = pd.Series(demand_vals, index=demand_df.index)
        diversion_by_real[realization_id] = pd.Series(diversion_vals, index=diversion_df.index)

        # Append to all data
        all_demand.extend(demand_vals)
        all_diversion.extend(diversion_vals)

    demand = np.array(all_demand)
    diversion = np.array(all_diversion)

    print(f"  Extracted {len(demand):,} demand-diversion pairs")
    print(f"  From {len(demand_by_real)} realizations")

    return demand, diversion, demand_by_real, diversion_by_real


def plot_distribution_comparison(demand, diversion, dataset_id, dataset_label):
    """
    Plot distribution comparison of demand vs diversion.

    Parameters
    ----------
    demand : np.ndarray
        Array of demand values
    diversion : np.ndarray
        Array of diversion values
    dataset_id : str
        Dataset identifier
    dataset_label : str
        Dataset label for plot title
    """
    print("Creating distribution comparison plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ### Left panel: Overlaid KDE plots
    # Remove NaN values
    demand_clean = demand[~np.isnan(demand)]
    diversion_clean = diversion[~np.isnan(diversion)]

    # Plot KDEs
    demand_clean_pos = demand_clean[demand_clean > 0]
    diversion_clean_pos = diversion_clean[diversion_clean > 0]

    if len(demand_clean_pos) > 1:
        sns.kdeplot(demand_clean_pos, ax=ax1, color='#1f77b4', linewidth=2.5,
                   label='Demand', fill=True, alpha=0.3)

    if len(diversion_clean_pos) > 1:
        sns.kdeplot(diversion_clean_pos, ax=ax1, color='#d62728', linewidth=2.5,
                   label='Diversion', fill=True, alpha=0.3)

    ax1.set_xlabel('Flow (MGD)')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('NYC Demand vs Diversion Distribution')
    ax1.legend(frameon=False)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=0)

    ### Right panel: Box plots
    box_data = [demand_clean, diversion_clean]
    box_labels = ['Demand', 'Diversion']

    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True,
                    widths=0.6, showfliers=False)

    colors = ['#1f77b4', '#d62728']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.set_ylabel('Flow (MGD)')
    ax2.set_title('NYC Demand vs Diversion Summary')
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Add statistics text
    stats_text = (
        f"Demand: mean={demand_clean.mean():.1f}, std={demand_clean.std():.1f}\n"
        f"Diversion: mean={diversion_clean.mean():.1f}, std={diversion_clean.std():.1f}"
    )
    ax2.text(0.02, 0.98, stats_text,
            transform=ax2.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.5))

    plt.suptitle(f'NYC Demand vs Diversion: {dataset_label}',
                fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()

    fname = f"{FIG_DIR_DEMAND_DIVERSION}/{dataset_id}_nyc_demand_vs_diversion_distributions.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


def plot_2d_histogram_with_marginals(demand, diversion, dataset_id, dataset_label):
    """
    Plot 2D histogram of demand vs diversion with joint marginals.

    Parameters
    ----------
    demand : np.ndarray
        Array of demand values
    diversion : np.ndarray
        Array of diversion values
    dataset_id : str
        Dataset identifier
    dataset_label : str
        Dataset label for plot title
    """
    print("Creating 2D histogram with marginals...")

    # Remove NaN values
    mask = ~(np.isnan(demand) | np.isnan(diversion))
    demand_clean = demand[mask]
    diversion_clean = diversion[mask]

    if len(demand_clean) == 0:
        print("  Warning: No valid data points for 2D histogram. Skipping plot.")
        return

    # Create figure with GridSpec for marginal plots
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 4, figure=fig, hspace=0.05, wspace=0.05)

    # Main 2D histogram
    ax_main = fig.add_subplot(gs[1:, :-1])

    # Marginal histograms
    ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)

    # 2D histogram
    h = ax_main.hist2d(demand_clean, diversion_clean, bins=100,
                       cmap='Blues', cmin=1)

    # Add 1:1 line
    max_val = max(demand_clean.max(), diversion_clean.max())
    min_val = min(demand_clean.min(), diversion_clean.min())
    ax_main.plot([min_val, max_val], [min_val, max_val],
                'r--', linewidth=2, label='1:1 Line', alpha=0.7)

    # Add colorbar
    cbar = plt.colorbar(h[3], ax=ax_main, pad=0.15)
    cbar.set_label('Count', rotation=270, labelpad=20)

    ax_main.set_xlabel('NYC Demand (MGD)', fontsize=11, fontweight='bold')
    ax_main.set_ylabel('NYC Diversion (MGD)', fontsize=11, fontweight='bold')
    ax_main.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax_main.grid(True, alpha=0.3, linestyle='--')

    # Top marginal (demand distribution)
    ax_top.hist(demand_clean, bins=100, color='#1f77b4', alpha=0.7, edgecolor='none')
    ax_top.set_ylabel('Count', fontsize=9)
    ax_top.tick_params(labelbottom=False)
    ax_top.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax_top.set_title(f'NYC Demand vs Diversion Joint Distribution: {dataset_label}',
                    fontsize=12, fontweight='bold', pad=10)

    # Right marginal (diversion distribution)
    ax_right.hist(diversion_clean, bins=100, orientation='horizontal',
                 color='#d62728', alpha=0.7, edgecolor='none')
    ax_right.set_xlabel('Count', fontsize=9)
    ax_right.tick_params(labelleft=False)
    ax_right.grid(True, alpha=0.3, linestyle='--', axis='x')

    fname = f"{FIG_DIR_DEMAND_DIVERSION}/{dataset_id}_nyc_demand_vs_diversion_2d_histogram.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


def plot_scatter_1to1(demand, diversion, dataset_id, dataset_label):
    """
    Plot 1-1 scatter plot of demand vs diversion.

    Parameters
    ----------
    demand : np.ndarray
        Array of demand values
    diversion : np.ndarray
        Array of diversion values
    dataset_id : str
        Dataset identifier
    dataset_label : str
        Dataset label for plot title
    """
    print("Creating 1-1 scatter plot...")

    # Remove NaN values
    mask = ~(np.isnan(demand) | np.isnan(diversion))
    demand_clean = demand[mask]
    diversion_clean = diversion[mask]

    if len(demand_clean) == 0:
        print("  Warning: No valid data points for scatter plot. Skipping plot.")
        return

    # Subsample for plotting (to avoid overplotting with millions of points)
    n_points = len(demand_clean)
    if n_points > 50000:
        print(f"  Subsampling {n_points:,} points to 50,000 for scatter plot...")
        indices = np.random.choice(n_points, 50000, replace=False)
        demand_plot = demand_clean[indices]
        diversion_plot = diversion_clean[indices]
    else:
        demand_plot = demand_clean
        diversion_plot = diversion_clean

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Scatter plot with transparency
    ax.scatter(demand_plot, diversion_plot, alpha=0.1, s=1, color='#1f77b4',
              rasterized=True)

    # Add 1:1 line
    max_val = max(demand_clean.max(), diversion_clean.max())
    min_val = min(demand_clean.min(), diversion_clean.min())
    ax.plot([min_val, max_val], [min_val, max_val],
           'r--', linewidth=2.5, label='1:1 Line (Perfect Delivery)', alpha=0.9)

    # Calculate shortage statistics
    shortage_mask = diversion_clean < demand_clean
    n_shortage = shortage_mask.sum()
    pct_shortage = 100 * n_shortage / n_points

    # Add statistics text
    stats_text = (
        f"Total Points: {n_points:,}\n"
        f"Shortage Events: {n_shortage:,} ({pct_shortage:.2f}%)\n"
        f"Points where diversion < demand"
    )
    ax.text(0.02, 0.98, stats_text,
           transform=ax.transAxes,
           fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=0.6))

    ax.set_xlabel('NYC Demand (MGD)', fontsize=12, fontweight='bold')
    ax.set_ylabel('NYC Diversion (MGD)', fontsize=12, fontweight='bold')
    ax.set_title(f'NYC Demand vs Diversion Scatter: {dataset_label}',
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    fname = f"{FIG_DIR_DEMAND_DIVERSION}/{dataset_id}_nyc_demand_vs_diversion_scatter.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


def main(dataset_id):
    """
    Main function to generate demand vs diversion diagnostic plots.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    """
    print("=" * 80)
    print(f"NYC DEMAND VS DIVERSION DIAGNOSTICS: {dataset_id}")
    print("=" * 80)

    # Verify dataset
    verify_dataset_id(dataset_id)
    dataset_config = DATASET_CONFIGS[dataset_id]
    dataset_label = f"{dataset_config['description']} ({dataset_config['type']})"

    # Load data
    print("\nLoading data...")
    data = load_shortage_data(dataset_id)

    # Extract demand and diversion data
    demand, diversion, demand_by_real, diversion_by_real = \
        extract_demand_diversion_data(data, dataset_id)

    # Check if we have data
    if len(demand) == 0 or len(diversion) == 0:
        print("\n" + "=" * 80)
        print("ERROR: No data extracted!")
        print("=" * 80)
        print("Please check that the postprocessed data contains 'demand_nyc' and 'delivery_nyc' columns.")
        sys.exit(1)

    # Create plots
    print("\nGenerating plots:")
    print("-" * 80)

    plot_distribution_comparison(demand, diversion, dataset_id, dataset_label)
    plot_2d_histogram_with_marginals(demand, diversion, dataset_id, dataset_label)
    plot_scatter_1to1(demand, diversion, dataset_id, dataset_label)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nFigures saved to: {FIG_DIR_DEMAND_DIVERSION}/")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        print(f"\nAvailable datasets: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)

    dataset_id = sys.argv[1]
    verify_dataset_id(dataset_id)

    main(dataset_id)