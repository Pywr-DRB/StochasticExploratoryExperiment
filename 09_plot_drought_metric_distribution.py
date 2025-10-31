"""
Plot drought metric distributions as 2D contour plots/heatmaps.

This script visualizes the joint distribution of any two drought metrics
(e.g., severity vs magnitude, duration vs magnitude, etc.) using:
- Seaborn kernel density estimate (KDE) contours/heatmaps
- Observed drought events overlaid as scatter points

Unlike 09_plot_drought_frequency.py, this script does NOT perform
return period or probability modeling - it simply shows the empirical
distribution of drought characteristics across synthetic realizations.

Usage:
  python 09_plot_drought_metric_distribution.py <dataset_id> <ssi_window> <x_metric> <y_metric>

Examples:
  python 09_plot_drought_metric_distribution.py stationary_ensemble 12 severity magnitude
  python 09_plot_drought_metric_distribution.py climate_adjusted_low 6 duration magnitude
  python 09_plot_drought_metric_distribution.py stationary_ensemble 12 duration severity

Available metrics:
  - severity: Minimum SSI value during drought
  - magnitude: Cumulative SSI deficit
  - duration: Drought length in months

SSI windows: 3, 6, 12 months
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
import warnings
warnings.filterwarnings("ignore")

from config import *


# Output directory
FIG_OUTPUT_DIR = f"{FIG_DIR}/drought_distributions"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

# Drought metrics directory
DROUGHT_METRICS_DIR = f"{ROOT_DIR}/pywrdrb/drought_metrics"

# Valid drought metrics
VALID_METRICS = ['severity', 'magnitude', 'duration']

# Metric display names
METRIC_DISPLAY_NAMES = {
    'severity': 'Severity (min SSI)',
    'magnitude': 'Magnitude (cumulative deficit)',
    'duration': 'Duration (months)'
}

# Metric units for axis labels
METRIC_UNITS = {
    'severity': '',  # Dimensionless (SSI)
    'magnitude': '',  # Dimensionless (cumulative SSI)
    'duration': 'months'
}


def load_drought_events(dataset_id, ssi_window, observed=False):
    """
    Load drought events from CSV file.

    Parameters:
    -----------
    dataset_id : str
        Dataset identifier ('stationary_ensemble', 'climate_adjusted_low', etc.)
    ssi_window : int
        SSI window size in months (3, 6, or 12)
    observed : bool
        If True, load observed droughts; if False, load synthetic droughts

    Returns:
    --------
    pd.DataFrame
        Drought events with columns: start, end, duration, severity, magnitude, realization_id
    """
    if observed:
        fname = f"{DROUGHT_METRICS_DIR}/observed_ssi{ssi_window}_drought_events.csv"
    else:
        fname = f"{DROUGHT_METRICS_DIR}/{dataset_id}_ssi{ssi_window}_drought_events.csv"

    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Drought events file not found: {fname}\n"
            f"Run 05_calculate_ssi_drought_metrics.py first!"
        )

    droughts = pd.read_csv(fname)

    # Convert dates to datetime
    droughts['start'] = pd.to_datetime(droughts['start'])
    droughts['end'] = pd.to_datetime(droughts['end'])

    # Take absolute values and ensure finite
    for metric in ['severity', 'magnitude']:
        if metric in droughts.columns:
            droughts[metric] = droughts[metric].abs()

    # Remove infinite or NaN values
    droughts = droughts.replace([np.inf, -np.inf], np.nan).dropna(
        subset=['severity', 'magnitude', 'duration']
    )

    return droughts


def validate_metric(metric):
    """Validate that metric is one of the allowed values."""
    if metric not in VALID_METRICS:
        raise ValueError(
            f"Invalid metric: {metric}\n"
            f"Must be one of: {VALID_METRICS}"
        )


def plot_drought_metric_distribution(
    syn_droughts,
    obs_droughts,
    x_metric='severity',
    y_metric='magnitude',
    dataset_id='stationary_ensemble',
    ssi_window=12,
    figsize=(10, 8),
    cmap='viridis',
    levels=10,
    plot_type='contourf',
    log_transform=False,
    kde_bw_adjust=1.0,
    kde_thresh=0.01,
    xlim=None,
    ylim=None,
    obs_marker='^',
    obs_color='red',
    obs_size=80,
    obs_alpha=0.9,
    fname=None
):
    """
    Plot 2D distribution of drought metrics using seaborn KDE.

    Parameters:
    -----------
    syn_droughts : pd.DataFrame
        Synthetic drought events
    obs_droughts : pd.DataFrame
        Observed drought events
    x_metric : str
        Metric for x-axis ('severity', 'magnitude', 'duration')
    y_metric : str
        Metric for y-axis ('severity', 'magnitude', 'duration')
    dataset_id : str
        Dataset identifier for title
    ssi_window : int
        SSI window size for title
    figsize : tuple
        Figure size (width, height)
    cmap : str
        Colormap name
    levels : int or list
        Number of contour levels or list of level values
    plot_type : str
        'contourf' for filled contours, 'contour' for line contours,
        'hexbin' for hexagonal binning
    log_transform : bool
        If True, apply log transform to metrics before plotting
    kde_bw_adjust : float
        Bandwidth adjustment for KDE (higher = smoother)
    kde_thresh : float
        Minimum density threshold for KDE contours (0-1). Lower values show more sparse regions.
        Default 0.01 (1% of max density). Use 0 to show all densities.
    xlim : tuple or None
        Manual x-axis limits (xmin, xmax). If None, auto-computed from data.
    ylim : tuple or None
        Manual y-axis limits (ymin, ymax). If None, auto-computed from data.
    obs_marker : str
        Marker style for observed droughts
    obs_color : str
        Color for observed drought markers
    obs_size : float
        Size of observed drought markers
    obs_alpha : float
        Alpha (transparency) for observed markers
    fname : str
        Output filename (if None, will auto-generate)

    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """

    # Validate metrics
    validate_metric(x_metric)
    validate_metric(y_metric)

    # Extract data
    x_syn = syn_droughts[x_metric].values
    y_syn = syn_droughts[y_metric].values

    x_obs = obs_droughts[x_metric].values if len(obs_droughts) > 0 else np.array([])
    y_obs = obs_droughts[y_metric].values if len(obs_droughts) > 0 else np.array([])

    # Apply log transform if requested
    if log_transform:
        x_syn = np.log10(x_syn + 1e-10)  # Add small epsilon to avoid log(0)
        y_syn = np.log10(y_syn + 1e-10)
        if len(x_obs) > 0:
            x_obs = np.log10(x_obs + 1e-10)
            y_obs = np.log10(y_obs + 1e-10)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create joint plot based on type
    if plot_type == 'hexbin':
        # Hexagonal binning (good for very large datasets)
        hb = ax.hexbin(x_syn, y_syn, gridsize=30, cmap=cmap, mincnt=1, bins='log')
        cb = plt.colorbar(hb, ax=ax, label='Count (log scale)')

    elif plot_type == 'contour':
        # Line contours only
        try:
            sns.kdeplot(
                x=x_syn, y=y_syn,
                ax=ax,
                levels=levels,
                color='black',
                linewidths=1.5,
                alpha=0.7,
                bw_adjust=kde_bw_adjust
            )
        except Exception as e:
            print(f"Warning: KDE failed, falling back to hexbin: {e}")
            hb = ax.hexbin(x_syn, y_syn, gridsize=30, cmap=cmap, mincnt=1, bins='log')
            cb = plt.colorbar(hb, ax=ax, label='Count (log scale)')

    else:  # 'contourf' (default)
        # Filled contours with colorbar
        try:
            # Use kdeplot with filled contours
            sns.kdeplot(
                x=x_syn, y=y_syn,
                ax=ax,
                levels=levels,
                cmap=cmap,
                fill=True,
                thresh=kde_thresh,
                bw_adjust=kde_bw_adjust,
                alpha=0.8
            )

            # Add contour lines for clarity
            sns.kdeplot(
                x=x_syn, y=y_syn,
                ax=ax,
                levels=levels,
                color='white',
                linewidths=0.8,
                alpha=0.5,
                thresh=kde_thresh,
                bw_adjust=kde_bw_adjust
            )

            # Create a colorbar proxy
            # (seaborn kdeplot doesn't return mappable, so we create one)
            norm = colors.Normalize(vmin=0, vmax=1)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cb = plt.colorbar(sm, ax=ax, label='Density')

        except Exception as e:
            print(f"Warning: KDE failed, falling back to hexbin: {e}")
            hb = ax.hexbin(x_syn, y_syn, gridsize=30, cmap=cmap, mincnt=1, bins='log')
            cb = plt.colorbar(hb, ax=ax, label='Count (log scale)')

    # Set axis limits to include ALL data points (synthetic + observed)
    if xlim is None or ylim is None:
        # Combine synthetic and observed data to determine full range
        all_x = np.concatenate([x_syn, x_obs]) if len(x_obs) > 0 else x_syn
        all_y = np.concatenate([y_syn, y_obs]) if len(y_obs) > 0 else y_syn

        # Add 5% padding on each side
        x_range = all_x.max() - all_x.min()
        y_range = all_y.max() - all_y.min()
        x_margin = x_range * 0.05
        y_margin = y_range * 0.05

        if xlim is None:
            xlim = (all_x.min() - x_margin, all_x.max() + x_margin)
        if ylim is None:
            ylim = (all_y.min() - y_margin, all_y.max() + y_margin)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Overlay observed droughts
    if len(x_obs) > 0:
        ax.scatter(
            x_obs, y_obs,
            s=obs_size,
            marker=obs_marker,
            c=obs_color,
            edgecolors='white',
            linewidths=1.0,
            alpha=obs_alpha,
            label='Observed',
            zorder=10
        )
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

    # Labels and title
    x_label = METRIC_DISPLAY_NAMES.get(x_metric, x_metric)
    y_label = METRIC_DISPLAY_NAMES.get(y_metric, y_metric)

    if log_transform:
        x_label = f"log₁₀({x_label})"
        y_label = f"log₁₀({y_label})"

    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')

    # Title
    dataset_name = DATASET_CONFIGS[dataset_id]['description']
    title = f"Drought Distribution: {dataset_name}\nSSI-{ssi_window} | {y_metric.title()} vs {x_metric.title()}"
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

    # Grid
    ax.grid(which='both', color='gray', alpha=0.2, linewidth=0.5, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save figure
    if fname is None:
        log_suffix = '_log' if log_transform else ''
        fname = (
            f"{FIG_OUTPUT_DIR}/{dataset_id}_ssi{ssi_window}_"
            f"{x_metric}_vs_{y_metric}{log_suffix}_{plot_type}.png"
        )

    plt.savefig(fname, dpi=400, bbox_inches='tight')
    print(f"Saved: {fname}")

    return fig, ax


def plot_all_metric_pairs(
    dataset_id,
    ssi_window=12,
    plot_type='contourf',
    log_transform=False
):
    """
    Generate distribution plots for all combinations of drought metrics.

    Parameters:
    -----------
    dataset_id : str
        Dataset identifier
    ssi_window : int
        SSI window size (3, 6, or 12)
    plot_type : str
        Type of plot ('contourf', 'contour', 'hexbin')
    log_transform : bool
        Whether to apply log transform
    """

    # Load data
    print(f"Loading drought events for {dataset_id}, SSI-{ssi_window}...")
    syn_droughts = load_drought_events(dataset_id, ssi_window, observed=False)
    obs_droughts = load_drought_events(dataset_id, ssi_window, observed=True)

    print(f"  Synthetic droughts: {len(syn_droughts):,}")
    print(f"  Observed droughts: {len(obs_droughts):,}")

    # Generate all pairs
    metrics = VALID_METRICS

    print(f"\nGenerating distribution plots...")
    for i, x_metric in enumerate(metrics):
        for j, y_metric in enumerate(metrics):
            if i >= j:  # Skip duplicates and self-pairs
                continue

            print(f"\n  Plotting {y_metric} vs {x_metric}...")

            try:
                plot_drought_metric_distribution(
                    syn_droughts=syn_droughts,
                    obs_droughts=obs_droughts,
                    x_metric=x_metric,
                    y_metric=y_metric,
                    dataset_id=dataset_id,
                    ssi_window=ssi_window,
                    plot_type=plot_type,
                    log_transform=log_transform
                )
                plt.close()  # Close to free memory

            except Exception as e:
                print(f"    ERROR: {e}")
                continue


def main():
    """Main execution function."""

    print("=" * 80)
    print("DROUGHT METRIC DISTRIBUTION PLOTTING")
    print("=" * 80)

    # Parse command line arguments
    if len(sys.argv) < 2:
        print("\nUsage: python 09_plot_drought_metric_distribution.py <dataset_id> [ssi_window] [x_metric] [y_metric]")
        print(f"\nAvailable datasets: {list(DATASET_CONFIGS.keys())}")
        print(f"Available SSI windows: {SSI_WINDOWS}")
        print(f"Available metrics: {VALID_METRICS}")
        print("\nExamples:")
        print("  python 09_plot_drought_metric_distribution.py stationary_ensemble")
        print("  python 09_plot_drought_metric_distribution.py stationary_ensemble 12")
        print("  python 09_plot_drought_metric_distribution.py stationary_ensemble 12 severity magnitude")
        sys.exit(1)

    dataset_id = sys.argv[1]
    verify_dataset_id(dataset_id)

    # Optional: SSI window (default: 12)
    ssi_window = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    if ssi_window not in SSI_WINDOWS:
        print(f"ERROR: Invalid SSI window: {ssi_window}")
        print(f"Must be one of: {SSI_WINDOWS}")
        sys.exit(1)

    # Optional: specific metric pair
    if len(sys.argv) >= 5:
        x_metric = sys.argv[3]
        y_metric = sys.argv[4]

        validate_metric(x_metric)
        validate_metric(y_metric)

        print(f"\nDataset: {dataset_id}")
        print(f"SSI Window: {ssi_window} months")
        print(f"Plotting: {y_metric} vs {x_metric}\n")

        # Load data
        syn_droughts = load_drought_events(dataset_id, ssi_window, observed=False)
        obs_droughts = load_drought_events(dataset_id, ssi_window, observed=True)

        print(f"Synthetic droughts: {len(syn_droughts):,}")
        print(f"Observed droughts: {len(obs_droughts):,}\n")

        # Plot single pair
        plot_drought_metric_distribution(
            syn_droughts=syn_droughts,
            obs_droughts=obs_droughts,
            x_metric=x_metric,
            y_metric=y_metric,
            dataset_id=dataset_id,
            ssi_window=ssi_window,
            plot_type='hexbin',
        )

    else:
        # Plot all pairs for this dataset and SSI window
        print(f"\nDataset: {dataset_id}")
        print(f"SSI Window: {ssi_window} months")
        print("Generating all metric pair combinations...\n")

        plot_all_metric_pairs(
            dataset_id=dataset_id,
            ssi_window=ssi_window,
            plot_type='contourf',
            log_transform=False
        )

    print("\n" + "=" * 80)
    print("COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
