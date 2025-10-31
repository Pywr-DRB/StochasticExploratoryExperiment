"""
Multi-dataset drought metric comparison using 2D histogram classification.

This script visualizes which datasets contain droughts in different regions
of the drought metric space using a clean, gridded approach. Each grid cell
is classified and colored based on which combination of datasets contain
droughts in that region.

Classification scheme:
- Grey: Droughts occur in ALL datasets (persistent)
- Dataset-specific colors: Droughts occur ONLY in that dataset
- Mixed colors: Droughts occur in specific combinations
- White/light: No droughts or very few

Usage:
  python 09_plot_drought_metric_comparison.py <ssi_window> <x_metric> <y_metric> [dataset_ids...]

Examples:
  # Compare all 4 datasets
  python 09_plot_drought_metric_comparison.py 12 severity magnitude

  # Compare specific datasets
  python 09_plot_drought_metric_comparison.py 12 severity magnitude stationary_ensemble climate_adjusted_low

  # Duration vs magnitude
  python 09_plot_drought_metric_comparison.py 6 duration magnitude

Available metrics: severity, magnitude, duration
SSI windows: 3, 6, 12 months
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
import itertools
import warnings
warnings.filterwarnings("ignore")

from config import *
from methods.plotting.styles import DATASET_COLORS, DATASET_LABELS_SHORT


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


def load_drought_events(dataset_id, ssi_window, observed=False):
    """
    Load drought events from CSV file.

    Parameters:
    -----------
    dataset_id : str
        Dataset identifier
    ssi_window : int
        SSI window size in months (3, 6, or 12)
    observed : bool
        If True, load observed droughts

    Returns:
    --------
    pd.DataFrame
        Drought events
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
    droughts['start'] = pd.to_datetime(droughts['start'])
    droughts['end'] = pd.to_datetime(droughts['end'])

    # Take absolute values
    for metric in ['severity', 'magnitude']:
        if metric in droughts.columns:
            droughts[metric] = droughts[metric].abs()

    # Remove infinite or NaN values
    droughts = droughts.replace([np.inf, -np.inf], np.nan).dropna(
        subset=['severity', 'magnitude', 'duration']
    )

    return droughts


def validate_metric(metric):
    """Validate that metric is allowed."""
    if metric not in VALID_METRICS:
        raise ValueError(f"Invalid metric: {metric}. Must be one of: {VALID_METRICS}")


def create_2d_histogram(x, y, bins=50, x_range=None, y_range=None):
    """
    Create 2D histogram of data points.

    Parameters:
    -----------
    x, y : np.ndarray
        Data coordinates
    bins : int or tuple
        Number of bins in each dimension
    x_range, y_range : tuple
        (min, max) for each dimension

    Returns:
    --------
    H : np.ndarray
        2D histogram (counts)
    xedges, yedges : np.ndarray
        Bin edges
    """
    if isinstance(bins, int):
        bins = (bins, bins)

    H, xedges, yedges = np.histogram2d(
        x, y,
        bins=bins,
        range=[x_range, y_range]
    )

    return H, xedges, yedges


def get_classification_colors(dataset_ids):
    """
    Generate distinct colors for each possible combination of datasets.

    Parameters:
    -----------
    dataset_ids : list
        List of dataset identifiers

    Returns:
    --------
    color_map : dict
        Maps classification code (integer) to RGB color
    code_to_datasets : dict
        Maps classification code to tuple of dataset_ids
    """
    n_datasets = len(dataset_ids)

    # Get base colors for individual datasets
    base_colors = {}
    for dataset_id in dataset_ids:
        if dataset_id in DATASET_COLORS:
            hex_color = DATASET_COLORS[dataset_id]
            rgb = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (1, 3, 5))
            base_colors[dataset_id] = rgb
        else:
            base_colors[dataset_id] = (0.5, 0.5, 0.5)

    # Generate all possible combinations and assign codes
    # Code 0 = empty (no datasets)
    # Code 1 = all datasets
    # Codes 2+ = specific combinations

    code_to_datasets = {0: ()}
    datasets_to_code = {(): 0}
    color_map = {0: (1.0, 1.0, 1.0, 0.0)}  # Transparent white for empty

    code = 1

    # All datasets -> grey
    all_datasets = tuple(sorted(dataset_ids))
    code_to_datasets[code] = all_datasets
    datasets_to_code[all_datasets] = code
    color_map[code] = (0.6, 0.6, 0.6, 1.0)  # Grey
    code += 1

    # Single datasets -> base colors
    for dataset_id in sorted(dataset_ids):
        combo = (dataset_id,)
        code_to_datasets[code] = combo
        datasets_to_code[combo] = code
        color_map[code] = base_colors[dataset_id] + (1.0,)  # Add alpha
        code += 1

    # Combinations of datasets -> blended colors
    for r in range(2, n_datasets):
        for combo in itertools.combinations(sorted(dataset_ids), r):
            code_to_datasets[code] = combo
            datasets_to_code[combo] = code

            # Blend colors
            colors_to_blend = [base_colors[did] for did in combo]
            blended = tuple(np.mean([c[i] for c in colors_to_blend]) for i in range(3))
            color_map[code] = blended + (1.0,)  # Add alpha
            code += 1

    return color_map, code_to_datasets, datasets_to_code


def classify_grid_cells(datasets_histograms, dataset_ids, min_count=1):
    """
    Classify each grid cell by which datasets have droughts in it.

    Parameters:
    -----------
    datasets_histograms : dict
        Maps dataset_id to 2D histogram array
    dataset_ids : list
        List of dataset identifiers
    min_count : int
        Minimum number of droughts to consider cell "occupied"

    Returns:
    --------
    classification_grid : np.ndarray
        2D array of classification codes
    """
    # Get grid shape from first histogram
    grid_shape = list(datasets_histograms.values())[0].shape

    # Initialize classification grid
    classification_grid = np.zeros(grid_shape, dtype=int)

    # Get color/code mapping
    _, _, datasets_to_code = get_classification_colors(dataset_ids)

    # Classify each cell
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            # Check which datasets have droughts in this cell
            present_datasets = []

            for dataset_id in sorted(dataset_ids):
                if datasets_histograms[dataset_id][i, j] >= min_count:
                    present_datasets.append(dataset_id)

            # Get classification code
            combo = tuple(sorted(present_datasets))
            classification_grid[i, j] = datasets_to_code[combo]

    return classification_grid


def plot_drought_metric_comparison(
    datasets_droughts,
    dataset_ids,
    x_metric='severity',
    y_metric='magnitude',
    ssi_window=12,
    bins=50,
    figsize=(12, 10),
    include_observed=True,
    obs_droughts=None,
    xlim=None,
    ylim=None,
    min_count=1,
    annotate_extremes=True,
    fname=None
):
    """
    Plot multi-dataset drought comparison using 2D histogram classification.

    Parameters:
    -----------
    datasets_droughts : dict
        Dictionary mapping dataset_id to drought DataFrame
    dataset_ids : list
        List of dataset identifiers to compare
    x_metric : str
        Metric for x-axis
    y_metric : str
        Metric for y-axis
    ssi_window : int
        SSI window size
    bins : int
        Number of bins in each dimension
    figsize : tuple
        Figure size (width, height)
    include_observed : bool
        Whether to overlay observed droughts
    obs_droughts : pd.DataFrame
        Observed drought events
    xlim, ylim : tuple or None
        Axis limits
    min_count : int
        Minimum drought count to consider cell occupied
    annotate_extremes : bool
        If True, annotate the observed droughts with largest magnitude and severity
    fname : str
        Output filename

    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    validate_metric(x_metric)
    validate_metric(y_metric)

    # Extract data for each dataset
    all_x = []
    all_y = []
    datasets_data = {}

    for dataset_id in dataset_ids:
        droughts = datasets_droughts[dataset_id]
        x = droughts[x_metric].values
        y = droughts[y_metric].values
        datasets_data[dataset_id] = (x, y)
        all_x.extend(x)
        all_y.extend(y)

    all_x = np.array(all_x)
    all_y = np.array(all_y)

    # Determine data range
    if xlim is None:
        x_range = all_x.max() - all_x.min()
        x_margin = x_range * 0.05
        xlim = (all_x.min() - x_margin, all_x.max() + x_margin)

    if ylim is None:
        y_range = all_y.max() - all_y.min()
        y_margin = y_range * 0.05
        ylim = (all_y.min() - y_margin, all_y.max() + y_margin)

    # Create 2D histograms for each dataset
    print(f"Creating 2D histograms ({bins}x{bins} bins)...")
    datasets_histograms = {}

    for dataset_id in dataset_ids:
        x, y = datasets_data[dataset_id]
        H, xedges, yedges = create_2d_histogram(
            x, y,
            bins=bins,
            x_range=xlim,
            y_range=ylim
        )
        datasets_histograms[dataset_id] = H
        print(f"  {DATASET_LABELS_SHORT.get(dataset_id, dataset_id)}: {len(x)} droughts")

    # Classify grid cells
    print(f"Classifying grid cells...")
    classification_grid = classify_grid_cells(
        datasets_histograms,
        dataset_ids,
        min_count=min_count
    )

    # Get colors
    color_map, code_to_datasets, _ = get_classification_colors(dataset_ids)

    # Count classifications
    from collections import Counter
    unique, counts = np.unique(classification_grid, return_counts=True)
    print(f"  Found {len(unique)} unique classifications:")

    for code in sorted(unique, key=lambda c: -counts[list(unique).index(c)])[:10]:
        combo = code_to_datasets[code]
        count = counts[list(unique).index(code)]

        if len(combo) == 0:
            label = "Empty"
        elif len(combo) == len(dataset_ids):
            label = "All datasets"
        elif len(combo) == 1:
            label = DATASET_LABELS_SHORT.get(combo[0], combo[0])
        else:
            labels = [DATASET_LABELS_SHORT.get(d, d) for d in combo]
            label = " + ".join(labels)

        print(f"    {label}: {count} cells")

    # Create colormap from classification codes
    n_codes = len(color_map)
    cmap_colors = [color_map.get(i, (1, 1, 1, 0)) for i in range(n_codes)]
    cmap = ListedColormap(cmap_colors)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot using imshow (cleaner than pcolormesh)
    im = ax.imshow(
        classification_grid.T,
        origin='lower',
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        cmap=cmap,
        vmin=0,
        vmax=n_codes-1,
        aspect='auto',
        interpolation='nearest',
        alpha=0.9
    )

    # Overlay observed droughts
    if include_observed and obs_droughts is not None and len(obs_droughts) > 0:
        x_obs = obs_droughts[x_metric].values
        y_obs = obs_droughts[y_metric].values
        ax.scatter(
            x_obs, y_obs,
            s=120,
            marker='^',
            c='black',
            edgecolors='white',
            linewidths=1.5,
            alpha=0.95,
            label='Observed',
            zorder=10
        )

        # Annotate extreme droughts
        if annotate_extremes:
            # Find drought with largest magnitude
            idx_max_mag = obs_droughts['magnitude'].idxmax()
            max_mag_drought = obs_droughts.loc[idx_max_mag]
            max_mag_year = pd.to_datetime(max_mag_drought['start']).year

            # Find drought with largest severity
            idx_max_sev = obs_droughts['severity'].idxmax()
            max_sev_drought = obs_droughts.loc[idx_max_sev]
            max_sev_year = pd.to_datetime(max_sev_drought['start']).year

            # Annotate largest magnitude
            ax.annotate(
                str(max_mag_year),
                xy=(max_mag_drought[x_metric], max_mag_drought[y_metric]),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=11,
                fontweight='bold',
                color='black',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', edgecolor='black', linewidth=1.5, alpha=0.9),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='black', linewidth=2),
                zorder=11
            )

            # Annotate largest severity (only if different from largest magnitude)
            if idx_max_sev != idx_max_mag:
                ax.annotate(
                    str(max_sev_year),
                    xy=(max_sev_drought[x_metric], max_sev_drought[y_metric]),
                    xytext=(10, -15),
                    textcoords='offset points',
                    fontsize=11,
                    fontweight='bold',
                    color='black',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='orange', edgecolor='black', linewidth=1.5, alpha=0.9),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3', color='black', linewidth=2),
                    zorder=11
                )

    # Labels and title
    x_label = METRIC_DISPLAY_NAMES.get(x_metric, x_metric)
    y_label = METRIC_DISPLAY_NAMES.get(y_metric, y_metric)

    ax.set_xlabel(x_label, fontsize=13, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=13, fontweight='bold')

    title = f"Drought Distribution Comparison: SSI-{ssi_window}\n{y_metric.title()} vs {x_metric.title()}"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # Set limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Grid
    ax.grid(which='both', color='white', alpha=0.3, linewidth=0.5, linestyle='--', zorder=1)
    ax.set_axisbelow(True)

    # Create legend
    legend_handles = []

    # Collect classifications that actually appear in the grid
    present_codes = set(unique)

    # Add legend entries in priority order:
    # 1. All datasets (if present)
    all_code = 1
    if all_code in present_codes and len(dataset_ids) > 1:
        patch = mpatches.Patch(
            facecolor=color_map[all_code][:3],
            edgecolor='none',
            label='All datasets'
        )
        legend_handles.append(patch)

    # 2. Individual datasets (always show)
    for i, dataset_id in enumerate(sorted(dataset_ids)):
        code = 2 + i  # Individual dataset codes start at 2
        if code in present_codes:
            patch = mpatches.Patch(
                facecolor=color_map[code][:3],
                edgecolor='none',
                label=f"{DATASET_LABELS_SHORT.get(dataset_id, dataset_id)} only"
            )
            legend_handles.append(patch)

    # 3. Most common combinations (show top 3-5)
    combo_codes = [c for c in present_codes if c > 1 + len(dataset_ids)]  # Skip empty, all, and individuals
    combo_counts = {c: counts[list(unique).index(c)] for c in combo_codes}

    for code in sorted(combo_counts.keys(), key=lambda c: -combo_counts[c])[:5]:
        combo = code_to_datasets[code]
        if len(combo) > 1 and len(combo) < len(dataset_ids):
            labels = [DATASET_LABELS_SHORT.get(d, d) for d in combo]
            label = " + ".join(labels)
            patch = mpatches.Patch(
                facecolor=color_map[code][:3],
                edgecolor='none',
                label=label
            )
            legend_handles.append(patch)

    # 4. Observed droughts
    if include_observed and obs_droughts is not None and len(obs_droughts) > 0:
        from matplotlib.lines import Line2D
        obs_handle = Line2D([0], [0], marker='^', color='w',
                           markerfacecolor='black', markeredgecolor='white',
                           markersize=11, label='Observed', linewidth=0)
        legend_handles.append(obs_handle)

    # Add legend
    ax.legend(
        handles=legend_handles,
        loc='upper right',
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=10,
        framealpha=0.95
    )

    plt.tight_layout()

    # Save figure
    if fname is None:
        dataset_str = "_".join([did.replace('climate_adjusted_', '').replace('_ensemble', '')
                                for did in dataset_ids])
        fname = (
            f"{FIG_OUTPUT_DIR}/comparison_grid_ssi{ssi_window}_"
            f"{x_metric}_vs_{y_metric}_{dataset_str}.png"
        )

    plt.savefig(fname, dpi=400, bbox_inches='tight')
    print(f"\nSaved: {fname}")

    return fig, ax


def main():
    """Main execution function."""

    print("=" * 80)
    print("MULTI-DATASET DROUGHT METRIC COMPARISON (GRID CLASSIFICATION)")
    print("=" * 80)

    # Parse command line arguments
    if len(sys.argv) < 4:
        print("\nUsage: python 09_plot_drought_metric_comparison.py <ssi_window> <x_metric> <y_metric> [dataset_ids...]")
        print(f"\nAvailable SSI windows: {SSI_WINDOWS}")
        print(f"Available metrics: {VALID_METRICS}")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        print("\nExamples:")
        print("  # Compare all 4 datasets")
        print("  python 09_plot_drought_metric_comparison.py 12 severity magnitude")
        print("\n  # Compare specific datasets")
        print("  python 09_plot_drought_metric_comparison.py 12 severity magnitude stationary_ensemble climate_adjusted_low")
        sys.exit(1)

    ssi_window = int(sys.argv[1])
    x_metric = sys.argv[2]
    y_metric = sys.argv[3]

    # Validate inputs
    if ssi_window not in SSI_WINDOWS:
        print(f"ERROR: Invalid SSI window: {ssi_window}")
        print(f"Must be one of: {SSI_WINDOWS}")
        sys.exit(1)

    validate_metric(x_metric)
    validate_metric(y_metric)

    # Get dataset IDs
    if len(sys.argv) > 4:
        dataset_ids = sys.argv[4:]
        for did in dataset_ids:
            verify_dataset_id(did)
    else:
        # Default: all datasets
        dataset_ids = list(DATASET_CONFIGS.keys())

    print(f"\nConfiguration:")
    print(f"  SSI Window: {ssi_window} months")
    print(f"  X-metric: {x_metric}")
    print(f"  Y-metric: {y_metric}")
    print(f"  Datasets: {dataset_ids}")
    print()

    # Load drought events for all datasets
    print("Loading drought events...")
    datasets_droughts = {}

    for dataset_id in dataset_ids:
        print(f"  Loading {dataset_id}...")
        droughts = load_drought_events(dataset_id, ssi_window, observed=False)
        datasets_droughts[dataset_id] = droughts
        print(f"    {len(droughts):,} droughts")

    # Load observed droughts
    print(f"\n  Loading observed droughts...")
    obs_droughts = load_drought_events(dataset_ids[0], ssi_window, observed=True)
    print(f"    {len(obs_droughts):,} droughts")

    # Generate plot
    print(f"\nGenerating comparison plot...")
    plot_drought_metric_comparison(
        datasets_droughts=datasets_droughts,
        dataset_ids=dataset_ids,
        x_metric=x_metric,
        y_metric=y_metric,
        ssi_window=ssi_window,
        bins=30,
        min_count=1,
        include_observed=True,
        obs_droughts=obs_droughts,
        annotate_extremes=False
    )

    print("\n" + "=" * 80)
    print("COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
