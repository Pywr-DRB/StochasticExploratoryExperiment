"""
Multi-dataset drought metric comparison using hexbin classification.

This script visualizes which datasets contain droughts in different regions
of the drought metric space (e.g., severity vs magnitude). Instead of showing
density via colormap, each hexbin is classified and colored based on which
combination of datasets contain droughts in that region:

- Grey: Droughts occur in ALL datasets
- Dataset-specific colors: Droughts occur in ONLY that dataset
- Mixed colors: Droughts occur in specific combinations of datasets

This reveals how climate change affects the distribution of drought characteristics:
- Which drought types emerge only under certain scenarios?
- Which drought types disappear?
- Which drought types persist across all scenarios?

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
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.collections import PolyCollection
from scipy.spatial import cKDTree
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


def create_hexbin_grid(x_all, y_all, gridsize=30):
    """
    Create hexagonal grid covering the data range.

    Parameters:
    -----------
    x_all : np.ndarray
        All x-values from all datasets
    y_all : np.ndarray
        All y-values from all datasets
    gridsize : int
        Number of hexagons along x-axis

    Returns:
    --------
    hex_centers : np.ndarray
        (N, 2) array of hexagon centers
    hex_radius : float
        Hexagon radius
    """
    # Determine data range with padding
    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()

    x_range = x_max - x_min
    y_range = y_max - y_min

    # Add 5% padding
    x_min -= x_range * 0.05
    x_max += x_range * 0.05
    y_min -= y_range * 0.05
    y_max += y_range * 0.05

    # Calculate hexagon dimensions
    hex_width = (x_max - x_min) / gridsize
    hex_height = hex_width * np.sqrt(3) / 2  # Regular hexagon height

    # Generate hexagon centers
    hex_centers = []
    y_gridsize = int((y_max - y_min) / hex_height) + 1

    for row in range(y_gridsize):
        y_center = y_min + row * hex_height
        x_offset = hex_width / 2 if row % 2 == 1 else 0

        for col in range(gridsize + 2):  # +2 to cover edges
            x_center = x_min + col * hex_width + x_offset
            hex_centers.append([x_center, y_center])

    hex_centers = np.array(hex_centers)
    hex_radius = hex_width / 2

    return hex_centers, hex_radius


def classify_hexbins(hex_centers, hex_radius, datasets_data, dataset_ids):
    """
    Classify each hexbin by which datasets contain points in it.

    Parameters:
    -----------
    hex_centers : np.ndarray
        (N, 2) array of hexagon centers
    hex_radius : float
        Hexagon radius
    datasets_data : dict
        Dictionary mapping dataset_id to (x, y) tuples
    dataset_ids : list
        List of dataset identifiers

    Returns:
    --------
    classifications : list
        List of tuples indicating which datasets are present in each hex
    """
    n_datasets = len(dataset_ids)
    n_hexbins = len(hex_centers)

    # Build KDTree for each dataset for fast lookup
    trees = {}
    for dataset_id in dataset_ids:
        x, y = datasets_data[dataset_id]
        if len(x) > 0:
            trees[dataset_id] = cKDTree(np.column_stack([x, y]))
        else:
            trees[dataset_id] = None

    # Classify each hexbin
    classifications = []
    search_radius = hex_radius * 1.2  # Slightly larger to ensure coverage

    for center in hex_centers:
        present_datasets = []

        for dataset_id in dataset_ids:
            if trees[dataset_id] is None:
                continue

            # Query points within radius
            indices = trees[dataset_id].query_ball_point(center, search_radius)

            if len(indices) > 0:
                present_datasets.append(dataset_id)

        # Store as tuple (hashable for dictionary lookup)
        classifications.append(tuple(sorted(present_datasets)))

    return classifications


def get_classification_colors(dataset_ids):
    """
    Generate colors for each possible combination of datasets.

    Parameters:
    -----------
    dataset_ids : list
        List of dataset identifiers

    Returns:
    --------
    color_map : dict
        Maps classification tuple to RGB color
    """
    n_datasets = len(dataset_ids)

    # Get base colors for individual datasets
    base_colors = {}
    for dataset_id in dataset_ids:
        if dataset_id in DATASET_COLORS:
            # Convert hex to RGB
            hex_color = DATASET_COLORS[dataset_id]
            rgb = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (1, 3, 5))
            base_colors[dataset_id] = rgb
        else:
            # Default color if not in config
            base_colors[dataset_id] = (0.5, 0.5, 0.5)

    # Build color map for all combinations
    color_map = {}

    # Empty (no datasets) -> white
    color_map[()] = (1.0, 1.0, 1.0)

    # All datasets -> grey
    all_datasets = tuple(sorted(dataset_ids))
    color_map[all_datasets] = (0.5, 0.5, 0.5)

    # Single dataset -> use dataset color
    for dataset_id in dataset_ids:
        color_map[(dataset_id,)] = base_colors[dataset_id]

    # Combinations of datasets -> blend colors
    for r in range(2, n_datasets):
        for combo in itertools.combinations(sorted(dataset_ids), r):
            # Average the colors
            colors_to_blend = [base_colors[did] for did in combo]
            blended = tuple(np.mean([c[i] for c in colors_to_blend]) for i in range(3))
            color_map[combo] = blended

    return color_map


def plot_drought_metric_comparison(
    datasets_droughts,
    dataset_ids,
    x_metric='severity',
    y_metric='magnitude',
    ssi_window=12,
    gridsize=30,
    figsize=(12, 10),
    include_observed=True,
    obs_droughts=None,
    xlim=None,
    ylim=None,
    fname=None
):
    """
    Plot multi-dataset drought comparison using hexbin classification.

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
    gridsize : int
        Number of hexagons along x-axis
    figsize : tuple
        Figure size (width, height)
    include_observed : bool
        Whether to overlay observed droughts
    obs_droughts : pd.DataFrame
        Observed drought events
    xlim : tuple or None
        X-axis limits (xmin, xmax)
    ylim : tuple or None
        Y-axis limits (ymin, ymax)
    fname : str
        Output filename

    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    validate_metric(x_metric)
    validate_metric(y_metric)

    # Extract data for each dataset
    datasets_data = {}
    all_x = []
    all_y = []

    for dataset_id in dataset_ids:
        droughts = datasets_droughts[dataset_id]
        x = droughts[x_metric].values
        y = droughts[y_metric].values
        datasets_data[dataset_id] = (x, y)
        all_x.extend(x)
        all_y.extend(y)

    all_x = np.array(all_x)
    all_y = np.array(all_y)

    # Create hexbin grid
    print(f"Creating hexagonal grid (gridsize={gridsize})...")
    hex_centers, hex_radius = create_hexbin_grid(all_x, all_y, gridsize=gridsize)
    print(f"  Generated {len(hex_centers)} hexagons")

    # Classify hexbins
    print(f"Classifying hexbins by dataset presence...")
    classifications = classify_hexbins(hex_centers, hex_radius, datasets_data, dataset_ids)

    # Count classifications
    from collections import Counter
    class_counts = Counter(classifications)
    print(f"  Found {len(class_counts)} unique classifications:")
    for classification, count in sorted(class_counts.items(), key=lambda x: -x[1])[:10]:
        if len(classification) == 0:
            label = "Empty"
        elif len(classification) == len(dataset_ids):
            label = "All datasets"
        elif len(classification) == 1:
            label = DATASET_LABELS_SHORT.get(classification[0], classification[0])
        else:
            labels = [DATASET_LABELS_SHORT.get(d, d) for d in classification]
            label = " + ".join(labels)
        print(f"    {label}: {count} hexbins")

    # Get colors for classifications
    color_map = get_classification_colors(dataset_ids)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot hexagons
    print(f"Plotting hexagons...")
    hex_patches = []
    hex_colors = []

    for center, classification in zip(hex_centers, classifications):
        # Skip empty hexbins
        if len(classification) == 0:
            continue

        # Create hexagon polygon
        angles = np.linspace(0, 2*np.pi, 7)
        vertices = np.column_stack([
            center[0] + hex_radius * np.cos(angles),
            center[1] + hex_radius * np.sin(angles)
        ])

        hex_patches.append(vertices)
        hex_colors.append(color_map[classification])

    # Add hexagons to plot
    poly_collection = PolyCollection(
        hex_patches,
        facecolors=hex_colors,
        edgecolors='white',
        linewidths=0.5,
        alpha=0.8
    )
    ax.add_collection(poly_collection)

    # Set axis limits
    if xlim is None or ylim is None:
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
    if include_observed and obs_droughts is not None and len(obs_droughts) > 0:
        x_obs = obs_droughts[x_metric].values
        y_obs = obs_droughts[y_metric].values
        ax.scatter(
            x_obs, y_obs,
            s=100,
            marker='^',
            c='black',
            edgecolors='white',
            linewidths=1.5,
            alpha=0.95,
            label='Observed',
            zorder=10
        )

    # Labels and title
    x_label = METRIC_DISPLAY_NAMES.get(x_metric, x_metric)
    y_label = METRIC_DISPLAY_NAMES.get(y_metric, y_metric)

    ax.set_xlabel(x_label, fontsize=13, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=13, fontweight='bold')

    title = f"Drought Distribution Comparison: SSI-{ssi_window}\n{y_metric.title()} vs {x_metric.title()}"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # Grid
    ax.grid(which='both', color='gray', alpha=0.2, linewidth=0.5, linestyle='--')
    ax.set_axisbelow(True)

    # Create legend
    legend_handles = []

    # All datasets (grey)
    if len(dataset_ids) > 1:
        all_patch = mpatches.Patch(
            facecolor=(0.5, 0.5, 0.5),
            edgecolor='white',
            linewidth=0.5,
            label='All datasets'
        )
        legend_handles.append(all_patch)

    # Individual datasets
    for dataset_id in dataset_ids:
        if (dataset_id,) in color_map:
            patch = mpatches.Patch(
                facecolor=color_map[(dataset_id,)],
                edgecolor='white',
                linewidth=0.5,
                label=f"{DATASET_LABELS_SHORT.get(dataset_id, dataset_id)} only"
            )
            legend_handles.append(patch)

    # Combinations (show most common ones)
    combo_counts = {k: v for k, v in class_counts.items()
                    if len(k) > 1 and len(k) < len(dataset_ids) and v > 5}

    for combo in sorted(combo_counts.keys(), key=lambda x: -combo_counts[x])[:5]:
        labels = [DATASET_LABELS_SHORT.get(d, d) for d in combo]
        label = " + ".join(labels)
        patch = mpatches.Patch(
            facecolor=color_map[combo],
            edgecolor='white',
            linewidth=0.5,
            label=label
        )
        legend_handles.append(patch)

    # Observed
    if include_observed and obs_droughts is not None and len(obs_droughts) > 0:
        from matplotlib.lines import Line2D
        obs_handle = Line2D([0], [0], marker='^', color='w',
                           markerfacecolor='black', markeredgecolor='white',
                           markersize=10, label='Observed', linewidth=0)
        legend_handles.append(obs_handle)

    # Add legend
    ax.legend(
        handles=legend_handles,
        loc='upper right',
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=10
    )

    plt.tight_layout()

    # Save figure
    if fname is None:
        dataset_str = "_".join([did.replace('climate_adjusted_', '').replace('_ensemble', '')
                                for did in dataset_ids])
        fname = (
            f"{FIG_OUTPUT_DIR}/comparison_hexbin_ssi{ssi_window}_"
            f"{x_metric}_vs_{y_metric}_{dataset_str}.png"
        )

    plt.savefig(fname, dpi=400, bbox_inches='tight')
    print(f"\nSaved: {fname}")

    # Also save vector version
    base = fname.rsplit('.', 1)[0]
    svg_fname = f"{base}.svg"
    plt.savefig(svg_fname, bbox_inches='tight')
    print(f"Saved: {svg_fname}")

    return fig, ax


def main():
    """Main execution function."""

    print("=" * 80)
    print("MULTI-DATASET DROUGHT METRIC COMPARISON (HEXBIN CLASSIFICATION)")
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
        gridsize=35,
        include_observed=True,
        obs_droughts=obs_droughts
    )

    print("\n" + "=" * 80)
    print("COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
