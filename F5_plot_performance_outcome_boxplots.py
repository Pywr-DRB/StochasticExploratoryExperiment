"""
F9: Performance metrics boxplot comparison.

Multi-panel figure showing quantile comparison (5th, 50th, 95th) of performance
metrics across datasets.

Usage:
  python F9_plot_performance_outcome_boxplots.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from methods.config import FIG_DIR, DATASET_CONFIGS
from methods.plotting.styles import (
    DATASET_COLORS, DATASET_LABELS,
    METRIC_DISPLAY_NAMES,
    METRICS_TO_SCALE,
    RECONSTRUCTION_SCALE_FACTOR,
    HISTORIC_MARKER_STYLE,
    get_ylabel_for_metrics,
    DPI_HIGH,
)
from methods.load import load_performance_metrics


# Output directory
FIG_OUTPUT_DIR = f"{FIG_DIR}/F9_performance_metrics"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

# ============================================================================
# CONFIGURABLE METRICS - 3x3 GRID BY CATEGORY
# ============================================================================
# The figure uses a 3x3 grid layout where each row corresponds to a category:
#   Row 1: NYC Reservoir Storage
#   Row 2: NYC Diversion/Demand Outcomes
#   Row 3: Montague Flow Outcomes
#
# Metrics were selected based on sensitivity analysis (relative differences
# between stationary and climate-adjusted datasets).

# Category 1: NYC Reservoir Storage (Row 1)
# Selected metrics with highest sensitivity to climate scenarios
METRICS_NYC_STORAGE = [
    'min_storage_pct',              # Absolute minimum storage (%) - 29% avg change
    'pct_days_storage_below_30',    # % days storage <30% - 59% avg change
    'years_below_30pct',            # Years with storage dropping ≤30% - 52% avg change
]

# Category 2: NYC Diversion/Demand Outcomes (Row 2)
METRICS_NYC_DIVERSION = [
    'mean_annual_nyc_diversion_shortage_mg',  # Mean annual NYC shortage - 53% avg change
    'pct_days_nyc_diversion_shortage',        # % days with NYC shortage - 11% avg change
    'mean_annual_nyc_contribution_mg',        # Mean annual NYC contribution - 12% avg change
]

# Category 3: Montague Flow Outcomes (Row 3)
METRICS_MONTAGUE = [
    'mean_annual_montague_shortage_mg',  # Mean annual Montague shortage - 62% avg change
    'max_1day_montague_shortage_mg',     # Max single-day shortage - 28% avg change
    'max_consecutive_drought_days',      # Longest drought duration - 21% avg change
]

# Combined list for compatibility with existing code
METRICS_TO_PLOT = METRICS_NYC_STORAGE + METRICS_NYC_DIVERSION + METRICS_MONTAGUE

# Category labels for row titles
CATEGORY_LABELS = {
    'NYC_STORAGE': 'NYC Reservoir Storage',
    'NYC_DIVERSION': 'NYC Diversion Outcomes',
    'MONTAGUE': 'Montague Flow Outcomes',
}

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
# Specify which dataset to use as baseline (for calculating percentage changes)
BASELINE_DATASET = 'stationary_ensemble'

# Option to manually specify which datasets to plot
# Set to None to automatically use all datasets from config.py
# Set to list of dataset_ids to manually specify
DATASETS_TO_PLOT = None  # None = use all datasets from config.py

# Option to show historical reconstruction values on plots
SHOW_HISTORIC = False  # Set to False to hide historic reconstruction points


def validate_metrics(metrics_df, dataset_id):
    """
    Validate that all requested metrics exist in the DataFrame.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with performance metrics
    dataset_id : str
        Dataset identifier (for error messages)

    Raises
    ------
    ValueError
        If any requested metrics are missing
    """
    available_metrics = set(metrics_df.columns)
    requested_metrics = set(METRICS_TO_PLOT)
    missing_metrics = requested_metrics - available_metrics

    if missing_metrics:
        raise ValueError(
            f"ERROR: Dataset '{dataset_id}' is missing requested metrics: {missing_metrics}\n"
            f"Available metrics: {sorted(available_metrics)}\n"
            f"Requested metrics: {sorted(requested_metrics)}\n"
            f"Please update METRICS_TO_PLOT or regenerate metrics CSV."
        )


def get_datasets_to_plot():
    """Get list of datasets from config.py."""
    all_datasets = list(DATASET_CONFIGS.keys())

    if DATASETS_TO_PLOT is None:
        datasets_to_plot = all_datasets
    else:
        datasets_to_plot = DATASETS_TO_PLOT
        for d in datasets_to_plot:
            if d not in all_datasets:
                raise ValueError(f"Dataset '{d}' not found in config.py!")

    # Use labels from styles module
    dataset_labels = {did: DATASET_LABELS.get(did, did) for did in datasets_to_plot}

    return datasets_to_plot, dataset_labels


def prepare_data_for_boxplots(all_metrics_dfs, datasets_to_plot, dataset_labels):
    """
    Prepare data in long format for seaborn box plots.

    Parameters
    ----------
    all_metrics_dfs : dict
        Dictionary mapping dataset_id to metrics DataFrame
    datasets_to_plot : list
        List of dataset IDs to include
    dataset_labels : dict
        Display labels for datasets

    Returns
    -------
    df_absolute : pd.DataFrame
        Long-format DataFrame for absolute values
    df_pct_change : pd.DataFrame
        Long-format DataFrame for percentage changes
    """
    # Prepare absolute values
    dfs_absolute = []
    for dataset_id in datasets_to_plot:
        df = all_metrics_dfs[dataset_id][METRICS_TO_PLOT].copy()
        df['dataset'] = dataset_labels[dataset_id]
        df['dataset_id'] = dataset_id
        df['realization'] = df.index
        dfs_absolute.append(df)

    df_absolute = pd.concat(dfs_absolute, ignore_index=True)

    # Melt to long format for seaborn
    df_absolute = df_absolute.melt(
        id_vars=['dataset', 'dataset_id', 'realization'],
        value_vars=METRICS_TO_PLOT,
        var_name='metric',
        value_name='value'
    )

    # Prepare percentage changes (relative to baseline)
    baseline_df = all_metrics_dfs[BASELINE_DATASET][METRICS_TO_PLOT]

    dfs_pct_change = []
    for dataset_id in datasets_to_plot:
        if dataset_id == BASELINE_DATASET:
            continue  # Skip baseline for percentage change

        df = all_metrics_dfs[dataset_id][METRICS_TO_PLOT].copy()

        # Calculate percentage change for each metric
        for metric in METRICS_TO_PLOT:
            baseline_values = baseline_df[metric].values
            current_values = df[metric].values

            # Calculate pairwise percentage changes
            # Use epsilon to avoid division by zero
            eps = 1e-8
            pct_change = 100.0 * (current_values - baseline_values) / np.maximum(np.abs(baseline_values), eps)

            df[metric] = pct_change

        df['dataset'] = dataset_labels[dataset_id]
        df['dataset_id'] = dataset_id
        df['realization'] = df.index
        dfs_pct_change.append(df)

    if dfs_pct_change:
        df_pct_change = pd.concat(dfs_pct_change, ignore_index=True)

        # Melt to long format
        df_pct_change = df_pct_change.melt(
            id_vars=['dataset', 'dataset_id', 'realization'],
            value_vars=METRICS_TO_PLOT,
            var_name='metric',
            value_name='pct_change'
        )
    else:
        df_pct_change = None

    return df_absolute, df_pct_change


def identify_top_changing_metrics(all_metrics_dfs, baseline_dataset, comparison_datasets,
                                  all_metrics=None, n_top=10):
    """
    Identify metrics with largest mean percentage changes relative to baseline.

    This function calculates the mean percentage change for each metric across
    all comparison datasets and ranks them to identify the most sensitive metrics.

    Parameters
    ----------
    all_metrics_dfs : dict
        Dictionary mapping dataset_id to metrics DataFrame
    baseline_dataset : str
        Dataset ID to use as baseline for comparison
    comparison_datasets : list
        List of dataset IDs to compare against baseline
    all_metrics : list, optional
        List of all available metrics to consider. If None, uses all metrics
        from the baseline dataset.
    n_top : int, optional
        Number of top changing metrics to return. Default: 10

    Returns
    -------
    top_metrics_df : pd.DataFrame
        DataFrame with columns:
        - metric: Metric name
        - mean_abs_pct_change: Mean absolute percentage change across comparison datasets
        - mean_pct_change: Mean percentage change (signed) across comparison datasets
        - max_abs_pct_change: Maximum absolute percentage change
        - direction: 'increase', 'decrease', or 'mixed'
    """
    import numpy as np
    import pandas as pd

    # Get all available metrics if not specified
    if all_metrics is None:
        all_metrics = [col for col in all_metrics_dfs[baseline_dataset].columns]

    # Calculate percentage changes for each metric across all comparison datasets
    metric_changes = {}

    baseline_df = all_metrics_dfs[baseline_dataset]

    for metric in all_metrics:
        if metric not in baseline_df.columns:
            continue

        baseline_values = baseline_df[metric].values
        baseline_mean = baseline_values.mean()

        # Skip if baseline mean is essentially zero
        if abs(baseline_mean) < 1e-8:
            continue

        pct_changes = []

        for comp_dataset in comparison_datasets:
            comp_df = all_metrics_dfs[comp_dataset]

            if metric not in comp_df.columns:
                continue

            comp_values = comp_df[metric].values
            comp_mean = comp_values.mean()

            # Calculate percentage change
            pct_change = 100.0 * (comp_mean - baseline_mean) / abs(baseline_mean)
            pct_changes.append(pct_change)

        if pct_changes:
            mean_pct_change = np.mean(pct_changes)
            mean_abs_pct_change = np.mean([abs(pc) for pc in pct_changes])
            max_abs_pct_change = max([abs(pc) for pc in pct_changes])

            # Determine direction
            if all(pc >= 0 for pc in pct_changes):
                direction = 'increase'
            elif all(pc <= 0 for pc in pct_changes):
                direction = 'decrease'
            else:
                direction = 'mixed'

            metric_changes[metric] = {
                'mean_abs_pct_change': mean_abs_pct_change,
                'mean_pct_change': mean_pct_change,
                'max_abs_pct_change': max_abs_pct_change,
                'direction': direction
            }

    # Convert to DataFrame and sort by mean absolute percentage change
    changes_df = pd.DataFrame(metric_changes).T
    changes_df.index.name = 'metric'
    changes_df = changes_df.reset_index()
    changes_df = changes_df.sort_values('mean_abs_pct_change', ascending=False)

    # Return top N
    top_metrics_df = changes_df.head(n_top)

    return top_metrics_df


def print_top_changing_metrics(top_metrics_df, n_display=10):
    """
    Print a formatted table of top changing metrics.

    Parameters
    ----------
    top_metrics_df : pd.DataFrame
        DataFrame from identify_top_changing_metrics()
    n_display : int, optional
        Number of metrics to display. Default: 10
    """
    print(f"\n{'='*80}")
    print(f"TOP {min(n_display, len(top_metrics_df))} METRICS WITH LARGEST CHANGES")
    print(f"{'='*80}")
    print(f"{'Rank':<6}{'Metric':<45}{'Mean Δ%':<12}{'Max |Δ%|':<12}{'Direction':<12}")
    print(f"{'-'*80}")

    for i, row in enumerate(top_metrics_df.head(n_display).itertuples(index=False)):
        rank = i + 1
        metric = row.metric
        mean_change = row.mean_pct_change
        max_abs_change = row.max_abs_pct_change
        direction = row.direction

        # Format with appropriate sign and color indicator
        if direction == 'increase':
            sign = '↑'
        elif direction == 'decrease':
            sign = '↓'
        else:
            sign = '±'

        print(f"{rank:<6}{metric:<45}{mean_change:>+10.1f}%  {max_abs_change:>10.1f}%  {sign} {direction:<10}")

    print(f"{'='*80}\n")


def calculate_quantiles_by_dataset(all_metrics_dfs, datasets_to_plot, metrics_to_plot):
    """
    Calculate 5th, 50th, and 95th percentiles for each metric and dataset.

    Parameters
    ----------
    all_metrics_dfs : dict
        Dictionary mapping dataset_id to metrics DataFrame
    datasets_to_plot : list
        List of dataset IDs
    metrics_to_plot : list
        List of metrics to calculate quantiles for

    Returns
    -------
    quantiles_data : dict
        Nested dict: {metric: {dataset_id: {'p5': val, 'p50': val, 'p95': val}}}
    """
    quantiles_data = {}

    for metric in metrics_to_plot:
        quantiles_data[metric] = {}

        for dataset_id in datasets_to_plot:
            df = all_metrics_dfs[dataset_id]

            if metric in df.columns:
                values = df[metric].values
                p5 = np.percentile(values, 5)
                p50 = np.percentile(values, 50)
                p95 = np.percentile(values, 95)

                quantiles_data[metric][dataset_id] = {
                    'p5': p5,
                    'p50': p50,
                    'p95': p95
                }

    return quantiles_data


def plot_boxplot_comparison():
    """
    Generate 3x3 multi-panel performance metrics figure.

    Layout:
    - 3 rows x 3 columns grid
    - Row 1: NYC Reservoir Storage metrics
    - Row 2: NYC Diversion/Demand Outcomes
    - Row 3: Montague Flow Outcomes
    - Within each subplot: grouped bars by quantile (5th, 50th, 95th)
    - Colors distinguish datasets
    """

    # Get datasets from config
    datasets_to_plot, dataset_labels = get_datasets_to_plot()

    # Load all metrics
    all_metrics_dfs = {}

    for dataset_id in datasets_to_plot:
        # Load pre-calculated metrics from CSV
        try:
            metrics_df = load_performance_metrics(dataset_id)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return None

        # Validate that requested metrics exist
        try:
            validate_metrics(metrics_df, dataset_id)
        except ValueError as e:
            print(str(e))
            return None

        all_metrics_dfs[dataset_id] = metrics_df

    # Load historic (reconstruction) metrics if requested
    historic_quantiles = None
    if SHOW_HISTORIC:
        try:
            historic_metrics_df = load_performance_metrics('reconstruction')
            historic_quantiles = {}
            for metric in METRICS_TO_PLOT:
                if metric in historic_metrics_df.columns:
                    raw_value = historic_metrics_df[metric].iloc[0]
                    if metric in METRICS_TO_SCALE:
                        scaled_value = raw_value * RECONSTRUCTION_SCALE_FACTOR
                    else:
                        scaled_value = raw_value
                    historic_quantiles[metric] = {
                        'p5': scaled_value,
                        'p50': scaled_value,
                        'p95': scaled_value
                    }
        except FileNotFoundError:
            historic_quantiles = None

    # Calculate quantiles for each metric and dataset
    quantiles_data = calculate_quantiles_by_dataset(
        all_metrics_dfs, datasets_to_plot, METRICS_TO_PLOT
    )

    # Define the 3x3 grid layout with categories
    category_metrics = [
        ('NYC Reservoir Storage', METRICS_NYC_STORAGE),
        ('NYC Diversion Outcomes', METRICS_NYC_DIVERSION),
        ('Montague Flow Outcomes', METRICS_MONTAGUE),
    ]

    n_rows = 3
    n_cols = 3

    # Create figure with extra space on left for row labels
    fig_width = 14
    fig_height = 11
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    # Set up color palette for datasets
    dataset_colors = {}
    for dataset_id in datasets_to_plot:
        if dataset_id in DATASET_COLORS:
            dataset_colors[dataset_id] = DATASET_COLORS[dataset_id]
        else:
            dataset_colors[dataset_id] = plt.cm.tab10(len(dataset_colors))

    # Plot settings
    quantile_labels = ['5th', '50th', '95th']
    quantile_keys = ['p5', 'p50', 'p95']
    n_quantiles = len(quantile_labels)
    n_datasets = len(datasets_to_plot)
    bar_width = 0.25

    # Plot each category row
    for row_idx, (category_name, metrics_list) in enumerate(category_metrics):
        for col_idx, metric in enumerate(metrics_list):
            ax = axes[row_idx, col_idx]

            x_positions = np.arange(n_quantiles)

            # Plot bars for each dataset
            for dataset_idx, dataset_id in enumerate(datasets_to_plot):
                dataset_label = dataset_labels[dataset_id]
                color = dataset_colors[dataset_id]

                # Get quantile values for this dataset and metric
                values = [
                    quantiles_data[metric][dataset_id][qkey]
                    for qkey in quantile_keys
                ]

                # Offset positions for grouped bars
                offset = (dataset_idx - (n_datasets - 1) / 2) * bar_width
                positions = x_positions + offset

                ax.bar(positions, values, bar_width,
                      label=dataset_label if (row_idx == 0 and col_idx == 0) else None,
                      color=color, alpha=0.8,
                      edgecolor='black', linewidth=0.5)

            # Add historic reconstruction points if available
            if historic_quantiles is not None and metric in historic_quantiles:
                historic_values = [
                    historic_quantiles[metric][qkey]
                    for qkey in quantile_keys
                ]
                ax.scatter(x_positions, historic_values,
                          **HISTORIC_MARKER_STYLE,
                          label='Historic' if (row_idx == 0 and col_idx == 0) else None)

            # Formatting
            metric_display_name = METRIC_DISPLAY_NAMES.get(metric, metric)
            ax.set_title(metric_display_name, fontsize=11, pad=8)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(quantile_labels, fontsize=9)

            # Only show x-axis label on bottom row
            if row_idx == n_rows - 1:
                ax.set_xlabel('Percentile', fontsize=10)
            else:
                ax.set_xlabel('')

            # Y-axis label
            ylabel = get_ylabel_for_metrics([metric])
            ax.set_ylabel(ylabel, fontsize=10)

            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            ax.tick_params(axis='both', labelsize=9)

        # Add category row label on the left side
        # Use the first axis in each row to add a label on the left
        ax_left = axes[row_idx, 0]
        ax_left.annotate(
            category_name,
            xy=(-0.35, 0.5),
            xycoords='axes fraction',
            fontsize=12,
            fontweight='bold',
            ha='center',
            va='center',
            rotation=90,
        )

    # Add legend to top-left subplot
    axes[0, 0].legend(loc='upper left', fontsize=8, frameon=True, fancybox=True)

    # Overall title
    fig.suptitle(
        'Water System Performance Metrics by Category',
        fontsize=14, fontweight='bold', y=0.98
    )

    plt.tight_layout(rect=[0.05, 0.02, 1, 0.96])

    # Save
    fname = f"{FIG_OUTPUT_DIR}/F9_performance_metrics_3x3.png"
    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")

    return fig, axes


def main():
    """Main entry point."""
    print("F9: Performance metrics boxplot")
    plot_boxplot_comparison()


if __name__ == "__main__":
    main()
