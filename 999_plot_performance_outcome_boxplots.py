"""
Plot performance metrics across ensembles with box plot distributions.

Shows:
- Left panel: Absolute performance distributions for all datasets (box plots)
- Right panel: Percentage change distributions relative to baseline (box plots)

This script uses pre-calculated metrics from postprocessing:
- shortage: Pre-calculated flow target violations
- mrf_target: Flow targets for calculating reliability
- res_storage: Reservoir storage for NYC system

Features:
- Box plots show full distribution of outcomes across realizations
- Flexible metric selection: Plot any list of performance metrics
- Dynamic dataset handling: Automatically adjusts based on config.py datasets
- Smart y-axis labeling: Detects metric types for appropriate labels

Usage:
  python 09_plot_performance_outcome_boxplots.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from methods.config import *
from methods.plotting.styles import (
    DATASET_COLORS,
    METRIC_DISPLAY_NAMES,
    METRICS_TO_SCALE,
    RECONSTRUCTION_SCALE_FACTOR,
    HISTORIC_MARKER_STYLE,
    get_ylabel_for_metrics,
    DPI_HIGH,
)
from methods.load import load_performance_metrics


# Output directory
FIG_OUTPUT_DIR = f"{FIG_DIR}/performance_metrics"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

# ============================================================================
# CONFIGURABLE METRICS
# ============================================================================
# Specify which metrics to plot and in what order
# The order of this list determines the order of boxes in the plot
#
# Available metrics (from methods/postprocess.py calculate_performance_metrics):
#
# See PERFORMANCE_METRICS_DOCUMENTATION.md for comprehensive descriptions.
#
# CATEGORY 1: Flow Reliability (Montague & Trenton)
#   - years_reliable_montague: Years Montague flow target met >90% of days
#   - years_reliable_montague_95: Years Montague flow target met >95% of days
#   - mean_annual_montague_reliability: Average annual Montague reliability (0-1)
#   - min_annual_montague_reliability: Worst annual Montague reliability
#   - total_montague_shortage_mg: Total Montague shortage (MG)
#   - mean_annual_montague_shortage_mg: Mean annual Montague shortage (MG/year)
#   - years_reliable_trenton: Years Trenton flow target met >90% of days
#   - years_reliable_trenton_95: Years Trenton flow target met >95% of days
#   - mean_annual_trenton_reliability: Average annual Trenton reliability (0-1)
#   - total_trenton_shortage_mg: Total Trenton shortage (MG)
#   - mean_annual_trenton_shortage_mg: Mean annual Trenton shortage (MG/year)
#
# CATEGORY 2: NYC Reservoir Storage
#   - years_above_30pct: Years min storage stays >30%
#   - years_above_20pct: Years min storage stays >20%
#   - years_above_10pct: Years min storage stays >10%
#   - years_below_10pct: Years min storage drops ≤10%
#   - years_high_storage_june1: Years ≥95% storage on June 1
#   - years_high_storage_june1_90: Years ≥90% storage on June 1
#   - mean_june1_storage_pct: Average June 1 storage (%)
#   - mean_sept1_storage_pct: Average Sept 1 storage (%)
#   - years_low_carryover: Years <50% storage on Sept 1
#   - years_low_carryover_40: Years <40% storage on Sept 1
#   - mean_storage_pct: Long-term average storage (%)
#   - median_storage_pct: Median storage (%)
#   - min_storage_pct: Absolute minimum storage (%)
#   - max_storage_pct: Maximum storage (%)
#   - std_storage_pct: Storage standard deviation (%)
#   - pct_days_storage_below_30: % days storage <30%
#   - pct_days_storage_below_20: % days storage <20%
#   - mean_annual_storage_range: Average annual storage swing (%)
#
# CATEGORY 3: Water Supply Reliability
#   - pct_days_nyc_diversion_shortage: % days NYC diversion shortage
#   - total_nyc_diversion_shortage_mg: Total NYC diversion shortage (MG)
#   - mean_annual_nyc_diversion_shortage_mg: Mean annual NYC shortage (MG/year)
#   - max_daily_nyc_diversion_shortage_mg: Max daily NYC shortage (MGD)
#   - years_no_nyc_shortage: Years with zero NYC shortage
#   - years_minor_nyc_shortage: Years with ≤365 MG shortage
#
# CATEGORY 4: Drought Characteristics
#   - max_consecutive_drought_days: Longest Montague drought (days)
#   - mean_drought_duration_days: Average Montague drought duration (days)
#   - n_drought_events: Number of Montague drought events
#   - n_major_droughts: Number of ≥90-day droughts
#   - n_severe_droughts: Number of ≥180-day droughts
#   - worst_drought_max_daily_shortage_mg: Peak shortage in worst drought (MGD)
#   - max_consecutive_drought_days_trenton: Longest Trenton drought (days)
#   - n_drought_events_trenton: Number of Trenton drought events
#   - pct_days_combined_stress: % days with both NYC & Montague shortage
#
# CATEGORY 5: NYC Contributions
#   - mean_annual_nyc_contribution_mg: Mean annual NYC contribution (MG/year)
#   - max_annual_nyc_contribution_mg: Max annual NYC contribution (MG/year)
#   - min_annual_nyc_contribution_mg: Min annual NYC contribution (MG/year)
#   - std_annual_nyc_contribution_mg: Std dev annual NYC contribution (MG/year)
#   - total_nyc_contribution_mg: Total NYC contribution (MG)
#   - pct_days_nyc_contribution: % days with NYC contribution
#   - n_days_high_nyc_contribution: Days with >100 MGD contribution
#
# CATEGORY 6: System Balance
#   - nyc_contribution_to_shortage_ratio: NYC contribution / Montague shortage
#   - years_high_storage_and_reliable: Years with high storage AND reliability
#   - years_vulnerable: Years with low storage OR low reliability
#
# LEGACY (backward compatibility):
#   - years_reliable: Alias for years_reliable_montague
#   - years_high_storage: Alias for years_high_storage_june1
#   - years_trenton_reliable: Alias for years_reliable_trenton

METRICS_TO_PLOT = [
    'years_drought_emergency',
    'max_1day_montague_shortage_mg',
    # 'years_below_30pct',
    # 'mean_annual_montague_shortage_mg',
    # 'years_reliable',
    # 'years_high_storage',
    # 'years_above_20pct',
    # 'years_above_10pct',
    # 'years_low_carryover',
    # 'years_trenton_reliable',
    'max_consecutive_drought_days',
    # 'mean_annual_nyc_contribution_mg',
    # 'pct_days_nyc_contribution',
]

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
    """
    Get list of datasets from config.py.

    Returns
    -------
    datasets_to_plot : list
        List of dataset IDs to plot
    dataset_labels : dict
        Display labels for each dataset
    """
    # Start with all datasets from config
    all_datasets = list(DATASET_CONFIGS.keys())

    # Determine which datasets to plot
    if DATASETS_TO_PLOT is None:
        # Use all datasets from config
        datasets_to_plot = all_datasets
    else:
        # Use manually specified datasets
        datasets_to_plot = DATASETS_TO_PLOT
        # Verify they exist
        for d in datasets_to_plot:
            if d not in all_datasets:
                raise ValueError(f"Dataset '{d}' not found in config.py!")

    # Create display labels from descriptions
    dataset_labels = {}
    for dataset_id in datasets_to_plot:
        config = DATASET_CONFIGS[dataset_id]
        desc = config.get('description', dataset_id)
        dataset_labels[dataset_id] = desc

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
    Generate multi-panel performance metrics figure.

    Layout:
    - One subplot per metric
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
                    # For reconstruction, we only have one realization (single value)
                    # Use this value for all quantiles since there's no distribution
                    raw_value = historic_metrics_df[metric].iloc[0]

                    # Scale metrics that count years to make them comparable
                    if metric in METRICS_TO_SCALE:
                        scaled_value = raw_value * RECONSTRUCTION_SCALE_FACTOR
                    else:
                        scaled_value = raw_value

                    # Use same value for all quantiles (no distribution)
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

    # Determine subplot layout
    n_metrics = len(METRICS_TO_PLOT)
    n_cols = min(3, n_metrics)  # Max 3 columns
    n_rows = int(np.ceil(n_metrics / n_cols))

    # Create figure with subplots
    fig_width = 6 * n_cols
    fig_height = 5 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    # Flatten axes array for easier iteration
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_metrics > 1 else [axes]

    # Set up color palette for datasets
    dataset_colors = {}
    for dataset_id in datasets_to_plot:
        if dataset_id in DATASET_COLORS:
            dataset_colors[dataset_id] = DATASET_COLORS[dataset_id]
        else:
            # Default colors if not specified
            dataset_colors[dataset_id] = plt.cm.tab10(len(dataset_colors))

    # Plot each metric in its own subplot
    quantile_labels = ['5th', '50th', '95th']
    quantile_keys = ['p5', 'p50', 'p95']
    n_quantiles = len(quantile_labels)
    n_datasets = len(datasets_to_plot)

    for idx, metric in enumerate(METRICS_TO_PLOT):
        ax = axes[idx]

        # Prepare data for this metric
        bar_width = 0.25
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
                  label=dataset_label, color=color, alpha=0.8,
                  edgecolor='black', linewidth=0.5)

        # Add historic reconstruction points if available
        if historic_quantiles is not None and metric in historic_quantiles:
            historic_values = [
                historic_quantiles[metric][qkey]
                for qkey in quantile_keys
            ]
            ax.scatter(x_positions, historic_values,
                      **HISTORIC_MARKER_STYLE,
                      label='Historic' if idx == 0 else None)

        # Formatting
        metric_display_name = METRIC_DISPLAY_NAMES.get(metric, metric)
        ax.set_title(metric_display_name, fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(quantile_labels, fontsize=10)
        ax.set_xlabel('Percentile', fontsize=11, fontweight='bold')

        # Y-axis label
        ylabel = get_ylabel_for_metrics([metric])
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')

        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Add legend to first subplot only
        if idx == 0:
            ax.legend(loc='upper left', fontsize=9, frameon=True, fancybox=True)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')

    # Overall title
    fig.suptitle(
        'Water System Performance Metrics - Quantile Comparison',
        fontsize=16, fontweight='bold', y=0.995
    )

    plt.tight_layout()

    # Save
    fname = f"{FIG_OUTPUT_DIR}/performance_metrics_boxplot_comparison.png"
    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"\nSaved: {fname}")

    return fig, axes


def main():
    """Main entry point."""
    plot_boxplot_comparison()


if __name__ == "__main__":
    main()
