"""
Plot performance metrics across ensembles with flexible comparison layout.

Shows:
- Left panel: Absolute performance for baseline dataset (with uncertainty bars)
- Right panels (stacked): Percentage change for each comparison dataset

This script uses pre-calculated metrics from postprocessing:
- shortage: Pre-calculated flow target violations
- mrf_target: Flow targets for calculating reliability
- res_storage: Reservoir storage for NYC system

Features:
- Flexible metric selection: Plot any list of performance metrics
- Dynamic dataset handling: Automatically adjusts panels based on config.py datasets
- Smart y-axis labeling: Detects metric types for appropriate labels

Usage:
  python 09_plot_performance_outcome_bars.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings("ignore")

from methods.config import *


# Output directory
FIG_OUTPUT_DIR = f"{FIG_DIR}/performance_metrics"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

# Performance metrics directory
PERFORMANCE_METRICS_DIR = f"{ROOT_DIR}/pywrdrb/performance_metrics"

# ============================================================================
# CONFIGURABLE METRICS
# ============================================================================
# Specify which metrics to plot and in what order
# The order of this list determines the order of bars in the plot
#
# Available metrics (from methods/postprocess.py calculate_performance_metrics):
#   - years_reliable: Years where Montague flow target met >90% of time
#   - years_high_storage: Years where NYC storage ≥95% on June 1
#   - years_above_20pct: Years where minimum NYC storage stays >20%
#   - years_above_10pct: Years where minimum NYC storage stays >10%
#   - mean_sept1_storage_pct: Average NYC storage on Sept 1 (%)
#   - years_low_carryover: Years with <50% storage on Sept 1
#   - years_trenton_reliable: Years where Trenton flow target met >90% of time
#   - pct_days_nyc_diversion_shortage: % of days NYC fails to meet diversion demand
#   - max_consecutive_drought_days: Longest drought period (days)
#   - mean_annual_nyc_contribution_mg: Average annual NYC downstream release (MG)
#   - max_annual_nyc_contribution_mg: Maximum annual NYC downstream release (MG)



# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
# Specify which dataset to use as baseline (left panel, absolute values)
# All other datasets from config.py will be shown as percentage changes
BASELINE_DATASET = 'stationary_ensemble'

# Option to manually specify which datasets to compare (in addition to baseline)
# Set to None to automatically use all datasets from config.py
# Set to list of dataset_ids to manually specify (e.g., ['climate_adjusted_low', 'climate_adjusted_high'])
COMPARISON_DATASETS = None  # None = use all datasets except baseline

# Metric display names (for plot labels)
METRIC_DISPLAY_NAMES = {
    'years_reliable': 'Years Montague\nReliable',
    'years_high_storage': 'Years NYC\nStorage High\non June 1',
    'years_above_20pct': 'Years Min\nStorage >20%',
    'years_above_10pct': 'Years Min\nStorage >10%',
    'mean_sept1_storage_pct': 'Mean Sept 1\nStorage (%)',
    'years_low_carryover': 'Years Low\nCarryover',
    'years_trenton_reliable': 'Years Trenton\nReliable',
    'pct_days_nyc_diversion_shortage': '% Days NYC\nDiversion Short',
    'max_consecutive_drought_days': 'Max Consecutive\nDrought (days)',
    'mean_annual_nyc_contribution_mg': 'Mean Annual NYC\nContribution (MG)',
    'max_annual_nyc_contribution_mg': 'Max Annual NYC\nContribution (MG)',
}

METRICS_TO_PLOT = [
    'years_reliable',
    'years_high_storage',
    'years_above_20pct',
    'years_above_10pct',
    # 'mean_sept1_storage_pct',
    'years_low_carryover',
    'years_trenton_reliable',
    # 'pct_days_nyc_diversion_shortage',
    'max_consecutive_drought_days',
    # 'mean_annual_nyc_contribution_mg',
    # 'max_annual_nyc_contribution_mg',
]

# Color palettes for bars
COLORS_ABSOLUTE = ['#A23B72', '#F18F01', '#2E86AB']  # Purple, Orange, Blue
COLORS_CHANGE = ['#D4399B', '#C73E1D', '#06A77D']  # Magenta, Red, Teal

# Reconstruction scaling factor
# Reconstruction has 79 years, ensembles have 70 years
# Scale reconstruction metrics to be comparable
RECONSTRUCTION_YEARS = 79
ENSEMBLE_YEARS = 70
RECONSTRUCTION_SCALE_FACTOR = ENSEMBLE_YEARS / RECONSTRUCTION_YEARS  # 70/79 ≈ 0.886

# Metrics that should be scaled (count of years)
# These metrics count years, so need to be scaled for comparison
METRICS_TO_SCALE = [
    'years_reliable',
    'years_high_storage',
    'years_above_20pct',
    'years_above_10pct',
    'years_low_carryover',
    'years_trenton_reliable',
]

# ============================================================================
# METRIC METADATA - For smart y-axis labeling
# ============================================================================
# Categorize metrics by their units/types
METRIC_UNITS = {
    # Year count metrics
    'years_reliable': 'years',
    'years_high_storage': 'years',
    'years_above_20pct': 'years',
    'years_above_10pct': 'years',
    'years_low_carryover': 'years',
    'years_trenton_reliable': 'years',

    # Percentage metrics
    'mean_sept1_storage_pct': 'percent',
    'pct_days_nyc_diversion_shortage': 'percent',

    # Duration metrics (days)
    'max_consecutive_drought_days': 'days',

    # Volume metrics (million gallons)
    'mean_annual_nyc_contribution_mg': 'million_gallons',
    'max_annual_nyc_contribution_mg': 'million_gallons',
}

# Y-axis labels for different metric types
Y_AXIS_LABELS = {
    'years': 'Number of Years (out of 70)',
    'percent': 'Percentage (%)',
    'days': 'Days',
    'million_gallons': 'Million Gallons (MG)',
}

def get_ylabel_for_metrics(metric_list):
    """
    Determine appropriate y-axis label for a list of metrics.
    If all metrics have same units, return that unit's label.
    Otherwise, return a generic label.
    """
    units = set(METRIC_UNITS.get(m, 'value') for m in metric_list)

    if len(units) == 1:
        unit = units.pop()
        return Y_AXIS_LABELS.get(unit, 'Value')
    else:
        return 'Value'


def load_performance_metrics(dataset_id):
    """
    Load pre-calculated performance metrics from CSV.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier

    Returns
    -------
    metrics_df : pd.DataFrame
        DataFrame with performance metrics for all realizations
    """
    csv_file = f"{PERFORMANCE_METRICS_DIR}/{dataset_id}_performance_metrics.csv"

    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"Performance metrics not found: {csv_file}\n"
            f"Run 04_postprocess_data.py first to calculate metrics!"
        )

    metrics_df = pd.read_csv(csv_file, index_col='realization_id')
    return metrics_df


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


def calculate_ensemble_percentiles(metrics_df, metrics_list):
    """
    Calculate p1, p50, p99 for each metric across realizations.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with performance metrics columns
    metrics_list : list
        List of metric column names to calculate percentiles for

    Returns
    -------
    percentiles : dict
        {'metric_name': [p1, p50, p99], ...}
    """
    percentiles = {}
    for metric in metrics_list:
        p1 = metrics_df[metric].quantile(0.01)
        p50 = metrics_df[metric].quantile(0.50)
        p99 = metrics_df[metric].quantile(0.99)
        percentiles[metric] = [p1, p50, p99]

    return percentiles


def get_datasets_from_config():
    """
    Get list of datasets from config.py and separate baseline from comparison datasets.

    Returns
    -------
    baseline_dataset : str
        Dataset to use as baseline (absolute values in left panel)
    comparison_datasets : list
        Datasets to compare (percentage change in right panels)
    dataset_labels : dict
        Display labels for each dataset
    """
    from methods.config import DATASET_CONFIGS

    # Start with all datasets from config
    all_datasets = list(DATASET_CONFIGS.keys())

    # Verify baseline exists
    if BASELINE_DATASET not in all_datasets:
        raise ValueError(
            f"Baseline dataset '{BASELINE_DATASET}' not found in config.py!\n"
            f"Available datasets: {all_datasets}"
        )

    # Determine comparison datasets
    if COMPARISON_DATASETS is None:
        # Use all datasets except baseline
        comparison_datasets = [d for d in all_datasets if d != BASELINE_DATASET]
    else:
        # Use manually specified comparison datasets
        comparison_datasets = COMPARISON_DATASETS
        # Verify they exist
        for d in comparison_datasets:
            if d not in all_datasets:
                raise ValueError(f"Comparison dataset '{d}' not found in config.py!")

    # Create display labels from descriptions
    dataset_labels = {}
    for dataset_id in [BASELINE_DATASET] + comparison_datasets:
        config = DATASET_CONFIGS[dataset_id]
        # Use first 3 words of description or full description if short
        desc = config.get('description', dataset_id)
        dataset_labels[dataset_id] = desc

    return BASELINE_DATASET, comparison_datasets, dataset_labels


def plot_performance_comparison():
    """
    Generate flexible multi-panel performance metrics figure.

    Layout dynamically adjusts based on number of datasets:
    - Left panel: Baseline dataset absolute values (with uncertainty bars)
    - Right panels (stacked): Percentage change for each comparison dataset
    """

    print("=" * 80)
    print("CALCULATING PERFORMANCE METRICS")
    print("=" * 80)
    print(f"Metrics to plot ({len(METRICS_TO_PLOT)}): {METRICS_TO_PLOT}")
    print("=" * 80)

    # Get datasets from config
    baseline_dataset, comparison_datasets, dataset_labels = get_datasets_from_config()

    print(f"\nBaseline dataset: {baseline_dataset}")
    print(f"Comparison datasets ({len(comparison_datasets)}): {comparison_datasets}")
    print("=" * 80)

    # Combine all datasets to load
    all_datasets = [baseline_dataset] + comparison_datasets

    all_percentiles = {}

    for dataset_id in all_datasets:
        label = dataset_labels[dataset_id]
        print(f"\nLoading {dataset_id} ({label})...")

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

        # Calculate percentiles for requested metrics only
        percentiles = calculate_ensemble_percentiles(metrics_df, METRICS_TO_PLOT)
        all_percentiles[dataset_id] = percentiles

        # Print summary for requested metrics
        for metric in METRICS_TO_PLOT:
            p1, p50, p99 = percentiles[metric]
            print(f"  {metric:40s}: p1={p1:6.1f}, p50={p50:6.1f}, p99={p99:6.1f}")

    # Load historic (reconstruction) metrics for comparison
    print(f"\nLoading historic (reconstruction) metrics...")
    print(f"  Note: Reconstruction has {RECONSTRUCTION_YEARS} years, ensembles have {ENSEMBLE_YEARS} years")
    print(f"  Scaling reconstruction year-count metrics by {RECONSTRUCTION_SCALE_FACTOR:.3f}")
    try:
        historic_metrics_df = load_performance_metrics('reconstruction')
        historic_values = {}
        for metric in METRICS_TO_PLOT:
            if metric in historic_metrics_df.columns:
                raw_value = historic_metrics_df[metric].iloc[0]

                # Scale metrics that count years to make them comparable
                if metric in METRICS_TO_SCALE:
                    scaled_value = raw_value * RECONSTRUCTION_SCALE_FACTOR
                    historic_values[metric] = scaled_value
                    print(f"  Historic {metric}: {raw_value:.1f} → {scaled_value:.1f} (scaled)")
                else:
                    historic_values[metric] = raw_value
                    print(f"  Historic {metric}: {raw_value:.1f}")
            else:
                print(f"  WARNING: Historic metric '{metric}' not found")

        if not historic_values:
            historic_values = None
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("Historic values will not be shown on plot.")
        historic_values = None

    # Calculate percentage changes from baseline
    print(f"\n{'='*80}")
    print(f"Calculating percentage changes from baseline ({baseline_dataset})...")
    print(f"{'='*80}")

    baseline_perc = all_percentiles[baseline_dataset]
    pct_changes = {}

    for dataset_id in comparison_datasets:
        pct_change = {}
        for metric in METRICS_TO_PLOT:
            # Calculate % change at each percentile
            eps = 1e-8
            pct_change[metric] = [
                100.0 * (all_percentiles[dataset_id][metric][i] - baseline_perc[metric][i]) /
                max(abs(baseline_perc[metric][i]), eps)
                for i in range(3)
            ]
        pct_changes[dataset_id] = pct_change

    # Create figure with dynamic layout
    n_comparison_datasets = len(comparison_datasets)
    print(f"\n{'='*60}")
    print(f"Creating {n_comparison_datasets + 1}-panel figure...")
    print(f"{'='*60}")

    # Dynamic figure sizing based on number of comparison datasets
    fig_height = max(8, 3 * n_comparison_datasets + 2)
    fig = plt.figure(figsize=(14, fig_height))

    # Create grid: left column for baseline, right column for comparisons (stacked)
    gs = fig.add_gridspec(
        n_comparison_datasets, 2,
        height_ratios=[1] * n_comparison_datasets,
        width_ratios=[1, 1],
        hspace=0.30, wspace=0.30,
        left=0.10, right=0.95, top=0.93, bottom=0.08
    )

    # Create axes
    # Left panel spans all rows (baseline)
    ax_baseline = fig.add_subplot(gs[:, 0])

    # Right panels (one for each comparison dataset)
    axes_comparison = [fig.add_subplot(gs[i, 1]) for i in range(n_comparison_datasets)]

    # Combine into list for iteration
    axes = [ax_baseline] + axes_comparison
    dataset_list = [baseline_dataset] + comparison_datasets

    # Generate panel labels dynamically
    panel_labels = [f'({chr(97 + i)})' for i in range(len(axes))]  # (a), (b), (c), ...

    # Use configured metrics
    metric_keys = METRICS_TO_PLOT
    metric_names = [METRIC_DISPLAY_NAMES.get(m, m) for m in metric_keys]
    n_metrics = len(metric_keys)

    # Generate colors dynamically based on number of metrics
    # Cycle through color palettes if more than 3 metrics
    colors_abs = (COLORS_ABSOLUTE * ((n_metrics // len(COLORS_ABSOLUTE)) + 1))[:n_metrics]
    colors_diff = (COLORS_CHANGE * ((n_metrics // len(COLORS_CHANGE)) + 1))[:n_metrics]

    # Calculate y-axis range for right panels (shared across all comparison datasets)
    all_pct_values = []
    for dataset_id in comparison_datasets:
        for metric in metric_keys:
            all_pct_values.append(pct_changes[dataset_id][metric][1])  # p50 values

    ymin_shared = min(all_pct_values) * 1.15  # 15% padding
    ymax_shared = max(all_pct_values) * 1.15
    # Make symmetric around 0 if it crosses zero
    if ymin_shared < 0 and ymax_shared > 0:
        y_abs_max = max(abs(ymin_shared), abs(ymax_shared))
        ymin_shared = -y_abs_max
        ymax_shared = y_abs_max

    # Determine appropriate y-axis label for baseline panel
    ylabel_baseline = get_ylabel_for_metrics(metric_keys)

    # Plot each panel
    for idx, (ax, dataset_id, panel_label) in enumerate(zip(axes, dataset_list, panel_labels)):

        if idx == 0:  # Baseline panel (absolute values)
            # Plot bars for each metric
            x_pos = np.arange(n_metrics)
            p50_values = [baseline_perc[m][1] for m in metric_keys]
            p1_values = [baseline_perc[m][0] for m in metric_keys]
            p99_values = [baseline_perc[m][2] for m in metric_keys]

            # Error bars (p1 to p99 range)
            yerr_low = [p50_values[i] - p1_values[i] for i in range(n_metrics)]
            yerr_high = [p99_values[i] - p50_values[i] for i in range(n_metrics)]

            bars = ax.bar(x_pos, p50_values, color=colors_abs, alpha=0.8,
                         yerr=[yerr_low, yerr_high], capsize=5,
                         error_kw={'linewidth': 2, 'ecolor': 'black', 'alpha': 0.6})

            # Add historic values as scatter points (if available)
            if historic_values is not None:
                historic_scatter_values = [historic_values.get(m, np.nan) for m in metric_keys]
                ax.scatter(x_pos, historic_scatter_values, color='red', s=100,
                          marker='D', edgecolors='darkred', linewidths=2, zorder=10,
                          label='Historic')

            ax.set_xticks(x_pos)
            ax.set_xticklabels(metric_names, fontsize=10)
            ax.set_ylabel(ylabel_baseline, fontsize=12, fontweight='bold')
            ax.set_ylim(bottom=0)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)

            # Add legend for historic points
            if historic_values is not None:
                ax.legend(loc='upper left', fontsize=9, frameon=True, fancybox=True)

        else:  # Comparison dataset panels (percentage change)
            pct_change = pct_changes[dataset_id]

            # Plot bars for each metric (median percentage change only)
            x_pos = np.arange(n_metrics)
            p50_values = [pct_change[m][1] for m in metric_keys]

            bars = ax.bar(x_pos, p50_values, color=colors_diff, alpha=0.8)

            ax.axhline(0, color='black', linewidth=1.5, linestyle='-', alpha=0.7)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(metric_names, fontsize=10)
            ax.set_ylabel('% Change', fontsize=12, fontweight='bold')
            ax.set_ylim(ymin_shared, ymax_shared)  # Shared y-axis range
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)

        # Panel title
        title_text = dataset_labels[dataset_id]
        ax.text(0.02, 0.98, f"{panel_label} {title_text}",
               transform=ax.transAxes, fontsize=13, fontweight='bold',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.3))

        ax.tick_params(labelsize=10)

    # Overall title
    fig.suptitle('Water System Performance Metrics', fontsize=16, fontweight='bold')

    # Add legend for error bars
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='_', color='black', linewidth=2, markersize=10,
               label='98% range (p1-p99)', linestyle='none'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
               markersize=10, label='Median (p50)', alpha=0.8)
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98),
              fontsize=10, frameon=True, fancybox=True, shadow=True)

    # Save with dynamic filename based on number of panels
    n_panels = len(axes)
    fname = f"{FIG_OUTPUT_DIR}/performance_metrics_comparison_{n_panels}panel.png"
    plt.savefig(fname, dpi=400, bbox_inches='tight')
    print(f"\nSaved: {fname}")

    # Also save as SVG
    fname_svg = fname.replace('.png', '.svg')
    plt.savefig(fname_svg, bbox_inches='tight')
    print(f"Saved: {fname_svg}")

    return fig, axes


def main():
    """Main entry point."""
    plot_performance_comparison()

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
