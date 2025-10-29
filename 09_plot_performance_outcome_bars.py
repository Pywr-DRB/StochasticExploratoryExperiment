"""
Plot performance metrics across ensembles with 4-panel comparison layout.

Shows:
- Left panel: Absolute performance for stationary ensemble (with uncertainty bars)
- Right panels (stacked): Percentage change for each climate scenario

This script uses pre-calculated metrics from postprocessing:
- shortage: Pre-calculated flow target violations
- mrf_target: Flow targets for calculating reliability
- res_storage: Reservoir storage for NYC system

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

from config import *


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
# Available metrics (from 04_postprocess_data.py):
#   - years_reliable: Years where Montague flow target met >90% of time
#   - years_high_storage: Years where NYC storage >90% on June 1
#   - years_above_20pct: Years where minimum NYC storage stays >20%
#   - years_above_10pct: Years where minimum NYC storage stays >10%
#   - mean_sept1_storage_pct: Average NYC storage on Sept 1 (%)
#   - years_low_carryover: Years with <50% storage on Sept 1
#   - years_trenton_reliable: Years where Trenton flow target met >90% of time
#   - pct_days_nyc_diversion_shortage: % of days NYC fails to meet diversion demand
#   - max_consecutive_drought_days: Longest drought period (days)
#   - mean_annual_nyc_contribution_mg: Average annual NYC downstream release (MG)
#   - max_annual_nyc_contribution_mg: Maximum annual NYC downstream release (MG)

METRICS_TO_PLOT = [
    'years_high_storage',     # Years NYC storage high on June 1
    'years_above_20pct',      # Years min storage stays >20%
    'years_reliable',         # Years Montague reliable
]

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
    Calculate p5, p50, p95 for each metric across realizations.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with performance metrics columns
    metrics_list : list
        List of metric column names to calculate percentiles for

    Returns
    -------
    percentiles : dict
        {'metric_name': [p5, p50, p95], ...}
    """
    percentiles = {}
    for metric in metrics_list:
        p5 = metrics_df[metric].quantile(0.05)
        p50 = metrics_df[metric].quantile(0.50)
        p95 = metrics_df[metric].quantile(0.95)
        percentiles[metric] = [p5, p50, p95]

    return percentiles


def plot_4panel_performance_comparison():
    """
    Generate complete 4-panel performance metrics figure.

    Layout:
    - Left panel: Stationary ensemble absolute values
    - Right panels (stacked): Low, Medium, High climate scenarios (% change)
    """

    print("=" * 80)
    print("CALCULATING PERFORMANCE METRICS")
    print("=" * 80)
    print(f"Metrics to plot ({len(METRICS_TO_PLOT)}): {METRICS_TO_PLOT}")
    print("=" * 80)

    # Load pre-calculated data for all datasets
    datasets = {
        'stationary_ensemble': 'Stationary',
        'climate_adjusted_low': 'Low',
        'climate_adjusted_medium': 'Medium',
        'climate_adjusted_high': 'High'
    }

    all_percentiles = {}

    for dataset_id, label in datasets.items():
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
            p5, p50, p95 = percentiles[metric]
            print(f"  {metric:40s}: p5={p5:6.1f}, p50={p50:6.1f}, p95={p95:6.1f}")

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

    # Calculate percentage changes from stationary
    print(f"\n{'='*80}")
    print("Calculating percentage changes from stationary...")
    print(f"{'='*80}")

    stat_perc = all_percentiles['stationary_ensemble']
    pct_changes = {}

    for dataset_id in ['climate_adjusted_low', 'climate_adjusted_medium', 'climate_adjusted_high']:
        pct_change = {}
        for metric in METRICS_TO_PLOT:
            # Calculate % change at each percentile
            eps = 1e-8
            pct_change[metric] = [
                100.0 * (all_percentiles[dataset_id][metric][i] - stat_perc[metric][i]) /
                max(abs(stat_perc[metric][i]), eps)
                for i in range(3)
            ]
        pct_changes[dataset_id] = pct_change

    # Create figure
    print(f"\n{'='*60}")
    print("Creating 4-panel figure...")
    print(f"{'='*60}")

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1],
                          hspace=0.30, wspace=0.30,
                          left=0.10, right=0.95, top=0.93, bottom=0.08)

    # Create axes
    ax_stat = fig.add_subplot(gs[:, 0])  # Left panel spans all rows
    ax_low = fig.add_subplot(gs[0, 1])   # Top right
    ax_med = fig.add_subplot(gs[1, 1])   # Middle right
    ax_high = fig.add_subplot(gs[2, 1])  # Bottom right

    axes = [ax_stat, ax_low, ax_med, ax_high]
    dataset_list = list(datasets.keys())
    panel_labels = ['(a)', '(b)', '(c)', '(d)']

    # Use configured metrics
    metric_keys = METRICS_TO_PLOT
    metric_names = [METRIC_DISPLAY_NAMES.get(m, m) for m in metric_keys]
    n_metrics = len(metric_keys)

    # Generate colors dynamically based on number of metrics
    # Cycle through color palettes if more than 3 metrics
    colors_abs = (COLORS_ABSOLUTE * ((n_metrics // len(COLORS_ABSOLUTE)) + 1))[:n_metrics]
    colors_diff = (COLORS_CHANGE * ((n_metrics // len(COLORS_CHANGE)) + 1))[:n_metrics]

    # Calculate y-axis range for right panels (shared across all 3 climate scenarios)
    all_pct_values = []
    for dataset_id in ['climate_adjusted_low', 'climate_adjusted_medium', 'climate_adjusted_high']:
        for metric in metric_keys:
            all_pct_values.append(pct_changes[dataset_id][metric][1])  # p50 values

    ymin_shared = min(all_pct_values) * 1.15  # 15% padding
    ymax_shared = max(all_pct_values) * 1.15
    # Make symmetric around 0 if it crosses zero
    if ymin_shared < 0 and ymax_shared > 0:
        y_abs_max = max(abs(ymin_shared), abs(ymax_shared))
        ymin_shared = -y_abs_max
        ymax_shared = y_abs_max

    # Plot each panel
    for idx, (ax, dataset_id, panel_label) in enumerate(zip(axes, dataset_list, panel_labels)):

        if idx == 0:  # Stationary panel (absolute values)
            # Plot bars for each metric
            x_pos = np.arange(n_metrics)
            p50_values = [stat_perc[m][1] for m in metric_keys]
            p5_values = [stat_perc[m][0] for m in metric_keys]
            p95_values = [stat_perc[m][2] for m in metric_keys]

            # Error bars (p5 to p95 range)
            yerr_low = [p50_values[i] - p5_values[i] for i in range(n_metrics)]
            yerr_high = [p95_values[i] - p50_values[i] for i in range(n_metrics)]

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
            ax.set_ylabel('Number of Years (out of 70)', fontsize=12, fontweight='bold')
            ax.set_ylim(bottom=0)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)

            # Add legend for historic points
            if historic_values is not None:
                ax.legend(loc='upper left', fontsize=9, frameon=True, fancybox=True)

        else:  # Climate scenario panels (percentage change)
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
        title_text = datasets[dataset_id]
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
               label='90% range (p5-p95)', linestyle='none'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
               markersize=10, label='Median (p50)', alpha=0.8)
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98),
              fontsize=10, frameon=True, fancybox=True, shadow=True)

    # Save
    fname = f"{FIG_OUTPUT_DIR}/performance_metrics_comparison_4panel.png"
    plt.savefig(fname, dpi=400, bbox_inches='tight')
    print(f"\nSaved: {fname}")

    # Also save as SVG
    fname_svg = fname.replace('.png', '.svg')
    plt.savefig(fname_svg, bbox_inches='tight')
    print(f"Saved: {fname_svg}")

    return fig, axes


def main():
    """Main entry point."""
    plot_4panel_performance_comparison()

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
