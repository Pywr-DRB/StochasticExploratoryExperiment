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

import pywrdrb
from config import *


# Output directory
FIG_OUTPUT_DIR = f"{FIG_DIR}/performance_metrics"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

# Storage capacities for NYC reservoirs (MG)
NYC_STORAGE_CAPACITIES = {
    'cannonsville': 95706,
    'pepacton': 140190,
    'neversink': 34941
}
NYC_TOTAL_CAPACITY = sum(NYC_STORAGE_CAPACITIES.values())


def calculate_performance_metrics(data, model, realizations):
    """
    Calculate three performance metrics for each realization using pre-calculated data.

    Parameters
    ----------
    data : pywrdrb.Data
        Data object with pre-calculated shortage, mrf_target, and res_storage
    model : str
        Model/dataset identifier
    realizations : list
        List of realization IDs

    Returns
    -------
    metrics_dict : dict
        {realization_id: {'metric1': value, 'metric2': value, 'metric3': value}}
    """
    metrics = {}
    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']

    for r in realizations:
        if (r % 100 == 0) and (r > 0):
            print(f"    Processed {r}/{len(realizations)} realizations...")

        # Use pre-calculated shortage and target data
        montague_shortage = data.shortage[model][r]['delMontague']
        montague_target = data.mrf_target[model][r]['delMontague']

        # Metric 1: # years where Montague flow target met >90% of time
        annual_shortage = montague_shortage.resample('YS').sum()
        annual_target = montague_target.resample('YS').sum()
        annual_reliability = 1 - (annual_shortage / annual_target)
        annual_reliability = annual_reliability.clip(0, 1)
        n_years_reliable = (annual_reliability > 0.90).sum()

        # Metric 2: # years where NYC storage >90% on June 1
        nyc_storage = data.res_storage[model][r][nyc_reservoirs].sum(axis=1)
        nyc_storage_pct = 100.0 * nyc_storage / NYC_TOTAL_CAPACITY

        # Filter for June 1 dates
        june1_storage = nyc_storage_pct[(nyc_storage_pct.index.month == 6) &
                                        (nyc_storage_pct.index.day == 1)]
        n_years_high_storage = (june1_storage > 90).sum()

        # Metric 3: Maximum daily shortage magnitude (MGD)
        max_shortage = montague_shortage.max()

        metrics[r] = {
            'years_reliable': n_years_reliable,
            'years_high_storage': n_years_high_storage,
            'max_shortage': max_shortage
        }

    return metrics


def calculate_ensemble_percentiles(metrics_dict):
    """
    Calculate p5, p50, p95 for each metric across realizations.

    Returns
    -------
    percentiles : dict
        {'years_reliable': [p5, p50, p95], ...}
    """
    df = pd.DataFrame(metrics_dict).T

    percentiles = {}
    for metric in ['years_reliable', 'years_high_storage', 'max_shortage']:
        p5 = df[metric].quantile(0.05)
        p50 = df[metric].quantile(0.50)
        p95 = df[metric].quantile(0.95)
        percentiles[metric] = [p5, p50, p95]

    return percentiles


def plot_4panel_performance_comparison():
    """
    Generate complete 4-panel performance metrics figure.

    Layout:
    - Left panel: Stationary ensemble absolute values
    - Right panels (stacked): Low, Medium, High climate scenarios (% change)
    """

    print("=" * 60)
    print("CALCULATING PERFORMANCE METRICS")
    print("=" * 60)

    # Load pre-calculated data for all datasets
    datasets = {
        'stationary_ensemble': 'Stationary',
        'climate_adjusted_low': 'Low',
        'climate_adjusted_medium': 'Medium',
        'climate_adjusted_high': 'High'
    }

    all_percentiles = {}

    for dataset_id, label in datasets.items():
        fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'

        if not os.path.exists(fname):
            print(f"ERROR: Postprocessed data not found: {fname}")
            print(f"Run postprocessing first for {dataset_id}!")
            return None

        print(f"\nLoading {dataset_id} ({label})...")
        data = pywrdrb.Data()
        data.load_from_export(fname, results_sets=['shortage', 'res_storage', 'mrf_target'])

        # Calculate metrics
        print(f"  Calculating metrics...")
        realizations = list(data.shortage[dataset_id].keys())
        metrics_dict = calculate_performance_metrics(data, dataset_id, realizations)

        # Calculate percentiles
        percentiles = calculate_ensemble_percentiles(metrics_dict)
        all_percentiles[dataset_id] = percentiles

        print(f"  Years Montague reliable: p5={percentiles['years_reliable'][0]:.1f}, "
              f"p50={percentiles['years_reliable'][1]:.1f}, p95={percentiles['years_reliable'][2]:.1f}")
        print(f"  Years NYC storage high: p5={percentiles['years_high_storage'][0]:.1f}, "
              f"p50={percentiles['years_high_storage'][1]:.1f}, p95={percentiles['years_high_storage'][2]:.1f}")
        print(f"  Max shortage (MGD): p5={percentiles['max_shortage'][0]:.0f}, "
              f"p50={percentiles['max_shortage'][1]:.0f}, p95={percentiles['max_shortage'][2]:.0f}")

    # Calculate percentage changes from stationary
    print(f"\n{'='*60}")
    print("Calculating percentage changes from stationary...")
    print(f"{'='*60}")

    stat_perc = all_percentiles['stationary_ensemble']
    pct_changes = {}

    for dataset_id in ['climate_adjusted_low', 'climate_adjusted_medium', 'climate_adjusted_high']:
        pct_change = {}
        for metric in ['years_reliable', 'years_high_storage', 'max_shortage']:
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

    # Metric info
    metric_names = ['Years Montague\nReliable', 'Years NYC\nStorage High', 'Max Shortage\n(MGD)']
    metric_keys = ['years_reliable', 'years_high_storage', 'max_shortage']
    colors_abs = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
    colors_diff = ['#06A77D', '#D4399B', '#C73E1D']  # Teal, Magenta, Red

    # Plot each panel
    for idx, (ax, dataset_id, panel_label) in enumerate(zip(axes, dataset_list, panel_labels)):

        if idx == 0:  # Stationary panel (absolute values)
            # Plot bars for each metric
            x_pos = np.arange(3)
            p50_values = [stat_perc[m][1] for m in metric_keys]
            p5_values = [stat_perc[m][0] for m in metric_keys]
            p95_values = [stat_perc[m][2] for m in metric_keys]

            # Error bars (p5 to p95 range)
            yerr_low = [p50_values[i] - p5_values[i] for i in range(3)]
            yerr_high = [p95_values[i] - p50_values[i] for i in range(3)]

            bars = ax.bar(x_pos, p50_values, color=colors_abs, alpha=0.8,
                         yerr=[yerr_low, yerr_high], capsize=5,
                         error_kw={'linewidth': 2, 'ecolor': 'black', 'alpha': 0.6})

            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, p50_values)):
                height = bar.get_height()
                if metric_keys[i] == 'max_shortage':
                    label = f'{val:.0f}'
                else:
                    label = f'{val:.0f}'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       label, ha='center', va='bottom', fontweight='bold', fontsize=10)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(metric_names, fontsize=10)
            ax.set_ylabel('Absolute Value', fontsize=12, fontweight='bold')
            ax.set_ylim(bottom=0)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)

        else:  # Climate scenario panels (percentage change)
            pct_change = pct_changes[dataset_id]

            # Plot bars for each metric
            x_pos = np.arange(3)
            p50_values = [pct_change[m][1] for m in metric_keys]
            p5_values = [pct_change[m][0] for m in metric_keys]
            p95_values = [pct_change[m][2] for m in metric_keys]

            # Error bars
            yerr_low = [p50_values[i] - p5_values[i] for i in range(3)]
            yerr_high = [p95_values[i] - p50_values[i] for i in range(3)]

            bars = ax.bar(x_pos, p50_values, color=colors_diff, alpha=0.8,
                         yerr=[yerr_low, yerr_high], capsize=5,
                         error_kw={'linewidth': 2, 'ecolor': 'black', 'alpha': 0.6})

            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, p50_values)):
                height = bar.get_height()
                offset = 2 if height >= 0 else -10
                label = f'{val:+.0f}%'
                ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                       label, ha='center', va='bottom' if height >= 0 else 'top',
                       fontweight='bold', fontsize=9)

            ax.axhline(0, color='black', linewidth=1.5, linestyle='-', alpha=0.7)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(metric_names, fontsize=10)
            ax.set_ylabel('% Change', fontsize=12, fontweight='bold')
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
