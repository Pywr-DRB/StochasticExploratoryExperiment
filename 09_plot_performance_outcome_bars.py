"""
Plot performance metrics across ensembles with parallel axes visualization.

Shows:
- Left panel: Absolute performance for stationary ensemble (p5, p50, p95)
- Right panels: Absolute change for each climate scenario (3x2 grid)

Usage:
  python 09_plot_performance_metrics.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from config import *


# Output directory
FIG_OUTPUT_DIR = f"{FIG_DIR}/performance_metrics"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)


def calculate_performance_metrics(data, model, realizations):
    """
    Calculate three performance metrics for each realization.
    
    Returns
    -------
    metrics_dict : dict
        {realization_id: {'metric1': value, 'metric2': value, 'metric3': value}}
    """
    metrics = {}
    
    for r in realizations:
        if (r % 100 == 0) and (r > 0):
            print(f"    Processed {r}/{len(realizations)} realizations...")
        
        # Metric 1: # years where Montague flow target met >90% of time
        montague_shortage = data.shortage[model][r]['delMontague']
        montague_target = data.mrf_target[model][r]['delMontague']
        
        # Annual reliability = 1 - (annual_shortage / annual_target)
        annual_shortage = montague_shortage.resample('YS').sum()
        annual_target = montague_target.resample('YS').sum()
        annual_reliability = 1 - (annual_shortage / annual_target)
        annual_reliability = annual_reliability.clip(0, 1)
        
        n_years_reliable = (annual_reliability > 0.90).sum()
        
        # Metric 2: # years where NYC storage >90% on June 1
        nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']
        nyc_storage = data.res_storage[model][r][nyc_reservoirs].sum(axis=1)
        
        # Get max possible storage (assume full storage is max observed)
        max_storage = nyc_storage.max()
        nyc_storage_pct = 100 * nyc_storage / max_storage
        
        # Filter for June 1 dates
        june1_storage = nyc_storage_pct[nyc_storage_pct.index.month == 6]
        june1_storage = june1_storage[june1_storage.index.day == 1]
        
        n_years_high_storage = (june1_storage > 90).sum()
        
        # Metric 3: Maximum daily shortage magnitude (INVERTED: lower is better → higher score)
        max_shortage = montague_shortage.max()
        # Invert so up = better (we'll use negative of max shortage)
        max_shortage_inverted = -max_shortage
        
        metrics[r] = {
            'metric1': n_years_reliable,
            'metric2': n_years_high_storage,
            'metric3': max_shortage_inverted
        }
    
    return metrics


def calculate_ensemble_percentiles(metrics_dict):
    """
    Calculate p5, p50, p95 for each metric across realizations.
    
    Returns
    -------
    percentiles : dict
        {'metric1': [p5, p50, p95], 'metric2': [...], 'metric3': [...]}
    """
    df = pd.DataFrame(metrics_dict).T
    
    percentiles = {}
    for metric in ['metric1', 'metric2', 'metric3']:
        p5 = df[metric].quantile(0.05)
        p50 = df[metric].quantile(0.50)
        p95 = df[metric].quantile(0.95)
        percentiles[metric] = [p5, p50, p95]
    
    return percentiles


def plot_parallel_performance(ax, percentiles_dict, 
                              metric_names,
                              title=None,
                              is_difference=False,
                              y_lims=None):
    """
    Plot parallel axes for performance metrics.
    
    Parameters
    ----------
    ax : matplotlib axis
    percentiles_dict : dict
        {'metric1': [p5, p50, p95], ...}
    metric_names : list
        Names for each metric axis
    is_difference : bool
        If True, plot as difference from baseline (centered at 0)
    y_lims : dict or None
        {'metric1': (ymin, ymax), ...} for shared limits
    """
    n_metrics = 3
    x_positions = np.arange(n_metrics)
    
    # Colors
    color_band = 'steelblue' if not is_difference else 'coral'
    color_median = 'darkblue' if not is_difference else 'darkred'
    
    # Plot each metric axis
    for i, metric in enumerate(['metric1', 'metric2', 'metric3']):
        p5, p50, p95 = percentiles_dict[metric]
        x = x_positions[i]
        
        # Vertical band (p5 to p95)
        ax.add_patch(Rectangle((x - 0.15, p5), 0.3, p95 - p5,
                               facecolor=color_band, alpha=0.3, 
                               edgecolor=color_band, linewidth=1.5))
        
        # Percentile dots
        ax.plot(x, p5, 'o', color=color_band, markersize=6, zorder=3)
        ax.plot(x, p95, 'o', color=color_band, markersize=6, zorder=3)
        ax.plot(x, p50, 'o', color=color_median, markersize=8, zorder=4)
        
        # Vertical line at axis position
        if y_lims and metric in y_lims:
            ymin, ymax = y_lims[metric]
        else:
            ymin, ymax = ax.get_ylim()
        ax.plot([x, x], [ymin, ymax], 'k-', linewidth=1, alpha=0.3, zorder=1)
    
    # Formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_xlim(-0.5, n_metrics - 0.5)
    
    # Set y-limits if provided
    if y_lims:
        for i, metric in enumerate(['metric1', 'metric2', 'metric3']):
            if metric in y_lims:
                current_ylim = ax.get_ylim()
                ax.set_ylim(y_lims[metric])
    
    # Add horizontal line at 0 for difference plots
    if is_difference:
        ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=2)
    
    # Remove spines
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_position(('data', ax.get_ylim()[0]))
    
    # Y-axis label
    ylabel = 'Δ Performance' if is_difference else 'Performance'
    ax.set_ylabel(ylabel, fontsize=11)
    
    if title:
        ax.set_title(title, fontsize=10, pad=10)
    
    ax.tick_params(left=True, bottom=False)
    ax.grid(axis='y', alpha=0.2, zorder=0)


def calculate_shared_ylims(all_percentiles, is_difference=False):
    """
    Calculate shared y-limits across all plots for each metric.
    
    Parameters
    ----------
    all_percentiles : dict
        {dataset_id: {'metric1': [p5, p50, p95], ...}}
    is_difference : bool
        If True, calculate symmetric limits around 0
    
    Returns
    -------
    y_lims : dict
        {'metric1': (ymin, ymax), ...}
    """
    y_lims = {}
    
    for metric in ['metric1', 'metric2', 'metric3']:
        all_values = []
        for dataset_id, percentiles in all_percentiles.items():
            all_values.extend(percentiles[metric])
        
        if is_difference:
            # Symmetric around 0
            max_abs = np.max(np.abs(all_values))
            y_lims[metric] = (-max_abs * 1.1, max_abs * 1.1)
        else:
            # Just add some padding
            vmin, vmax = np.min(all_values), np.max(all_values)
            padding = (vmax - vmin) * 0.1
            y_lims[metric] = (vmin - padding, vmax + padding)
    
    return y_lims


def plot_all_performance_metrics():
    """
    Generate complete performance metrics figure.
    """
    print("=" * 60)
    print("CALCULATING PERFORMANCE METRICS")
    print("=" * 60)
    
    # Load data for all datasets
    all_data = {}
    all_percentiles = {}
    
    for dataset_id in DATASET_CONFIGS.keys():
        fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
        
        
        if not os.path.exists(fname):
            print(f"Skipping {dataset_id}: file not found")
            continue
        
        print(f"\nLoading {dataset_id}...")
        data = pywrdrb.Data()
        data.load_from_export(fname,
                              results_sets = ['shortage', 'res_storage', 'mrf_target'],)
        
        # Check if data has attributes for each results_set
        for rs in ['shortage', 'res_storage', 'mrf_target']:
            if not hasattr(data, rs):
                print(f"  ERROR: {dataset_id} missing results_set '{rs}'")
                continue
            else:
                print(f"  Loaded results_set '{rs}' for {dataset_id}")
        
        all_data[dataset_id] = data
        
        # Calculate metrics
        print(f"  Calculating metrics...")
        realizations = list(data.shortage[dataset_id].keys())
        metrics_dict = calculate_performance_metrics(data, dataset_id, realizations)
        
        # Calculate percentiles
        percentiles = calculate_ensemble_percentiles(metrics_dict)
        all_percentiles[dataset_id] = percentiles
        
        print(f"  Metric 1 (# years Montague reliable): p5={percentiles['metric1'][0]:.1f}, "
              f"p50={percentiles['metric1'][1]:.1f}, p95={percentiles['metric1'][2]:.1f}")
        print(f"  Metric 2 (# years NYC storage high): p5={percentiles['metric2'][0]:.1f}, "
              f"p50={percentiles['metric2'][1]:.1f}, p95={percentiles['metric2'][2]:.1f}")
        print(f"  Metric 3 (-max shortage): p5={percentiles['metric3'][0]:.1f}, "
              f"p50={percentiles['metric3'][1]:.1f}, p95={percentiles['metric3'][2]:.1f}")
    
    if 'stationary_ensemble' not in all_percentiles:
        print("ERROR: stationary_ensemble required but not found!")
        return
    
    # Calculate differences from stationary
    stationary = all_percentiles['stationary_ensemble']
    differences = {}
    
    for dataset_id in all_percentiles.keys():
        if dataset_id == 'stationary_ensemble':
            continue
        
        diff = {}
        for metric in ['metric1', 'metric2', 'metric3']:
            # Absolute difference at each percentile
            diff[metric] = [
                all_percentiles[dataset_id][metric][i] - stationary[metric][i]
                for i in range(3)
            ]
        differences[dataset_id] = diff
    
    # Calculate shared y-limits
    print("\nCalculating shared y-limits...")
    y_lims_absolute = calculate_shared_ylims({'stationary': stationary}, is_difference=False)
    y_lims_difference = calculate_shared_ylims(differences, is_difference=True)
    
    # Create figure
    print("\nGenerating figure...")
    fig = plt.figure(figsize=(16, 10))
    
    # Left panel: Stationary baseline
    ax_baseline = plt.subplot2grid((3, 4), (0, 0), rowspan=3)
    
    metric_names = [
        'Years w/\nMontague\nreliable',
        'Years w/\nNYC storage\nhigh',
        'Min. max\nshortage\n(inverted)'
    ]
    
    plot_parallel_performance(
        ax_baseline,
        stationary,
        metric_names,
        title='Stationary Ensemble',
        is_difference=False,
        y_lims=y_lims_absolute
    )
    
    # Right panels: Climate scenarios (3x2 grid)
    climate_datasets = [k for k in DATASET_CONFIGS.keys() if k != 'stationary_ensemble']
    climate_datasets.sort()  # Consistent ordering
    
    for idx, dataset_id in enumerate(climate_datasets):
        if dataset_id not in differences:
            continue
        
        row = idx // 2
        col = idx % 2 + 2  # Start at column 2
        
        ax = plt.subplot2grid((3, 4), (row, col))
        
        # Clean title
        title = dataset_id.replace('climate_adjusted_', '').replace('_', ' ').upper()
        
        plot_parallel_performance(
            ax,
            differences[dataset_id],
            metric_names,
            title=title,
            is_difference=True,
            y_lims=y_lims_difference
        )
    
    # Overall title
    fig.suptitle('Water System Performance Metrics', fontsize=14, fontweight='bold', y=0.98)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', 
               markersize=8, label='p5, p95'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', 
               markersize=8, label='p50 (median)'),
        Rectangle((0, 0), 1, 1, facecolor='steelblue', alpha=0.3, 
                 edgecolor='steelblue', label='90% range')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, 
              bbox_to_anchor=(0.15, 0.96), frameon=False, fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    fname = f"{FIG_OUTPUT_DIR}/performance_metrics_comparison.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {fname}")
    
    # Also save as SVG
    fname_svg = fname.replace('.png', '.svg')
    plt.savefig(fname_svg, bbox_inches='tight')
    print(f"Saved: {fname_svg}")


def main():
    """Main entry point."""
    plot_all_performance_metrics()
    
    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()