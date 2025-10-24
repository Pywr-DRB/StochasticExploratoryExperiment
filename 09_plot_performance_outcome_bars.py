"""
Plot performance metrics across ensembles with parallel axes visualization.

Shows:
- Left panel: Absolute performance for stationary ensemble (p5, p50, p95)
- Right panels: Absolute change for each climate scenario (3x2 grid)

This script uses pre-calculated metrics from postprocessing:
- shortage: Pre-calculated flow target violations
- mrf_target: Flow targets for calculating reliability
- res_storage: Reservoir storage for NYC system

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
        
        # Metric 3: Maximum daily shortage magnitude (positive value, lower is better)
        # For visualization, we'll show the actual value (not inverted)
        max_shortage = montague_shortage.max()
        
        metrics[r] = {
            'metric1': n_years_reliable,
            'metric2': n_years_high_storage,
            'metric3': max_shortage  # Positive value
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
                              metric_scales=None):
    """
    Plot parallel axes for performance metrics with independent y-scales.
    
    Parameters
    ----------
    ax : matplotlib axis
    percentiles_dict : dict
        {'metric1': [p5, p50, p95], ...}
    metric_names : list
        Names for each metric axis
    is_difference : bool
        If True, plot as difference from baseline (centered at 0)
    metric_scales : dict or None
        {'metric1': (ymin, ymax), ...} for independent axis scales
    """
    n_metrics = 3
    x_positions = np.arange(n_metrics)
    
    # Colors
    color_band = 'steelblue' if not is_difference else 'coral'
    color_median = 'darkblue' if not is_difference else 'darkred'
    
    # Calculate normalization for each metric to [0, 1] range for plotting
    # but keep track of original values for labels
    normalized_percentiles = {}
    original_values = {}
    
    for metric in ['metric1', 'metric2', 'metric3']:
        p5, p50, p95 = percentiles_dict[metric]
        original_values[metric] = [p5, p50, p95]
        
        # Determine scale for this metric
        if metric_scales and metric in metric_scales:
            vmin, vmax = metric_scales[metric]
        else:
            if is_difference:
                max_abs = max(abs(p5), abs(p95))
                vmin, vmax = -max_abs * 1.2, max_abs * 1.2
            else:
                vmin = min(0, p5 * 0.9)
                vmax = p95 * 1.1
        
        # Normalize to [0, 1] for plotting
        value_range = vmax - vmin
        if value_range > 0:
            norm_p5 = (p5 - vmin) / value_range
            norm_p50 = (p50 - vmin) / value_range
            norm_p95 = (p95 - vmin) / value_range
        else:
            norm_p5 = norm_p50 = norm_p95 = 0.5
        
        normalized_percentiles[metric] = [norm_p5, norm_p50, norm_p95]
    
    # Plot each metric axis with normalized values
    for i, metric in enumerate(['metric1', 'metric2', 'metric3']):
        norm_p5, norm_p50, norm_p95 = normalized_percentiles[metric]
        x = x_positions[i]
        
        # Vertical band (p5 to p95)
        ax.add_patch(Rectangle((x - 0.15, norm_p5), 0.3, norm_p95 - norm_p5,
                               facecolor=color_band, alpha=0.3, 
                               edgecolor=color_band, linewidth=1.5))
        
        # Percentile dots
        ax.plot(x, norm_p5, 'o', color=color_band, markersize=6, zorder=3)
        ax.plot(x, norm_p95, 'o', color=color_band, markersize=6, zorder=3)
        ax.plot(x, norm_p50, 'o', color=color_median, markersize=8, zorder=4)
        
        # Add value labels below each metric
        orig_p5, orig_p50, orig_p95 = original_values[metric]
        label_text = f"{orig_p50:.0f}"
        ax.text(x, -0.15, label_text, ha='center', va='top', fontsize=8, 
               fontweight='bold', color=color_median)
        
        # Vertical line at axis position
        ax.plot([x, x], [0, 1], 'k-', linewidth=1, alpha=0.3, zorder=1)
    
    # Formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_xlim(-0.5, n_metrics - 0.5)
    ax.set_ylim(-0.2, 1.05)
    
    # Add horizontal line at middle for difference plots
    if is_difference:
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=2)
    
    # Hide y-axis (we're using normalized values)
    ax.set_yticks([])
    
    # Remove spines
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)
    
    # Y-axis label
    ylabel = 'Δ Performance (normalized)' if is_difference else 'Performance (normalized)'
    ax.set_ylabel(ylabel, fontsize=11)
    
    if title:
        ax.set_title(title, fontsize=10, pad=10)
    
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
    Generate complete performance metrics figure using pre-calculated data.
    """
    print("=" * 60)
    print("CALCULATING PERFORMANCE METRICS")
    print("=" * 60)
    
    # Load pre-calculated data for all datasets
    all_data = {}
    all_percentiles = {}
    
    for dataset_id in DATASET_CONFIGS.keys():
        fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
        
        if not os.path.exists(fname):
            print(f"Skipping {dataset_id}: postprocessed data not found")
            continue
        
        print(f"\nLoading {dataset_id}...")
        data = pywrdrb.Data()
        # Load only pre-calculated metrics needed
        data.load_from_export(fname, results_sets=['shortage', 'res_storage', 'mrf_target'])
        
        all_data[dataset_id] = data
        
        # Calculate metrics using pre-calculated data
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
        print(f"  Metric 3 (max shortage): p5={percentiles['metric3'][0]:.1f}, "
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
    
    # Calculate metric scales for normalization
    print("\nCalculating metric scales for normalization...")
    metric_scales_absolute = calculate_shared_ylims({'stationary': stationary}, is_difference=False)
    metric_scales_difference = calculate_shared_ylims(differences, is_difference=True)
    
    # Create figure
    print("\nGenerating figure...")
    fig = plt.figure(figsize=(16, 10))
    
    # Left panel: Stationary baseline
    ax_baseline = plt.subplot2grid((3, 4), (0, 0), rowspan=3)
    
    metric_names = [
        'Years w/\nMontague\nreliable',
        'Years w/\nNYC storage\nhigh',
        'Max.\nshortage\n(MGD)'
    ]
    
    plot_parallel_performance(
        ax_baseline,
        stationary,
        metric_names,
        title='Stationary Ensemble',
        is_difference=False,
        metric_scales=metric_scales_absolute
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
            metric_scales=metric_scales_difference
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