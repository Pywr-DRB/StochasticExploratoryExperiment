"""
Plot NYC inflow vs Montague contributions colored by satisficing conditions.

Satisficing conditions:
1. NYC storage >= 20% throughout June-Dec period
2. Montague flow target violations <= 3 continuous days

This script uses pre-calculated metrics from the postprocessing output:
- shortage: Pre-calculated flow target violations for each node
- contribution: Pre-calculated NYC downstream contributions to Montague
- inflow: Reservoir inflows with aggregated NYC values
- res_storage: Reservoir storage levels

Creates a 4-panel comparison figure showing satisficing scatter plots
for all climate scenarios.

Usage:
------
python 09_plot_satisficing_scatter.py

Output:
-------
figures/satisficing/satisficing_4panel_comparison.png
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from config import *
from methods.plotting.styles import (
    DATASET_COLORS, DATASET_LABELS, DATASET_LABELS_SHORT,
    DATASET_ORDER, FIGSIZE_QUAD, DPI_HIGH
)
from methods.metrics.satisficing import calculate_satisficing_conditions


def _legacy_calculate_satisficing_conditions(data, dataset_id, storage_threshold=20.0, violation_days=3):
    """
    Calculate satisficing conditions for each (year, realization) pair using pre-calculated metrics.

    Parameters
    ----------
    data : pywrdrb.Data
        Data object with pre-calculated shortage, contribution, res_storage, and inflow
    dataset_id : str
        Dataset identifier
    storage_threshold : float, optional
        Minimum acceptable NYC storage percentage (default: 20%)
    violation_days : int, optional
        Maximum acceptable continuous Montague violation days (default: 3)

    Returns
    -------
    pd.DataFrame
        Results with satisficing status and aggregated metrics
    """

    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']

    # Storage capacities for NYC reservoirs (MG)
    storage_capacities = {
        'cannonsville': 95706,
        'pepacton': 140190,
        'neversink': 34941
    }
    total_capacity = sum(storage_capacities.values())

    # Get realizations from shortage data (all dicts should have same realizations)
    realizations = list(data.shortage[dataset_id].keys())

    results = {
        'year': [],
        'realization': [],
        'nyc_inflow_jun_dec': [],
        'montague_contrib_jun_dec': [],
        'satisficing': [],
        'min_storage_pct': [],
        'max_violation_days': []
    }

    for r in realizations:
        # Use pre-calculated data directly from postprocessing
        nyc_storage = data.res_storage[dataset_id][r][nyc_reservoirs].sum(axis=1)
        nyc_storage_pct = 100.0 * nyc_storage / total_capacity

        # Use pre-calculated shortage
        montague_shortage = data.shortage[dataset_id][r]['delMontague']

        # Use pre-calculated NYC inflow (aggregated in postprocessing)
        nyc_inflow = data.inflow[dataset_id][r]['nyc']

        # Use pre-calculated NYC contribution to Montague
        montague_contrib = data.contribution[dataset_id][r]['mrf_montagueTrenton_nyc']

        # Align all time series to the same index (use storage as reference)
        common_index = nyc_storage.index
        montague_shortage = montague_shortage.reindex(common_index, fill_value=0)
        nyc_inflow = nyc_inflow.reindex(common_index, fill_value=0)
        montague_contrib = montague_contrib.reindex(common_index, fill_value=0)

        # Get years in data
        years = pd.DatetimeIndex(common_index).year.unique()

        for year in years:
            # Filter June 1 - Dec 31
            mask = (common_index >= f'{year}-06-01') & (common_index <= f'{year}-12-31')

            if not mask.any():
                continue

            # Check storage condition
            min_storage = nyc_storage_pct[mask].min()
            storage_ok = min_storage >= storage_threshold

            # Check Montague violation condition using pre-calculated shortage
            violations = montague_shortage[mask] > 0
            if violations.any():
                # Calculate max consecutive violation days
                groups = (violations != violations.shift()).cumsum()
                max_consec = violations.groupby(groups).sum().max()
            else:
                max_consec = 0

            montague_ok = max_consec <= violation_days

            # Calculate aggregates for Jun-Dec period
            total_inflow = nyc_inflow[mask].sum()
            total_contrib = montague_contrib[mask].sum()

            # Store results
            results['year'].append(year)
            results['realization'].append(r)
            results['nyc_inflow_jun_dec'].append(total_inflow)
            results['montague_contrib_jun_dec'].append(total_contrib)
            results['satisficing'].append(storage_ok and montague_ok)
            results['min_storage_pct'].append(min_storage)
            results['max_violation_days'].append(max_consec)

    return pd.DataFrame(results)


def plot_4panel_satisficing_comparison(all_results, figsize=FIGSIZE_QUAD, fname=None):
    """
    Create a 4-panel comparison figure showing satisficing scatter plots.

    Layout matches other 4-panel figures:
    - Left panel: Stationary ensemble
    - Right panels (stacked): Climate Low, Medium, High

    Parameters
    ----------
    all_results : dict
        Dictionary mapping dataset_id to results DataFrame
    figsize : tuple, optional
        Figure size (default: from styles)
    fname : str, optional
        Output filename

    Returns
    -------
    tuple
        (fig, axes)
    """

    print(f"\n{'='*60}")
    print("Creating 4-Panel Satisficing Comparison Figure")
    print(f"{'='*60}")

    # Dataset order and labels
    datasets = {
        'stationary_ensemble': DATASET_LABELS_SHORT['stationary_ensemble'],
        'climate_adjusted_low': DATASET_LABELS_SHORT['climate_adjusted_low'],
        'climate_adjusted_medium': DATASET_LABELS_SHORT['climate_adjusted_medium'],
        'climate_adjusted_high': DATASET_LABELS_SHORT['climate_adjusted_high']
    }

    # Set up figure with GridSpec for flexible layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1],
                          hspace=0.20, wspace=0.25,
                          left=0.08, right=0.95, top=0.93, bottom=0.10)

    # Create axes
    ax_stat = fig.add_subplot(gs[:, 0])  # Left panel spans all rows
    ax_low = fig.add_subplot(gs[0, 1])   # Top right
    ax_med = fig.add_subplot(gs[1, 1])   # Middle right
    ax_high = fig.add_subplot(gs[2, 1])  # Bottom right

    axes = [ax_stat, ax_low, ax_med, ax_high]
    dataset_list = list(datasets.keys())
    panel_labels = ['(a)', '(b)', '(c)', '(d)']

    # Find global axis limits for consistency
    all_inflows = []
    all_contribs = []
    for results_df in all_results.values():
        all_inflows.extend(results_df['nyc_inflow_jun_dec'].values)
        all_contribs.extend(results_df['montague_contrib_jun_dec'].values)

    x_min, x_max = np.percentile(all_inflows, [0.5, 99.5])
    y_min, y_max = np.percentile(all_contribs, [0.5, 99.5])
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_lim = [x_min - 0.05 * x_range, x_max + 0.05 * x_range]
    y_lim = [y_min - 0.05 * y_range, y_max + 0.05 * y_range]

    # Plot each panel
    for idx, (ax, dataset_id, panel_label) in enumerate(zip(axes, dataset_list, panel_labels)):

        if dataset_id not in all_results:
            print(f"  Skipping {dataset_id} (no data)")
            continue

        results_df = all_results[dataset_id]

        # Separate satisficing and non-satisficing
        satisficing = results_df[results_df['satisficing']]
        non_satisficing = results_df[~results_df['satisficing']]

        # Colors: satisficing (dataset color), non-satisficing (gray)
        color_satisficing = DATASET_COLORS[dataset_id]
        color_nonsatisficing = '#808080'  # Gray

        # Plot non-satisficing first (background)
        ax.scatter(non_satisficing['nyc_inflow_jun_dec'],
                  non_satisficing['montague_contrib_jun_dec'],
                  c=color_nonsatisficing, alpha=0.4, s=12,
                  edgecolors='none', label='Non-satisficing', zorder=1)

        # Plot satisficing on top
        ax.scatter(satisficing['nyc_inflow_jun_dec'],
                  satisficing['montague_contrib_jun_dec'],
                  c=color_satisficing, alpha=0.6, s=15,
                  edgecolors='none', label='Satisficing', zorder=2)

        # Set axis limits
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
        ax.set_axisbelow(True)

        # Panel title with satisficing percentage
        pct_satisficing = 100 * len(satisficing) / len(results_df)
        title_text = f"{panel_label} {datasets[dataset_id]}"
        ax.text(0.02, 0.98, title_text,
               transform=ax.transAxes, fontsize=13, fontweight='bold',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.3))

        # Add satisficing percentage
        stats_text = f'{pct_satisficing:.1f}% satisficing'
        ax.text(0.98, 0.02, stats_text,
               transform=ax.transAxes, fontsize=10,
               horizontalalignment='right', verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7, pad=0.3))

        # Axis labels
        if idx == 0:  # Stationary (left panel)
            ax.set_xlabel('NYC Inflow (Jun-Dec) [MG]', fontsize=12, fontweight='bold')
            ax.set_ylabel('NYC → Montague Contributions (Jun-Dec) [MG]',
                         fontsize=12, fontweight='bold')
        elif idx == 3:  # Bottom right panel
            ax.set_xlabel('NYC Inflow (Jun-Dec) [MG]', fontsize=12, fontweight='bold')
        else:  # Other right panels
            ax.set_xticklabels([])

        # Y-axis labels only for left panel
        if idx != 0:
            ax.set_ylabel('')

        # Tick label sizes
        ax.tick_params(labelsize=10)

    # Overall title
    fig.suptitle('Satisficing Conditions: NYC Inflow vs. Montague Contributions',
                fontsize=16, fontweight='bold')

    # Add criteria text box at bottom
    criteria_text = (
        'Satisficing Criteria: NYC storage ≥ 20% (Jun-Dec) AND '
        'Montague violations ≤ 3 consecutive days'
    )
    fig.text(0.5, 0.02, criteria_text,
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, pad=0.5))

    # Save figure
    if fname:
        plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
        print(f"\nSaved: {fname}")

        # Also save as SVG
        fname_svg = fname.replace('.png', '.svg')
        plt.savefig(fname_svg, bbox_inches='tight')
        print(f"Saved: {fname_svg}")

    return fig, axes


def print_summary_statistics(all_results):
    """
    Print summary statistics comparing satisficing rates across datasets.

    Parameters
    ----------
    all_results : dict
        Dictionary mapping dataset_id to results DataFrame
    """
    print(f"\n{'='*60}")
    print("SATISFICING SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"{'Dataset':<30} {'Total':>10} {'Satisficing':>12} {'%':>8}")
    print("-" * 60)

    for dataset_id in DATASET_ORDER:
        if dataset_id not in all_results:
            continue

        results_df = all_results[dataset_id]
        n_total = len(results_df)
        n_satisficing = results_df['satisficing'].sum()
        pct_satisficing = 100 * n_satisficing / n_total

        label = DATASET_LABELS[dataset_id]
        print(f"{label:<30} {n_total:>10,} {n_satisficing:>12,} {pct_satisficing:>7.1f}%")

    print("=" * 60)

    # Print failure breakdown for each dataset
    print(f"\n{'='*60}")
    print("FAILURE BREAKDOWN BY TYPE")
    print(f"{'='*60}")

    for dataset_id in DATASET_ORDER:
        if dataset_id not in all_results:
            continue

        results_df = all_results[dataset_id]
        n_total = len(results_df)

        # Categorize failures
        storage_fail = results_df['min_storage_pct'] < 20
        montague_fail = results_df['max_violation_days'] > 3
        both_fail = storage_fail & montague_fail

        print(f"\n{DATASET_LABELS[dataset_id]}:")
        print(f"  Storage < 20% only:        {(storage_fail & ~montague_fail).sum():>6,} "
              f"({100*(storage_fail & ~montague_fail).sum()/n_total:>5.1f}%)")
        print(f"  Montague > 3 days only:    {(montague_fail & ~storage_fail).sum():>6,} "
              f"({100*(montague_fail & ~storage_fail).sum()/n_total:>5.1f}%)")
        print(f"  Both failures:             {both_fail.sum():>6,} "
              f"({100*both_fail.sum()/n_total:>5.1f}%)")
        print(f"  Total non-satisficing:     {(~results_df['satisficing']).sum():>6,} "
              f"({100*(~results_df['satisficing']).sum()/n_total:>5.1f}%)")

    print("=" * 60)


def main():
    """Main function to generate 4-panel satisficing comparison."""

    print("=" * 60)
    print("SATISFICING CONDITIONS ANALYSIS - 4-PANEL COMPARISON")
    print("=" * 60)

    all_results = {}

    # Calculate for each dataset
    for dataset_id in DATASET_ORDER:
        fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'

        if not os.path.exists(fname):
            print(f"\nSkipping {dataset_id} (postprocessed data not found)")
            continue

        print(f"\nProcessing {dataset_id}...")

        # Load data
        data = pywrdrb.Data()
        data.load_from_export(fname, results_sets=['res_storage', 'inflow', 'shortage', 'contribution'])

        # Calculate satisficing conditions using new module
        results = calculate_satisficing_conditions(
            data, dataset_id,
            period_type='year',
            evaluate_all_years=True,
            storage_threshold=20.0,
            violation_days=3
        )

        # Rename columns to match expected format for plotting
        results = results.rename(columns={
            'nyc_inflow': 'nyc_inflow_jun_dec',
            'montague_contrib': 'montague_contrib_jun_dec'
        })

        all_results[dataset_id] = results

        # Quick summary
        pct_sat = 100 * results['satisficing'].sum() / len(results)
        print(f"  {pct_sat:.1f}% satisficing ({results['satisficing'].sum():,}/{len(results):,})")

    if len(all_results) == 0:
        print("\nERROR: No datasets found!")
        print("Run postprocessing (04_postprocess_data.py) first!")
        return

    # Print detailed statistics
    print_summary_statistics(all_results)

    # Create output directory
    output_dir = f"{FIG_DIR}/satisficing"
    os.makedirs(output_dir, exist_ok=True)

    # Generate 4-panel comparison figure
    print(f"\nGenerating 4-panel comparison figure...")
    fname = f"{output_dir}/satisficing_4panel_comparison.png"
    fig, axes = plot_4panel_satisficing_comparison(all_results, fname=fname)

    # Save individual results to CSV
    for dataset_id, results_df in all_results.items():
        csv_fname = f"{output_dir}/{dataset_id}_satisficing_results.csv"
        results_df.to_csv(csv_fname, index=False)
        print(f"Saved results: {csv_fname}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

    return all_results


if __name__ == "__main__":
    main()
