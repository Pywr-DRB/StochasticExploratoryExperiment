"""
Plot contribution ratio vs minimum storage scatter plot.

This script creates a scatter plot showing the relationship between:
- X-axis: Ratio of NYC Montague contributions to NYC total inflow
  (calculated for N days prior to minimum storage event)
- Y-axis: Minimum NYC reservoir storage for each year

By default, the script automatically finds the optimal N (number of days) that
maximizes the correlation magnitude between contribution ratio and minimum storage.

Usage:
------
python 10_plot_contribution_ratio_vs_min_storage.py [dataset_id] [options]

Examples:
---------
# Auto-optimize N for maximum correlation (default)
python 10_plot_contribution_ratio_vs_min_storage.py stationary_ensemble

# Use specific N value
python 10_plot_contribution_ratio_vs_min_storage.py stationary_ensemble --days 90

# Custom optimization range
python 10_plot_contribution_ratio_vs_min_storage.py stationary_ensemble --n-range 20 150 5

# All scenarios with auto-optimization
python 10_plot_contribution_ratio_vs_min_storage.py --all

Output:
-------
figures/contribution_analysis/contribution_ratio_vs_min_storage_{dataset_id}_{N}days.png
figures/contribution_analysis/correlation_optimization_{dataset_id}.csv (if optimizing)
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from methods.config import *
from methods.plotting.styles import (
    DATASET_COLORS, DATASET_LABELS,
    FIGSIZE_DOUBLE, DPI_HIGH
)

# Output directory
FIGURE_DIR = "./figures/contribution_analysis"
os.makedirs(FIGURE_DIR, exist_ok=True)


def calculate_contribution_ratio_metrics(data, dataset_id, n_days=90):
    """
    Calculate contribution ratio and minimum storage for each year-realization pair.

    For each year, we:
    1. Find the date of minimum storage
    2. Calculate sum of NYC Montague contributions for N days prior
    3. Calculate sum of NYC inflow for N days prior
    4. Calculate ratio: sum(contributions) / sum(inflow)

    Parameters
    ----------
    data : pywrdrb.Data
        Data object with res_storage, contribution, and inflow
    dataset_id : str
        Dataset identifier
    n_days : int
        Number of days prior to minimum storage to use for sums

    Returns
    -------
    pd.DataFrame
        Columns: year, realization, annual_min_storage_pct, contribution_ratio,
                 min_storage_date, nyc_inflow_sum, montague_contrib_sum
    """

    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']

    # Get realizations
    realizations = sorted(data.res_storage[dataset_id].keys())

    print(f"\nCalculating contribution ratio metrics for {dataset_id}")
    print(f"  Using {n_days} days prior to minimum storage")
    print(f"  Processing {len(realizations)} realizations...")

    results = []
    skipped_count = 0
    total_years = 0

    for r_idx, r in enumerate(realizations):
        if (r_idx + 1) % 100 == 0:
            print(f"    Processed {r_idx + 1}/{len(realizations)} realizations...")

        # Get NYC storage percentage
        nyc_storage = data.res_storage[dataset_id][r][nyc_reservoirs].sum(axis=1)
        nyc_storage_pct = 100.0 * nyc_storage / NYC_TOTAL_CAPACITY

        # Get NYC inflow and Montague contributions
        nyc_inflow = data.inflow[dataset_id][r]['nyc']
        montague_contrib = data.contribution[dataset_id][r]['mrf_montagueTrenton_nyc']

        # Get all unique years in the data
        years = sorted(nyc_storage_pct.index.year.unique())
        total_years += len(years)

        for year in years:
            # Get data for this year
            year_mask = (nyc_storage_pct.index.year == year)
            storage_year = nyc_storage_pct[year_mask]

            # Also get corresponding inflow and contribution data for the year
            inflow_year = nyc_inflow[year_mask]
            contrib_year = montague_contrib[year_mask]

            # Skip if insufficient data for this year
            if len(storage_year) < 30:  # At least 30 days of data
                skipped_count += 1
                continue

            # Find minimum storage date within this year
            min_storage_date = storage_year.idxmin()
            min_storage_value = storage_year.min()

            # Calculate start date (N days prior to minimum)
            start_date = min_storage_date - pd.Timedelta(days=n_days)

            # Get sums for N-day window prior to minimum
            # Use loc to get data between start_date and min_storage_date
            try:
                # Get the N-day window (can span across year boundary)
                inflow_window = nyc_inflow.loc[start_date:min_storage_date]
                contrib_window = montague_contrib.loc[start_date:min_storage_date]

                # Check if we have enough data
                if len(inflow_window) < n_days * 0.8:  # Allow 20% missing
                    skipped_count += 1
                    continue

                inflow_sum = inflow_window.sum()
                contrib_sum = contrib_window.sum()

                # Calculate ratio (handle division by zero)
                if inflow_sum > 0:
                    contrib_ratio = contrib_sum / inflow_sum
                else:
                    contrib_ratio = np.nan
                    skipped_count += 1
                    continue

                results.append({
                    'year': year,
                    'realization': r,
                    'annual_min_storage_pct': min_storage_value,
                    'contribution_ratio': contrib_ratio,
                    'min_storage_date': min_storage_date,
                    'nyc_inflow_sum': inflow_sum,
                    'montague_contrib_sum': contrib_sum,
                    'n_days': n_days
                })

            except (KeyError, IndexError) as e:
                # Skip if date range issues (e.g., start_date before data begins)
                skipped_count += 1
                continue

    df = pd.DataFrame(results)
    print(f"  Total year-realization pairs processed: {total_years}")
    print(f"  Calculated metrics for: {len(df)} pairs")
    print(f"  Skipped (insufficient data): {skipped_count} pairs")

    return df


def plot_contribution_ratio_scatter(df, dataset_id, n_days=90, ax=None, show_stats=True):
    """
    Create scatter plot of contribution ratio vs minimum storage.

    Parameters
    ----------
    df : pd.DataFrame
        Results from calculate_contribution_ratio_metrics
    dataset_id : str
        Dataset identifier
    n_days : int
        Number of days used for calculation
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    show_stats : bool
        Whether to show correlation statistics

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    # Remove NaN values
    df_clean = df.dropna(subset=['contribution_ratio', 'annual_min_storage_pct'])

    # Create scatter plot
    color = DATASET_COLORS.get(dataset_id, 'steelblue')

    ax.scatter(
        df_clean['contribution_ratio'],
        df_clean['annual_min_storage_pct'],
        alpha=0.3,
        s=20,
        color=color,
        edgecolors='none'
    )

    # Calculate and display statistics
    if show_stats and len(df_clean) > 0:
        # Pearson correlation
        corr = df_clean[['contribution_ratio', 'annual_min_storage_pct']].corr().iloc[0, 1]

        # Add text box with statistics
        stats_text = f"n = {len(df_clean):,}\nr = {corr:.3f}"
        ax.text(0.05, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)

        # Add trend line
        z = np.polyfit(df_clean['contribution_ratio'], df_clean['annual_min_storage_pct'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df_clean['contribution_ratio'].min(),
                             df_clean['contribution_ratio'].max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.5, linewidth=2, label='Linear fit')

    # Labels and formatting
    ax.set_xlabel(f'NYC Contribution Ratio\n(Sum of Montague contrib. / Sum of NYC inflow)\n{n_days} days prior to min storage',
                  fontsize=11)
    ax.set_ylabel('Minimum NYC Storage (%)', fontsize=11)

    dataset_label = DATASET_LABELS.get(dataset_id, dataset_id)
    ax.set_title(f'{dataset_label}', fontsize=12, fontweight='bold')

    # Add reference lines
    ax.axhline(y=20, color='red', linestyle=':', alpha=0.5, linewidth=1.5, label='20% storage threshold')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Legend
    if show_stats:
        ax.legend(loc='lower right', fontsize=9)

    # Set reasonable axis limits
    ax.set_xlim(left=0, right=min(1.0, df_clean['contribution_ratio'].quantile(0.99) * 1.1))
    ax.set_ylim(bottom=0, top=100)

    return fig, ax


def plot_4panel_comparison(datasets, n_days=90):
    """
    Create 4-panel comparison plot for multiple datasets.

    Parameters
    ----------
    datasets : list of str
        Dataset identifiers to plot
    n_days : int
        Number of days for calculation

    Returns
    -------
    fig : matplotlib figure
    """

    # Load data for all datasets
    print("\n" + "=" * 80)
    print("LOADING DATA FOR COMPARISON PLOT")
    print("=" * 80)

    all_data = {}
    all_results = {}

    for dataset_id in datasets:
        print(f"\nLoading {dataset_id}...")
        fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'

        if not os.path.exists(fname):
            print(f"  WARNING: File not found: {fname}")
            continue

        data = pywrdrb.Data()
        data.load_from_export(fname, results_sets=['res_storage', 'inflow', 'contribution'])
        all_data[dataset_id] = data

        # Calculate metrics
        df = calculate_contribution_ratio_metrics(data, dataset_id, n_days)
        all_results[dataset_id] = df

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_DOUBLE)
    axes = axes.flatten()

    for idx, dataset_id in enumerate(datasets):
        if dataset_id not in all_results:
            axes[idx].text(0.5, 0.5, f'Data not available\n{dataset_id}',
                          ha='center', va='center', transform=axes[idx].transAxes)
            continue

        plot_contribution_ratio_scatter(
            all_results[dataset_id],
            dataset_id,
            n_days=n_days,
            ax=axes[idx],
            show_stats=True
        )

    plt.tight_layout()

    return fig


def find_optimal_n_days(data, dataset_id, n_range=(30, 180, 10)):
    """
    Find the optimal number of days that maximizes correlation magnitude.

    Parameters
    ----------
    data : pywrdrb.Data
        Data object
    dataset_id : str
        Dataset identifier
    n_range : tuple
        (min_days, max_days, step) for testing

    Returns
    -------
    optimal_n : int
        Number of days with maximum correlation magnitude
    correlations : dict
        Dictionary of {n_days: correlation} for all tested values
    """
    print(f"\n{'='*80}")
    print(f"FINDING OPTIMAL N FOR: {dataset_id}")
    print(f"{'='*80}")
    print(f"Testing range: {n_range[0]} to {n_range[1]} days (step={n_range[2]})")

    n_values = range(n_range[0], n_range[1] + 1, n_range[2])
    correlations = {}

    for n in n_values:
        print(f"  Testing n={n} days...", end=' ')
        df = calculate_contribution_ratio_metrics(data, dataset_id, n_days=n)

        if len(df) > 10:  # Need minimum sample size
            df_clean = df.dropna(subset=['contribution_ratio', 'annual_min_storage_pct'])
            if len(df_clean) > 10:
                corr = df_clean[['contribution_ratio', 'annual_min_storage_pct']].corr().iloc[0, 1]
                correlations[n] = corr
                print(f"r = {corr:.4f}")
            else:
                print("insufficient clean data")
        else:
            print("insufficient data")

    if not correlations:
        print("\nWARNING: Could not calculate correlations for any n value!")
        return n_range[0], correlations

    # Find n with maximum correlation magnitude (most negative)
    optimal_n = min(correlations.keys(), key=lambda k: correlations[k])
    optimal_corr = correlations[optimal_n]

    print(f"\n{'='*80}")
    print(f"OPTIMAL N FOUND: {optimal_n} days")
    print(f"  Correlation: {optimal_corr:.4f}")
    print(f"{'='*80}")

    return optimal_n, correlations


def main():
    """Main function with argument parsing."""

    import argparse

    parser = argparse.ArgumentParser(
        description='Plot contribution ratio vs minimum storage',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'dataset_id',
        type=str,
        nargs='?',
        default=None,
        help='Dataset identifier (or --all for 4-panel comparison)'
    )

    parser.add_argument(
        '--days',
        type=int,
        default=None,
        help='Number of days prior to minimum storage to use for sums (default: auto-optimize)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Create 4-panel comparison plot for all climate scenarios'
    )

    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Optimize N days for maximum correlation (default: True if --days not specified)'
    )

    parser.add_argument(
        '--n-range',
        type=int,
        nargs=3,
        default=[30, 180, 10],
        metavar=('MIN', 'MAX', 'STEP'),
        help='Range for N optimization: min max step (default: 30 180 10)'
    )

    args = parser.parse_args()

    # Determine if we should optimize
    should_optimize = args.optimize or (args.days is None)

    # Determine which datasets to plot
    if args.all or args.dataset_id == '--all':
        # Plot all scenarios in 4-panel
        datasets = [
            'stationary_ensemble',
            'climate_adjusted_low',
            'climate_adjusted_medium',
            'climate_adjusted_high'
        ]

        print("=" * 80)
        print("CREATING 4-PANEL COMPARISON PLOT")
        print("=" * 80)

        # Use first dataset to find optimal N if optimizing
        if should_optimize:
            # Load first dataset to optimize
            fname = f'./pywrdrb/outputs/{datasets[0]}_with_postprocessing.hdf5'
            if os.path.exists(fname):
                data = pywrdrb.Data()
                data.load_from_export(fname, results_sets=['res_storage', 'inflow', 'contribution'])
                optimal_n, _ = find_optimal_n_days(data, datasets[0], n_range=tuple(args.n_range))
                n_days = optimal_n
            else:
                print(f"WARNING: Could not find {fname}, using default n=90")
                n_days = 90
        else:
            n_days = args.days

        fig = plot_4panel_comparison(datasets, n_days=n_days)

        output_file = f"{FIGURE_DIR}/contribution_ratio_vs_min_storage_4panel_{n_days}days.png"
        fig.savefig(output_file, dpi=DPI_HIGH, bbox_inches='tight')
        print(f"\n{'='*80}")
        print(f"Saved: {output_file}")
        print("=" * 80)

    elif args.dataset_id is not None:
        # Plot single dataset
        dataset_id = args.dataset_id
        verify_dataset_id(dataset_id)

        print("=" * 80)
        print(f"CONTRIBUTION RATIO VS MIN STORAGE: {dataset_id}")
        print("=" * 80)

        # Try loading pre-computed metrics first (FAST PATH)
        use_cached = False
        try:
            from methods.load import load_contribution_metrics
            from methods.metrics.contribution import (
                find_optimal_window_for_correlation, get_metrics_for_window
            )

            print("\nAttempting to load pre-computed metrics...")

            if should_optimize:
                print("  Optimizing window using pre-computed metrics (fast)...")
                result = find_optimal_window_for_correlation(
                    dataset_id,
                    target_metric='annual_min_storage_pct',
                    source_metric='contribution_ratio',
                    window_range=tuple(args.n_range)
                )
                n_days = result['optimal_window']
                correlations = result['all_correlations']

                # Save correlation results
                corr_df = pd.DataFrame(list(correlations.items()), columns=['n_days', 'correlation'])
                corr_df = corr_df.sort_values('n_days')
                corr_file = f"{FIGURE_DIR}/correlation_optimization_{dataset_id}.csv"
                corr_df.to_csv(corr_file, index=False)
                print(f"\nSaved correlation results: {corr_file}")
            else:
                n_days = args.days

            # Load metrics for the chosen window
            print(f"\nLoading metrics for {n_days}-day window...")
            df_full = load_contribution_metrics(dataset_id)
            df = get_metrics_for_window(df_full, n_days)

            # Rename columns for compatibility with plotting functions
            df = df.rename(columns={
                f'contribution_ratio_{n_days}d': 'contribution_ratio',
                f'worst_1mo_demand_sat_{n_days}d': 'worst_1mo_demand_sat'
            })

            use_cached = True
            print("  ✓ Successfully loaded pre-computed metrics (fast mode)")

        except (ImportError, FileNotFoundError) as e:
            print(f"\n  ⚠ Pre-computed metrics not available: {e}")
            print("  ℹ Falling back to on-the-fly calculation (slower, ~40-60 min)...")
            print("  ℹ To optimize: re-run postprocessing (sbatch S3_run_postprocessing.sh)\n")

            # FALLBACK: Original code path
            fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
            print(f"Loading data from: {fname}")

            data = pywrdrb.Data()
            data.load_from_export(fname, results_sets=['res_storage', 'inflow', 'contribution',
                                                       'ibt_diversions', 'ibt_demands', 'res_level'])

            # Optimize N if requested
            if should_optimize:
                optimal_n, correlations = find_optimal_n_days(data, dataset_id, n_range=tuple(args.n_range))
                n_days = optimal_n

                # Save correlation results
                corr_df = pd.DataFrame(list(correlations.items()), columns=['n_days', 'correlation'])
                corr_df = corr_df.sort_values('n_days')
                corr_file = f"{FIGURE_DIR}/correlation_optimization_{dataset_id}.csv"
                corr_df.to_csv(corr_file, index=False)
                print(f"\nSaved correlation results: {corr_file}")
            else:
                n_days = args.days

            # Calculate metrics with optimal/specified N
            print(f"\nCalculating final metrics with n={n_days} days...")
            df = calculate_contribution_ratio_metrics(data, dataset_id, n_days=n_days)

            use_cached = False

        # Create plot (same regardless of data source)
        fig, ax = plot_contribution_ratio_scatter(df, dataset_id, n_days=n_days)

        # Save
        output_file = f"{FIGURE_DIR}/contribution_ratio_vs_min_storage_{dataset_id}_{n_days}days.png"
        fig.savefig(output_file, dpi=DPI_HIGH, bbox_inches='tight')

        print(f"\n{'='*80}")
        if use_cached:
            print(f"✓ Plot created using pre-computed metrics (fast mode)")
        else:
            print(f"✓ Plot created using on-the-fly calculation (slow mode)")
        print(f"Saved: {output_file}")
        print("=" * 80)

    else:
        parser.print_help()
        print(f"\nAvailable datasets: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)


if __name__ == "__main__":
    main()
