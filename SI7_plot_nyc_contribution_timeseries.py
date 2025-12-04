"""
SI7: NYC Contribution Percentage Timeseries

This script creates publication-quality timeseries plots showing the distribution
of NYC contributions as a percentage of Montague streamflow across an ensemble.

Features:
- Shows water year timeseries (June-May, day of water year)
- Distribution bands: 1-99% (light fill), 25-75% (darker fill), and median line
- Three modes:
  1. Single dataset: Shows contribution distribution for one ensemble
  2. Comparison: Shows difference between two ensembles using pairwise quantile comparison
  3. Multi-panel: Stationary distribution on left, stacked difference plots on right
- Optional filtering by NYC storage drought zone classification
- Optional representative year trace overlay
- Clean, publication-quality styling
- OPTIMIZED: Column-specific loading and optional weekly resampling for faster runtime

Difference Calculation:
- For each day of year, creates a matrix of quantiles (101 values from 0-100%)
  for both baseline and comparison ensembles. Then computes the pairwise
  differences at each quantile level, and plots the 1st, 25th, 50th, 75th,
  and 99th percentiles of those differences.

Drought Zone Filtering:
- Set FILTER_BY_ZONES to filter years by drought severity
- None: Include all years (default behavior)
- [6]: Only years with Drought Emergency
- [5, 6]: Only years with Drought Watch or Emergency
- [4, 5, 6]: Only years with Drought Warning, Watch, or Emergency

Usage:
    # Single dataset mode
    python SI7_plot_nyc_contribution_timeseries.py <dataset_id>

    # Comparison mode (shows difference: comparison - baseline)
    python SI7_plot_nyc_contribution_timeseries.py --comparison <baseline_id> <comparison_id>

    # Multi-panel comparison mode (stationary on left, differences on right)
    python SI7_plot_nyc_contribution_timeseries.py --multipanel

Examples:
    python SI7_plot_nyc_contribution_timeseries.py stationary_ensemble
    python SI7_plot_nyc_contribution_timeseries.py --comparison stationary_ensemble climate_adjusted_low
    python SI7_plot_nyc_contribution_timeseries.py --multipanel
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from methods.config import *
from methods.plotting.styles import DPI_HIGH

# Output directory
FIG_DIR_CONTRIBUTION = f"{FIG_DIR}/nyc_contribution_timeseries"
os.makedirs(FIG_DIR_CONTRIBUTION, exist_ok=True)

# NYC reservoir parameters
NYC_RESERVOIRS = ['cannonsville', 'pepacton', 'neversink']

# Minimum inflow threshold for filtering (MG) - same as SI6
MIN_INFLOW_THRESHOLD = 1000

# Minimum number of time periods required for a water year to be considered complete
# A full water year has 365-366 days (or ~52 weeks); this threshold filters out partial years
# at the beginning/end of the simulation period
MIN_DAYS_FOR_COMPLETE_WATER_YEAR = 360  # For daily data
MIN_WEEKS_FOR_COMPLETE_WATER_YEAR = 50  # For weekly data (~52 weeks per year)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Drought zone filtering
# Options:
#   None: Include all years (default behavior)
#   [6]: Only years with Drought Emergency
#   [5, 6]: Only years with Drought Watch or Emergency
#   [4, 5, 6]: Only years with Drought Warning, Watch, or Emergency
FILTER_BY_ZONES = None #[4,5,6]  # Set to list of zones or None for all years

# Representative year trace (shows contribution for year closest to mean)
SHOW_REPRESENTATIVE_YEAR = True

# Y-axis scaling
Y_SCALE_FIXED = True  # True for 0-100%, False for auto-scale

# Performance optimization: Use weekly resampling instead of daily
# Set to True to resample to weekly data before processing (faster but less granular)
USE_WEEKLY_RESAMPLING = False  # Default: daily data for full resolution

# Drought zone mapping (from SI6)
ZONE_NAMES = {
    6: 'Drought Emergency',
    5: 'Drought Watch',
    4: 'Drought Warning',
    3: 'Normal',
    2: 'Flood Watch',
    1: 'Flood Warning',
}


def get_water_year(date):
    """
    Get the water year for a date (June 1 - May 31).

    Water year N runs from June 1 of year N to May 31 of year N+1.
    """
    if date.month >= 6:
        return date.year
    else:
        return date.year - 1


def get_water_year_doy(date):
    """
    Get day-of-water-year (1-366) for a date.

    Day 1 = June 1, Day 366 = May 31 (leap year).
    """
    water_year = get_water_year(date)
    june1 = pd.Timestamp(year=water_year, month=6, day=1)
    return (date - june1).days + 1


def classify_water_years_by_max_zone(res_level_df):
    """
    Classify each water year (June-May) by the maximum drought zone reached.
    """
    df = res_level_df.copy()
    df['water_year'] = df.index.map(get_water_year)

    water_year_classifications = {}

    for wy in df['water_year'].unique():
        wy_data = df[df['water_year'] == wy]
        max_zone = wy_data['nyc'].max()
        water_year_classifications[wy] = max_zone

    return water_year_classifications


def get_zone_filter_label(zone_list):
    """Generate a human-readable label for a zone filter."""
    if zone_list is None:
        return "All Water Years"

    zone_labels = [ZONE_NAMES.get(z, f"Zone {z}") for z in sorted(zone_list, reverse=True)]

    if len(zone_labels) == 1:
        return f"Water Years with {zone_labels[0]}"
    else:
        return f"Water Years with {', '.join(zone_labels[:-1])}, or {zone_labels[-1]}"


def calculate_daily_contribution_percentage(data, dataset_id, zone_filter=None):
    """
    Calculate NYC contribution as percentage of Montague flow for each time period.

    Uses water years (June 1 - May 31) for analysis and day-of-water-year indexing.
    Optimized for performance with vectorized operations.
    Supports both daily and weekly resampled data.

    Returns
    -------
    all_years_data : pd.DataFrame
        DataFrame with day_of_water_year as index (1-366 for daily, 1-52 for weekly)
        and each column as a year-realization
    n_years_total : int
        Total number of water years before filtering
    n_years_filtered : int
        Number of water years after filtering
    """
    realization_ids = list(data.contribution[dataset_id].keys())

    all_series = []
    n_years_total = 0
    n_years_filtered = 0

    # Determine minimum periods threshold based on data frequency
    # Check the first realization to determine frequency
    first_real_id = realization_ids[0]
    sample_df = data.contribution[dataset_id][first_real_id]
    is_weekly = len(sample_df) < 10000  # Weekly data has ~3600 rows vs ~25000 for daily
    min_periods = MIN_WEEKS_FOR_COMPLETE_WATER_YEAR if is_weekly else MIN_DAYS_FOR_COMPLETE_WATER_YEAR

    for real_id in realization_ids:
        contribution_df = data.contribution[dataset_id][real_id]
        nyc_contribution = contribution_df['mrf_montagueTrenton_nyc']

        major_flow_df = data.major_flow[dataset_id][real_id]
        montague_flow = major_flow_df['delMontague']

        # For weekly data with sum aggregation, calculate percentage from totals
        contrib_pct = np.where(montague_flow > 0,
                               100.0 * nyc_contribution / montague_flow,
                               np.nan)

        # Vectorized water year and day-of-water-year calculation
        dates = nyc_contribution.index
        months = dates.month.values
        years = dates.year.values
        water_years_arr = np.where(months >= 6, years, years - 1)

        # Compute day-of-water-year (or week-of-water-year) vectorized using numpy
        # June 1 of water year is day 1
        june1_dates = pd.DatetimeIndex(
            pd.to_datetime({'year': water_years_arr, 'month': 6, 'day': 1})
        )
        doy_arr = (dates - june1_dates).days.values + 1

        # For weekly data, convert to week-of-water-year
        if is_weekly:
            doy_arr = (doy_arr - 1) // 7 + 1  # Convert days to weeks (1-52)

        # Build DataFrame for vectorized groupby
        df_temp = pd.DataFrame({
            'contrib_pct': contrib_pct,
            'water_year': water_years_arr,
            'doy': doy_arr
        }, index=dates)

        wy_zone_map = None
        if zone_filter is not None:
            if not hasattr(data, 'res_level') or dataset_id not in data.res_level:
                raise ValueError("Zone filtering requires res_level data.")
            res_level_df = data.res_level[dataset_id][real_id]
            wy_zone_map = classify_water_years_by_max_zone(res_level_df)

        # Group by water year
        for wy, group in df_temp.groupby('water_year'):
            if len(group) < min_periods:
                continue

            n_years_total += 1

            if zone_filter is not None:
                max_zone = wy_zone_map.get(wy)
                if max_zone not in zone_filter:
                    continue

            n_years_filtered += 1

            wy_series = pd.Series(group['contrib_pct'].values, index=group['doy'].values,
                                  name=f"r{real_id}_wy{wy}")
            wy_series = wy_series.sort_index()
            all_series.append(wy_series)

    all_years_df = pd.concat(all_series, axis=1)
    return all_years_df, n_years_total, n_years_filtered


def find_representative_year_for_zone(data, dataset_id, zone_filter=None):
    """
    Find the realization/water year with contribution ratio closest to mean.

    Returns contribution trace for the representative year.

    Optimized version: computes water years vectorized once per realization,
    avoiding repeated .map() calls inside loops.
    """
    zone_label = get_zone_filter_label(zone_filter)
    print(f"  Finding representative water year for {zone_label}...")

    realization_ids = list(data.res_level[dataset_id].keys())
    records = []

    for real_id in realization_ids:
        res_level_df = data.res_level[dataset_id][real_id]
        inflow_df = data.inflow[dataset_id][real_id]
        contribution_df = data.contribution[dataset_id][real_id]

        nyc_inflow = inflow_df[NYC_RESERVOIRS].sum(axis=1)
        nyc_contributions = contribution_df['mrf_montagueTrenton_nyc']

        # Compute water years vectorized ONCE (not inside loop)
        dates = res_level_df.index
        months = dates.month.values
        years = dates.year.values
        water_years_arr = np.where(months >= 6, years, years - 1)

        # Add water year column to work with
        res_level_df = res_level_df.copy()
        res_level_df['water_year'] = water_years_arr

        # Group by water year for efficient processing
        for wy, wy_data in res_level_df.groupby('water_year'):
            if len(wy_data) < MIN_DAYS_FOR_COMPLETE_WATER_YEAR:
                continue

            max_zone = wy_data['nyc'].max()

            if zone_filter is not None:
                if max_zone not in zone_filter:
                    continue

            max_zone_date = wy_data[wy_data['nyc'] == max_zone].index[0]
            start_date = max_zone_date - pd.DateOffset(months=6)

            inflow_total = nyc_inflow.loc[start_date:max_zone_date].sum()
            contribution_total = nyc_contributions.loc[start_date:max_zone_date].sum()

            if inflow_total <= MIN_INFLOW_THRESHOLD:
                continue

            contribution_ratio = 100.0 * contribution_total / inflow_total

            records.append({
                'realization_id': real_id,
                'water_year': wy,
                'contribution_ratio': contribution_ratio
            })

    if len(records) == 0:
        print(f"  Warning: No water years found matching zone filter {zone_filter}")
        return None

    df = pd.DataFrame(records)
    mean_ratio = df['contribution_ratio'].mean()
    df['distance_to_mean'] = abs(df['contribution_ratio'] - mean_ratio)
    closest_idx = df['distance_to_mean'].idxmin()
    closest_row = df.loc[closest_idx]

    real_id = int(closest_row['realization_id'])
    wy = int(closest_row['water_year'])

    print(f"  Representative: Realization {real_id}, WY {wy}, "
          f"Ratio {closest_row['contribution_ratio']:.1f}% (mean: {mean_ratio:.1f}%)")

    # Get contribution trace
    contribution_df = data.contribution[dataset_id][real_id]
    nyc_contribution = contribution_df['mrf_montagueTrenton_nyc']

    major_flow_df = data.major_flow[dataset_id][real_id]
    montague_flow = major_flow_df['delMontague']

    contrib_pct = np.where(montague_flow > 0,
                           100.0 * nyc_contribution / montague_flow,
                           np.nan)
    contrib_pct_series = pd.Series(contrib_pct, index=nyc_contribution.index)

    # Vectorized water year mask
    months = contrib_pct_series.index.month.values
    years = contrib_pct_series.index.year.values
    wy_arr = np.where(months >= 6, years, years - 1)
    wy_mask = wy_arr == wy

    wy_contrib_data = contrib_pct_series[wy_mask]
    doy_contrib = wy_contrib_data.index.map(get_water_year_doy)
    contribution_trace = pd.Series(wy_contrib_data.values, index=doy_contrib,
                                   name=f'r{real_id}_wy{wy}_contrib')
    contribution_trace = contribution_trace.sort_index()

    return {
        'realization_id': real_id,
        'year': wy,
        'ratio': closest_row['contribution_ratio'],
        'mean_ratio': mean_ratio,
        'contribution_trace': contribution_trace
    }


# ============================================================================
# PLOTTING HELPER FUNCTIONS
# ============================================================================

# Month boundaries for x-axis formatting (water year: June 1 - May 31)
# Daily data: days 1-366
MONTH_STARTS_WY = [1, 31, 62, 93, 123, 154, 184, 215, 246, 274, 305, 335]
MONTH_LABELS_WY = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',
                   'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']
# Weekly data: weeks 1-52
MONTH_STARTS_WY_WEEKLY = [1, 5, 9, 14, 18, 22, 27, 31, 36, 40, 44, 48]


def _format_xaxis(ax, is_weekly=False):
    """Format x-axis with month labels for water year (June-May)."""
    if is_weekly:
        ax.set_xticks(MONTH_STARTS_WY_WEEKLY)
        ax.set_xticklabels(MONTH_LABELS_WY, fontsize=10)
        ax.set_xlim(1, 52)
    else:
        ax.set_xticks(MONTH_STARTS_WY)
        ax.set_xticklabels(MONTH_LABELS_WY, fontsize=10)
        ax.set_xlim(1, 366)


def _calculate_percentiles(df):
    """Calculate percentiles (1, 25, 50, 75, 99) for each day."""
    percentiles = df.T.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    return percentiles


def _calculate_pairwise_difference_percentiles(baseline_df, comparison_df):
    """
    Calculate difference percentiles by comparing all quantiles pairwise.

    For each day:
    1. Compute quantiles (0-100%) for both baseline and comparison
    2. Compute differences at each quantile level (comparison - baseline)
    3. Return percentiles of those differences (1%, 25%, 50%, 75%, 99%)

    Vectorized implementation for performance.
    """
    common_idx = baseline_df.index.intersection(comparison_df.index)
    baseline_arr = baseline_df.loc[common_idx].values  # (n_days, n_realizations)
    comparison_arr = comparison_df.loc[common_idx].values

    quantile_levels = np.linspace(0, 1, 101)
    output_percentiles = np.array([0.01, 0.25, 0.50, 0.75, 0.99])

    # Compute quantiles for all days at once using nanquantile
    # Shape: (101, n_days)
    baseline_quantiles = np.nanquantile(baseline_arr, quantile_levels, axis=1)
    comparison_quantiles = np.nanquantile(comparison_arr, quantile_levels, axis=1)

    # Differences at each quantile level: shape (101, n_days)
    differences = comparison_quantiles - baseline_quantiles

    # Compute output percentiles of the differences for each day
    # Shape: (5, n_days)
    diff_percentiles = np.percentile(differences, output_percentiles * 100, axis=0)

    return pd.DataFrame({
        '1%': diff_percentiles[0],
        '25%': diff_percentiles[1],
        '50%': diff_percentiles[2],
        '75%': diff_percentiles[3],
        '99%': diff_percentiles[4]
    }, index=common_idx)


def _plot_contribution_bands(ax, percentiles, color='steelblue', label_prefix='',
                              representative_year=None, alpha_outer=0.2, alpha_inner=0.4):
    """Plot contribution bands with 1-99% outer band, 25-75% inner band, and median."""
    doy = percentiles.index.values

    # Plot 1-99% band (lightest)
    ax.fill_between(doy, percentiles['1%'], percentiles['99%'],
                    color=color, alpha=alpha_outer, label=f'{label_prefix}1st-99th %ile')

    # Plot 25-75% band (darker)
    ax.fill_between(doy, percentiles['25%'], percentiles['75%'],
                    color=color, alpha=alpha_inner, label=f'{label_prefix}25th-75th %ile')

    # Plot median line
    ax.plot(doy, percentiles['50%'], color=color, linewidth=2,
            label=f'{label_prefix}Median')

    # Plot representative year trace if provided
    if representative_year is not None and 'contribution_trace' in representative_year:
        trace = representative_year['contribution_trace']
        year = representative_year['year']
        ax.plot(trace.index, trace.values, color=color, linewidth=1.5,
                linestyle='--', alpha=0.8, label=f'Rep. Year ({year})')


def _plot_difference_bands(ax, diff_percentiles, color='steelblue', label_prefix='',
                           alpha_outer=0.2, alpha_inner=0.4):
    """Plot difference bands for comparison mode."""
    doy = diff_percentiles.index.values

    ax.fill_between(doy, diff_percentiles['1%'], diff_percentiles['99%'],
                    color=color, alpha=alpha_outer, label=f'{label_prefix}1st-99th %ile')

    ax.fill_between(doy, diff_percentiles['25%'], diff_percentiles['75%'],
                    color=color, alpha=alpha_inner, label=f'{label_prefix}25th-75th %ile')

    ax.plot(doy, diff_percentiles['50%'], color=color, linewidth=2,
            label=f'{label_prefix}Median')


# ============================================================================
# SINGLE DATASET MODE
# ============================================================================

def plot_single_dataset(contrib_df, dataset_id, representative_year=None,
                        zone_filter=None, n_years_total=None, n_years_filtered=None):
    """
    Plot contribution distribution for a single dataset.
    """
    print("Creating single dataset contribution plot...")

    percentiles = _calculate_percentiles(contrib_df)

    # Detect if using weekly data based on index range
    is_weekly = contrib_df.index.max() <= 52

    _, ax = plt.subplots(1, 1, figsize=(12, 5))

    _plot_contribution_bands(ax, percentiles, representative_year=representative_year)

    _format_xaxis(ax, is_weekly=is_weekly)
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('NYC Contribution to Montague Flow (%)', fontsize=12, fontweight='bold')

    if Y_SCALE_FIXED:
        ax.set_ylim(0, 100)

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=9, frameon=True, fancybox=True)

    # Annotation
    zone_label = get_zone_filter_label(zone_filter) if zone_filter else "All Water Years"
    if n_years_total is not None and n_years_filtered is not None:
        annotation_text = f'{zone_label}\nn = {n_years_filtered} / {n_years_total} water year-realizations'
    else:
        annotation_text = f'{zone_label}\nn = {contrib_df.shape[1]} water year-realizations'

    ax.text(0.02, 0.98, annotation_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    zone_suffix = '_zones_' + '_'.join(map(str, sorted(zone_filter, reverse=True))) if zone_filter else ''
    fname = f"{FIG_DIR_CONTRIBUTION}/{dataset_id}_contribution{zone_suffix}.png"
    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


def main_single(dataset_id):
    """Main function for single dataset mode."""
    print("=" * 80)
    print(f"NYC CONTRIBUTION TIMESERIES: {dataset_id}")
    print("=" * 80)

    print("\nConfiguration:")
    print(f"  Zone Filter: {get_zone_filter_label(FILTER_BY_ZONES)}")
    print(f"  Show Representative Year: {SHOW_REPRESENTATIVE_YEAR}")

    verify_dataset_id(dataset_id)

    # Determine which results sets to load
    results_sets = ['contribution', 'major_flow']
    if FILTER_BY_ZONES is not None or SHOW_REPRESENTATIVE_YEAR:
        results_sets.append('res_level')
    if SHOW_REPRESENTATIVE_YEAR:
        results_sets.append('inflow')

    print("\nLoading data...")
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Data not found: {fname}")

    data = pywrdrb.Data()
    data.load_from_export(fname, results_sets=results_sets)
    print(f"  Data loaded: {dataset_id}")

    print("\nCalculating contribution percentages...")
    contrib_df, n_years_total, n_years_filtered = calculate_daily_contribution_percentage(
        data, dataset_id, zone_filter=FILTER_BY_ZONES
    )
    print(f"  {n_years_filtered} / {n_years_total} year-realizations")

    representative_year = None
    if SHOW_REPRESENTATIVE_YEAR:
        print("\nFinding representative year...")
        representative_year = find_representative_year_for_zone(
            data, dataset_id, zone_filter=FILTER_BY_ZONES
        )

    print("\nCreating plot...")
    plot_single_dataset(
        contrib_df=contrib_df,
        dataset_id=dataset_id,
        representative_year=representative_year,
        zone_filter=FILTER_BY_ZONES,
        n_years_total=n_years_total,
        n_years_filtered=n_years_filtered
    )

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)


# ============================================================================
# COMPARISON MODE
# ============================================================================

def plot_comparison(baseline_contrib, comparison_contrib,
                    baseline_id, comparison_id,
                    zone_filter=None):
    """
    Create comparison plot showing differences between two ensembles.
    """
    print("Creating comparison plot with pairwise quantile differences...")

    print("  Computing pairwise quantile differences...")
    contrib_diff = _calculate_pairwise_difference_percentiles(baseline_contrib, comparison_contrib)

    # Detect if using weekly data based on index range
    is_weekly = baseline_contrib.index.max() <= 52

    _, ax = plt.subplots(1, 1, figsize=(12, 5))

    _plot_difference_bands(ax, contrib_diff, color='steelblue', label_prefix='')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    _format_xaxis(ax, is_weekly=is_weekly)
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Contribution Change (% points)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    ax.legend(loc='upper right', fontsize=8, frameon=True, fancybox=True)

    zone_label = get_zone_filter_label(zone_filter) if zone_filter else "All Years"
    annotation_text = f'Change: {comparison_id}\nrelative to {baseline_id}\n{zone_label}'
    ax.text(0.02, 0.98, annotation_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    zone_suffix = '_zones_' + '_'.join(map(str, sorted(zone_filter, reverse=True))) if zone_filter else ''
    fname = f"{FIG_DIR_CONTRIBUTION}/{comparison_id}_vs_{baseline_id}_comparison{zone_suffix}.png"
    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


def main_comparison(baseline_id, comparison_id):
    """Main function for comparison mode."""
    print("=" * 80)
    print(f"NYC CONTRIBUTION TIMESERIES COMPARISON")
    print(f"  Baseline: {baseline_id}")
    print(f"  Comparison: {comparison_id}")
    print("=" * 80)

    print("\nConfiguration:")
    print(f"  Zone Filter: {get_zone_filter_label(FILTER_BY_ZONES)}")
    print(f"  Weekly Resampling: {USE_WEEKLY_RESAMPLING}")

    verify_dataset_id(baseline_id)
    verify_dataset_id(comparison_id)

    need_res_level = FILTER_BY_ZONES is not None

    print("\nLoading baseline data (optimized)...")
    baseline_fname = f'./pywrdrb/outputs/{baseline_id}_with_postprocessing.hdf5'
    if not os.path.exists(baseline_fname):
        raise FileNotFoundError(f"Baseline data not found: {baseline_fname}")

    baseline_data = OptimizedDataContainer()
    baseline_data.load_for_contribution_analysis(
        baseline_fname, baseline_id,
        need_res_level=need_res_level,
        need_inflow=False,
        use_weekly=USE_WEEKLY_RESAMPLING
    )
    print(f"  Baseline loaded: {baseline_id}")

    print("\nLoading comparison data (optimized)...")
    comparison_fname = f'./pywrdrb/outputs/{comparison_id}_with_postprocessing.hdf5'
    if not os.path.exists(comparison_fname):
        raise FileNotFoundError(f"Comparison data not found: {comparison_fname}")

    comparison_data = OptimizedDataContainer()
    comparison_data.load_for_contribution_analysis(
        comparison_fname, comparison_id,
        need_res_level=need_res_level,
        need_inflow=False,
        use_weekly=USE_WEEKLY_RESAMPLING
    )
    print(f"  Comparison loaded: {comparison_id}")

    print("\nCalculating contribution percentages...")
    baseline_contrib, n_base_total, n_base_filtered = calculate_daily_contribution_percentage(
        baseline_data, baseline_id, zone_filter=FILTER_BY_ZONES
    )
    print(f"  Baseline: {n_base_filtered} / {n_base_total} year-realizations")

    comparison_contrib, n_comp_total, n_comp_filtered = calculate_daily_contribution_percentage(
        comparison_data, comparison_id, zone_filter=FILTER_BY_ZONES
    )
    print(f"  Comparison: {n_comp_filtered} / {n_comp_total} year-realizations")

    print("\nCreating comparison plot...")
    plot_comparison(
        baseline_contrib=baseline_contrib,
        comparison_contrib=comparison_contrib,
        baseline_id=baseline_id,
        comparison_id=comparison_id,
        zone_filter=FILTER_BY_ZONES
    )

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE!")
    print("=" * 80)


# ============================================================================
# MULTI-PANEL COMPARISON MODE
# ============================================================================

def _load_and_process_dataset(args):
    """Helper function to load and process a single dataset (for parallel execution)."""
    dataset_id, label, zone_filter, use_weekly = args

    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Data not found: {fname}")

    # Use optimized loading
    data = OptimizedDataContainer()
    data.load_for_contribution_analysis(
        fname, dataset_id,
        need_res_level=(zone_filter is not None),
        need_inflow=False,
        use_weekly=use_weekly
    )

    contrib_df, n_total, n_filtered = calculate_daily_contribution_percentage(
        data, dataset_id, zone_filter=zone_filter
    )

    return dataset_id, label, contrib_df, n_total, n_filtered


def plot_multipanel_comparison(zone_filter=None, figsize=(12, 6)):
    """
    Create a 3-panel comparison figure.

    Layout:
    - Left panel: Stationary ensemble (absolute distribution)
    - Right panels (stacked): Low, High climate scenarios (difference from stationary)
    """
    from concurrent.futures import ThreadPoolExecutor

    print("=" * 80)
    print("Creating Multi-Panel NYC Contribution Comparison")
    print("=" * 80)

    print("\nConfiguration:")
    print(f"  Zone Filter: {get_zone_filter_label(zone_filter)}")
    print(f"  Weekly Resampling: {USE_WEEKLY_RESAMPLING}")

    datasets = {
        'stationary_ensemble': 'Stationary',
        'climate_adjusted_low': 'Low Climate',
        'climate_adjusted_high': 'High Climate'
    }

    # Load all datasets in parallel (using optimized loading)
    print("\nLoading datasets in parallel (optimized)...")
    all_contrib_dfs = {}

    args_list = [(dataset_id, label, zone_filter, USE_WEEKLY_RESAMPLING) for dataset_id, label in datasets.items()]

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(_load_and_process_dataset, args_list))

    for dataset_id, label, contrib_df, n_total, n_filtered in results:
        print(f"  {dataset_id} ({label}): {n_filtered} / {n_total} year-realizations")
        all_contrib_dfs[dataset_id] = contrib_df

    # Calculate percentiles for stationary
    print("\nCalculating percentiles...")
    stationary_percentiles = _calculate_percentiles(all_contrib_dfs['stationary_ensemble'])

    # Calculate differences for climate scenarios
    print("Computing pairwise differences...")
    diff_low = _calculate_pairwise_difference_percentiles(
        all_contrib_dfs['stationary_ensemble'],
        all_contrib_dfs['climate_adjusted_low']
    )
    diff_high = _calculate_pairwise_difference_percentiles(
        all_contrib_dfs['stationary_ensemble'],
        all_contrib_dfs['climate_adjusted_high']
    )

    # Detect if using weekly data based on index range
    is_weekly = all_contrib_dfs['stationary_ensemble'].index.max() <= 52

    # Create figure with GridSpec
    print("\nCreating multi-panel figure...")
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
                          hspace=0.08, wspace=0.25,
                          left=0.08, right=0.95, top=0.95, bottom=0.12)

    # Create axes
    ax_stat = fig.add_subplot(gs[:, 0])  # Left panel spans both rows
    ax_high = fig.add_subplot(gs[0, 1])  # Top right
    ax_low = fig.add_subplot(gs[1, 1], sharex=ax_high)   # Bottom right shares x with top

    # Hide x-tick labels on top right panel
    plt.setp(ax_high.get_xticklabels(), visible=False)

    # Left panel: Stationary distribution
    _plot_contribution_bands(ax_stat, stationary_percentiles, color='steelblue')
    _format_xaxis(ax_stat, is_weekly=is_weekly)
    ax_stat.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax_stat.set_ylabel('NYC Contribution to Montague Flow (%)', fontsize=12, fontweight='bold')
    if Y_SCALE_FIXED:
        ax_stat.set_ylim(0, 100)
    ax_stat.grid(axis='y', alpha=0.3, linestyle='--')
    ax_stat.set_axisbelow(True)
    ax_stat.set_title('Stationary Ensemble', fontsize=11, fontweight='bold')
    ax_stat.legend(loc='upper right', fontsize=8, frameon=True, fancybox=True)

    # Top right panel: High climate difference
    _plot_difference_bands(ax_high, diff_high, color='firebrick', label_prefix='')
    ax_high.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax_high.set_ylabel('Δ Contribution\n(% points)', fontsize=10, fontweight='bold')
    ax_high.grid(axis='y', alpha=0.3, linestyle='--')
    ax_high.set_axisbelow(True)
    ax_high.set_title('High Climate - Stationary', fontsize=10, fontweight='bold')
    ax_high.legend(loc='upper right', fontsize=7, frameon=True, fancybox=True)

    # Bottom right panel: Low climate difference
    _plot_difference_bands(ax_low, diff_low, color='darkorange', label_prefix='')
    ax_low.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    _format_xaxis(ax_low, is_weekly=is_weekly)
    ax_low.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax_low.set_ylabel('Δ Contribution\n(% points)', fontsize=10, fontweight='bold')
    ax_low.grid(axis='y', alpha=0.3, linestyle='--')
    ax_low.set_axisbelow(True)
    ax_low.set_title('Low Climate - Stationary', fontsize=10, fontweight='bold')
    ax_low.legend(loc='upper right', fontsize=7, frameon=True, fancybox=True)

    # Match y-axis limits for difference panels
    y_min = min(ax_high.get_ylim()[0], ax_low.get_ylim()[0])
    y_max = max(ax_high.get_ylim()[1], ax_low.get_ylim()[1])
    ax_high.set_ylim(y_min, y_max)
    ax_low.set_ylim(y_min, y_max)

    # Annotation
    zone_label = get_zone_filter_label(zone_filter) if zone_filter else "All Water Years"
    fig.text(0.5, 0.02, zone_label, ha='center', va='bottom', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Save
    zone_suffix = '_zones_' + '_'.join(map(str, sorted(zone_filter, reverse=True))) if zone_filter else ''
    fname = f"{FIG_DIR_CONTRIBUTION}/multipanel_contribution_comparison{zone_suffix}.png"
    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"\nSaved: {fname}")

    # Also save SVG
    base = fname.rsplit('.', 1)[0]
    plt.savefig(f"{base}.svg", bbox_inches='tight')
    print(f"Saved: {base}.svg")

    plt.close()

    print("\n" + "=" * 80)
    print("MULTI-PANEL COMPARISON COMPLETE!")
    print("=" * 80)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='NYC Contribution Percentage Timeseries Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Single dataset mode
  python SI7_plot_nyc_contribution_timeseries.py stationary_ensemble

  # Comparison mode (shows difference: comparison - baseline)
  python SI7_plot_nyc_contribution_timeseries.py --comparison stationary_ensemble climate_adjusted_low

  # Multi-panel comparison (stationary on left, differences on right)
  python SI7_plot_nyc_contribution_timeseries.py --multipanel

Available datasets: {list(DATASET_CONFIGS.keys())}
        """
    )

    parser.add_argument(
        'dataset_id',
        nargs='?',
        help='Dataset identifier for single dataset mode'
    )

    parser.add_argument(
        '--comparison', '-c',
        nargs=2,
        metavar=('BASELINE', 'COMPARISON'),
        help='Comparison mode: show differences between two ensembles (comparison - baseline)'
    )

    parser.add_argument(
        '--multipanel', '-m',
        action='store_true',
        help='Multi-panel comparison mode: stationary on left, stacked differences on right'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.multipanel:
        # Multi-panel comparison mode
        plot_multipanel_comparison(zone_filter=FILTER_BY_ZONES)
    elif args.comparison:
        # Comparison mode
        baseline_id, comparison_id = args.comparison
        verify_dataset_id(baseline_id)
        verify_dataset_id(comparison_id)
        main_comparison(baseline_id, comparison_id)
    elif args.dataset_id:
        # Single dataset mode
        verify_dataset_id(args.dataset_id)
        main_single(args.dataset_id)
    else:
        print(__doc__)
        print(f"\nAvailable datasets: {list(DATASET_CONFIGS.keys())}")
