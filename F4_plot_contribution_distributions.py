"""
F4: NYC Contribution Percentage Timeseries

Timeseries plots showing the distribution of NYC contributions as a percentage
of Montague streamflow across an ensemble.

Modes:
  1. Single dataset: Shows contribution distribution for one ensemble
  2. Comparison: Shows difference between two ensembles
  3. Multi-panel: Stationary distribution on left, stacked difference plots on right

Usage:
    python F4_plot_contribution_distributions.py <dataset_id>
    python F4_plot_contribution_distributions.py --comparison <baseline_id> <comparison_id>
    python F4_plot_contribution_distributions.py --multipanel
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from methods.config import FIG_DIR, DATASET_CONFIGS, verify_dataset_id
from methods.plotting.styles import DPI_HIGH, DATASET_COLORS, DATASET_LABELS

# Output directory
FIG_OUTPUT_DIR = f"{FIG_DIR}/F4_contribution_timeseries"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

NYC_RESERVOIRS = ['cannonsville', 'pepacton', 'neversink']

# Minimum inflow threshold for filtering (MG) - same as SI6
MIN_INFLOW_THRESHOLD = 1000

# Minimum number of days required for a water year to be considered complete
# A full water year has 365-366 days; this threshold filters out partial years
# at the beginning/end of the simulation period
MIN_DAYS_FOR_COMPLETE_WATER_YEAR = 360

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

# Multi-panel layout: 'side_by_side' (2-col) or 'stacked' (3-row vertical)
MULTIPANEL_LAYOUT = 'stacked'

# Representative year trace (shows contribution for year closest to mean)
SHOW_REPRESENTATIVE_YEAR = True

# Y-axis scaling
Y_SCALE_FIXED = True  # True for 0-100%, False for auto-scale

# Drought zone mapping (from SI6)
ZONE_NAMES = {
    6: 'Drought Emergency',
    5: 'Drought Warning',
    4: 'Drought Watch',
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
    Calculate NYC contribution as percentage of Montague flow for each day.

    Uses water years (June 1 - May 31) for analysis and day-of-water-year indexing.

    Returns
    -------
    all_years_data : pd.DataFrame
        DataFrame with day_of_water_year as index (1-366) and each column as a year-realization
    n_years_total : int
        Total number of water years before filtering
    n_years_filtered : int
        Number of water years after filtering
    """
    realization_ids = list(data.contribution[dataset_id].keys())

    # Pre-allocate: collect (doy, contrib_pct) per water-year into a 2D array
    # Each accepted water year becomes one column of shape (366,) with NaN fill
    col_arrays = []
    n_years_total = 0
    n_years_filtered = 0

    for real_id in realization_ids:
        contribution_df = data.contribution[dataset_id][real_id]
        nyc_contribution = contribution_df['mrf_montagueTrenton_nyc']

        major_flow_df = data.major_flow[dataset_id][real_id]
        montague_flow = major_flow_df['delMontague']

        contrib_pct = np.where(montague_flow > 0,
                               100.0 * nyc_contribution / montague_flow,
                               np.nan)

        # Vectorized water year and day-of-water-year
        dates = nyc_contribution.index
        months = dates.month.values
        years = dates.year.values
        water_years_arr = np.where(months >= 6, years, years - 1)

        june1_years = water_years_arr
        june1_ordinals = (
            pd.Timestamp(1970, 6, 1).toordinal()
            + (june1_years - 1970) * 365
            + ((june1_years - 1968) // 4)  # leap year approx
        )
        # Accurate: use numpy datetime64 for speed
        june1_dates_np = np.array(
            [np.datetime64(f'{y}-06-01') for y in june1_years],
            dtype='datetime64[D]'
        )
        dates_np = dates.values.astype('datetime64[D]')
        doy_arr = (dates_np - june1_dates_np).astype(int) + 1

        # Zone filter map (once per realization)
        wy_zone_map = None
        if zone_filter is not None:
            if not hasattr(data, 'res_level') or dataset_id not in data.res_level:
                raise ValueError("Zone filtering requires res_level data.")
            res_level_df = data.res_level[dataset_id][real_id]
            wy_zone_map = classify_water_years_by_max_zone(res_level_df)

        # Process water years vectorized: find unique water years and their boundaries
        unique_wys, wy_indices = np.unique(water_years_arr, return_inverse=True)

        for wy_idx, wy in enumerate(unique_wys):
            mask = wy_indices == wy_idx
            n_days = mask.sum()

            if n_days < MIN_DAYS_FOR_COMPLETE_WATER_YEAR:
                continue

            n_years_total += 1

            if zone_filter is not None:
                max_zone = wy_zone_map.get(wy)
                if max_zone not in zone_filter:
                    continue

            n_years_filtered += 1

            # Scatter into a 366-element array by doy
            col = np.full(366, np.nan)
            wy_doy = doy_arr[mask]
            wy_pct = contrib_pct[mask]
            valid = (wy_doy >= 1) & (wy_doy <= 366)
            col[wy_doy[valid] - 1] = wy_pct[valid]
            col_arrays.append(col)

    # Build DataFrame in one shot from stacked array
    if col_arrays:
        result = pd.DataFrame(
            np.column_stack(col_arrays),
            index=np.arange(1, 367),
        )
    else:
        result = pd.DataFrame(index=np.arange(1, 367))

    return result, n_years_total, n_years_filtered


def find_representative_year_for_zone(data, dataset_id, zone_filter=None):
    """
    Find the realization/water year with contribution ratio closest to mean.

    Returns contribution trace for the representative year.

    Optimized version: computes water years vectorized once per realization,
    avoiding repeated .map() calls inside loops.
    """
    zone_label = get_zone_filter_label(zone_filter)

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
        return None

    df = pd.DataFrame(records)
    mean_ratio = df['contribution_ratio'].mean()
    df['distance_to_mean'] = abs(df['contribution_ratio'] - mean_ratio)
    closest_idx = df['distance_to_mean'].idxmin()
    closest_row = df.loc[closest_idx]

    real_id = int(closest_row['realization_id'])
    wy = int(closest_row['water_year'])


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
MONTH_STARTS_WY = [1, 31, 62, 93, 123, 154, 184, 215, 246, 274, 305, 335]
MONTH_LABELS_WY = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',
                   'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']


def _format_xaxis(ax):
    """Format x-axis with month labels for water year (June-May)."""
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
    """Plot contribution distribution for a single dataset."""

    percentiles = _calculate_percentiles(contrib_df)

    _, ax = plt.subplots(1, 1, figsize=(12, 5))

    _plot_contribution_bands(ax, percentiles, representative_year=representative_year)

    _format_xaxis(ax)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('NYC Contribution to Montague Flow (%)', fontsize=12)

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
    fname = f"{FIG_OUTPUT_DIR}/F4_{dataset_id}_contribution{zone_suffix}.png"
    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close()


def main_single(dataset_id):
    """Main function for single dataset mode."""
    print(f"F4: NYC contribution timeseries - {dataset_id}")

    verify_dataset_id(dataset_id)

    results_sets = ['contribution', 'major_flow']
    if FILTER_BY_ZONES is not None or SHOW_REPRESENTATIVE_YEAR:
        results_sets.append('res_level')
    if SHOW_REPRESENTATIVE_YEAR:
        results_sets.append('inflow')

    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Data not found: {fname}")

    data = pywrdrb.Data()
    data.load_from_export(fname, results_sets=results_sets)

    contrib_df, n_years_total, n_years_filtered = calculate_daily_contribution_percentage(
        data, dataset_id, zone_filter=FILTER_BY_ZONES
    )

    representative_year = None
    if SHOW_REPRESENTATIVE_YEAR:
        representative_year = find_representative_year_for_zone(
            data, dataset_id, zone_filter=FILTER_BY_ZONES
        )

    plot_single_dataset(
        contrib_df=contrib_df,
        dataset_id=dataset_id,
        representative_year=representative_year,
        zone_filter=FILTER_BY_ZONES,
        n_years_total=n_years_total,
        n_years_filtered=n_years_filtered
    )


# ============================================================================
# COMPARISON MODE
# ============================================================================

def plot_comparison(baseline_contrib, comparison_contrib,
                    baseline_id, comparison_id,
                    zone_filter=None):
    """Create comparison plot showing differences between two ensembles."""
    contrib_diff = _calculate_pairwise_difference_percentiles(baseline_contrib, comparison_contrib)

    _, ax = plt.subplots(1, 1, figsize=(12, 5))

    _plot_difference_bands(ax, contrib_diff, color='steelblue', label_prefix='')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    _format_xaxis(ax)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Contribution Change (% points)', fontsize=12)
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
    fname = f"{FIG_OUTPUT_DIR}/F4_{comparison_id}_vs_{baseline_id}_comparison{zone_suffix}.png"
    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close()


def main_comparison(baseline_id, comparison_id):
    """Main function for comparison mode."""
    print(f"F4: Contribution comparison - {comparison_id} vs {baseline_id}")

    verify_dataset_id(baseline_id)
    verify_dataset_id(comparison_id)

    results_sets = ['contribution', 'major_flow']
    if FILTER_BY_ZONES is not None:
        results_sets.append('res_level')

    baseline_fname = f'./pywrdrb/outputs/{baseline_id}_with_postprocessing.hdf5'
    if not os.path.exists(baseline_fname):
        raise FileNotFoundError(f"Baseline data not found: {baseline_fname}")

    baseline_data = pywrdrb.Data()
    baseline_data.load_from_export(baseline_fname, results_sets=results_sets)

    comparison_fname = f'./pywrdrb/outputs/{comparison_id}_with_postprocessing.hdf5'
    if not os.path.exists(comparison_fname):
        raise FileNotFoundError(f"Comparison data not found: {comparison_fname}")

    comparison_data = pywrdrb.Data()
    comparison_data.load_from_export(comparison_fname, results_sets=results_sets)

    baseline_contrib, n_base_total, n_base_filtered = calculate_daily_contribution_percentage(
        baseline_data, baseline_id, zone_filter=FILTER_BY_ZONES
    )

    comparison_contrib, n_comp_total, n_comp_filtered = calculate_daily_contribution_percentage(
        comparison_data, comparison_id, zone_filter=FILTER_BY_ZONES
    )

    plot_comparison(
        baseline_contrib=baseline_contrib,
        comparison_contrib=comparison_contrib,
        baseline_id=baseline_id,
        comparison_id=comparison_id,
        zone_filter=FILTER_BY_ZONES
    )


# ============================================================================
# MULTI-PANEL COMPARISON MODE
# ============================================================================

def _load_and_process_dataset(args):
    """Helper function to load and process a single dataset (for parallel execution)."""
    dataset_id, label, zone_filter = args

    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Data not found: {fname}")

    # Determine which results sets to load
    results_sets = ['contribution', 'major_flow']
    if zone_filter is not None:
        results_sets.append('res_level')

    data = pywrdrb.Data()
    data.load_from_export(fname, results_sets=results_sets)

    contrib_df, n_total, n_filtered = calculate_daily_contribution_percentage(
        data, dataset_id, zone_filter=zone_filter
    )

    return dataset_id, label, contrib_df, n_total, n_filtered


def plot_multipanel_comparison(zone_filter=None, figsize=None, layout=None):
    """
    Create a 3-panel comparison figure.

    Parameters
    ----------
    zone_filter : list of int, optional
        Drought zones to filter by.
    figsize : tuple, optional
        Figure size. If None, chosen based on layout.
    layout : str, optional
        'side_by_side' (stationary left, deltas stacked right) or
        'stacked' (all 3 panels stacked vertically).
        If None, uses module-level MULTIPANEL_LAYOUT.
    """
    if layout is None:
        layout = MULTIPANEL_LAYOUT

    print(f"F4: Multi-panel contribution comparison (layout={layout})")

    datasets = {
        'stationary_ensemble': DATASET_LABELS.get('stationary_ensemble', 'Stationary'),
        'climate_adjusted_low': DATASET_LABELS.get('climate_adjusted_low', 'Climate Low'),
        'climate_adjusted_high': DATASET_LABELS.get('climate_adjusted_high', 'Climate High')
    }

    all_contrib_dfs = {}

    for dataset_id, label in datasets.items():
        load_args = (dataset_id, label, zone_filter)
        dataset_id, label, contrib_df, n_total, n_filtered = _load_and_process_dataset(load_args)
        all_contrib_dfs[dataset_id] = contrib_df

    stationary_percentiles = _calculate_percentiles(all_contrib_dfs['stationary_ensemble'])

    diff_low = _calculate_pairwise_difference_percentiles(
        all_contrib_dfs['stationary_ensemble'],
        all_contrib_dfs['climate_adjusted_low']
    )
    diff_high = _calculate_pairwise_difference_percentiles(
        all_contrib_dfs['stationary_ensemble'],
        all_contrib_dfs['climate_adjusted_high']
    )

    # ---- Layout-specific figure construction ----
    if layout == 'stacked':
        if figsize is None:
            figsize = (12, 10)
        fig, (ax_stat, ax_low, ax_high) = plt.subplots(
            3, 1, figsize=figsize, sharex=True,
            gridspec_kw={'hspace': 0.12,
                         'left': 0.08, 'right': 0.95,
                         'top': 0.96, 'bottom': 0.08},
        )
        # Hide x-tick labels on top two panels
        plt.setp(ax_stat.get_xticklabels(), visible=False)
        plt.setp(ax_low.get_xticklabels(), visible=False)

    else:  # side_by_side (original)
        if figsize is None:
            figsize = (12, 6)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
                              hspace=0.08, wspace=0.25,
                              left=0.08, right=0.95, top=0.95, bottom=0.12)
        ax_stat = fig.add_subplot(gs[:, 0])
        ax_low = fig.add_subplot(gs[0, 1])
        ax_high = fig.add_subplot(gs[1, 1], sharex=ax_low)
        plt.setp(ax_low.get_xticklabels(), visible=False)

    # ---- Panel content (shared across layouts) ----

    # (a) Stationary distribution
    _plot_contribution_bands(ax_stat, stationary_percentiles, color=DATASET_COLORS['stationary_ensemble'])
    _format_xaxis(ax_stat)
    ax_stat.set_ylabel('NYC Contribution to Montague Flow (%)', fontsize=12)
    if Y_SCALE_FIXED:
        ax_stat.set_ylim(0, 100)
    ax_stat.grid(axis='y', alpha=0.3, linestyle='--')
    ax_stat.set_axisbelow(True)
    stat_label = DATASET_LABELS.get('stationary_ensemble', 'Stationary')
    ax_stat.text(0.02, 0.97, f'(a) {stat_label}', transform=ax_stat.transAxes, fontsize=12, va='top', ha='left')

    # (b) Low climate difference
    _plot_difference_bands(ax_low, diff_low, color=DATASET_COLORS['climate_adjusted_low'], label_prefix='')
    ax_low.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax_low.set_ylabel('Change in Distribution\nRelative to Baseline Ensemble', fontsize=10)
    ax_low.grid(axis='y', alpha=0.3, linestyle='--')
    ax_low.set_axisbelow(True)
    low_label = DATASET_LABELS.get('climate_adjusted_low', 'Climate Low')
    ax_low.text(0.02, 0.97, f'(b) {low_label}', transform=ax_low.transAxes, fontsize=12, va='top', ha='left')

    # (c) High climate difference
    _plot_difference_bands(ax_high, diff_high, color=DATASET_COLORS['climate_adjusted_high'], label_prefix='')
    ax_high.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    _format_xaxis(ax_high)
    ax_high.set_xlabel('Month', fontsize=12)
    ax_high.set_ylabel('Change in Distribution\nRelative to Baseline Ensemble', fontsize=10)
    ax_high.grid(axis='y', alpha=0.3, linestyle='--')
    ax_high.set_axisbelow(True)
    high_label = DATASET_LABELS.get('climate_adjusted_high', 'Climate High')
    ax_high.text(0.02, 0.97, f'(c) {high_label}', transform=ax_high.transAxes, fontsize=12, va='top', ha='left')

    # x-label only on stationary panel for side_by_side layout
    if layout != 'stacked':
        ax_stat.set_xlabel('Month', fontsize=12)

    # Shared grey-scale legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='grey', alpha=0.2, label='Scenario 1st-99th %ile'),
        Patch(facecolor='grey', alpha=0.4, label='Scenario 25th-75th %ile'),
        Line2D([0], [0], color='grey', linewidth=2, label='Scenario Median'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.03))

    # Match y-axis limits for difference panels
    y_min = min(ax_high.get_ylim()[0], ax_low.get_ylim()[0])
    y_max = max(ax_high.get_ylim()[1], ax_low.get_ylim()[1])
    ax_high.set_ylim(y_min, y_max)
    ax_low.set_ylim(y_min, y_max)

    # Save
    zone_suffix = '_zones_' + '_'.join(map(str, sorted(zone_filter, reverse=True))) if zone_filter else ''
    layout_suffix = f'_{layout}' if layout != 'side_by_side' else ''
    fname = f"{FIG_OUTPUT_DIR}/F4_multipanel_contribution_comparison{layout_suffix}{zone_suffix}.png"
    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")

    plt.close()


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
  python F4_plot_contribution_distributions.py stationary_ensemble
  python F4_plot_contribution_distributions.py --comparison stationary_ensemble climate_adjusted_low
  python F4_plot_contribution_distributions.py --multipanel

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
