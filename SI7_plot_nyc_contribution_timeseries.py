"""
SI7: NYC Contribution Percentage Timeseries

This script creates a publication-quality timeseries plot showing the distribution
of NYC contributions as a percentage of Montague streamflow across an ensemble.

Features:
- Shows Jan-Dec timeseries (day of year)
- Distribution bands: 5-95% (light fill), 25-75% (darker fill), and median line
- Aggregates across all realizations and years in the ensemble
- Optional filtering by NYC storage drought zone classification
- Clean, publication-quality styling
- Comparison mode to show differences between two ensembles

Drought Zone Filtering:
- Set FILTER_BY_ZONES to filter years by drought severity
- None: Include all years (default behavior)
- [6]: Only years with Drought Emergency
- [5, 6]: Only years with Drought Watch or Emergency
- [4, 5, 6]: Only years with Drought Warning, Watch, or Emergency

Configuration:
- Edit FILTER_BY_ZONES constant in the script to change zone filtering

Usage:
    # Single ensemble mode
    python SI7_plot_nyc_contribution_timeseries.py <dataset_id>

    # Comparison mode (shows difference: comparison - baseline)
    python SI7_plot_nyc_contribution_timeseries.py --comparison <baseline_id> <comparison_id>

Examples:
    python SI7_plot_nyc_contribution_timeseries.py stationary_ensemble
    python SI7_plot_nyc_contribution_timeseries.py --comparison stationary_ensemble climate_adjusted_low
"""

import sys
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
from methods.config import RECONSTRUCTION_OUTPUT_FNAME, NYC_TOTAL_CAPACITY

# Output directory
FIG_DIR_CONTRIBUTION = f"{FIG_DIR}/nyc_contribution_timeseries"
os.makedirs(FIG_DIR_CONTRIBUTION, exist_ok=True)

# NYC reservoir parameters
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

# Drought zone filtering (applies to BOTH contribution and storage)
# Options:
#   None: Include all years (default behavior)
#   [6]: Only years with Drought Emergency
#   [5, 6]: Only years with Drought Watch or Emergency
#   [4, 5, 6]: Only years with Drought Warning, Watch, or Emergency
FILTER_BY_ZONES = [4,5,6]  # Set to list of zones or None for all years

# Storage display mode
SHOW_NYC_STORAGE = True          # Set to False to show contribution only
STORAGE_SUBPLOT = True           # True: separate subplot (storage top, contribution bottom)
                                 # False: dual-axis overlay on single plot

# Representative year trace (shows BOTH contribution AND storage for that year)
SHOW_REPRESENTATIVE_YEAR = True  # Show trace for year closest to mean

# Y-axis scaling (both axes)
Y_SCALE_FIXED = True  # True for 0-100%, False for auto-scale

# Linked traces mode (subplot mode only)
# When True, plots individual year traces color-coded by mean contribution
# This allows visual linkage between storage and contribution panels
SHOW_LINKED_TRACES = True  # Set to False for standard percentile bands
SMOOTHING_WINDOW = 7  # Days for rolling mean smoothing of contribution traces

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

    Parameters
    ----------
    date : datetime-like
        Date to get water year for

    Returns
    -------
    int
        Water year
    """
    if date.month >= 6:
        return date.year
    else:
        return date.year - 1


def get_water_year_doy(date):
    """
    Get day-of-water-year (1-366) for a date.

    Day 1 = June 1, Day 366 = May 31 (leap year).

    Parameters
    ----------
    date : datetime-like
        Date to convert

    Returns
    -------
    int
        Day of water year (1-366)
    """
    water_year = get_water_year(date)
    june1 = pd.Timestamp(year=water_year, month=6, day=1)
    return (date - june1).days + 1


def classify_water_years_by_max_zone(res_level_df):
    """
    Classify each water year (June-May) by the maximum drought zone reached.

    Water year N runs from June 1 of year N to May 31 of year N+1.
    This is more appropriate for drought analysis since droughts typically
    develop over summer/fall (Sept-Dec).

    Parameters
    ----------
    res_level_df : pd.DataFrame
        Reservoir level DataFrame with 'nyc' column and datetime index

    Returns
    -------
    water_year_classifications : dict
        Dictionary mapping water_year -> max_zone (int)
    """
    df = res_level_df.copy()
    # Assign water year to each date
    df['water_year'] = df.index.map(get_water_year)

    water_year_classifications = {}

    for wy in df['water_year'].unique():
        wy_data = df[df['water_year'] == wy]
        # Find maximum zone value (higher zone = more severe drought)
        max_zone = wy_data['nyc'].max()
        water_year_classifications[wy] = max_zone

    return water_year_classifications


def get_zone_filter_label(zone_list):
    """
    Generate a human-readable label for a zone filter.

    Parameters
    ----------
    zone_list : list of int
        List of zone numbers

    Returns
    -------
    str
        Human-readable label
    """
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

    Parameters
    ----------
    data : pywrdrb.Data
        Data object containing contribution, major_flow, and optionally res_level
    dataset_id : str
        Dataset identifier
    zone_filter : list of int or None
        If provided, only include water years where NYC reservoirs experienced
        one of these drought zones. Requires res_level data to be loaded.
        None means include all water years.

    Returns
    -------
    all_years_data : pd.DataFrame
        DataFrame with day_of_water_year as index (1-366, starting June 1)
        and each column as a water_year-realization
    n_years_total : int
        Total number of water years before filtering
    n_years_filtered : int
        Number of water years after filtering
    """
    realization_ids = list(data.contribution[dataset_id].keys())

    all_series = []
    n_years_total = 0
    n_years_filtered = 0

    for real_id in realization_ids:
        # Get contribution data
        contribution_df = data.contribution[dataset_id][real_id]
        nyc_contribution = contribution_df['mrf_montagueTrenton_nyc']

        # Get Montague flow from major_flow
        major_flow_df = data.major_flow[dataset_id][real_id]
        montague_flow = major_flow_df['delMontague']

        # Calculate percentage (handle division by zero)
        contrib_pct = np.where(montague_flow > 0,
                               100.0 * nyc_contribution / montague_flow,
                               np.nan)
        contrib_pct_series = pd.Series(contrib_pct, index=nyc_contribution.index)

        # Get unique water years
        water_years = contrib_pct_series.index.map(get_water_year).unique()

        # If zone filtering is enabled, classify water years by drought zone
        wy_zone_map = None
        if zone_filter is not None:
            if not hasattr(data, 'res_level') or dataset_id not in data.res_level:
                raise ValueError(
                    "Zone filtering requires res_level data to be loaded. "
                    "Please load res_level results set."
                )
            res_level_df = data.res_level[dataset_id][real_id]
            wy_zone_map = classify_water_years_by_max_zone(res_level_df)

        for wy in water_years:
            # Get data for this water year (June 1 of wy to May 31 of wy+1)
            wy_mask = contrib_pct_series.index.map(get_water_year) == wy
            wy_data = contrib_pct_series[wy_mask]

            # Skip incomplete water years (e.g., first year starting in Jan,
            # or last year ending in Dec)
            if len(wy_data) < MIN_DAYS_FOR_COMPLETE_WATER_YEAR:
                continue

            n_years_total += 1

            # Apply zone filter if specified
            if zone_filter is not None:
                max_zone = wy_zone_map.get(wy)
                if max_zone not in zone_filter:
                    continue

            n_years_filtered += 1

            # Create day of water year index (1-366, starting June 1)
            doy = wy_data.index.map(get_water_year_doy)
            wy_series = pd.Series(wy_data.values, index=doy, name=f"r{real_id}_wy{wy}")

            # Sort by index to ensure continuous plotting from June -> May
            wy_series = wy_series.sort_index()

            all_series.append(wy_series)

    # Combine all series into DataFrame
    all_years_df = pd.concat(all_series, axis=1)

    return all_years_df, n_years_total, n_years_filtered


def calculate_daily_storage_percentage(data, dataset_id, zone_filter=None):
    """
    Calculate NYC storage as percentage of capacity for each day.

    Uses water years (June 1 - May 31) for analysis and day-of-water-year indexing.

    Parameters
    ----------
    data : pywrdrb.Data
        Data object containing res_storage and optionally res_level
    dataset_id : str
        Dataset identifier
    zone_filter : list of int or None
        If provided, only include water years where NYC reservoirs experienced
        one of these drought zones. Requires res_level data to be loaded.
        None means include all water years.

    Returns
    -------
    all_years_data : pd.DataFrame
        DataFrame with day_of_water_year as index (1-366, starting June 1)
        and each column as a water_year-realization
    n_years_total : int
        Total number of water years before filtering
    n_years_filtered : int
        Number of water years after filtering
    """
    realization_ids = list(data.res_storage[dataset_id].keys())

    all_series = []
    n_years_total = 0
    n_years_filtered = 0

    for real_id in realization_ids:
        # Get storage data
        storage_df = data.res_storage[dataset_id][real_id]

        # Calculate total NYC storage as percentage of capacity
        nyc_storage = storage_df[NYC_RESERVOIRS].sum(axis=1)
        storage_pct = 100.0 * nyc_storage / NYC_TOTAL_CAPACITY

        # Get unique water years
        water_years = storage_pct.index.map(get_water_year).unique()

        # If zone filtering is enabled, classify water years by drought zone
        wy_zone_map = None
        if zone_filter is not None:
            if not hasattr(data, 'res_level') or dataset_id not in data.res_level:
                raise ValueError(
                    "Zone filtering requires res_level data to be loaded. "
                    "Please load res_level results set."
                )
            res_level_df = data.res_level[dataset_id][real_id]
            wy_zone_map = classify_water_years_by_max_zone(res_level_df)

        for wy in water_years:
            # Get data for this water year (June 1 of wy to May 31 of wy+1)
            wy_mask = storage_pct.index.map(get_water_year) == wy
            wy_data = storage_pct[wy_mask]

            # Skip incomplete water years (e.g., first year starting in Jan,
            # or last year ending in Dec)
            if len(wy_data) < MIN_DAYS_FOR_COMPLETE_WATER_YEAR:
                continue

            n_years_total += 1

            # Apply zone filter if specified
            if zone_filter is not None:
                max_zone = wy_zone_map.get(wy)
                if max_zone not in zone_filter:
                    continue

            n_years_filtered += 1

            # Create day of water year index (1-366, starting June 1)
            doy = wy_data.index.map(get_water_year_doy)
            wy_series = pd.Series(wy_data.values, index=doy, name=f"r{real_id}_wy{wy}")

            # Sort by index to ensure continuous plotting from June -> May
            wy_series = wy_series.sort_index()

            all_series.append(wy_series)

    # Combine all series into DataFrame
    all_years_df = pd.concat(all_series, axis=1)

    return all_years_df, n_years_total, n_years_filtered


def get_1964_reconstruction_contribution_trace():
    """
    Get the daily NYC contribution percentage for 1964 from reconstruction data.

    Returns
    -------
    pd.Series or None
        Series with day of year as index and contribution percentage as values
    """
    reconstruction_file = RECONSTRUCTION_OUTPUT_FNAME

    if not os.path.exists(reconstruction_file):
        print(f"  Warning: Reconstruction file not found: {reconstruction_file}")
        return None

    print("  Loading 1964 reconstruction data...")

    try:
        data = pywrdrb.Data()
        data.load_output(output_filenames=[reconstruction_file],
                        results_sets=['major_flow', 'nyc_release_components'])

        dataset_name = 'reconstruction'
        if dataset_name not in data.major_flow:
            available_keys = list(data.major_flow.keys())
            if len(available_keys) == 1:
                dataset_name = available_keys[0]
            else:
                print(f"  Warning: Could not find reconstruction data")
                return None

        realization_id = 0
        if realization_id not in data.major_flow[dataset_name]:
            available_reals = list(data.major_flow[dataset_name].keys())
            if len(available_reals) > 0:
                realization_id = available_reals[0]
            else:
                return None

        # Get Montague flow
        major_flow_df = data.major_flow[dataset_name][realization_id]
        montague_flow = major_flow_df['delMontague']

        # Get NYC contributions
        nyc_release_df = data.nyc_release_components[dataset_name][realization_id]
        contribution_columns = [f'mrf_montagueTrenton_{res}' for res in NYC_RESERVOIRS]
        nyc_contribution = nyc_release_df[contribution_columns].sum(axis=1)

        # Calculate percentage
        contrib_pct = np.where(montague_flow > 0,
                               100.0 * nyc_contribution / montague_flow,
                               np.nan)
        contrib_pct_series = pd.Series(contrib_pct, index=nyc_contribution.index)

        # Filter to 1964
        year_1964 = contrib_pct_series[contrib_pct_series.index.year == 1964]

        if len(year_1964) == 0:
            print(f"  Warning: No 1964 data found")
            return None

        # Convert to day of year
        doy = year_1964.index.dayofyear
        trace_1964 = pd.Series(year_1964.values, index=doy, name='1964_reconstruction')

        print(f"  1964 reconstruction trace loaded ({len(trace_1964)} days)")
        return trace_1964

    except Exception as e:
        print(f"  Warning: Error loading 1964 reconstruction: {e}")
        return None


def find_representative_year_for_zone(data, dataset_id, zone_filter=None):
    """
    Find the realization/water year with contribution ratio closest to mean for the specified zone filter.

    Uses the same methodology as SI6 to classify water years by drought zone and find
    the representative year. Returns BOTH contribution AND storage traces.
    Uses water years (June 1 - May 31) for analysis.

    Parameters
    ----------
    data : pywrdrb.Data
        Data object with res_level, inflow, contribution, res_storage, major_flow
    dataset_id : str
        Dataset identifier
    zone_filter : list of int or None
        List of zone numbers to filter by. If None, includes all water years.

    Returns
    -------
    dict or None
        Dictionary with:
        - realization_id: int
        - year: int (water year)
        - ratio: float (contribution ratio for this water year)
        - mean_ratio: float (mean contribution ratio across filtered water years)
        - contribution_trace: pd.Series (daily contribution % by day of water year)
        - storage_trace: pd.Series (daily storage % by day of water year)
    """
    zone_label = get_zone_filter_label(zone_filter)
    print(f"  Finding representative water year for {zone_label}...")

    # Get all realization IDs
    realization_ids = list(data.res_level[dataset_id].keys())

    # Collect data for all water years across all realizations
    records = []

    for real_id in realization_ids:
        res_level_df = data.res_level[dataset_id][real_id]
        inflow_df = data.inflow[dataset_id][real_id]
        contribution_df = data.contribution[dataset_id][real_id]

        nyc_inflow = inflow_df[NYC_RESERVOIRS].sum(axis=1)
        nyc_contributions = contribution_df['mrf_montagueTrenton_nyc']

        # Get unique water years
        water_years = res_level_df.index.map(get_water_year).unique()

        for wy in water_years:
            wy_mask = res_level_df.index.map(get_water_year) == wy
            wy_data = res_level_df[wy_mask]

            # Skip incomplete water years
            if len(wy_data) < MIN_DAYS_FOR_COMPLETE_WATER_YEAR:
                continue

            # Find max zone (most severe drought) in this water year
            max_zone = wy_data['nyc'].max()

            # Apply zone filter if specified
            if zone_filter is not None:
                if max_zone not in zone_filter:
                    continue

            # Find date of max zone
            max_zone_date = wy_data[wy_data['nyc'] == max_zone].index[0]

            # Calculate 6-month prior aggregates (matching SI6 default)
            start_date = max_zone_date - pd.DateOffset(months=6)

            inflow_mask = (nyc_inflow.index >= start_date) & (nyc_inflow.index <= max_zone_date)
            contribution_mask = (nyc_contributions.index >= start_date) & (nyc_contributions.index <= max_zone_date)

            inflow_total = nyc_inflow[inflow_mask].sum()
            contribution_total = nyc_contributions[contribution_mask].sum()

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

    # Find water year closest to mean
    mean_ratio = df['contribution_ratio'].mean()
    df['distance_to_mean'] = abs(df['contribution_ratio'] - mean_ratio)
    closest_idx = df['distance_to_mean'].idxmin()
    closest_row = df.loc[closest_idx]

    real_id = int(closest_row['realization_id'])
    wy = int(closest_row['water_year'])

    print(f"  Representative water year: Realization {real_id}, WY {wy}, "
          f"Ratio {closest_row['contribution_ratio']:.1f}% (mean: {mean_ratio:.1f}%)")

    # Get the daily CONTRIBUTION trace for this water year
    contribution_df = data.contribution[dataset_id][real_id]
    nyc_contribution = contribution_df['mrf_montagueTrenton_nyc']

    major_flow_df = data.major_flow[dataset_id][real_id]
    montague_flow = major_flow_df['delMontague']

    contrib_pct = np.where(montague_flow > 0,
                           100.0 * nyc_contribution / montague_flow,
                           np.nan)
    contrib_pct_series = pd.Series(contrib_pct, index=nyc_contribution.index)

    wy_mask = contrib_pct_series.index.map(get_water_year) == wy
    wy_contrib_data = contrib_pct_series[wy_mask]
    doy_contrib = wy_contrib_data.index.map(get_water_year_doy)
    contribution_trace = pd.Series(wy_contrib_data.values, index=doy_contrib,
                                   name=f'r{real_id}_wy{wy}_contrib')
    # Sort by index to ensure continuous plotting from June -> May
    contribution_trace = contribution_trace.sort_index()

    # Get the daily STORAGE trace for this water year
    storage_df = data.res_storage[dataset_id][real_id]
    nyc_storage = storage_df[NYC_RESERVOIRS].sum(axis=1)
    storage_pct = 100.0 * nyc_storage / NYC_TOTAL_CAPACITY

    wy_mask_storage = storage_pct.index.map(get_water_year) == wy
    wy_storage_data = storage_pct[wy_mask_storage]
    doy_storage = wy_storage_data.index.map(get_water_year_doy)
    storage_trace = pd.Series(wy_storage_data.values, index=doy_storage,
                              name=f'r{real_id}_wy{wy}_storage')
    # Sort by index to ensure continuous plotting from June -> May
    storage_trace = storage_trace.sort_index()

    return {
        'realization_id': real_id,
        'year': wy,
        'ratio': closest_row['contribution_ratio'],
        'mean_ratio': mean_ratio,
        'contribution_trace': contribution_trace,
        'storage_trace': storage_trace
    }


# ============================================================================
# PLOTTING HELPER FUNCTIONS
# ============================================================================

# Month boundaries for x-axis formatting (water year: June 1 - May 31)
# Day 1 = June 1, Day 366 = May 31
# Approximate days at start of each month in water year:
# Jun=1, Jul=31, Aug=62, Sep=93, Oct=123, Nov=154, Dec=184, Jan=215, Feb=246, Mar=274, Apr=305, May=335
MONTH_STARTS_WY = [1, 31, 62, 93, 123, 154, 184, 215, 246, 274, 305, 335]
MONTH_LABELS_WY = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',
                   'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']


def _format_xaxis(ax):
    """Format x-axis with month labels for water year (June-May)."""
    ax.set_xticks(MONTH_STARTS_WY)
    ax.set_xticklabels(MONTH_LABELS_WY, fontsize=10)
    ax.set_xlim(1, 366)


def _plot_contribution_bands(ax, contrib_percentiles, representative_year=None):
    """
    Plot contribution bands and optional representative year trace on an axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    contrib_percentiles : pd.DataFrame
        Percentiles DataFrame with columns '5%', '25%', '50%', '75%', '95%'
    representative_year : dict, optional
        Representative year info with 'contribution_trace' key
    """
    doy = contrib_percentiles.index.values

    # Plot 5-95% band (lightest)
    ax.fill_between(doy, contrib_percentiles['5%'], contrib_percentiles['95%'],
                    color='steelblue', alpha=0.2, label='Contribution 5th-95th %ile')

    # Plot 25-75% band (darker)
    ax.fill_between(doy, contrib_percentiles['25%'], contrib_percentiles['75%'],
                    color='steelblue', alpha=0.4, label='Contribution 25th-75th %ile')

    # Plot median line
    ax.plot(doy, contrib_percentiles['50%'], color='steelblue', linewidth=2,
            label='Contribution Median')

    # Plot representative year contribution trace
    if representative_year is not None and 'contribution_trace' in representative_year:
        trace = representative_year['contribution_trace']
        year = representative_year['year']
        ax.plot(trace.index, trace.values, color='steelblue', linewidth=1.5,
                linestyle='--', alpha=0.8, label=f'Representative Year ({year})')


def _plot_storage_bands(ax, storage_percentiles, representative_year=None):
    """
    Plot storage bands and optional representative year trace on an axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on (can be twin axis for dual-axis mode)
    storage_percentiles : pd.DataFrame
        Percentiles DataFrame with columns '5%', '25%', '50%', '75%', '95%'
    representative_year : dict, optional
        Representative year info with 'storage_trace' key
    """
    doy = storage_percentiles.index.values

    # Plot 5-95% band (lightest)
    ax.fill_between(doy, storage_percentiles['5%'], storage_percentiles['95%'],
                    color='gray', alpha=0.15, label='Storage 5th-95th %ile')

    # Plot 25-75% band (darker)
    ax.fill_between(doy, storage_percentiles['25%'], storage_percentiles['75%'],
                    color='gray', alpha=0.3, label='Storage 25th-75th %ile')

    # Plot median line
    ax.plot(doy, storage_percentiles['50%'], color='gray', linewidth=1.5,
            label='Storage Median')

    # Plot representative year storage trace
    if representative_year is not None and 'storage_trace' in representative_year:
        trace = representative_year['storage_trace']
        year = representative_year['year']
        ax.plot(trace.index, trace.values, color='gray', linewidth=1.5,
                linestyle='--', alpha=0.8, label=f'Representative Storage ({year})')


def _get_annotation_text(zone_filter, n_years_total, n_years_filtered, n_samples):
    """Generate annotation text for sample size and zone filter."""
    if zone_filter is not None and n_years_total is not None and n_years_filtered is not None:
        zone_label = get_zone_filter_label(zone_filter)
        return (f'{zone_label}\n'
                f'n = {n_years_filtered} / {n_years_total} water year-realizations')
    else:
        return f'n = {n_samples} water year-realizations'


def _get_output_filename(dataset_id, zone_filter, mode_suffix=''):
    """Generate output filename based on zone filter and mode."""
    if zone_filter is not None:
        zone_suffix = '_zones_' + '_'.join(map(str, sorted(zone_filter, reverse=True)))
    else:
        zone_suffix = ''
    return f"{FIG_DIR_CONTRIBUTION}/{dataset_id}_nyc_contribution{mode_suffix}{zone_suffix}.png"


def _rank_years_by_mean_contribution(contrib_df):
    """
    Rank year-realizations by mean contribution (highest to lowest).

    Parameters
    ----------
    contrib_df : pd.DataFrame
        Contribution percentage data (day of year as index, year-realizations as columns)

    Returns
    -------
    ranked_columns : list
        Column names sorted by mean contribution (highest first)
    mean_contributions : pd.Series
        Mean contribution for each year-realization, sorted highest to lowest
    """
    # Calculate mean contribution for each year-realization
    mean_contrib = contrib_df.mean(axis=0)

    # Sort from highest to lowest
    mean_contrib_sorted = mean_contrib.sort_values(ascending=False)

    return list(mean_contrib_sorted.index), mean_contrib_sorted


def _apply_smoothing(series, window=7):
    """
    Apply rolling mean smoothing to a series.

    Parameters
    ----------
    series : pd.Series
        Time series to smooth
    window : int
        Window size for rolling mean

    Returns
    -------
    pd.Series
        Smoothed series
    """
    return series.rolling(window=window, center=True, min_periods=1).mean()


def _plot_subplots_linked(contrib_df, storage_df, dataset_id,
                          zone_filter=None, n_years_total=None, n_years_filtered=None):
    """
    Mode 2b: Plot storage and contribution with linked color-coded traces.

    Each year-realization is plotted as an individual trace, color-coded by
    its mean contribution. This allows visual comparison between panels -
    years with high contribution should show lower storage.

    Parameters
    ----------
    contrib_df : pd.DataFrame
        Contribution percentage data (day of year x year-realizations)
    storage_df : pd.DataFrame
        Storage percentage data (day of year x year-realizations)
    dataset_id : str
        Dataset identifier
    zone_filter : list of int or None
        Zone filter applied
    n_years_total, n_years_filtered : int
        Year counts for annotation
    """
    print("Creating linked traces subplot layout (Mode 2b)...")

    # Rank years by mean contribution
    ranked_columns, mean_contributions = _rank_years_by_mean_contribution(contrib_df)
    n_years = len(ranked_columns)

    print(f"  Ranked {n_years} year-realizations by mean contribution")
    print(f"  Mean contribution range: {mean_contributions.min():.1f}% - {mean_contributions.max():.1f}%")

    # Create colormap - high contribution = warm colors, low contribution = cool colors
    cmap = plt.get_cmap('RdYlBu_r')  # Red (high) -> Yellow -> Blue (low)

    # Normalize mean contributions to [0, 1] for colormap
    contrib_min = mean_contributions.min()
    contrib_max = mean_contributions.max()
    contrib_range = contrib_max - contrib_min if contrib_max > contrib_min else 1

    # Create figure with 2 rows
    fig, (ax_storage, ax_contrib) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot each year-realization with color based on mean contribution
    for col in ranked_columns:
        # Get normalized position for colormap (0 = low contribution, 1 = high contribution)
        norm_val = (mean_contributions[col] - contrib_min) / contrib_range
        color = cmap(norm_val)

        # Get traces and ensure they're sorted by day-of-water-year for continuous plotting
        contrib_trace = contrib_df[col].dropna().sort_index()
        storage_trace = storage_df[col].dropna().sort_index()

        # Apply smoothing to contribution trace
        contrib_smoothed = _apply_smoothing(contrib_trace, window=SMOOTHING_WINDOW)

        # Plot on both panels with same color
        ax_storage.plot(storage_trace.index, storage_trace.values,
                       color=color, alpha=0.4, linewidth=0.5)
        ax_contrib.plot(contrib_smoothed.index, contrib_smoothed.values,
                       color=color, alpha=0.4, linewidth=0.5)

    # Add median lines for reference
    contrib_median = contrib_df.median(axis=1)
    storage_median = storage_df.median(axis=1)

    # Smooth the contribution median too
    contrib_median_smoothed = _apply_smoothing(contrib_median, window=SMOOTHING_WINDOW)

    ax_storage.plot(storage_median.index, storage_median.values,
                   color='black', linewidth=2, label='Median', zorder=10)
    ax_contrib.plot(contrib_median_smoothed.index, contrib_median_smoothed.values,
                   color='black', linewidth=2, label='Median (smoothed)', zorder=10)

    # Format top panel (Storage)
    ax_storage.set_ylabel('NYC Storage\n(% of capacity)', fontsize=11, fontweight='bold')
    if Y_SCALE_FIXED:
        ax_storage.set_ylim(0, 100)
    ax_storage.grid(axis='y', alpha=0.3, linestyle='--')
    ax_storage.set_axisbelow(True)
    ax_storage.text(0.02, 0.95, '(a)', transform=ax_storage.transAxes, fontsize=12,
                    fontweight='bold', verticalalignment='top')
    ax_storage.legend(loc='lower right', fontsize=8, frameon=True, fancybox=True)

    # Format bottom panel (Contribution)
    _format_xaxis(ax_contrib)
    ax_contrib.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax_contrib.set_ylabel('NYC Contribution\n(% of Montague)', fontsize=11, fontweight='bold')
    if Y_SCALE_FIXED:
        ax_contrib.set_ylim(0, 100)
    ax_contrib.grid(axis='y', alpha=0.3, linestyle='--')
    ax_contrib.set_axisbelow(True)
    ax_contrib.text(0.02, 0.95, '(b)', transform=ax_contrib.transAxes, fontsize=12,
                    fontweight='bold', verticalalignment='top')
    ax_contrib.legend(loc='upper right', fontsize=8, frameon=True, fancybox=True)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=contrib_min, vmax=contrib_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_storage, ax_contrib], location='right', pad=0.02, aspect=30)
    cbar.set_label('Mean NYC Contribution (%)', fontsize=10, fontweight='bold')

    # Annotation on bottom panel
    annotation_text = _get_annotation_text(zone_filter, n_years_total, n_years_filtered,
                                           contrib_df.shape[1])
    ax_contrib.text(0.98, 0.02, annotation_text, transform=ax_contrib.transAxes, fontsize=9,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save
    fname = _get_output_filename(dataset_id, zone_filter, '_storage_linked')
    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


def _plot_contribution_only(contrib_df, dataset_id, representative_year=None,
                            zone_filter=None, n_years_total=None, n_years_filtered=None):
    """
    Mode 3: Plot contribution only (backward compatible with original SI7).

    Parameters
    ----------
    contrib_df : pd.DataFrame
        Contribution percentage data (day of year x year-realizations)
    dataset_id : str
        Dataset identifier
    representative_year : dict, optional
        Representative year info with contribution_trace
    zone_filter : list of int or None
        Zone filter applied
    n_years_total, n_years_filtered : int
        Year counts for annotation
    """
    print("Creating contribution-only timeseries plot (Mode 3)...")

    # Calculate percentiles
    contrib_percentiles = contrib_df.T.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    # Plot contribution bands
    _plot_contribution_bands(ax, contrib_percentiles, representative_year)

    # Format axes
    _format_xaxis(ax)
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('NYC Contribution to Montague Flow (%)', fontsize=12, fontweight='bold')

    if Y_SCALE_FIXED:
        ax.set_ylim(0, 100)

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc='upper right', fontsize=9, frameon=True, fancybox=True)

    # Annotation
    annotation_text = _get_annotation_text(zone_filter, n_years_total, n_years_filtered,
                                           contrib_df.shape[1])
    ax.text(0.02, 0.98, annotation_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save
    fname = _get_output_filename(dataset_id, zone_filter, '_timeseries')
    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


def _plot_dual_axis(contrib_df, storage_df, dataset_id, representative_year=None,
                    zone_filter=None, n_years_total=None, n_years_filtered=None):
    """
    Mode 1: Plot contribution and storage on dual Y-axes.

    Parameters
    ----------
    contrib_df : pd.DataFrame
        Contribution percentage data (day of year x year-realizations)
    storage_df : pd.DataFrame
        Storage percentage data (day of year x year-realizations)
    dataset_id : str
        Dataset identifier
    representative_year : dict, optional
        Representative year info with contribution_trace and storage_trace
    zone_filter : list of int or None
        Zone filter applied
    n_years_total, n_years_filtered : int
        Year counts for annotation
    """
    print("Creating dual-axis overlay plot (Mode 1)...")

    # Calculate percentiles
    contrib_percentiles = contrib_df.T.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    storage_percentiles = storage_df.T.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T

    # Create figure with dual Y-axes
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    ax2 = ax1.twinx()

    # Plot contribution on left axis (ax1)
    _plot_contribution_bands(ax1, contrib_percentiles, representative_year)

    # Plot storage on right axis (ax2)
    _plot_storage_bands(ax2, storage_percentiles, representative_year)

    # Format x-axis
    _format_xaxis(ax1)
    ax1.set_xlabel('Month', fontsize=12, fontweight='bold')

    # Left Y-axis (Contribution)
    ax1.set_ylabel('NYC Contribution to Montague Flow (%)', fontsize=12,
                   fontweight='bold', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    if Y_SCALE_FIXED:
        ax1.set_ylim(0, 100)

    # Right Y-axis (Storage)
    ax2.set_ylabel('NYC Storage (% of capacity)', fontsize=12,
                   fontweight='bold', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    if Y_SCALE_FIXED:
        ax2.set_ylim(0, 100)

    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8,
               frameon=True, fancybox=True, ncol=2)

    # Annotation
    annotation_text = _get_annotation_text(zone_filter, n_years_total, n_years_filtered,
                                           contrib_df.shape[1])
    ax1.text(0.02, 0.98, annotation_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save
    fname = _get_output_filename(dataset_id, zone_filter, '_storage_overlay')
    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


def _plot_subplots(contrib_df, storage_df, dataset_id, representative_year=None,
                   zone_filter=None, n_years_total=None, n_years_filtered=None):
    """
    Mode 2: Plot storage and contribution in separate subplots.

    Parameters
    ----------
    contrib_df : pd.DataFrame
        Contribution percentage data (day of year x year-realizations)
    storage_df : pd.DataFrame
        Storage percentage data (day of year x year-realizations)
    dataset_id : str
        Dataset identifier
    representative_year : dict, optional
        Representative year info with contribution_trace and storage_trace
    zone_filter : list of int or None
        Zone filter applied
    n_years_total, n_years_filtered : int
        Year counts for annotation
    """
    print("Creating subplot layout (Mode 2)...")

    # Calculate percentiles
    contrib_percentiles = contrib_df.T.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    storage_percentiles = storage_df.T.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T

    # Create figure with 2 rows
    fig, (ax_storage, ax_contrib) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top panel: Storage
    _plot_storage_bands(ax_storage, storage_percentiles, representative_year)
    ax_storage.set_ylabel('NYC Storage\n(% of capacity)', fontsize=11, fontweight='bold')
    if Y_SCALE_FIXED:
        ax_storage.set_ylim(0, 100)
    ax_storage.grid(axis='y', alpha=0.3, linestyle='--')
    ax_storage.set_axisbelow(True)
    ax_storage.text(0.02, 0.95, '(a)', transform=ax_storage.transAxes, fontsize=12,
                    fontweight='bold', verticalalignment='top')
    ax_storage.legend(loc='upper right', fontsize=8, frameon=True, fancybox=True)

    # Bottom panel: Contribution
    _plot_contribution_bands(ax_contrib, contrib_percentiles, representative_year)
    _format_xaxis(ax_contrib)
    ax_contrib.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax_contrib.set_ylabel('NYC Contribution\n(% of Montague)', fontsize=11, fontweight='bold')
    if Y_SCALE_FIXED:
        ax_contrib.set_ylim(0, 100)
    ax_contrib.grid(axis='y', alpha=0.3, linestyle='--')
    ax_contrib.set_axisbelow(True)
    ax_contrib.text(0.02, 0.95, '(b)', transform=ax_contrib.transAxes, fontsize=12,
                    fontweight='bold', verticalalignment='top')
    ax_contrib.legend(loc='upper right', fontsize=8, frameon=True, fancybox=True)

    # Annotation on bottom panel
    annotation_text = _get_annotation_text(zone_filter, n_years_total, n_years_filtered,
                                           contrib_df.shape[1])
    ax_contrib.text(0.98, 0.02, annotation_text, transform=ax_contrib.transAxes, fontsize=9,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save
    fname = _get_output_filename(dataset_id, zone_filter, '_storage_subplots')
    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


def plot_contribution_timeseries(contrib_df, dataset_id, storage_df=None,
                                 representative_year=None, zone_filter=None,
                                 n_years_total=None, n_years_filtered=None):
    """
    Main plotting function - dispatches to appropriate mode based on configuration.

    Parameters
    ----------
    contrib_df : pd.DataFrame
        Contribution percentage data (day of year x year-realizations)
    dataset_id : str
        Dataset identifier
    storage_df : pd.DataFrame, optional
        Storage percentage data (required if SHOW_NYC_STORAGE=True)
    representative_year : dict, optional
        Representative year info from find_representative_year_for_zone()
    zone_filter : list of int or None
        Zone filter applied
    n_years_total, n_years_filtered : int
        Year counts for annotation
    """
    if SHOW_NYC_STORAGE and storage_df is not None:
        if STORAGE_SUBPLOT:
            if SHOW_LINKED_TRACES:
                # Mode 2b: Linked color-coded traces
                _plot_subplots_linked(contrib_df, storage_df, dataset_id,
                                      zone_filter, n_years_total, n_years_filtered)
            else:
                # Mode 2: Separate subplots with percentile bands
                _plot_subplots(contrib_df, storage_df, dataset_id, representative_year,
                              zone_filter, n_years_total, n_years_filtered)
        else:
            # Mode 1: Dual-axis overlay
            _plot_dual_axis(contrib_df, storage_df, dataset_id, representative_year,
                           zone_filter, n_years_total, n_years_filtered)
    else:
        # Mode 3: Contribution only
        _plot_contribution_only(contrib_df, dataset_id, representative_year,
                               zone_filter, n_years_total, n_years_filtered)


# ============================================================================
# COMPARISON MODE PLOTTING FUNCTIONS
# ============================================================================

def _calculate_percentile_differences(baseline_df, comparison_df):
    """
    Calculate the difference in percentiles between two ensembles.

    Parameters
    ----------
    baseline_df : pd.DataFrame
        Baseline ensemble data (day of year x year-realizations)
    comparison_df : pd.DataFrame
        Comparison ensemble data (day of year x year-realizations)

    Returns
    -------
    diff_percentiles : pd.DataFrame
        DataFrame with percentile differences (comparison - baseline)
    """
    baseline_pctl = baseline_df.T.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    comparison_pctl = comparison_df.T.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T

    # Align indices (some days may be missing in one or the other)
    common_idx = baseline_pctl.index.intersection(comparison_pctl.index)
    baseline_pctl = baseline_pctl.loc[common_idx]
    comparison_pctl = comparison_pctl.loc[common_idx]

    # Calculate differences
    diff_percentiles = comparison_pctl - baseline_pctl

    return diff_percentiles


def _plot_difference_bands(ax, diff_percentiles, color, label_prefix, alpha_outer=0.2, alpha_inner=0.4):
    """
    Plot difference bands for a single variable.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    diff_percentiles : pd.DataFrame
        Percentile differences with columns '5%', '25%', '50%', '75%', '95%'
    color : str
        Color for the bands
    label_prefix : str
        Prefix for legend labels
    alpha_outer : float
        Alpha for 5-95% band
    alpha_inner : float
        Alpha for 25-75% band
    """
    doy = diff_percentiles.index.values

    # Plot 5-95% band (lightest)
    ax.fill_between(doy, diff_percentiles['5%'], diff_percentiles['95%'],
                    color=color, alpha=alpha_outer, label=f'{label_prefix} 5th-95th %ile')

    # Plot 25-75% band (darker)
    ax.fill_between(doy, diff_percentiles['25%'], diff_percentiles['75%'],
                    color=color, alpha=alpha_inner, label=f'{label_prefix} 25th-75th %ile')

    # Plot median line
    ax.plot(doy, diff_percentiles['50%'], color=color, linewidth=2,
            label=f'{label_prefix} Median')


def _plot_comparison_subplots(baseline_contrib, comparison_contrib,
                               baseline_storage, comparison_storage,
                               baseline_id, comparison_id, zone_filter=None):
    """
    Plot comparison as separate subplots showing differences.

    Parameters
    ----------
    baseline_contrib, comparison_contrib : pd.DataFrame
        Contribution data for baseline and comparison ensembles
    baseline_storage, comparison_storage : pd.DataFrame
        Storage data for baseline and comparison ensembles
    baseline_id, comparison_id : str
        Dataset identifiers
    zone_filter : list of int or None
        Zone filter applied
    """
    print("Creating comparison subplot layout...")

    # Calculate differences
    contrib_diff = _calculate_percentile_differences(baseline_contrib, comparison_contrib)
    storage_diff = _calculate_percentile_differences(baseline_storage, comparison_storage)

    # Create figure with 2 rows
    fig, (ax_storage, ax_contrib) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top panel: Storage difference
    _plot_difference_bands(ax_storage, storage_diff, color='gray', label_prefix='Storage',
                          alpha_outer=0.15, alpha_inner=0.3)
    ax_storage.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax_storage.set_ylabel('Storage Change\n(% points)', fontsize=11, fontweight='bold')
    ax_storage.grid(axis='y', alpha=0.3, linestyle='--')
    ax_storage.set_axisbelow(True)
    ax_storage.text(0.02, 0.95, '(a)', transform=ax_storage.transAxes, fontsize=12,
                    fontweight='bold', verticalalignment='top')
    ax_storage.legend(loc='upper right', fontsize=8, frameon=True, fancybox=True)

    # Bottom panel: Contribution difference
    _plot_difference_bands(ax_contrib, contrib_diff, color='steelblue', label_prefix='Contribution',
                          alpha_outer=0.2, alpha_inner=0.4)
    ax_contrib.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    _format_xaxis(ax_contrib)
    ax_contrib.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax_contrib.set_ylabel('Contribution Change\n(% points)', fontsize=11, fontweight='bold')
    ax_contrib.grid(axis='y', alpha=0.3, linestyle='--')
    ax_contrib.set_axisbelow(True)
    ax_contrib.text(0.02, 0.95, '(b)', transform=ax_contrib.transAxes, fontsize=12,
                    fontweight='bold', verticalalignment='top')
    ax_contrib.legend(loc='upper right', fontsize=8, frameon=True, fancybox=True)

    # Annotation
    zone_label = get_zone_filter_label(zone_filter) if zone_filter else "All Years"
    annotation_text = (f'Change: {comparison_id}\nrelative to {baseline_id}\n{zone_label}')
    ax_contrib.text(0.98, 0.02, annotation_text, transform=ax_contrib.transAxes, fontsize=9,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save
    zone_suffix = '_zones_' + '_'.join(map(str, sorted(zone_filter, reverse=True))) if zone_filter else ''
    fname = f"{FIG_DIR_CONTRIBUTION}/{comparison_id}_vs_{baseline_id}_comparison_subplots{zone_suffix}.png"
    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


def _plot_comparison_dual_axis(baseline_contrib, comparison_contrib,
                                baseline_storage, comparison_storage,
                                baseline_id, comparison_id, zone_filter=None):
    """
    Plot comparison as dual-axis overlay showing differences.

    Parameters
    ----------
    baseline_contrib, comparison_contrib : pd.DataFrame
        Contribution data for baseline and comparison ensembles
    baseline_storage, comparison_storage : pd.DataFrame
        Storage data for baseline and comparison ensembles
    baseline_id, comparison_id : str
        Dataset identifiers
    zone_filter : list of int or None
        Zone filter applied
    """
    print("Creating comparison dual-axis overlay plot...")

    # Calculate differences
    contrib_diff = _calculate_percentile_differences(baseline_contrib, comparison_contrib)
    storage_diff = _calculate_percentile_differences(baseline_storage, comparison_storage)

    # Create figure with dual Y-axes
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    ax2 = ax1.twinx()

    # Plot contribution difference on left axis (ax1)
    _plot_difference_bands(ax1, contrib_diff, color='steelblue', label_prefix='Contribution',
                          alpha_outer=0.2, alpha_inner=0.4)

    # Plot storage difference on right axis (ax2)
    _plot_difference_bands(ax2, storage_diff, color='gray', label_prefix='Storage',
                          alpha_outer=0.15, alpha_inner=0.3)

    # Add zero lines
    ax1.axhline(y=0, color='steelblue', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

    # Format x-axis
    _format_xaxis(ax1)
    ax1.set_xlabel('Month', fontsize=12, fontweight='bold')

    # Left Y-axis (Contribution)
    ax1.set_ylabel('Contribution Change (% points)', fontsize=12,
                   fontweight='bold', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')

    # Right Y-axis (Storage)
    ax2.set_ylabel('Storage Change (% points)', fontsize=12,
                   fontweight='bold', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8,
               frameon=True, fancybox=True, ncol=2)

    # Annotation
    zone_label = get_zone_filter_label(zone_filter) if zone_filter else "All Years"
    annotation_text = (f'Change: {comparison_id}\nrelative to {baseline_id}\n{zone_label}')
    ax1.text(0.02, 0.98, annotation_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save
    zone_suffix = '_zones_' + '_'.join(map(str, sorted(zone_filter, reverse=True))) if zone_filter else ''
    fname = f"{FIG_DIR_CONTRIBUTION}/{comparison_id}_vs_{baseline_id}_comparison_overlay{zone_suffix}.png"
    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


def _plot_comparison_contribution_only(baseline_contrib, comparison_contrib,
                                        baseline_id, comparison_id, zone_filter=None):
    """
    Plot comparison showing contribution difference only.

    Parameters
    ----------
    baseline_contrib, comparison_contrib : pd.DataFrame
        Contribution data for baseline and comparison ensembles
    baseline_id, comparison_id : str
        Dataset identifiers
    zone_filter : list of int or None
        Zone filter applied
    """
    print("Creating comparison contribution-only plot...")

    # Calculate differences
    contrib_diff = _calculate_percentile_differences(baseline_contrib, comparison_contrib)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    # Plot contribution difference bands
    _plot_difference_bands(ax, contrib_diff, color='steelblue', label_prefix='Contribution',
                          alpha_outer=0.2, alpha_inner=0.4)

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    # Format axes
    _format_xaxis(ax)
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Contribution Change (% points)', fontsize=12, fontweight='bold')

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc='upper right', fontsize=9, frameon=True, fancybox=True)

    # Annotation
    zone_label = get_zone_filter_label(zone_filter) if zone_filter else "All Years"
    annotation_text = (f'Change: {comparison_id}\nrelative to {baseline_id}\n{zone_label}')
    ax.text(0.02, 0.98, annotation_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save
    zone_suffix = '_zones_' + '_'.join(map(str, sorted(zone_filter, reverse=True))) if zone_filter else ''
    fname = f"{FIG_DIR_CONTRIBUTION}/{comparison_id}_vs_{baseline_id}_comparison{zone_suffix}.png"
    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


def plot_comparison(baseline_contrib, comparison_contrib,
                    baseline_storage, comparison_storage,
                    baseline_id, comparison_id, zone_filter=None):
    """
    Main comparison plotting function - dispatches to appropriate mode based on configuration.

    Parameters
    ----------
    baseline_contrib, comparison_contrib : pd.DataFrame
        Contribution data for baseline and comparison ensembles
    baseline_storage, comparison_storage : pd.DataFrame or None
        Storage data for baseline and comparison ensembles
    baseline_id, comparison_id : str
        Dataset identifiers
    zone_filter : list of int or None
        Zone filter applied
    """
    if SHOW_NYC_STORAGE and baseline_storage is not None and comparison_storage is not None:
        if STORAGE_SUBPLOT:
            # Mode 2: Separate subplots
            _plot_comparison_subplots(baseline_contrib, comparison_contrib,
                                      baseline_storage, comparison_storage,
                                      baseline_id, comparison_id, zone_filter)
        else:
            # Mode 1: Dual-axis overlay
            _plot_comparison_dual_axis(baseline_contrib, comparison_contrib,
                                       baseline_storage, comparison_storage,
                                       baseline_id, comparison_id, zone_filter)
    else:
        # Mode 3: Contribution only
        _plot_comparison_contribution_only(baseline_contrib, comparison_contrib,
                                           baseline_id, comparison_id, zone_filter)


def main(dataset_id):
    """
    Main function to generate NYC contribution timeseries plot.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    """
    print("=" * 80)
    print(f"NYC CONTRIBUTION TIMESERIES: {dataset_id}")
    print("=" * 80)

    # Print configuration
    print("\nConfiguration:")
    print(f"  Zone Filter: {get_zone_filter_label(FILTER_BY_ZONES)}")
    print(f"  Show NYC Storage: {SHOW_NYC_STORAGE}")
    print(f"  Storage Subplot: {STORAGE_SUBPLOT}")
    print(f"  Show Linked Traces: {SHOW_LINKED_TRACES}")
    print(f"  Smoothing Window: {SMOOTHING_WINDOW} days")
    print(f"  Show Representative Year: {SHOW_REPRESENTATIVE_YEAR}")
    print(f"  Fixed Y-Scale (0-100%): {Y_SCALE_FIXED}")

    # Determine plotting mode
    if SHOW_NYC_STORAGE:
        if STORAGE_SUBPLOT:
            if SHOW_LINKED_TRACES:
                print(f"  Mode: 2b (Linked Color-Coded Traces)")
            else:
                print(f"  Mode: 2 (Separate Subplots)")
        else:
            print(f"  Mode: 1 (Dual-Axis Overlay)")
    else:
        print(f"  Mode: 3 (Contribution Only)")

    # Verify dataset
    verify_dataset_id(dataset_id)

    # Load data
    print("\nLoading data...")
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'

    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Postprocessed data not found: {fname}\n"
            "Run 04_postprocess_data.py first!"
        )

    print(f"  Loading from: {fname}")
    data = pywrdrb.Data()

    # Determine which results sets to load based on configuration
    results_sets = ['contribution', 'major_flow', 'res_level', 'inflow']
    if SHOW_NYC_STORAGE:
        results_sets.append('res_storage')

    data.load_from_export(fname, results_sets=results_sets)
    print("  Data loaded successfully")

    # Calculate daily contribution percentages
    print("\nCalculating daily contribution percentages...")
    contrib_df, n_years_total, n_years_filtered = calculate_daily_contribution_percentage(
        data, dataset_id, zone_filter=FILTER_BY_ZONES
    )
    print(f"  Total year-realizations before filtering: {n_years_total}")
    print(f"  Total year-realizations after filtering: {n_years_filtered}")
    if FILTER_BY_ZONES is not None:
        pct_kept = 100.0 * n_years_filtered / n_years_total if n_years_total > 0 else 0
        print(f"  Percentage kept: {pct_kept:.1f}%")

    # Calculate daily storage percentages (if enabled)
    storage_df = None
    if SHOW_NYC_STORAGE:
        print("\nCalculating daily storage percentages...")
        storage_df, _, _ = calculate_daily_storage_percentage(
            data, dataset_id, zone_filter=FILTER_BY_ZONES
        )
        print(f"  Storage data calculated for {storage_df.shape[1]} year-realizations")

    # Find representative year for the current zone filter (if enabled)
    representative_year = None
    if SHOW_REPRESENTATIVE_YEAR:
        print("\nFinding representative year...")
        representative_year = find_representative_year_for_zone(
            data, dataset_id, zone_filter=FILTER_BY_ZONES
        )

    # Create plot
    print("\nCreating plot...")
    plot_contribution_timeseries(
        contrib_df=contrib_df,
        dataset_id=dataset_id,
        storage_df=storage_df,
        representative_year=representative_year,
        zone_filter=FILTER_BY_ZONES,
        n_years_total=n_years_total,
        n_years_filtered=n_years_filtered
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)


def main_comparison(baseline_id, comparison_id):
    """
    Main function for comparison mode - shows differences between two ensembles.

    Parameters
    ----------
    baseline_id : str
        Baseline dataset identifier (e.g., 'stationary_ensemble')
    comparison_id : str
        Comparison dataset identifier (e.g., 'climate_adjusted_low')
    """
    print("=" * 80)
    print(f"NYC CONTRIBUTION TIMESERIES COMPARISON")
    print(f"  Baseline: {baseline_id}")
    print(f"  Comparison: {comparison_id}")
    print("=" * 80)

    # Print configuration
    print("\nConfiguration:")
    print(f"  Zone Filter: {get_zone_filter_label(FILTER_BY_ZONES)}")
    print(f"  Show NYC Storage: {SHOW_NYC_STORAGE}")
    print(f"  Storage Subplot: {STORAGE_SUBPLOT}")

    # Determine plotting mode
    if SHOW_NYC_STORAGE:
        if STORAGE_SUBPLOT:
            print(f"  Mode: 2 (Separate Subplots)")
        else:
            print(f"  Mode: 1 (Dual-Axis Overlay)")
    else:
        print(f"  Mode: 3 (Contribution Only)")

    # Verify datasets
    verify_dataset_id(baseline_id)
    verify_dataset_id(comparison_id)

    # Determine which results sets to load
    results_sets = ['contribution', 'major_flow', 'res_level', 'inflow']
    if SHOW_NYC_STORAGE:
        results_sets.append('res_storage')

    # Load baseline data
    print("\nLoading baseline data...")
    baseline_fname = f'./pywrdrb/outputs/{baseline_id}_with_postprocessing.hdf5'
    if not os.path.exists(baseline_fname):
        raise FileNotFoundError(f"Baseline data not found: {baseline_fname}")

    baseline_data = pywrdrb.Data()
    baseline_data.load_from_export(baseline_fname, results_sets=results_sets)
    print(f"  Baseline loaded: {baseline_id}")

    # Load comparison data
    print("\nLoading comparison data...")
    comparison_fname = f'./pywrdrb/outputs/{comparison_id}_with_postprocessing.hdf5'
    if not os.path.exists(comparison_fname):
        raise FileNotFoundError(f"Comparison data not found: {comparison_fname}")

    comparison_data = pywrdrb.Data()
    comparison_data.load_from_export(comparison_fname, results_sets=results_sets)
    print(f"  Comparison loaded: {comparison_id}")

    # Calculate contribution percentages for both ensembles
    print("\nCalculating contribution percentages...")
    baseline_contrib, n_base_total, n_base_filtered = calculate_daily_contribution_percentage(
        baseline_data, baseline_id, zone_filter=FILTER_BY_ZONES
    )
    print(f"  Baseline: {n_base_filtered} / {n_base_total} year-realizations")

    comparison_contrib, n_comp_total, n_comp_filtered = calculate_daily_contribution_percentage(
        comparison_data, comparison_id, zone_filter=FILTER_BY_ZONES
    )
    print(f"  Comparison: {n_comp_filtered} / {n_comp_total} year-realizations")

    # Calculate storage percentages if enabled
    baseline_storage = None
    comparison_storage = None
    if SHOW_NYC_STORAGE:
        print("\nCalculating storage percentages...")
        baseline_storage, _, _ = calculate_daily_storage_percentage(
            baseline_data, baseline_id, zone_filter=FILTER_BY_ZONES
        )
        comparison_storage, _, _ = calculate_daily_storage_percentage(
            comparison_data, comparison_id, zone_filter=FILTER_BY_ZONES
        )
        print(f"  Storage data calculated for both ensembles")

    # Create comparison plot
    print("\nCreating comparison plot...")
    plot_comparison(
        baseline_contrib=baseline_contrib,
        comparison_contrib=comparison_contrib,
        baseline_storage=baseline_storage,
        comparison_storage=comparison_storage,
        baseline_id=baseline_id,
        comparison_id=comparison_id,
        zone_filter=FILTER_BY_ZONES
    )

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE!")
    print("=" * 80)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='NYC Contribution Percentage Timeseries Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Single ensemble mode
  python SI7_plot_nyc_contribution_timeseries.py stationary_ensemble

  # Comparison mode (shows difference: comparison - baseline)
  python SI7_plot_nyc_contribution_timeseries.py --comparison stationary_ensemble climate_adjusted_low

Available datasets: {list(DATASET_CONFIGS.keys())}
        """
    )

    parser.add_argument(
        'dataset_id',
        nargs='?',
        help='Dataset identifier for single ensemble mode'
    )

    parser.add_argument(
        '--comparison', '-c',
        nargs=2,
        metavar=('BASELINE', 'COMPARISON'),
        help='Comparison mode: show differences between two ensembles (comparison - baseline)'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.comparison:
        # Comparison mode
        baseline_id, comparison_id = args.comparison
        verify_dataset_id(baseline_id)
        verify_dataset_id(comparison_id)
        main_comparison(baseline_id, comparison_id)
    elif args.dataset_id:
        # Single ensemble mode
        verify_dataset_id(args.dataset_id)
        main(args.dataset_id)
    else:
        print(__doc__)
        print(f"\nAvailable datasets: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)
