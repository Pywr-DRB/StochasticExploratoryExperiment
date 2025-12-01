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

Drought Zone Filtering:
- Set FILTER_BY_ZONES to filter years by drought severity
- None: Include all years (default behavior)
- [6]: Only years with Drought Emergency
- [5, 6]: Only years with Drought Watch or Emergency
- [4, 5, 6]: Only years with Drought Warning, Watch, or Emergency

Configuration:
- Edit FILTER_BY_ZONES constant in the script to change zone filtering

Usage:
    python SI7_plot_nyc_contribution_timeseries.py <dataset_id>

Example:
    python SI7_plot_nyc_contribution_timeseries.py stationary_ensemble
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
from methods.plotting.styles import DPI_HIGH
from methods.config import RECONSTRUCTION_OUTPUT_FNAME

# Output directory
FIG_DIR_CONTRIBUTION = f"{FIG_DIR}/nyc_contribution_timeseries"
os.makedirs(FIG_DIR_CONTRIBUTION, exist_ok=True)

# NYC reservoir parameters
NYC_RESERVOIRS = ['cannonsville', 'pepacton', 'neversink']

# Minimum inflow threshold for filtering (MG) - same as SI6
MIN_INFLOW_THRESHOLD = 1000

# ============================================================================
# DROUGHT ZONE FILTERING CONFIGURATION
# ============================================================================

# Filter years by drought zone (set to None to include all years)
# Options:
#   None: Include all years (default behavior)
#   [6]: Only years with Drought Emergency
#   [5, 6]: Only years with Drought Watch or Emergency
#   [4, 5, 6]: Only years with Drought Warning, Watch, or Emergency
FILTER_BY_ZONES = [4, 5, 6]  # Set to list of zones or None for all years

# Drought zone mapping (from SI6)
ZONE_NAMES = {
    6: 'Drought Emergency',
    5: 'Drought Watch',
    4: 'Drought Warning',
    3: 'Normal',
    2: 'Flood Watch',
    1: 'Flood Warning',
}


def classify_years_by_max_zone(res_level_df):
    """
    Classify each year by the maximum drought zone reached (most severe drought).

    Based on SI6's classify_years_by_min_zone function.

    Parameters
    ----------
    res_level_df : pd.DataFrame
        Reservoir level DataFrame with 'nyc' column and datetime index

    Returns
    -------
    year_classifications : dict
        Dictionary mapping year -> max_zone (int)
    """
    df = res_level_df.copy()
    df['year'] = df.index.year

    year_classifications = {}

    for year in df['year'].unique():
        year_data = df[df['year'] == year]
        # Find maximum zone value (higher zone = more severe drought)
        max_zone = year_data['nyc'].max()
        year_classifications[year] = max_zone

    return year_classifications


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
        return "All Years"

    zone_labels = [ZONE_NAMES.get(z, f"Zone {z}") for z in sorted(zone_list, reverse=True)]

    if len(zone_labels) == 1:
        return f"Years with {zone_labels[0]}"
    else:
        return f"Years with {', '.join(zone_labels[:-1])}, or {zone_labels[-1]}"


def calculate_daily_contribution_percentage(data, dataset_id, zone_filter=None):
    """
    Calculate NYC contribution as percentage of Montague flow for each day.

    Parameters
    ----------
    data : pywrdrb.Data
        Data object containing contribution, major_flow, and optionally res_level
    dataset_id : str
        Dataset identifier
    zone_filter : list of int or None
        If provided, only include years where NYC reservoirs experienced
        one of these drought zones. Requires res_level data to be loaded.
        None means include all years.

    Returns
    -------
    all_years_data : pd.DataFrame
        DataFrame with day_of_year as index and each column as a year-realization
    n_years_total : int
        Total number of years before filtering
    n_years_filtered : int
        Number of years after filtering
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

        # Get unique years
        years = contrib_pct_series.index.year.unique()

        # If zone filtering is enabled, classify years by drought zone
        year_zone_map = None
        if zone_filter is not None:
            if not hasattr(data, 'res_level') or dataset_id not in data.res_level:
                raise ValueError(
                    "Zone filtering requires res_level data to be loaded. "
                    "Please load res_level results set."
                )
            res_level_df = data.res_level[dataset_id][real_id]
            year_zone_map = classify_years_by_max_zone(res_level_df)

        for year in years:
            n_years_total += 1

            # Apply zone filter if specified
            if zone_filter is not None:
                max_zone = year_zone_map.get(year)
                if max_zone not in zone_filter:
                    continue

            n_years_filtered += 1

            year_data = contrib_pct_series[contrib_pct_series.index.year == year]

            # Create day of year index (1-366)
            doy = year_data.index.dayofyear
            year_series = pd.Series(year_data.values, index=doy, name=f"r{real_id}_y{year}")

            all_series.append(year_series)

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


def find_representative_drought_emergency_year(data, dataset_id):
    """
    Find the realization/year with contribution ratio closest to mean for Drought Emergency.

    Uses the same methodology as SI6 to classify years by drought zone and find
    the representative year.

    Parameters
    ----------
    data : pywrdrb.Data
        Data object with res_level, inflow, contribution
    dataset_id : str
        Dataset identifier

    Returns
    -------
    dict or None
        Dictionary with realization_id, year, and contribution trace
    """
    print("  Finding representative Drought Emergency year...")

    # Get all realization IDs
    realization_ids = list(data.res_level[dataset_id].keys())

    # Collect data for all years across all realizations
    records = []

    for real_id in realization_ids:
        res_level_df = data.res_level[dataset_id][real_id]
        inflow_df = data.inflow[dataset_id][real_id]
        contribution_df = data.contribution[dataset_id][real_id]

        nyc_inflow = inflow_df[NYC_RESERVOIRS].sum(axis=1)
        nyc_contributions = contribution_df['mrf_montagueTrenton_nyc']

        years = res_level_df.index.year.unique()

        for year in years:
            year_data = res_level_df[res_level_df.index.year == year]

            # Find max zone (most severe drought)
            max_zone = year_data['nyc'].max()

            # Only consider Drought Emergency years (zone 6)
            if max_zone != 6:
                continue

            # Find date of max zone
            max_zone_date = year_data[year_data['nyc'] == max_zone].index[0]

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
                'year': year,
                'contribution_ratio': contribution_ratio
            })

    if len(records) == 0:
        print(f"  Warning: No Drought Emergency years found")
        return None

    df = pd.DataFrame(records)

    # Find year closest to mean
    mean_ratio = df['contribution_ratio'].mean()
    df['distance_to_mean'] = abs(df['contribution_ratio'] - mean_ratio)
    closest_idx = df['distance_to_mean'].idxmin()
    closest_row = df.loc[closest_idx]

    real_id = int(closest_row['realization_id'])
    year = int(closest_row['year'])

    print(f"  Representative year: Realization {real_id}, Year {year}, "
          f"Ratio {closest_row['contribution_ratio']:.1f}% (mean: {mean_ratio:.1f}%)")

    # Get the daily contribution trace for this year
    contribution_df = data.contribution[dataset_id][real_id]
    nyc_contribution = contribution_df['mrf_montagueTrenton_nyc']

    major_flow_df = data.major_flow[dataset_id][real_id]
    montague_flow = major_flow_df['delMontague']

    contrib_pct = np.where(montague_flow > 0,
                           100.0 * nyc_contribution / montague_flow,
                           np.nan)
    contrib_pct_series = pd.Series(contrib_pct, index=nyc_contribution.index)

    year_data = contrib_pct_series[contrib_pct_series.index.year == year]
    doy = year_data.index.dayofyear
    trace = pd.Series(year_data.values, index=doy, name=f'r{real_id}_y{year}')

    return {
        'realization_id': real_id,
        'year': year,
        'ratio': closest_row['contribution_ratio'],
        'mean_ratio': mean_ratio,
        'trace': trace
    }


def plot_contribution_timeseries(all_years_df, dataset_id, dataset_label,
                                  trace_1964=None, representative_emergency=None,
                                  zone_filter=None, n_years_total=None, n_years_filtered=None):
    """
    Create publication-quality timeseries plot with distribution bands.

    Parameters
    ----------
    all_years_df : pd.DataFrame
        Output from calculate_daily_contribution_percentage()
    dataset_id : str
        Dataset identifier
    dataset_label : str
        Dataset display label
    trace_1964 : pd.Series, optional
        1964 reconstruction daily contribution trace
    representative_emergency : dict, optional
        Dictionary with representative Drought Emergency year info and trace
    zone_filter : list of int or None
        Zone filter applied (for filename and title)
    n_years_total : int, optional
        Total years before filtering
    n_years_filtered : int, optional
        Total years after filtering
    """
    print("Creating NYC contribution timeseries plot...")

    # Calculate percentiles for each day of year
    percentiles = all_years_df.T.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T

    # Get day of year index
    doy = percentiles.index.values

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    # Plot 5-95% band (lightest)
    ax.fill_between(doy, percentiles['5%'], percentiles['95%'],
                   color='steelblue', alpha=0.2, label='5th-95th percentile')

    # Plot 25-75% band (darker)
    ax.fill_between(doy, percentiles['25%'], percentiles['75%'],
                   color='steelblue', alpha=0.4, label='25th-75th percentile')

    # Plot median line
    ax.plot(doy, percentiles['50%'], color='steelblue', linewidth=2,
           label='Median')

    # Plot 1964 reconstruction trace
    if trace_1964 is not None:
        ax.plot(trace_1964.index, trace_1964.values, color='black', linewidth=1.5,
               linestyle='-', alpha=0.9, label='1964 Drought')

    # Plot representative Drought Emergency year
    if representative_emergency is not None:
        trace = representative_emergency['trace']
        year = representative_emergency['year']
        ax.plot(trace.index, trace.values, color='#8B0000', linewidth=1.5,
               linestyle='--', alpha=0.8, label=f'Representative Emergency ({year})')

    # Format x-axis with month labels
    # Create month boundaries
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels, fontsize=10)
    ax.set_xlim(1, 366)

    # Axis labels
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('NYC Contribution to Montague Flow (%)', fontsize=12, fontweight='bold')

    # Y-axis limits
    ax.set_ylim(0, 100)

    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc='upper right', fontsize=9, frameon=True, fancybox=True)

    # Add sample size annotation with zone filter info
    n_samples = all_years_df.shape[1]
    if zone_filter is not None and n_years_total is not None and n_years_filtered is not None:
        zone_label = get_zone_filter_label(zone_filter)
        annotation_text = (f'{zone_label}\n'
                          f'n = {n_years_filtered} / {n_years_total} year-realizations')
    else:
        annotation_text = f'n = {n_samples} year-realizations'

    ax.text(0.02, 0.98, annotation_text,
           transform=ax.transAxes, fontsize=9,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Generate filename based on zone filter
    if zone_filter is not None:
        zone_suffix = '_zones_' + '_'.join(map(str, sorted(zone_filter, reverse=True)))
    else:
        zone_suffix = ''

    # Save
    fname = f"{FIG_DIR_CONTRIBUTION}/{dataset_id}_nyc_contribution_timeseries{zone_suffix}.png"
    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


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

    # Print zone filter configuration
    if FILTER_BY_ZONES is not None:
        zone_label = get_zone_filter_label(FILTER_BY_ZONES)
        print(f"\nZone Filter: {zone_label}")
        print(f"  Filtering to zones: {FILTER_BY_ZONES}")
    else:
        print("\nZone Filter: None (including all years)")

    # Verify dataset
    verify_dataset_id(dataset_id)
    dataset_config = DATASET_CONFIGS[dataset_id]
    dataset_label = f"{dataset_config['description']} ({dataset_config['type']})"

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
    # Load additional results sets needed for representative year finding and zone filtering
    data.load_from_export(
        fname,
        results_sets=['contribution', 'major_flow', 'res_level', 'inflow']
    )
    print("  Data loaded successfully")

    # Calculate daily contribution percentages
    print("\nCalculating daily contribution percentages...")
    all_years_df, n_years_total, n_years_filtered = calculate_daily_contribution_percentage(
        data, dataset_id, zone_filter=FILTER_BY_ZONES
    )
    print(f"  Total year-realizations before filtering: {n_years_total}")
    print(f"  Total year-realizations after filtering: {n_years_filtered}")
    if FILTER_BY_ZONES is not None:
        pct_kept = 100.0 * n_years_filtered / n_years_total if n_years_total > 0 else 0
        print(f"  Percentage kept: {pct_kept:.1f}%")

    # Get 1964 reconstruction trace
    print("\nLoading 1964 reconstruction trace...")
    trace_1964 = get_1964_reconstruction_contribution_trace()

    # Find representative Drought Emergency year
    print("\nFinding representative Drought Emergency year...")
    representative_emergency = find_representative_drought_emergency_year(data, dataset_id)

    # Create plot
    print("\nCreating plot...")
    plot_contribution_timeseries(
        all_years_df, dataset_id, dataset_label,
        trace_1964=None,
        representative_emergency=representative_emergency,
        zone_filter=FILTER_BY_ZONES,
        n_years_total=n_years_total,
        n_years_filtered=n_years_filtered
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        print(f"\nAvailable datasets: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)

    dataset_id = sys.argv[1]
    verify_dataset_id(dataset_id)

    main(dataset_id)
