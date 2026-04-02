"""
NYC contribution timeseries computation.

Two metrics:
1. Montague contribution percentage (daily):
   100 * NYC_contribution / Montague_flow

2. Contribution-to-inflow ratio (rolling window):
   100 * rolling_sum(NYC_contribution) / rolling_sum(NYC_inflow)
"""

import numpy as np
import pandas as pd

from methods.config import NYC_RESERVOIRS
from methods.water_year import (
    MIN_DAYS_FOR_COMPLETE_WATER_YEAR,
    vectorized_water_year,
    vectorized_water_year_doy,
    get_water_year,
    get_water_year_doy,
)

# Minimum thresholds
MIN_INFLOW_THRESHOLD = 1000   # MG — for representative-year search
MIN_ROLLING_INFLOW = 100      # MG — for ratio low-inflow masking

# Default rolling window (days)
DEFAULT_WINDOW_DAYS = 90

# Drought zone names (for zone-filter labelling)
ZONE_NAMES = {
    6: 'Drought Emergency',
    5: 'Drought Warning',
    4: 'Drought Watch',
    3: 'Normal',
    2: 'Flood Watch',
    1: 'Flood Warning',
}


def _scatter_to_doy_columns(values, dates, col_arrays,
                             zone_filter=None, wy_zone_map=None,
                             n_years_total=None, n_years_filtered=None):
    """
    Scatter a daily timeseries into per-water-year columns (366 rows).

    Shared scaffolding for both contribution metrics.

    Parameters
    ----------
    values : np.ndarray
        Daily metric values (same length as dates).
    dates : pd.DatetimeIndex
        Date index.
    col_arrays : list
        Accumulator list — columns are appended in-place.
    zone_filter : list of int, optional
        Drought zones to keep.
    wy_zone_map : dict, optional
        {water_year: max_zone} mapping (required when zone_filter is set).
    n_years_total : list of length 1, optional
        Mutable counter for total water years.
    n_years_filtered : list of length 1, optional
        Mutable counter for filtered water years.
    """
    water_years = vectorized_water_year(dates)
    doy = vectorized_water_year_doy(dates)

    unique_wys, wy_inv = np.unique(water_years, return_inverse=True)

    for wy_idx, wy in enumerate(unique_wys):
        mask = wy_inv == wy_idx
        if mask.sum() < MIN_DAYS_FOR_COMPLETE_WATER_YEAR:
            continue

        if n_years_total is not None:
            n_years_total[0] += 1

        if zone_filter is not None:
            max_zone = wy_zone_map.get(wy)
            if max_zone not in zone_filter:
                continue

        if n_years_filtered is not None:
            n_years_filtered[0] += 1

        col = np.full(366, np.nan)
        wy_doy = doy[mask]
        wy_vals = values[mask]
        valid = (wy_doy >= 1) & (wy_doy <= 366)
        col[wy_doy[valid] - 1] = wy_vals[valid]
        col_arrays.append(col)


def _build_doy_dataframe(col_arrays):
    """Build a (366 × N) DataFrame from collected column arrays."""
    if col_arrays:
        return pd.DataFrame(
            np.column_stack(col_arrays),
            index=np.arange(1, 367),
        )
    return pd.DataFrame(index=np.arange(1, 367))


# =========================================================================
# Metric 1: Daily contribution / Montague flow
# =========================================================================

def calculate_daily_contribution_percentage(data, dataset_id, zone_filter=None):
    """
    Daily NYC contribution as % of Montague flow, per water year.

    Returns
    -------
    result : pd.DataFrame
        366 rows (day-of-water-year) × N columns.
    n_years_total : int
    n_years_filtered : int
    """
    realization_ids = list(data.contribution[dataset_id].keys())
    col_arrays = []
    n_total = [0]
    n_filtered = [0]

    for real_id in realization_ids:
        nyc_contribution = data.contribution[dataset_id][real_id]['mrf_montagueTrenton_nyc']
        montague_flow = data.major_flow[dataset_id][real_id]['delMontague']

        contrib_pct = np.where(
            montague_flow > 0,
            100.0 * nyc_contribution / montague_flow,
            np.nan,
        )

        wy_zone_map = None
        if zone_filter is not None:
            if not hasattr(data, 'res_level') or dataset_id not in data.res_level:
                raise ValueError("Zone filtering requires res_level data.")
            wy_zone_map = classify_water_years_by_max_zone(
                data.res_level[dataset_id][real_id])

        _scatter_to_doy_columns(
            contrib_pct, nyc_contribution.index, col_arrays,
            zone_filter=zone_filter, wy_zone_map=wy_zone_map,
            n_years_total=n_total, n_years_filtered=n_filtered,
        )

    return _build_doy_dataframe(col_arrays), n_total[0], n_filtered[0]


# =========================================================================
# Metric 2: Rolling-window contribution / NYC inflow
# =========================================================================

def calculate_daily_contribution_ratio(data, dataset_id, window=None):
    """
    Rolling-window NYC contribution / NYC inflow ratio (%), per water year.

    Returns
    -------
    result : pd.DataFrame
        366 rows (day-of-water-year) × N columns.
    """
    if window is None:
        window = DEFAULT_WINDOW_DAYS

    realization_ids = list(data.contribution[dataset_id].keys())
    col_arrays = []

    for real_id in realization_ids:
        nyc_contribution = data.contribution[dataset_id][real_id]['mrf_montagueTrenton_nyc']
        nyc_inflow = data.inflow[dataset_id][real_id][NYC_RESERVOIRS].sum(axis=1)

        rolling_contrib = nyc_contribution.rolling(window, min_periods=window).sum()
        rolling_inflow = nyc_inflow.rolling(window, min_periods=window).sum()

        ratio = np.where(
            rolling_inflow > MIN_ROLLING_INFLOW,
            100.0 * rolling_contrib / rolling_inflow,
            np.nan,
        )

        _scatter_to_doy_columns(ratio, nyc_contribution.index, col_arrays)

    return _build_doy_dataframe(col_arrays)


# =========================================================================
# Zone classification helpers
# =========================================================================

def classify_water_years_by_max_zone(res_level_df):
    """Classify each water year (June-May) by the maximum drought zone reached."""
    df = res_level_df.copy()
    df['water_year'] = df.index.map(get_water_year)

    water_year_classifications = {}
    for wy in df['water_year'].unique():
        wy_data = df[df['water_year'] == wy]
        max_zone = wy_data['nyc'].max()
        water_year_classifications[wy] = max_zone

    return water_year_classifications


def classify_years_by_max_zone(res_level_df):
    """
    Classify each calendar year by the maximum drought zone reached.

    Parameters
    ----------
    res_level_df : pd.DataFrame
        Reservoir level DataFrame with 'nyc' column and datetime index

    Returns
    -------
    dict
        Mapping year -> {'max_zone': int, 'max_zone_date': pd.Timestamp}
    """
    nyc = res_level_df['nyc']
    years = nyc.index.year
    max_zone_per_year = nyc.groupby(years).max()
    max_zone_date_per_year = nyc.groupby(years).idxmax()

    return {
        year: {'max_zone': max_zone_per_year[year],
               'max_zone_date': max_zone_date_per_year[year]}
        for year in max_zone_per_year.index
    }


def get_zone_filter_label(zone_list):
    """Generate a human-readable label for a zone filter."""
    if zone_list is None:
        return "All Water Years"

    zone_labels = [ZONE_NAMES.get(z, f"Zone {z}") for z in sorted(zone_list, reverse=True)]

    if len(zone_labels) == 1:
        return f"Water Years with {zone_labels[0]}"
    return f"Water Years with {', '.join(zone_labels[:-1])}, or {zone_labels[-1]}"


# =========================================================================
# Representative year
# =========================================================================

def find_representative_year_for_zone(data, dataset_id, zone_filter=None):
    """
    Find the realization/water year with contribution ratio closest to mean.

    Returns dict with 'realization_id', 'year', 'ratio', 'mean_ratio',
    'contribution_trace', or None if no data.
    """
    realization_ids = list(data.res_level[dataset_id].keys())
    records = []

    for real_id in realization_ids:
        res_level_df = data.res_level[dataset_id][real_id]
        inflow_df = data.inflow[dataset_id][real_id]
        contribution_df = data.contribution[dataset_id][real_id]

        nyc_inflow = inflow_df[NYC_RESERVOIRS].sum(axis=1)
        nyc_contributions = contribution_df['mrf_montagueTrenton_nyc']

        dates = res_level_df.index
        water_years_arr = vectorized_water_year(dates)

        res_level_df = res_level_df.copy()
        res_level_df['water_year'] = water_years_arr

        for wy, wy_data in res_level_df.groupby('water_year'):
            if len(wy_data) < MIN_DAYS_FOR_COMPLETE_WATER_YEAR:
                continue

            max_zone = wy_data['nyc'].max()
            if zone_filter is not None and max_zone not in zone_filter:
                continue

            max_zone_date = wy_data[wy_data['nyc'] == max_zone].index[0]
            start_date = max_zone_date - pd.DateOffset(months=6)

            inflow_total = nyc_inflow.loc[start_date:max_zone_date].sum()
            contribution_total = nyc_contributions.loc[start_date:max_zone_date].sum()

            if inflow_total <= MIN_INFLOW_THRESHOLD:
                continue

            records.append({
                'realization_id': real_id,
                'water_year': wy,
                'contribution_ratio': 100.0 * contribution_total / inflow_total,
            })

    if len(records) == 0:
        return None

    df = pd.DataFrame(records)
    mean_ratio = df['contribution_ratio'].mean()
    df['distance_to_mean'] = abs(df['contribution_ratio'] - mean_ratio)
    closest_row = df.loc[df['distance_to_mean'].idxmin()]

    real_id = int(closest_row['realization_id'])
    wy = int(closest_row['water_year'])

    # Build contribution trace for the representative year
    contribution_df = data.contribution[dataset_id][real_id]
    nyc_contribution = contribution_df['mrf_montagueTrenton_nyc']
    montague_flow = data.major_flow[dataset_id][real_id]['delMontague']

    contrib_pct = np.where(montague_flow > 0,
                           100.0 * nyc_contribution / montague_flow,
                           np.nan)
    contrib_pct_series = pd.Series(contrib_pct, index=nyc_contribution.index)

    wy_arr = vectorized_water_year(contrib_pct_series.index)
    wy_mask = wy_arr == wy

    wy_contrib_data = contrib_pct_series[wy_mask]
    doy_contrib = wy_contrib_data.index.map(get_water_year_doy)
    contribution_trace = pd.Series(
        wy_contrib_data.values, index=doy_contrib,
        name=f'r{real_id}_wy{wy}_contrib',
    ).sort_index()

    return {
        'realization_id': real_id,
        'year': wy,
        'ratio': closest_row['contribution_ratio'],
        'mean_ratio': mean_ratio,
        'contribution_trace': contribution_trace,
    }
