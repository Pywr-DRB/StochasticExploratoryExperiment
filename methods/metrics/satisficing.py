"""
Satisficing conditions analysis for water system performance.

Evaluates whether annual performance metrics meet satisficing thresholds,
and annotates each year with drought overlap information.

Satisficing conditions:
1. NYC storage >= threshold throughout the calendar year
2. Montague flow target violations <= maximum consecutive days
"""

import numpy as np
import pandas as pd


# NYC reservoir storage capacities (MG)
NYC_RESERVOIRS = ['cannonsville', 'pepacton', 'neversink']
NYC_STORAGE_CAPACITIES = {
    'cannonsville': 95706,
    'pepacton': 140190,
    'neversink': 34941,
}
NYC_TOTAL_CAPACITY = sum(NYC_STORAGE_CAPACITIES.values())


def add_satisficing_category(df, storage_threshold=20.0, violation_days=3):
    """
    Annotate a DataFrame with satisficing pass/fail columns and a category label.

    Adds boolean columns ``storage_pass`` and ``montague_pass``, plus a
    ``satisficing_category`` column with one of:
    ``'all_pass'``, ``'storage_fail'``, ``'montague_fail'``, ``'multiple_fail'``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``nyc_min_storage_pct`` and
        ``montague_max_consec_shortage_days`` columns.
    storage_threshold : float
        Minimum acceptable NYC storage percentage (default: 20%).
    violation_days : int
        Maximum acceptable consecutive Montague violation days (default: 3).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional columns added in-place.
    """
    df['storage_pass'] = df['nyc_min_storage_pct'] >= storage_threshold
    df['montague_pass'] = df['montague_max_consec_shortage_days'] <= violation_days

    def _category(row):
        failures = []
        if not row['storage_pass']:
            failures.append('storage')
        if not row['montague_pass']:
            failures.append('montague')
        if len(failures) == 0:
            return 'all_pass'
        if len(failures) > 1:
            return 'multiple_fail'
        return 'storage_fail' if failures[0] == 'storage' else 'montague_fail'

    df['satisficing_category'] = df.apply(_category, axis=1)
    return df


def calculate_annual_satisficing(data, dataset_id, drought_events_df=None,
                                 storage_threshold=20.0, violation_days=3):
    """
    Calculate annual satisficing conditions for all realizations,
    annotated with drought overlap counts.

    Evaluates Jan 1 - Dec 31 of each year for every realization.
    If drought_events_df is provided, each (year, realization) row
    includes n_droughts_in_year (0 = non-drought year).

    Parameters
    ----------
    data : pywrdrb.Data
        Data object with pre-calculated shortage, contribution,
        res_storage, and inflow.
    dataset_id : str
        Dataset identifier.
    drought_events_df : pd.DataFrame or None
        Drought events with columns: start, end, realization_id.
        If None, n_droughts_in_year is set to 0 for all rows.
    storage_threshold : float
        Minimum acceptable NYC storage percentage (default: 20%).
    violation_days : int
        Maximum acceptable consecutive Montague violation days (default: 3).

    Returns
    -------
    pd.DataFrame
        Columns: year, realization, satisficing, nyc_min_storage_pct,
        montague_max_consec_shortage_days, nyc_inflow, montague_contrib, n_droughts_in_year
    """
    # Pre-compute drought counts per (realization, year)
    drought_counts = _count_droughts_per_year(drought_events_df)

    realizations = list(data.shortage[dataset_id].keys())
    rows = []

    for r in realizations:
        # Extract and align timeseries once per realization
        nyc_storage = data.res_storage[dataset_id][r][NYC_RESERVOIRS].sum(axis=1)
        nyc_storage_pct = 100.0 * nyc_storage / NYC_TOTAL_CAPACITY
        montague_shortage = data.shortage[dataset_id][r]['delMontague']
        nyc_inflow = data.inflow[dataset_id][r]['nyc']
        montague_contrib = data.contribution[dataset_id][r]['mrf_montagueTrenton_nyc']

        common_index = nyc_storage.index
        montague_shortage = montague_shortage.reindex(common_index, fill_value=0)
        nyc_inflow = nyc_inflow.reindex(common_index, fill_value=0)
        montague_contrib = montague_contrib.reindex(common_index, fill_value=0)

        years = pd.DatetimeIndex(common_index).year.unique()

        for year in years:
            mask = (common_index >= f'{year}-01-01') & (common_index <= f'{year}-12-31')
            if not mask.any():
                continue

            satisficing, min_storage, max_consec, total_inflow, total_contrib = \
                _evaluate_period(nyc_storage_pct[mask], montague_shortage[mask],
                                 nyc_inflow[mask], montague_contrib[mask],
                                 storage_threshold, violation_days)

            rows.append({
                'year': year,
                'realization': r,
                'satisficing': satisficing,
                'nyc_min_storage_pct': min_storage,
                'montague_max_consec_shortage_days': max_consec,
                'nyc_inflow': total_inflow,
                'montague_contrib': total_contrib,
                'n_droughts_in_year': drought_counts.get((r, year), 0),
            })

    return pd.DataFrame(rows)


def _count_droughts_per_year(drought_events_df):
    """Count drought events overlapping each (realization, year) pair.

    Returns dict mapping (realization_id, year) -> count.
    """
    counts = {}
    if drought_events_df is None or len(drought_events_df) == 0:
        return counts

    df = drought_events_df.copy()
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])

    for _, row in df.iterrows():
        r = row['realization_id']
        for year in range(row['start'].year, row['end'].year + 1):
            year_start = pd.Timestamp(f'{year}-01-01')
            year_end = pd.Timestamp(f'{year}-12-31')
            # Check overlap
            if not (year_end < row['start'] or year_start > row['end']):
                counts[(r, year)] = counts.get((r, year), 0) + 1

    return counts


def _evaluate_period(storage_pct, shortage, inflow, contrib,
                     storage_threshold, violation_days,
                     shortage_tolerance=0.0):
    """
    Evaluate satisficing conditions for a single time period.

    Parameters
    ----------
    shortage_tolerance : float
        Minimum shortage (MGD) to count as a violation (default: 0.0).
        The tolerance is normally already applied upstream when
        creating the shortage series via calculate_shortage_series()
        (DEFAULT_SHORTAGE_TOLERANCE_MGD = 1.0 MGD).  Set > 0 here
        only if additional filtering is desired at analysis time.

    Returns
    -------
    tuple
        (satisficing, min_storage, max_consecutive_violations,
         total_inflow, total_contribution)
    """
    min_storage = storage_pct.min()
    storage_ok = min_storage >= storage_threshold

    violations = shortage > shortage_tolerance
    if violations.any():
        groups = (violations != violations.shift()).cumsum()
        max_consec = violations.groupby(groups).sum().max()
    else:
        max_consec = 0

    montague_ok = max_consec <= violation_days

    total_inflow = inflow.sum()
    total_contrib = contrib.sum()

    satisficing = storage_ok and montague_ok

    return satisficing, min_storage, max_consec, total_inflow, total_contrib
