"""
Satisficing conditions analysis for water system performance.

This module provides functions for calculating whether performance metrics
meet satisficing thresholds across different time periods and scenarios.

Satisficing conditions typically include:
1. NYC storage >= threshold throughout evaluation period
2. Montague flow target violations <= maximum consecutive days
"""

import numpy as np
import pandas as pd


def calculate_satisficing_conditions(data, dataset_id,
                                     period_start=None, period_end=None,
                                     period_type='year',
                                     storage_threshold=20.0,
                                     violation_days=3,
                                     evaluate_all_years=True):
    """
    Calculate satisficing conditions for specified time periods using pre-calculated metrics.

    This function can calculate satisficing for:
    - All simulation years (period_type='year', evaluate_all_years=True)
    - Specific time periods (provide period_start and period_end)
    - Custom evaluation periods (period_type='custom')

    Parameters
    ----------
    data : pywrdrb.Data
        Data object with pre-calculated shortage, contribution, res_storage, and inflow
    dataset_id : str
        Dataset identifier
    period_start : str, pd.Timestamp, or None
        Start date for evaluation period. If None and evaluate_all_years=True,
        uses June 1 of each year.
    period_end : str, pd.Timestamp, or None
        End date for evaluation period. If None and evaluate_all_years=True,
        uses Dec 31 of each year.
    period_type : str
        Type of period evaluation:
        - 'year': Evaluate June-Dec for each year
        - 'custom': Evaluate single custom period (requires period_start, period_end)
        - 'full': Evaluate entire simulation period
    storage_threshold : float, optional
        Minimum acceptable NYC storage percentage (default: 20%)
    violation_days : int, optional
        Maximum acceptable continuous Montague violation days (default: 3)
    evaluate_all_years : bool, optional
        If True, evaluate each year independently. If False, only evaluate the
        specified period_start to period_end (default: True)

    Returns
    -------
    pd.DataFrame
        Results with satisficing status and aggregated metrics.
        Columns depend on period_type:
        - 'year': year, realization, nyc_inflow, montague_contrib, satisficing,
                  min_storage_pct, max_violation_days
        - 'custom' or 'full': realization, period_start, period_end, nyc_inflow,
                              montague_contrib, satisficing, min_storage_pct,
                              max_violation_days
    """

    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']

    # Storage capacities for NYC reservoirs (MG)
    storage_capacities = {
        'cannonsville': 95706,
        'pepacton': 140190,
        'neversink': 34941
    }
    total_capacity = sum(storage_capacities.values())

    # Get realizations
    realizations = list(data.shortage[dataset_id].keys())

    results = {
        'realization': [],
        'nyc_inflow': [],
        'montague_contrib': [],
        'satisficing': [],
        'min_storage_pct': [],
        'max_violation_days': []
    }

    # Add period-specific columns
    if period_type == 'year':
        results['year'] = []
    else:
        results['period_start'] = []
        results['period_end'] = []

    for r in realizations:
        # Use pre-calculated data
        nyc_storage = data.res_storage[dataset_id][r][nyc_reservoirs].sum(axis=1)
        nyc_storage_pct = 100.0 * nyc_storage / total_capacity
        montague_shortage = data.shortage[dataset_id][r]['delMontague']
        nyc_inflow = data.inflow[dataset_id][r]['nyc']
        montague_contrib = data.contribution[dataset_id][r]['mrf_montagueTrenton_nyc']

        # Align all time series
        common_index = nyc_storage.index
        montague_shortage = montague_shortage.reindex(common_index, fill_value=0)
        nyc_inflow = nyc_inflow.reindex(common_index, fill_value=0)
        montague_contrib = montague_contrib.reindex(common_index, fill_value=0)

        if period_type == 'year' and evaluate_all_years:
            # Evaluate each year independently (Jun-Dec)
            years = pd.DatetimeIndex(common_index).year.unique()

            for year in years:
                # Define period (June 1 - Dec 31)
                if period_start is None:
                    p_start = f'{year}-06-01'
                else:
                    # Use custom start but within this year
                    p_start = pd.to_datetime(period_start).replace(year=year)

                if period_end is None:
                    p_end = f'{year}-12-31'
                else:
                    # Use custom end but within this year
                    p_end = pd.to_datetime(period_end).replace(year=year)

                mask = (common_index >= p_start) & (common_index <= p_end)

                if not mask.any():
                    continue

                # Calculate metrics for this period
                satisficing, min_storage, max_consec, total_inflow, total_contrib = \
                    _evaluate_period(nyc_storage_pct[mask], montague_shortage[mask],
                                    nyc_inflow[mask], montague_contrib[mask],
                                    storage_threshold, violation_days)

                # Store results
                results['year'].append(year)
                results['realization'].append(r)
                results['nyc_inflow'].append(total_inflow)
                results['montague_contrib'].append(total_contrib)
                results['satisficing'].append(satisficing)
                results['min_storage_pct'].append(min_storage)
                results['max_violation_days'].append(max_consec)

        else:
            # Evaluate single period (custom or full)
            if period_type == 'full':
                # Use entire simulation period
                p_start = common_index[0]
                p_end = common_index[-1]
            else:
                # Use specified period
                if period_start is None or period_end is None:
                    raise ValueError("period_start and period_end required for period_type='custom'")
                p_start = pd.to_datetime(period_start)
                p_end = pd.to_datetime(period_end)

            mask = (common_index >= p_start) & (common_index <= p_end)

            if mask.any():
                # Calculate metrics for this period
                satisficing, min_storage, max_consec, total_inflow, total_contrib = \
                    _evaluate_period(nyc_storage_pct[mask], montague_shortage[mask],
                                    nyc_inflow[mask], montague_contrib[mask],
                                    storage_threshold, violation_days)

                # Store results
                results['period_start'].append(p_start)
                results['period_end'].append(p_end)
                results['realization'].append(r)
                results['nyc_inflow'].append(total_inflow)
                results['montague_contrib'].append(total_contrib)
                results['satisficing'].append(satisficing)
                results['min_storage_pct'].append(min_storage)
                results['max_violation_days'].append(max_consec)

    return pd.DataFrame(results)


def _evaluate_period(storage_pct, shortage, inflow, contrib,
                     storage_threshold, violation_days):
    """
    Evaluate satisficing conditions for a single time period.

    Parameters
    ----------
    storage_pct : pd.Series
        NYC storage percentage time series for period
    shortage : pd.Series
        Montague shortage time series for period
    inflow : pd.Series
        NYC inflow time series for period
    contrib : pd.Series
        NYC contribution to Montague time series for period
    storage_threshold : float
        Minimum acceptable storage percentage
    violation_days : int
        Maximum acceptable consecutive violation days

    Returns
    -------
    tuple
        (satisficing, min_storage, max_consecutive_violations,
         total_inflow, total_contribution)
    """
    # Check storage condition
    min_storage = storage_pct.min()
    storage_ok = min_storage >= storage_threshold

    # Check Montague violation condition
    violations = shortage > 0
    if violations.any():
        # Calculate max consecutive violation days
        groups = (violations != violations.shift()).cumsum()
        max_consec = violations.groupby(groups).sum().max()
    else:
        max_consec = 0

    montague_ok = max_consec <= violation_days

    # Calculate aggregates
    total_inflow = inflow.sum()
    total_contrib = contrib.sum()

    # Satisficing = both conditions met
    satisficing = storage_ok and montague_ok

    return satisficing, min_storage, max_consec, total_inflow, total_contrib


def calculate_satisficing_during_droughts(data, dataset_id, drought_events_df,
                                          storage_threshold=20.0,
                                          violation_days=3):
    """
    Calculate satisficing conditions during specific drought events.

    Parameters
    ----------
    data : pywrdrb.Data
        Data object with pre-calculated metrics
    dataset_id : str
        Dataset identifier
    drought_events_df : pd.DataFrame
        Drought events with columns: start, end, realization_id, (other characteristics)
    storage_threshold : float, optional
        Minimum acceptable NYC storage percentage (default: 20%)
    violation_days : int, optional
        Maximum acceptable consecutive Montague violation days (default: 3)

    Returns
    -------
    pd.DataFrame
        Results with satisficing status for each drought event.
        Includes all columns from drought_events_df plus:
        nyc_inflow, montague_contrib, satisficing, min_storage_pct, max_violation_days
    """

    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']

    # Storage capacities
    storage_capacities = {
        'cannonsville': 95706,
        'pepacton': 140190,
        'neversink': 34941
    }
    total_capacity = sum(storage_capacities.values())

    # Convert date columns to datetime
    drought_events_df = drought_events_df.copy()
    drought_events_df['start'] = pd.to_datetime(drought_events_df['start'])
    drought_events_df['end'] = pd.to_datetime(drought_events_df['end'])

    results = []

    for idx, row in drought_events_df.iterrows():
        r = row['realization_id']
        drought_start = row['start']
        drought_end = row['end']

        # Get data for this realization
        nyc_storage = data.res_storage[dataset_id][r][nyc_reservoirs].sum(axis=1)
        nyc_storage_pct = 100.0 * nyc_storage / total_capacity
        montague_shortage = data.shortage[dataset_id][r]['delMontague']
        nyc_inflow = data.inflow[dataset_id][r]['nyc']
        montague_contrib = data.contribution[dataset_id][r]['mrf_montagueTrenton_nyc']

        # Align time series
        common_index = nyc_storage.index
        montague_shortage = montague_shortage.reindex(common_index, fill_value=0)
        nyc_inflow = nyc_inflow.reindex(common_index, fill_value=0)
        montague_contrib = montague_contrib.reindex(common_index, fill_value=0)

        # Filter to drought period
        mask = (common_index >= drought_start) & (common_index <= drought_end)

        if not mask.any():
            # No data for this drought period - skip
            continue

        # Evaluate satisficing for this drought period
        satisficing, min_storage, max_consec, total_inflow, total_contrib = \
            _evaluate_period(nyc_storage_pct[mask], montague_shortage[mask],
                            nyc_inflow[mask], montague_contrib[mask],
                            storage_threshold, violation_days)

        # Combine original drought characteristics with satisficing results
        result_row = row.to_dict()
        result_row['nyc_inflow'] = total_inflow
        result_row['montague_contrib'] = total_contrib
        result_row['satisficing'] = satisficing
        result_row['min_storage_pct'] = min_storage
        result_row['max_violation_days'] = max_consec

        results.append(result_row)

    return pd.DataFrame(results)


def calculate_satisficing_non_drought_periods(data, dataset_id, drought_events_df,
                                               storage_threshold=20.0,
                                               violation_days=3):
    """
    Calculate satisficing conditions during non-drought periods.

    This function identifies all time periods NOT covered by drought events
    and evaluates satisficing on a per-year basis for non-drought periods only.

    Parameters
    ----------
    data : pywrdrb.Data
        Data object with pre-calculated metrics
    dataset_id : str
        Dataset identifier
    drought_events_df : pd.DataFrame
        Drought events with columns: start, end, realization_id
    storage_threshold : float, optional
        Minimum acceptable NYC storage percentage (default: 20%)
    violation_days : int, optional
        Maximum acceptable consecutive Montague violation days (default: 3)

    Returns
    -------
    pd.DataFrame
        Results with satisficing status for non-drought periods.
        Columns: year, realization, nyc_inflow, montague_contrib, satisficing,
                min_storage_pct, max_violation_days, in_drought
    """

    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']

    # Storage capacities
    storage_capacities = {
        'cannonsville': 95706,
        'pepacton': 140190,
        'neversink': 34941
    }
    total_capacity = sum(storage_capacities.values())

    # Convert date columns
    drought_events_df = drought_events_df.copy()
    drought_events_df['start'] = pd.to_datetime(drought_events_df['start'])
    drought_events_df['end'] = pd.to_datetime(drought_events_df['end'])

    # Get realizations
    realizations = list(data.shortage[dataset_id].keys())

    results = []

    for r in realizations:
        # Get data for this realization
        nyc_storage = data.res_storage[dataset_id][r][nyc_reservoirs].sum(axis=1)
        nyc_storage_pct = 100.0 * nyc_storage / total_capacity
        montague_shortage = data.shortage[dataset_id][r]['delMontague']
        nyc_inflow = data.inflow[dataset_id][r]['nyc']
        montague_contrib = data.contribution[dataset_id][r]['mrf_montagueTrenton_nyc']

        # Align time series
        common_index = nyc_storage.index
        montague_shortage = montague_shortage.reindex(common_index, fill_value=0)
        nyc_inflow = nyc_inflow.reindex(common_index, fill_value=0)
        montague_contrib = montague_contrib.reindex(common_index, fill_value=0)

        # Get drought periods for this realization
        realization_droughts = drought_events_df[
            drought_events_df['realization_id'] == r
        ]

        # Get years
        years = pd.DatetimeIndex(common_index).year.unique()

        for year in years:
            # Define evaluation period (Jun-Dec)
            period_start = pd.to_datetime(f'{year}-06-01')
            period_end = pd.to_datetime(f'{year}-12-31')

            # Check if this period overlaps with any drought
            overlaps_drought = False
            for _, drought in realization_droughts.iterrows():
                drought_start = drought['start']
                drought_end = drought['end']

                # Check for overlap
                if not (period_end < drought_start or period_start > drought_end):
                    overlaps_drought = True
                    break

            # Only evaluate if period does NOT overlap with drought
            if not overlaps_drought:
                mask = (common_index >= period_start) & (common_index <= period_end)

                if not mask.any():
                    continue

                # Evaluate satisficing
                satisficing, min_storage, max_consec, total_inflow, total_contrib = \
                    _evaluate_period(nyc_storage_pct[mask], montague_shortage[mask],
                                    nyc_inflow[mask], montague_contrib[mask],
                                    storage_threshold, violation_days)

                results.append({
                    'year': year,
                    'realization': r,
                    'nyc_inflow': total_inflow,
                    'montague_contrib': total_contrib,
                    'satisficing': satisficing,
                    'min_storage_pct': min_storage,
                    'max_violation_days': max_consec,
                    'in_drought': False
                })

    return pd.DataFrame(results)
