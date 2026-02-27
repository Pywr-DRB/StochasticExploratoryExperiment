"""
Per-drought-event metric calculator for Sankey-Parallel Coordinate figure.

Computes metrics over exact drought event windows from raw HDF5 timeseries.
Each drought event (defined by SSI start/end dates) becomes one sample row
with hazard characteristics, system actions, and outcome metrics.

Reuses _evaluate_period() logic from methods.metrics.satisficing for
consistent satisficing classification.
"""

import numpy as np
import pandas as pd

from methods.config import NYC_RESERVOIRS, NYC_TOTAL_CAPACITY


def calculate_all_event_metrics(data, dataset_id, drought_events_df,
                                 storage_threshold=20.0, violation_days=3):
    """
    Calculate per-drought-event metrics from raw HDF5 timeseries.

    Parameters
    ----------
    data : pywrdrb.Data
        Data object loaded with result sets:
        res_storage, shortage, inflow, contribution, ibt_diversions, ibt_demands
    dataset_id : str
        Dataset identifier (e.g., 'stationary_ensemble')
    drought_events_df : pd.DataFrame
        Drought events with columns: start, end, realization_id, severity,
        magnitude, duration, avg_severity, etc.
    storage_threshold : float
        Minimum acceptable NYC storage percentage for satisficing (default: 20%)
    violation_days : int
        Maximum acceptable consecutive Montague violation days (default: 3)

    Returns
    -------
    pd.DataFrame
        One row per drought event with all computed metrics and classifications.
    """
    drought_events_df = drought_events_df.copy()
    drought_events_df['start'] = pd.to_datetime(drought_events_df['start'])
    drought_events_df['end'] = pd.to_datetime(drought_events_df['end'])

    realizations = drought_events_df['realization_id'].unique()
    all_results = []

    for r in realizations:
        # Load timeseries once per realization
        ts = _load_realization_timeseries(data, dataset_id, r)
        if ts is None:
            continue

        # Get this realization's drought events
        r_events = drought_events_df[drought_events_df['realization_id'] == r]

        for idx, event in r_events.iterrows():
            metrics = _calculate_single_event(
                event, ts, storage_threshold, violation_days
            )
            if metrics is not None:
                all_results.append(metrics)

    if not all_results:
        return pd.DataFrame()

    return pd.DataFrame(all_results)


def _load_realization_timeseries(data, dataset_id, realization_id):
    """
    Extract aligned timeseries for a single realization from pywrdrb.Data.

    Uses the same accessor pattern as satisficing.py and 06_calculate_satisficing.

    Returns dict with keys: storage_pct, montague_shortage, nyc_inflow,
    contribution, nyc_diversion, nyc_demand. All pd.Series with common index.
    Returns None if data unavailable.
    """
    try:
        # NYC aggregate storage as percentage (same as satisficing.py)
        storage_raw = data.res_storage[dataset_id][realization_id][NYC_RESERVOIRS].sum(axis=1)
        storage_pct = 100.0 * storage_raw / NYC_TOTAL_CAPACITY

        montague_shortage = data.shortage[dataset_id][realization_id]['delMontague']
        nyc_inflow = data.inflow[dataset_id][realization_id]['nyc']
        contribution = data.contribution[dataset_id][realization_id]

        # Handle contribution as Series or DataFrame
        if isinstance(contribution, pd.DataFrame):
            contribution = contribution['mrf_montagueTrenton_nyc']

        # NYC diversions
        nyc_diversion = data.ibt_diversions[dataset_id][realization_id]['delivery_nyc']
        nyc_demand = data.ibt_demands[dataset_id][realization_id]['demand_nyc']

        # Align to common index (same pattern as satisficing.py lines 98-108)
        common_idx = storage_pct.index
        montague_shortage = montague_shortage.reindex(common_idx, fill_value=0)
        nyc_inflow = nyc_inflow.reindex(common_idx, fill_value=0)
        contribution = contribution.reindex(common_idx, fill_value=0)
        nyc_diversion = nyc_diversion.reindex(common_idx, fill_value=0)
        nyc_demand = nyc_demand.reindex(common_idx, fill_value=0)

        return {
            'storage_pct': storage_pct,
            'montague_shortage': montague_shortage,
            'nyc_inflow': nyc_inflow,
            'contribution': contribution,
            'nyc_diversion': nyc_diversion,
            'nyc_demand': nyc_demand,
        }
    except (KeyError, IndexError) as e:
        print(f"  Warning: Could not load realization {realization_id}: {e}")
        return None


def _max_consecutive_positive(series):
    """
    Count maximum consecutive days where series > 0.

    Same logic as satisficing._evaluate_period() violation counting.
    """
    violations = series > 0
    if not violations.any():
        return 0
    groups = (violations != violations.shift()).cumsum()
    return int(violations.groupby(groups).sum().max())


def _calculate_single_event(event, ts, storage_threshold, violation_days):
    """
    Compute all metrics for a single drought event over its exact window.

    Parameters
    ----------
    event : pd.Series
        Single row from drought_events_df
    ts : dict
        Timeseries dict from _load_realization_timeseries
    storage_threshold : float
        Satisficing storage threshold (%)
    violation_days : int
        Satisficing max consecutive violation days

    Returns
    -------
    dict or None
        Metrics dict, or None if event window has no data
    """
    start = pd.Timestamp(event['start'])
    end = pd.Timestamp(event['end'])

    # Slice all timeseries to event window
    mask = (ts['storage_pct'].index >= start) & (ts['storage_pct'].index <= end)
    if not mask.any():
        return None

    stor = ts['storage_pct'][mask]
    short = ts['montague_shortage'][mask]
    inflow = ts['nyc_inflow'][mask]
    contrib = ts['contribution'][mask]
    diversion = ts['nyc_diversion'][mask]
    demand = ts['nyc_demand'][mask]

    # --- Drought characteristics (passed through from CSV) ---
    duration_days = (end - start).days
    start_month = start.month

    # --- Storage at drought onset ---
    start_idx = ts['storage_pct'].index.searchsorted(start)
    storage_at_start = ts['storage_pct'].iloc[min(start_idx, len(ts['storage_pct']) - 1)]

    # --- Metrics during drought window ---
    min_storage = stor.min()
    total_contribution = contrib.sum()
    total_inflow = inflow.sum()
    contribution_ratio = total_contribution / total_inflow if total_inflow > 0 else 0.0
    total_montague_shortage = short.sum()

    # Max consecutive Montague shortage days (same as satisficing._evaluate_period)
    max_consec = _max_consecutive_positive(short)

    # NYC diversion satisfaction ratio
    total_diversion = diversion.sum()
    total_demand = demand.sum()
    diversion_sat_ratio = total_diversion / total_demand if total_demand > 0 else 1.0

    # --- Classifications (matching F11 satisficing_category logic) ---
    storage_ok = bool(min_storage >= storage_threshold)
    montague_ok = bool(max_consec <= violation_days)

    if storage_ok and montague_ok:
        classification = 'all_pass'
    elif not storage_ok and not montague_ok:
        classification = 'both_fail'
    elif not storage_ok:
        classification = 'storage_fail'
    else:
        classification = 'montague_fail'

    return {
        'realization_id': event['realization_id'],
        'start': start,
        'end': end,
        'duration_days': duration_days,
        'start_month': start_month,
        'severity': event.get('severity', np.nan),
        'magnitude': event.get('magnitude', np.nan),
        'avg_severity': event.get('avg_severity', np.nan),
        'storage_at_start_pct': storage_at_start,
        'min_storage_pct': min_storage,
        'total_nyc_contribution_mg': total_contribution,
        'total_inflow_mg': total_inflow,
        'contribution_ratio': contribution_ratio,
        'max_consec_montague_days': int(max_consec),
        'total_montague_shortage_mg': total_montague_shortage,
        'nyc_diversion_sat_ratio': diversion_sat_ratio,
        'storage_ok': storage_ok,
        'montague_ok': montague_ok,
        'classification': classification,
    }
