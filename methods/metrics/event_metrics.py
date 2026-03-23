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
from methods.load import load_ffmp_boundaries


def _build_ffmp_doy_lookup():
    """Build day-of-year lookup table for FFMP zone boundaries (%)."""
    fb = load_ffmp_boundaries()
    fb['doy'] = fb.index.dayofyear
    # level5 = Emergency, level4 = Warning, level3 = Watch
    cols = ['level5', 'level4', 'level3']
    return fb.groupby('doy')[cols].median()


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

    # Load FFMP boundaries once
    ffmp_doy = _build_ffmp_doy_lookup()

    realizations = drought_events_df['realization_id'].unique()
    all_results = []

    for r in realizations:
        # Load timeseries once per realization
        ts = _load_realization_timeseries(data, dataset_id, r)

        # Get this realization's drought events
        r_events = drought_events_df[drought_events_df['realization_id'] == r]

        for idx, event in r_events.iterrows():
            metrics = _calculate_single_event(
                event, ts, storage_threshold, violation_days, ffmp_doy
            )
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
        trenton_shortage = data.shortage[dataset_id][realization_id]['delTrenton']
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
        trenton_shortage = trenton_shortage.reindex(common_idx, fill_value=0)
        nyc_inflow = nyc_inflow.reindex(common_idx, fill_value=0)
        contribution = contribution.reindex(common_idx, fill_value=0)
        nyc_diversion = nyc_diversion.reindex(common_idx, fill_value=0)
        nyc_demand = nyc_demand.reindex(common_idx, fill_value=0)

        return {
            'storage_pct': storage_pct,
            'montague_shortage': montague_shortage,
            'trenton_shortage': trenton_shortage,
            'nyc_inflow': nyc_inflow,
            'contribution': contribution,
            'nyc_diversion': nyc_diversion,
            'nyc_demand': nyc_demand,
        }
    except (KeyError, IndexError) as e:
        raise RuntimeError(
            f"Failed to load timeseries for realization {realization_id}: {e}"
        ) from e


def _max_consecutive_positive(series, tolerance=0.0):
    """
    Count maximum consecutive days where series exceeds tolerance.

    Same logic as satisficing._evaluate_period() violation counting.

    Parameters
    ----------
    tolerance : float
        Minimum shortage (MGD) to count as a violation (default: 0.0).
        The tolerance is normally already applied upstream when
        creating the shortage series via calculate_shortage_series()
        (DEFAULT_SHORTAGE_TOLERANCE_MGD = 1.0 MGD).
    """
    violations = series > tolerance
    if not violations.any():
        return 0
    groups = (violations != violations.shift()).cumsum()
    return int(violations.groupby(groups).sum().max())


def _calculate_single_event(event, ts, storage_threshold, violation_days, ffmp_doy):
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
    ffmp_doy : pd.DataFrame
        FFMP boundaries by day-of-year (columns: level5, level4, level3)

    Returns
    -------
    dict
        Metrics dict for this drought event.

    Raises
    ------
    ValueError
        If the event window has no overlapping data in the timeseries.
    """
    start = pd.Timestamp(event['start'])
    end = pd.Timestamp(event['end'])

    # Slice all timeseries to event window
    mask = (ts['storage_pct'].index >= start) & (ts['storage_pct'].index <= end)
    if not mask.any():
        raise ValueError(
            f"No timeseries data found for drought event "
            f"{start.date()} to {end.date()} "
            f"(realization {event.get('realization_id', '?')}). "
            f"Data range: {ts['storage_pct'].index[0].date()} to "
            f"{ts['storage_pct'].index[-1].date()}."
        )

    stor = ts['storage_pct'][mask]
    mont_short = ts['montague_shortage'][mask]
    tren_short = ts['trenton_shortage'][mask]
    inflow = ts['nyc_inflow'][mask]
    contrib = ts['contribution'][mask]
    diversion = ts['nyc_diversion'][mask]
    demand = ts['nyc_demand'][mask]

    # --- Drought characteristics (passed through from CSV) ---
    duration_days = (end - start).days
    start_month = start.month

    # --- Peak severity date and month ---
    max_severity_date = pd.Timestamp(event.get('max_severity_date', pd.NaT))
    peak_severity_month = max_severity_date.month if pd.notna(max_severity_date) else np.nan

    # --- Storage at drought onset ---
    start_idx = ts['storage_pct'].index.searchsorted(start)
    storage_at_start = ts['storage_pct'].iloc[min(start_idx, len(ts['storage_pct']) - 1)]

    # --- Metrics during drought window ---
    min_storage = stor.min()
    min_storage_date = stor.idxmin()
    min_storage_month = min_storage_date.month if pd.notna(min_storage_date) else np.nan
    storage_drawdown = storage_at_start - min_storage

    # --- FFMP zone at min storage date (dynamic seasonal thresholds) ---
    min_doy = min_storage_date.dayofyear if pd.notna(min_storage_date) else 1
    # Handle leap year DOY > 365
    if min_doy > 365:
        min_doy = 365
    if min_doy in ffmp_doy.index:
        ffmp_at_min = ffmp_doy.loc[min_doy]
    else:
        ffmp_at_min = ffmp_doy.iloc[min(min_doy - 1, len(ffmp_doy) - 1)]

    emergency_threshold = ffmp_at_min['level5']
    warning_threshold = ffmp_at_min['level4']
    watch_threshold = ffmp_at_min['level3']

    if min_storage < emergency_threshold:
        ffmp_zone_at_min = 'Emergency'
    elif min_storage < warning_threshold:
        ffmp_zone_at_min = 'Warning'
    elif min_storage < watch_threshold:
        ffmp_zone_at_min = 'Watch'
    else:
        ffmp_zone_at_min = 'Normal'

    total_contribution = contrib.sum()
    total_inflow = inflow.sum()
    contribution_ratio = total_contribution / total_inflow if total_inflow > 0 else 0.0

    # Hazard rate
    severity_rate = float(event.get('magnitude', 0)) / duration_days if duration_days > 0 else 0.0

    # --- System action: NYC diversions ---
    total_diversion = diversion.sum()
    total_demand = demand.sum()
    diversion_sat_ratio = total_diversion / total_demand if total_demand > 0 else 1.0
    diversion_inflow_ratio = total_diversion / total_inflow if total_inflow > 0 else 0.0

    # --- Outcome: NYC shortage ---
    nyc_shortage = (demand - diversion).clip(lower=0)
    total_nyc_shortage = nyc_shortage.sum()
    nyc_shortage_pct = 100.0 * total_nyc_shortage / total_demand if total_demand > 0 else 0.0

    # --- Outcome: Montague shortage ---
    total_montague_shortage = mont_short.sum()
    max_consec_montague = _max_consecutive_positive(mont_short)
    # Peak 3-day rolling average shortage (MGD)
    if len(mont_short) >= 3:
        max_3day_avg_montague = mont_short.rolling(window=3, min_periods=1).mean().max()
    else:
        max_3day_avg_montague = mont_short.max() if len(mont_short) > 0 else 0.0

    # --- Outcome: Trenton shortage ---
    total_trenton_shortage = tren_short.sum()
    max_consec_trenton = _max_consecutive_positive(tren_short)

    # --- Classifications (matching satisficing_category logic) ---
    storage_ok = bool(min_storage >= storage_threshold)
    montague_ok = bool(max_consec_montague <= violation_days)

    if storage_ok and montague_ok:
        classification = 'all_pass'
    elif not storage_ok and not montague_ok:
        classification = 'both_fail'
    elif not storage_ok:
        classification = 'storage_fail'
    else:
        classification = 'montague_fail'

    return {
        # Identity
        'realization_id': event['realization_id'],
        'start': start,
        'end': end,
        # Antecedent
        'storage_at_start_pct': storage_at_start,
        'start_month': start_month,
        # Hazard
        'duration_days': duration_days,
        'severity': event.get('severity', np.nan),
        'magnitude': event.get('magnitude', np.nan),
        'avg_severity': event.get('avg_severity', np.nan),
        'severity_rate': severity_rate,
        'peak_severity_month': peak_severity_month,
        'total_inflow_mg': total_inflow,
        # Action
        'total_nyc_contribution_mg': total_contribution,
        'contribution_ratio': contribution_ratio,
        'total_nyc_diversion_mg': total_diversion,
        'total_nyc_demand_mg': total_demand,
        'nyc_diversion_inflow_ratio': diversion_inflow_ratio,
        'nyc_diversion_sat_ratio': diversion_sat_ratio,
        # Outcome: storage
        'event_min_storage_pct': min_storage,
        'min_storage_date': min_storage_date,
        'min_storage_month': min_storage_month,
        'storage_drawdown_pct': storage_drawdown,
        # FFMP zone classification at min storage (dynamic thresholds)
        'ffmp_zone_at_min': ffmp_zone_at_min,
        'ffmp_emergency_threshold': emergency_threshold,
        'ffmp_warning_threshold': warning_threshold,
        # Outcome: NYC shortage
        'total_nyc_shortage_mg': total_nyc_shortage,
        'nyc_shortage_pct': nyc_shortage_pct,
        # Outcome: Montague
        'max_consec_montague_days': int(max_consec_montague),
        'max_3day_avg_montague_mgd': float(max_3day_avg_montague),
        'total_montague_shortage_mg': total_montague_shortage,
        # Outcome: Trenton
        'max_consec_trenton_days': int(max_consec_trenton),
        'total_trenton_shortage_mg': total_trenton_shortage,
        # Classification
        'storage_ok': storage_ok,
        'montague_ok': montague_ok,
        'classification': classification,
    }
