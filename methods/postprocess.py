"""
Core functions for postprocessing Pywr-DRB simulation outputs.

This module contains functions for:
- Loading and combining ensemble set outputs
- Calculating performance metrics
- Adding derived metrics (shortages, contributions)
- Exporting combined datasets
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from methods.metrics.shortfall import (
    add_trenton_equiv_flow,
    get_flow_and_target_values,
    calculate_shortage_series,
    calculate_hashimoto_metrics,
)
from methods.zone_duration_metrics import calculate_drought_zone_events
from methods.config import (
    N_REALIZATIONS_PER_ENSEMBLE_SET,
    N_ENSEMBLE_SETS,
    NYC_TOTAL_CAPACITY,
    NYC_RESERVOIRS,
    get_ensemble_set_spec
)


def assign_water_year(index):
    """
    Assign water year labels to a datetime index.

    Water year starts June 1. Dates June 1 - Dec 31 belong to that calendar
    year's water year; dates Jan 1 - May 31 belong to the previous year's.

    Example: WY2030 = June 1 2030 through May 31 2031.
    """
    return np.where(index.month >= 6, index.year, index.year - 1)


def _max_consec_in_mask(series, mask):
    """
    Find longest consecutive run of True values in *series* within days
    where *mask* is True.

    If mask has no True values, returns 0.
    """
    if not mask.any():
        return 0
    masked = series[mask]
    if not masked.any():
        return 0
    groups = (masked != masked.shift()).cumsum()
    return int(masked.groupby(groups).sum().max())


def _build_drought_day_mask(index, drought_events_df, realization_id):
    """
    Build a boolean array marking which days fall within SSI drought events.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Full timeseries index.
    drought_events_df : pd.DataFrame
        Columns: start, end, realization_id.
    realization_id : int
        Realization to filter events for.

    Returns
    -------
    np.ndarray of bool, same length as index
    """
    mask = np.zeros(len(index), dtype=bool)
    r_events = drought_events_df[drought_events_df['realization_id'] == realization_id]

    for _, event in r_events.iterrows():
        start = pd.Timestamp(event['start'])
        end = pd.Timestamp(event['end'])
        mask |= (index >= start) & (index <= end)

    return mask


def _compute_period_metrics(shortage_dict, target_dict, nyc_storage_pct,
                            nyc_contribution, nyc_zone_level,
                            period_mask, period_name, water_year):
    """
    Compute all 20 annual metrics for a single (water_year, period) slice.

    Parameters
    ----------
    shortage_dict : dict
        {loc: pd.Series} for 'montague', 'trenton', 'nyc'.
    target_dict : dict
        {loc: pd.Series} for 'montague', 'trenton', 'nyc'.
    nyc_storage_pct : pd.Series
        Daily NYC combined storage as % of capacity.
    nyc_contribution : pd.Series
        Daily NYC releases for downstream targets (MG).
    nyc_zone_level : pd.Series or None
        Daily NYC drought zone level.
    period_mask : np.ndarray of bool
        Which days within the water year belong to this period.
    period_name : str
        'all', 'drought', or 'nondrought'.
    water_year : int
        Water year label.

    Returns
    -------
    dict with all metric values for this period.
    """
    ndays = int(period_mask.sum())

    record = {
        'period': period_name,
        'ndays_in_period': ndays,
    }

    # Pre-compute 7-day week bins for the water year (used for reliability).
    # Week boundaries are defined on the full water year, not the period subset.
    n_total_days = len(period_mask)
    week_ids = np.arange(n_total_days) // 7  # 0-indexed week number

    # Per-location shortage metrics (4 x 3 locations = 12 columns)
    for loc in ['montague', 'trenton', 'nyc']:
        shortage = shortage_dict[loc]
        target = target_dict[loc]

        if ndays == 0:
            record[f'{loc}_reliability'] = np.nan
            record[f'{loc}_shortage_mg'] = np.nan
            record[f'{loc}_max_consec_shortage_days'] = np.nan
            record[f'{loc}_max_1day_shortage_mg'] = np.nan
            continue

        period_shortage = shortage[period_mask]

        # --- Weekly reliability ---
        # For each 7-day week in the water year, count deficit days
        # (shortage > 0) that fall within the period. A week is "failed"
        # if it has >= 3 deficit days within the period. Only weeks with
        # at least 1 day in the period contribute.
        is_deficit_full = (shortage.values > 0) & period_mask
        period_day_in_week = period_mask.copy()

        n_weeks = week_ids[-1] + 1 if n_total_days > 0 else 0
        weeks_counted = 0
        weeks_failed = 0
        for w in range(n_weeks):
            week_mask = week_ids == w
            days_in_period = int(period_day_in_week[week_mask].sum())
            if days_in_period == 0:
                continue
            deficit_days = int(is_deficit_full[week_mask].sum())
            weeks_counted += 1
            if deficit_days >= 3:
                weeks_failed += 1

        if weeks_counted > 0:
            reliability = 1.0 - weeks_failed / weeks_counted
        else:
            reliability = np.nan
        record[f'{loc}_reliability'] = reliability

        # --- Shortage volume, peak, consecutive days ---
        shortage_sum = period_shortage.sum()
        record[f'{loc}_shortage_mg'] = float(shortage_sum)
        record[f'{loc}_max_1day_shortage_mg'] = float(period_shortage.max()) if len(period_shortage) > 0 else 0.0

        # Max consecutive shortage days within period
        shortage_positive = period_shortage > 0
        if shortage_positive.any():
            groups = (shortage_positive != shortage_positive.shift()).cumsum()
            max_consec = int(shortage_positive.groupby(groups).sum().max())
        else:
            max_consec = 0
        record[f'{loc}_max_consec_shortage_days'] = max_consec

    # NYC storage metrics (5 columns)
    if ndays == 0:
        record['nyc_min_storage_pct'] = np.nan
        record['ndays_storage_below_20pct'] = np.nan
        record['ndays_storage_below_30pct'] = np.nan
    else:
        period_storage = nyc_storage_pct[period_mask]
        record['nyc_min_storage_pct'] = float(period_storage.min())
        record['ndays_storage_below_20pct'] = int((period_storage < 20).sum())
        record['ndays_storage_below_30pct'] = int((period_storage < 30).sum())

    # Point-in-time storage (only for period='all')
    if period_name == 'all':
        june1 = nyc_storage_pct[(nyc_storage_pct.index.month == 6) & (nyc_storage_pct.index.day == 1)]
        sept1 = nyc_storage_pct[(nyc_storage_pct.index.month == 9) & (nyc_storage_pct.index.day == 1)]
        record['june1_storage_pct'] = float(june1.iloc[0]) if len(june1) > 0 else np.nan
        record['sept1_storage_pct'] = float(sept1.iloc[0]) if len(sept1) > 0 else np.nan
    else:
        record['june1_storage_pct'] = np.nan
        record['sept1_storage_pct'] = np.nan

    # System metrics (3 columns)
    if ndays == 0:
        record['nyc_contribution_mg'] = np.nan
        record['ndays_combined_stress'] = np.nan
        record['max_zone'] = np.nan
    else:
        period_contribution = nyc_contribution[period_mask]
        record['nyc_contribution_mg'] = float(period_contribution.sum())

        montague_shortage_positive = shortage_dict['montague'][period_mask] > 0
        nyc_shortage_positive = shortage_dict['nyc'][period_mask] > 0
        record['ndays_combined_stress'] = int((montague_shortage_positive & nyc_shortage_positive).sum())

        if nyc_zone_level is not None:
            period_zone = nyc_zone_level[period_mask]
            record['max_zone'] = int(period_zone.max()) if len(period_zone) > 0 else np.nan
        else:
            record['max_zone'] = np.nan

    return record


def calculate_annual_metrics(data, dataset_id, realizations, drought_events_df):
    """
    Calculate annual performance metrics per water year, split by period.

    Produces one row per (realization_id, water_year, period) with 20 metrics.
    Period is one of 'all', 'drought', 'nondrought'.

    Parameters
    ----------
    data : pywrdrb.Data
        Data object with shortage, mrf_target, res_storage, ibt_diversions,
        ibt_demands, contribution, res_level.
    dataset_id : str
        Dataset identifier.
    realizations : list
        List of realization IDs.
    drought_events_df : pd.DataFrame
        SSI drought events with columns: start, end, realization_id.
        Required — raises ValueError if None.

    Returns
    -------
    pd.DataFrame
        Columns: realization_id, water_year, period, 20 metric columns,
        ndays_in_period, n_droughts_in_year, drought_days_in_year.
    """
    if drought_events_df is None:
        raise ValueError(
            "drought_events_df is required. Run SSI drought identification first."
        )

    # Ensure datetime types
    drought_events_df = drought_events_df.copy()
    drought_events_df['start'] = pd.to_datetime(drought_events_df['start'])
    drought_events_df['end'] = pd.to_datetime(drought_events_df['end'])

    print(f"  Calculating annual metrics for {len(realizations)} realizations...")

    all_records = []
    shortage_locations = ['montague', 'trenton', 'nyc']
    # Map location names to get_flow_and_target_values node names
    loc_to_node = {'montague': 'delMontague', 'trenton': 'delTrenton', 'nyc': 'nyc'}

    for i, r in enumerate(realizations):
        if (i > 0) and (i % 100 == 0):
            print(f"    Processed {i}/{len(realizations)} realizations...")

        # Build shortage and target series for all locations
        shortage_dict = {}
        target_dict = {}
        for loc in shortage_locations:
            node = loc_to_node[loc]
            flow_series, target_series = get_flow_and_target_values(
                data, node, dataset_id, r, start_date=None, end_date=None
            )
            shortage_series = calculate_shortage_series(
                target_series, flow_series,
                min_duration=0 if loc == 'nyc' else 3,
                warmup_days=0 if loc == 'nyc' else 3,
            )
            shortage_dict[loc] = shortage_series
            target_dict[loc] = target_series

        # NYC storage
        nyc_storage = data.res_storage[dataset_id][r][NYC_RESERVOIRS].sum(axis=1)
        nyc_storage_pct = 100.0 * nyc_storage / NYC_TOTAL_CAPACITY

        # NYC contribution
        nyc_contribution = data.contribution[dataset_id][r]['mrf_montagueTrenton_nyc']

        # NYC zone level
        nyc_zone_level = None
        if hasattr(data, 'res_level') and dataset_id in data.res_level and r in data.res_level[dataset_id]:
            nyc_zone_level = data.res_level[dataset_id][r]['nyc']

        # Use a common index (storage is typically the reference)
        common_idx = nyc_storage_pct.index

        # Align all series to common index
        for loc in shortage_locations:
            shortage_dict[loc] = shortage_dict[loc].reindex(common_idx, fill_value=0)
            target_dict[loc] = target_dict[loc].reindex(common_idx, fill_value=0)
        nyc_contribution = nyc_contribution.reindex(common_idx, fill_value=0)
        if nyc_zone_level is not None:
            nyc_zone_level = nyc_zone_level.reindex(common_idx, fill_value=3)

        # Assign water years
        wy_labels = assign_water_year(common_idx)

        # Build drought day mask for this realization
        drought_mask_full = _build_drought_day_mask(common_idx, drought_events_df, r)

        # Count droughts per water year
        r_events = drought_events_df[drought_events_df['realization_id'] == r]

        # Process each water year
        unique_wys = np.unique(wy_labels)
        for wy in unique_wys:
            wy_mask = wy_labels == wy

            # Subset all series to this water year
            wy_idx = common_idx[wy_mask]
            wy_shortage = {loc: shortage_dict[loc][wy_mask] for loc in shortage_locations}
            wy_target = {loc: target_dict[loc][wy_mask] for loc in shortage_locations}
            wy_storage = nyc_storage_pct[wy_mask]
            wy_contribution = nyc_contribution[wy_mask]
            wy_zone = nyc_zone_level[wy_mask] if nyc_zone_level is not None else None

            # Drought mask within this water year
            wy_drought_mask = drought_mask_full[wy_mask]

            # Count drought events overlapping this water year
            wy_start = wy_idx[0]
            wy_end = wy_idx[-1]
            n_droughts = 0
            for _, ev in r_events.iterrows():
                if not (ev['end'] < wy_start or ev['start'] > wy_end):
                    n_droughts += 1

            drought_days = int(wy_drought_mask.sum())
            all_mask = np.ones(len(wy_idx), dtype=bool)

            # Compute metrics for each period
            for period_name, period_mask in [('all', all_mask),
                                              ('drought', wy_drought_mask),
                                              ('nondrought', ~wy_drought_mask)]:
                record = _compute_period_metrics(
                    wy_shortage, wy_target, wy_storage,
                    wy_contribution, wy_zone,
                    period_mask, period_name, wy
                )
                record['realization_id'] = r
                record['water_year'] = int(wy)
                record['n_droughts_in_year'] = n_droughts
                record['drought_days_in_year'] = drought_days
                all_records.append(record)

    metrics_df = pd.DataFrame(all_records)

    # Reorder columns: index cols first, then annotations, then metrics
    index_cols = ['realization_id', 'water_year', 'period']
    annotation_cols = ['ndays_in_period', 'n_droughts_in_year', 'drought_days_in_year']
    metric_cols = [c for c in metrics_df.columns if c not in index_cols + annotation_cols]
    metrics_df = metrics_df[index_cols + annotation_cols + metric_cols]

    print(f"  Calculated {len(metrics_df)} rows × {len(metrics_df.columns)} annual metrics")
    return metrics_df


def calculate_hashimoto_all(data, dataset_id, realizations):
    """
    Calculate Hashimoto (1982) RRV metrics for all realizations.

    Returns two DataFrames:
    1. Simulation-level metrics (reliability, resiliency) per realization
    2. Per-event detail (start, end, duration, severity, intensity, vulnerability)

    Parameters
    ----------
    data : pywrdrb.Data
        Data object with major_flow and mrf_target.
    dataset_id : str
        Dataset identifier.
    realizations : list
        List of realization IDs.

    Returns
    -------
    hashimoto_metrics_df : pd.DataFrame
        One row per realization with reliability and resiliency for
        Montague and Trenton.
    hashimoto_events_df : pd.DataFrame
        One row per shortage event per location per realization.
    """
    print(f"  Calculating Hashimoto RRV metrics for {len(realizations)} realizations...")

    sim_records = []
    event_records = []

    loc_config = {
        'montague': {'flow_col': 'delMontague', 'target_node': 'delMontague'},
        'trenton': {'flow_col': 'delTrenton_equiv', 'target_node': 'delTrenton'},
    }

    for i, r in enumerate(realizations):
        if (i > 0) and (i % 100 == 0):
            print(f"    Processed {i}/{len(realizations)} realizations...")

        sim_record = {'realization_id': r}

        for loc, config in loc_config.items():
            flows = data.major_flow[dataset_id][r][config['flow_col']]
            thresholds = data.mrf_target[dataset_id][r][config['target_node']]

            result = calculate_hashimoto_metrics(
                flows, thresholds,
                shortfall_break_length=7,
            )

            sim_record[f'hashimoto_reliability_{loc}'] = result['reliability']
            sim_record[f'hashimoto_resiliency_{loc}'] = result['resiliency']

            # Collect per-event detail
            events_df = result['events']
            if len(events_df) > 0:
                for _, ev in events_df.iterrows():
                    event_records.append({
                        'realization_id': r,
                        'location': loc,
                        'start': ev['start'].isoformat() if hasattr(ev['start'], 'isoformat') else str(ev['start']),
                        'end': ev['end'].isoformat() if hasattr(ev['end'], 'isoformat') else str(ev['end']),
                        'duration_days': int(ev['duration']),
                        'severity_mg': float(ev['severity']),
                        'intensity_mgd': float(ev['intensity']),
                        'vulnerability_mgd': float(ev['vulnerability']),
                    })

        sim_records.append(sim_record)

    hashimoto_metrics_df = pd.DataFrame(sim_records)
    hashimoto_events_df = pd.DataFrame(event_records)

    print(f"  Hashimoto: {len(hashimoto_metrics_df)} realizations, "
          f"{len(hashimoto_events_df)} shortage events")

    return hashimoto_metrics_df, hashimoto_events_df


def classify_years_by_max_zone(res_level_df):
    """
    Classify each year by the maximum drought zone reached.

    Parameters
    ----------
    res_level_df : pd.DataFrame
        Reservoir level DataFrame with 'nyc' column and datetime index

    Returns
    -------
    year_classifications : dict
        Dictionary mapping year -> {'max_zone': int, 'max_zone_date': pd.Timestamp}
    """
    nyc = res_level_df['nyc']
    years = nyc.index.year

    # Vectorized: find max zone per year and its first occurrence date
    max_zone_per_year = nyc.groupby(years).max()
    max_zone_date_per_year = nyc.groupby(years).idxmax()

    return {
        year: {'max_zone': max_zone_per_year[year],
               'max_zone_date': max_zone_date_per_year[year]}
        for year in max_zone_per_year.index
    }


def _contribution_metrics_for_realization(args):
    """
    Calculate contribution metrics for a single realization (worker function).

    Designed to be called from ProcessPoolExecutor or directly in a loop.
    Uses cumulative sums for O(1) window-sum lookups instead of repeated masking.

    Parameters
    ----------
    args : tuple
        (r, res_level, res_storage_nyc, contribution, inflow_nyc,
         diversion, demand, nyc_total_capacity, window_days)

    Returns
    -------
    records : list of dict
        One record per year with all window metrics
    """
    (r, res_level, res_storage_nyc, contribution, inflow_nyc,
     diversion, demand, nyc_total_capacity, window_days) = args

    # NYC combined storage percentage
    nyc_storage_pct = 100.0 * res_storage_nyc / nyc_total_capacity

    # Classify years by drought zone (vectorized groupby)
    nyc_zone = res_level['nyc']
    years = nyc_zone.index.year
    max_zone_per_year = nyc_zone.groupby(years).max()
    max_zone_date_per_year = nyc_zone.groupby(years).idxmax()

    # Min storage per year
    min_storage_per_year = nyc_storage_pct.groupby(nyc_storage_pct.index.year).min()

    # Build cumulative sums for O(1) window lookups
    # Align all series to same index (they should already be aligned)
    idx = contribution.index
    contrib_cumsum = contribution.values.cumsum()
    inflow_cumsum = inflow_nyc.values.cumsum()
    div_cumsum = diversion.values.cumsum()
    dem_cumsum = demand.values.cumsum()

    # Precompute 30-day rolling demand satisfaction for worst-1mo calc
    rolling_div_30 = diversion.rolling(30).sum()
    rolling_dem_30 = demand.rolling(30).sum()
    rolling_sat_30 = (rolling_div_30 / rolling_dem_30).clip(upper=1.0)

    records = []

    for year in max_zone_per_year.index:
        annual_max_zone = max_zone_per_year[year]
        annual_max_zone_date = max_zone_date_per_year[year]
        annual_min_storage = min_storage_per_year[year]

        record = {
            'realization_id': r,
            'year': year,
            'annual_max_zone': annual_max_zone,
            'annual_max_zone_date': annual_max_zone_date.isoformat(),
            'annual_min_storage_pct': annual_min_storage
        }

        # Find the position of annual_max_zone_date in the index
        # Use searchsorted for O(log n) lookup
        end_pos = idx.searchsorted(annual_max_zone_date, side='right') - 1
        if end_pos < 0:
            # annual_max_zone_date is before the start of the timeseries
            for W in window_days:
                record.update({
                    f'contribution_total_{W}d': np.nan,
                    f'contribution_ratio_{W}d': np.nan,
                    f'inflow_total_{W}d': np.nan,
                    f'demand_satisfaction_{W}d': np.nan,
                    f'worst_1mo_demand_sat_{W}d': np.nan,
                })
            records.append(record)
            continue

        # Compute metrics for each window using cumsum differences
        for W in window_days:
            start_date = annual_max_zone_date - pd.Timedelta(days=W)
            start_pos = idx.searchsorted(start_date, side='left')

            # Cumsum-based window sums: sum[start:end+1] = cumsum[end] - cumsum[start-1]
            if start_pos > 0:
                contrib_total = contrib_cumsum[end_pos] - contrib_cumsum[start_pos - 1]
                inflow_total = inflow_cumsum[end_pos] - inflow_cumsum[start_pos - 1]
                total_div = div_cumsum[end_pos] - div_cumsum[start_pos - 1]
                total_dem = dem_cumsum[end_pos] - dem_cumsum[start_pos - 1]
            else:
                contrib_total = contrib_cumsum[end_pos]
                inflow_total = inflow_cumsum[end_pos]
                total_div = div_cumsum[end_pos]
                total_dem = dem_cumsum[end_pos]

            contrib_ratio = 100.0 * contrib_total / inflow_total if inflow_total > 0 else np.nan
            demand_sat = min(total_div / total_dem, 1.0) if total_dem > 0 else 1.0

            # Worst 1-month rolling demand satisfaction within window
            # Skip first 29 positions: rolling values there use pre-window data.
            # This matches the original which computed rolling on the window subset
            # (where the first 29 values were NaN and excluded by .min()).
            window_len = end_pos - start_pos + 1
            if window_len >= 30:
                worst_1mo = 100.0 * rolling_sat_30.iloc[start_pos + 29:end_pos + 1].min()
            elif window_len > 0:
                worst_1mo = 100.0 * demand_sat
            else:
                worst_1mo = np.nan

            record.update({
                f'contribution_total_{W}d': contrib_total,
                f'contribution_ratio_{W}d': contrib_ratio,
                f'inflow_total_{W}d': inflow_total,
                f'demand_satisfaction_{W}d': demand_sat,
                f'worst_1mo_demand_sat_{W}d': worst_1mo,
            })

        records.append(record)

    return records


def calculate_contribution_analysis_metrics(data, dataset_id, realizations,
                                            window_days=[30, 60, 90, 120, 150, 180, 270]):
    """
    Calculate year-level contribution analysis metrics for multiple aggregation windows.

    This function pre-computes metrics used by contribution analysis plotting scripts,
    eliminating the need to recalculate on-the-fly during figure generation.

    Uses cumulative-sum based window calculations and optional multiprocessing
    for significantly faster execution on large ensembles.

    Parameters
    ----------
    data : pywrdrb.Data
        Data object with res_level, res_storage, contribution, inflow, ibt_diversions, ibt_demands
    dataset_id : str
        Dataset identifier
    realizations : list
        List of realization IDs
    window_days : list of int
        Window lengths in days to compute metrics for (default: [30, 60, 90, 120, 150, 180, 270])

    Returns
    -------
    metrics_df : pd.DataFrame
        DataFrame with columns:
        - realization_id, year, annual_max_zone, annual_max_zone_date, annual_min_storage_pct
        - For each window W in window_days:
          - contribution_total_{W}d: NYC→Montague contributions sum (MG)
          - contribution_ratio_{W}d: (contribution/inflow) × 100 (%)
          - inflow_total_{W}d: NYC reservoir inflow sum (MG)
          - demand_satisfaction_{W}d: volumetric diversion/demand ratio (≤1.0)
          - worst_1mo_demand_sat_{W}d: minimum 30-day rolling demand satisfaction (%)
    """
    print(f"  Calculating contribution analysis metrics for {len(realizations)} realizations...")

    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']

    # Prepare arguments for each realization
    all_records = []
    n_done = 0

    for r in realizations:
        # Extract per-realization data (already in memory)
        res_level = data.res_level[dataset_id][r]
        res_storage_nyc = data.res_storage[dataset_id][r][nyc_reservoirs].sum(axis=1)
        contribution = data.contribution[dataset_id][r]['mrf_montagueTrenton_nyc']
        inflow_nyc = data.inflow[dataset_id][r][nyc_reservoirs].sum(axis=1)
        diversion = data.ibt_diversions[dataset_id][r]['delivery_nyc']
        demand = data.ibt_demands[dataset_id][r]['demand_nyc']

        records = _contribution_metrics_for_realization(
            (r, res_level, res_storage_nyc, contribution, inflow_nyc,
             diversion, demand, NYC_TOTAL_CAPACITY, window_days)
        )
        all_records.extend(records)

        n_done += 1
        if n_done % 500 == 0:
            print(f"    Processed {n_done}/{len(realizations)} realizations...")

    metrics_df = pd.DataFrame(all_records)

    print(f"  Calculated {len(metrics_df)} year-realization pairs × {len(metrics_df.columns)} metrics")

    return metrics_df


def save_metrics_csv(df, dataset_id, suffix, output_dir):
    """
    Save a metrics DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
    dataset_id : str
    suffix : str
        e.g. 'annual_metrics', 'hashimoto_metrics'
    output_dir : str
    """
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/{dataset_id}_{suffix}.csv"
    df.to_csv(fname, index=False)
    print(f"  Saved: {fname} ({len(df)} rows)")
    return fname


def calculate_and_save_zone_duration_events(data, dataset_id, realizations, output_dir="./pywrdrb/performance_metrics"):
    """
    Calculate drought zone episode durations for all realizations and save to CSV.

    Each contiguous drought episode (NYC zone >= 4, ending after 7+ consecutive days
    below zone 4) is recorded as one row. The episode is attributed to the maximum
    zone level reached during it; lower zones passed through are not recorded separately.
    Duration is the full episode length from the first to the last drought day.

    Parameters
    ----------
    data : pywrdrb.Data
        Data object with res_level containing the 'nyc' zone timeseries.
    dataset_id : str
        Dataset identifier.
    realizations : list
        List of realization IDs.
    output_dir : str
        Output directory for the CSV file.

    Returns
    -------
    events_df : pd.DataFrame
        DataFrame with columns: realization_id, start_date, end_date,
        duration_days, max_zone.
    """
    print(f"  Calculating zone drought episode durations...")

    records = []
    for r in realizations:
        zone_series = data.res_level[dataset_id][r]['nyc']
        episodes = calculate_drought_zone_events(zone_series, min_end_days=7)
        for ep in episodes:
            records.append({
                'realization_id': r,
                'start_date': ep['start_date'].isoformat(),
                'end_date': ep['end_date'].isoformat(),
                'duration_days': ep['duration_days'],
                'max_zone': ep['max_zone'],
            })

    events_df = pd.DataFrame(records)

    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/{dataset_id}_zone_duration_events.csv"
    events_df.to_csv(fname, index=False)
    print(f"  Saved: {fname} ({len(events_df)} episodes across {len(realizations)} realizations)")

    return events_df




def _compute_shortage(data, dataset_id):
    """Compute shortage from major_flow and mrf_target, matching MPI postprocessing logic."""
    realizations = sorted(data.major_flow[dataset_id].keys())
    nodes = ['delMontague', 'delTrenton', 'nyc', 'nj']

    shortage_dict = {}
    for r in realizations:
        node_shortages = {}
        for node in nodes:
            flow_series, target_series = get_flow_and_target_values(
                data, node, dataset_id, r, start_date=None, end_date=None
            )
            node_shortages[node] = calculate_shortage_series(target_series, flow_series)

        shortage_dict[r] = pd.DataFrame(node_shortages)

    data.shortage = {dataset_id: shortage_dict}


def _compute_contribution(data, dataset_id):
    """Compute NYC contribution to downstream targets from nyc_release_components."""
    realizations = sorted(data.nyc_release_components[dataset_id].keys())
    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']
    contribution_columns = [f'mrf_montagueTrenton_{res}' for res in nyc_reservoirs]

    contribution_dict = {}
    for r in realizations:
        release_components = data.nyc_release_components[dataset_id][r]
        total_nyc_contribution = release_components.loc[:, contribution_columns].sum(axis=1)
        contribution_dict[r] = total_nyc_contribution.to_frame(name='mrf_montagueTrenton_nyc')

    data.contribution = {dataset_id: contribution_dict}


def _load_gage_flow(data, dataset_id):
    """Load gage flow from hydrologic model flow files and combine with global realization IDs."""
    from methods.load import load_gage_flow_data
    data.gage_flow = {dataset_id: load_gage_flow_data(dataset_id)}


def combine_ensemble_sets(dataset_id, recombine=True):
    """
    Load and combine all ensemble sets into a single unified dataset.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    recombine : bool
        If True, reload from individual ensemble sets.
        If False, try to load pre-combined file.

    Returns
    -------
    data : pywrdrb.Data
        Combined data object with all realizations
    """
    fname_combined = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'

    # Check if combined file exists and we don't need to recombine
    if not recombine and os.path.exists(fname_combined):
        print(f"  Loading pre-combined data from {fname_combined}")
        data = pywrdrb.Data()
        data.load_from_export(fname_combined)
        return data

    print(f"  Combining {N_ENSEMBLE_SETS} ensemble sets...")

    # Load data from all ensemble sets
    data = pywrdrb.Data()

    # Results to load from raw pywrdrb output files
    # Note: shortage and contribution are computed after loading (not in raw output)
    results_sets = [
        'major_flow', 'mrf_target', 'res_storage', 'res_release',
        'inflow', 'ibt_diversions', 'ibt_demands',
        'nyc_release_components', 'res_level',
    ]

    for results_set in results_sets:
        print(f"    Loading {results_set}...")

        full_results_set_dict = {}
        full_results_set_dict[dataset_id] = {}

        for i in range(N_ENSEMBLE_SETS):
            set_spec = get_ensemble_set_spec(i, dataset_id)

            if not os.path.exists(set_spec.output_file):
                raise FileNotFoundError(
                    f"Output file not found for set {i}: {set_spec.output_file}"
                )

            # Load this ensemble set (raw pywrdrb output format)
            temp_data = pywrdrb.Data()
            temp_data.load_output(
                output_filenames=[set_spec.output_file],
                results_sets=[results_set],
            )

            # Extract data using filename stem as key
            file_label = os.path.splitext(os.path.basename(set_spec.output_file))[0]
            set_data = getattr(temp_data, results_set)[file_label]

            # Get local realization IDs for this set
            local_ids = sorted(set_data.keys())
            min_local_id = min(local_ids)

            # Renumber to global IDs and combine
            combined_data = full_results_set_dict[dataset_id]
            for local_id, df in set_data.items():
                # Calculate global realization ID
                local_id_normalized = local_id - min_local_id
                global_id = i * N_REALIZATIONS_PER_ENSEMBLE_SET + local_id_normalized
                combined_data[global_id] = df

        # Store combined data back
        full_results_set_dict[dataset_id] = combined_data
        setattr(data, results_set, full_results_set_dict)

    # Add Trenton equivalent flow AFTER combining datasets
    data = add_trenton_equiv_flow(data)

    # Add NYC aggregate inflow column (matching MPI postprocessing logic)
    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']
    for r, df in data.inflow[dataset_id].items():
        df['nyc'] = df[nyc_reservoirs].sum(axis=1)

    # Compute shortage and contribution from loaded data
    print("    Computing shortage...")
    _compute_shortage(data, dataset_id)
    print("    Computing contribution...")
    _compute_contribution(data, dataset_id)

    # Load gage_flow from hydrologic model flow files (input data, not simulation output)
    print("    Loading gage_flow...")
    _load_gage_flow(data, dataset_id)

    print("  Data loading complete")

    return data


