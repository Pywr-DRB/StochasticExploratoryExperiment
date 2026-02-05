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
from .metrics.shortfall import add_trenton_equiv_flow, get_flow_and_target_values
from .config import (
    N_REALIZATIONS_PER_ENSEMBLE_SET,
    N_ENSEMBLE_SETS,
    NYC_TOTAL_CAPACITY,
    get_ensemble_set_spec
)
from .print_summary import print_performance_metrics_summary


def calculate_performance_metrics(data, dataset_id, realizations):
    """
    Calculate comprehensive performance metrics for all realizations.

    This function calculates metrics across multiple categories:
    - Flow reliability (Montague, Trenton)
    - NYC reservoir storage (levels, frequencies, extremes)
    - Water supply reliability (diversions, shortages)
    - Drought characteristics (frequency, duration, severity)
    - System operations (releases, contributions, balances)
    - Drought zone classifications (watch, warning, emergency)

    Parameters
    ----------
    data : pywrdrb.Data
        Data object with shortage, mrf_target, res_storage, ibt_diversions, ibt_demands, contribution, res_level
    dataset_id : str
        Dataset identifier
    realizations : list
        List of realization IDs

    Returns
    -------
    metrics_df : pd.DataFrame
        DataFrame with performance metrics for all realizations
    """
    print(f"  Calculating comprehensive performance metrics...")

    metrics = {}
    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']

    for r in realizations:
        if (r % 100 == 0) and (r > 0):
            print(f"    Processed {r}/{len(realizations)} realizations...")

        # =====================================================================
        # LOAD DATA FOR THIS REALIZATION
        # =====================================================================
        # Flow targets and shortages
        montague_shortage = data.shortage[dataset_id][r]['delMontague']
        montague_target = data.mrf_target[dataset_id][r]['delMontague']
        trenton_shortage = data.shortage[dataset_id][r]['delTrenton']
        trenton_target = data.mrf_target[dataset_id][r]['delTrenton']

        # NYC reservoir storage
        nyc_storage = data.res_storage[dataset_id][r][nyc_reservoirs].sum(axis=1)
        nyc_storage_pct = 100.0 * nyc_storage / NYC_TOTAL_CAPACITY

        # NYC diversions
        nyc_diversion_actual = data.ibt_diversions[dataset_id][r]['delivery_nyc']
        nyc_diversion_demand = data.ibt_demands[dataset_id][r]['demand_nyc']
        nyc_diversion_shortage = (nyc_diversion_demand - nyc_diversion_actual).clip(lower=0)

        # NYC contributions to downstream targets
        total_nyc_contribution = data.contribution[dataset_id][r]['mrf_montagueTrenton_nyc']

        # NYC drought zone levels (if available)
        nyc_zone_level = None
        if hasattr(data, 'res_level') and dataset_id in data.res_level and r in data.res_level[dataset_id]:
            nyc_zone_level = data.res_level[dataset_id][r]['nyc']

        # =====================================================================
        # CATEGORY 1: FLOW RELIABILITY METRICS
        # =====================================================================

        # Montague reliability
        annual_shortage = montague_shortage.resample('YS').sum()
        annual_target = montague_target.resample('YS').sum()
        annual_reliability = (1 - (annual_shortage / annual_target)).clip(0, 1)

        years_reliable_montague = (annual_reliability > 0.90).sum()
        years_reliable_montague_95 = (annual_reliability > 0.95).sum()
        years_reliable_montague_99 = (annual_reliability > 0.99).sum()
        mean_annual_montague_reliability = annual_reliability.mean()
        min_annual_montague_reliability = annual_reliability.min()

        # Trenton reliability
        annual_trenton_shortage = trenton_shortage.resample('YS').sum()
        annual_trenton_target = trenton_target.resample('YS').sum()
        trenton_reliability = (1 - (annual_trenton_shortage / annual_trenton_target)).clip(0, 1)

        years_reliable_trenton = (trenton_reliability > 0.90).sum()
        years_reliable_trenton_95 = (trenton_reliability > 0.95).sum()
        mean_annual_trenton_reliability = trenton_reliability.mean()

        # Total shortage volumes
        total_montague_shortage_mg = montague_shortage.sum()
        total_trenton_shortage_mg = trenton_shortage.sum()
        mean_annual_montague_shortage_mg = montague_shortage.resample('YS').sum().mean()
        mean_annual_trenton_shortage_mg = trenton_shortage.resample('YS').sum().mean()

        # =====================================================================
        # CATEGORY 2: NYC RESERVOIR STORAGE METRICS
        # =====================================================================

        # Critical storage thresholds throughout year
        min_annual_storage = nyc_storage_pct.resample('YS').min()
        years_above_30pct = (min_annual_storage > 30).sum()
        years_below_30pct = (min_annual_storage <= 30).sum()
        years_above_20pct = (min_annual_storage > 20).sum()
        years_below_20pct = (min_annual_storage <= 20).sum()
        years_above_10pct = (min_annual_storage > 10).sum()
        years_below_10pct = (min_annual_storage <= 10).sum()

        # Storage on key dates
        june1_storage = nyc_storage_pct[(nyc_storage_pct.index.month == 6) &
                                        (nyc_storage_pct.index.day == 1)]
        sept1_storage = nyc_storage_pct[(nyc_storage_pct.index.month == 9) &
                                        (nyc_storage_pct.index.day == 1)]

        years_high_storage_june1 = (june1_storage >= 95).sum()
        years_high_storage_june1_90 = (june1_storage >= 90).sum()
        mean_june1_storage_pct = june1_storage.mean()
        mean_sept1_storage_pct = sept1_storage.mean()
        years_low_carryover = (sept1_storage < 50).sum()
        years_low_carryover_40 = (sept1_storage < 40).sum()

        # Overall storage statistics
        mean_storage_pct = nyc_storage_pct.mean()
        median_storage_pct = nyc_storage_pct.median()
        min_storage_pct = nyc_storage_pct.min()
        max_storage_pct = nyc_storage_pct.max()
        pct_days_storage_below_30 = 100.0 * (nyc_storage_pct < 30).sum() / len(nyc_storage_pct)
        pct_days_storage_below_20 = 100.0 * (nyc_storage_pct < 20).sum() / len(nyc_storage_pct)
        
        # Storage variability
        std_storage_pct = nyc_storage_pct.std()
        annual_storage_range = nyc_storage_pct.resample('YS').apply(lambda x: x.max() - x.min())
        mean_annual_storage_range = annual_storage_range.mean()

        # =====================================================================
        # CATEGORY 3: WATER SUPPLY RELIABILITY METRICS
        # =====================================================================

        # NYC diversion performance
        n_days_nyc_shortage = (nyc_diversion_shortage > 0).sum()
        pct_days_nyc_diversion_shortage = 100.0 * n_days_nyc_shortage / len(nyc_diversion_shortage)
        total_nyc_diversion_shortage_mg = nyc_diversion_shortage.sum()
        mean_annual_nyc_diversion_shortage_mg = nyc_diversion_shortage.resample('YS').sum().mean()
        max_daily_nyc_diversion_shortage_mg = nyc_diversion_shortage.max()

        # NYC diversion reliability by year
        annual_nyc_diversion_shortage = nyc_diversion_shortage.resample('YS').sum()
        years_no_nyc_shortage = (annual_nyc_diversion_shortage == 0).sum()
        years_minor_nyc_shortage = (annual_nyc_diversion_shortage <= 365).sum()  # <1 MGD avg

        # =====================================================================
        # CATEGORY 4: DROUGHT CHARACTERISTICS
        # =====================================================================

        # Montague drought events (consecutive days with shortage)
        montague_shortage_binary = (montague_shortage > 0).astype(int)
        drought_events = montague_shortage_binary.groupby(
            (montague_shortage_binary != montague_shortage_binary.shift()).cumsum()
        ).sum()
        drought_events = drought_events[drought_events > 0]

        if len(drought_events) > 0:
            max_consecutive_drought_days = drought_events.max()
            mean_drought_duration_days = drought_events.mean()
            n_drought_events = len(drought_events)
            n_major_droughts = (drought_events >= 90).sum()  # 3+ months
            n_severe_droughts = (drought_events >= 180).sum()  # 6+ months
        else:
            max_consecutive_drought_days = 0
            mean_drought_duration_days = 0
            n_drought_events = 0
            n_major_droughts = 0
            n_severe_droughts = 0

        # Trenton drought events
        trenton_shortage_binary = (trenton_shortage > 0).astype(int)
        trenton_drought_events = trenton_shortage_binary.groupby(
            (trenton_shortage_binary != trenton_shortage_binary.shift()).cumsum()
        ).sum()
        trenton_drought_events = trenton_drought_events[trenton_drought_events > 0]

        if len(trenton_drought_events) > 0:
            max_consecutive_drought_days_trenton = trenton_drought_events.max()
            n_drought_events_trenton = len(trenton_drought_events)
        else:
            max_consecutive_drought_days_trenton = 0
            n_drought_events_trenton = 0

        # Drought severity (maximum shortage during worst drought)
        if len(drought_events) > 0:
            # Find the worst drought period
            drought_groups = (montague_shortage_binary != montague_shortage_binary.shift()).cumsum()
            max_shortage_by_event = montague_shortage.groupby(drought_groups).max()
            max_shortage_by_event = max_shortage_by_event[montague_shortage.groupby(drought_groups).sum() > 0]
            worst_drought_max_daily_shortage_mg = max_shortage_by_event.max()
        else:
            worst_drought_max_daily_shortage_mg = 0

        # Combined system stress (simultaneous NYC shortage + Montague shortage)
        combined_stress_days = ((nyc_diversion_shortage > 0) & (montague_shortage > 0)).sum()
        pct_days_combined_stress = 100.0 * combined_stress_days / len(nyc_diversion_shortage)

        # Maximum Montague shortage metrics (1-day, 3-day, 7-day rolling means)
        max_1day_montague_shortage_mg = montague_shortage.max()
        max_3day_montague_shortage_mg = montague_shortage.rolling(window=3, min_periods=1).mean().max()
        max_7day_montague_shortage_mg = montague_shortage.rolling(window=7, min_periods=1).mean().max()

        # =====================================================================
        # CATEGORY 4b: DROUGHT ZONE CLASSIFICATIONS
        # =====================================================================

        # NYC drought zone year counts (based on res_level data)
        # Zone definitions: 6=Emergency, 5=Watch, 4=Warning, 3=Normal, 1-2=Flood
        if nyc_zone_level is not None:
            # Get maximum zone reached in each year
            annual_max_zone = nyc_zone_level.resample('YS').max()

            # Count years reaching each drought zone level
            years_drought_emergency = (annual_max_zone >= 6).sum()  # Zone 6
            years_drought_watch = (annual_max_zone >= 5).sum()  # Zone 5 or higher
            years_drought_warning = (annual_max_zone >= 4).sum()  # Zone 4 or higher

            # Count years reaching exactly each zone (not higher)
            years_exactly_emergency = (annual_max_zone == 6).sum()
            years_exactly_watch = (annual_max_zone == 5).sum()
            years_exactly_warning = (annual_max_zone == 4).sum()
        else:
            years_drought_emergency = np.nan
            years_drought_watch = np.nan
            years_drought_warning = np.nan
            years_exactly_emergency = np.nan
            years_exactly_watch = np.nan
            years_exactly_warning = np.nan

        # =====================================================================
        # CATEGORY 5: NYC CONTRIBUTION TO DOWNSTREAM TARGETS
        # =====================================================================

        annual_nyc_contribution = total_nyc_contribution.resample('YS').sum()
        mean_annual_nyc_contribution_mg = annual_nyc_contribution.mean()
        max_annual_nyc_contribution_mg = annual_nyc_contribution.max()
        min_annual_nyc_contribution_mg = annual_nyc_contribution.min()
        std_annual_nyc_contribution_mg = annual_nyc_contribution.std()

        # Days with significant contributions
        n_days_nyc_contribution = (total_nyc_contribution > 0).sum()
        pct_days_nyc_contribution = 100.0 * n_days_nyc_contribution / len(total_nyc_contribution)
        n_days_high_nyc_contribution = (total_nyc_contribution > 100).sum()  # >100 MGD

        # Total NYC contribution over simulation
        total_nyc_contribution_mg = total_nyc_contribution.sum()

        # =====================================================================
        # CATEGORY 6: SYSTEM BALANCE METRICS
        # =====================================================================

        # Ratio of NYC contribution to Montague shortage
        if total_montague_shortage_mg > 0:
            nyc_contribution_to_shortage_ratio = total_nyc_contribution_mg / total_montague_shortage_mg
        else:
            nyc_contribution_to_shortage_ratio = np.nan

        # Years with simultaneous high storage and high reliability
        high_storage_years = (june1_storage >= 90)
        reliable_years = (annual_reliability > 0.90)
        years_high_storage_and_reliable = (high_storage_years.values & reliable_years.values).sum()

        # Years with low storage OR low reliability (vulnerability)
        low_storage_years = (min_annual_storage <= 30)
        unreliable_years = (annual_reliability <= 0.85)
        years_vulnerable = (low_storage_years.values | unreliable_years.values).sum()

        # =====================================================================
        # STORE ALL METRICS
        # =====================================================================
        metrics[r] = {
            # Flow Reliability - Montague
            'years_reliable_montague': years_reliable_montague,
            'years_reliable_montague_95': years_reliable_montague_95,
            'years_reliable_montague_99': years_reliable_montague_99,
            'mean_annual_montague_reliability': mean_annual_montague_reliability,
            'min_annual_montague_reliability': min_annual_montague_reliability,
            'total_montague_shortage_mg': total_montague_shortage_mg,
            'mean_annual_montague_shortage_mg': mean_annual_montague_shortage_mg,

            # Flow Reliability - Trenton
            'years_reliable_trenton': years_reliable_trenton,
            'years_reliable_trenton_95': years_reliable_trenton_95,
            'mean_annual_trenton_reliability': mean_annual_trenton_reliability,
            'total_trenton_shortage_mg': total_trenton_shortage_mg,
            'mean_annual_trenton_shortage_mg': mean_annual_trenton_shortage_mg,

            # NYC Storage - Critical Thresholds
            'years_above_30pct': years_above_30pct,
            'years_above_20pct': years_above_20pct,
            'years_above_10pct': years_above_10pct,
            'years_below_30pct': years_below_30pct,
            'years_below_20pct': years_below_20pct,
            'years_below_10pct': years_below_10pct,

            # NYC Storage - Key Dates
            'years_high_storage_june1': years_high_storage_june1,
            'years_high_storage_june1_90': years_high_storage_june1_90,
            'mean_june1_storage_pct': mean_june1_storage_pct,
            'mean_sept1_storage_pct': mean_sept1_storage_pct,
            'years_low_carryover': years_low_carryover,
            'years_low_carryover_40': years_low_carryover_40,

            # NYC Storage - Statistics
            'mean_storage_pct': mean_storage_pct,
            'median_storage_pct': median_storage_pct,
            'min_storage_pct': min_storage_pct,
            'max_storage_pct': max_storage_pct,
            'std_storage_pct': std_storage_pct,
            'pct_days_storage_below_30': pct_days_storage_below_30,
            'pct_days_storage_below_20': pct_days_storage_below_20,
            'mean_annual_storage_range': mean_annual_storage_range,

            # Water Supply Reliability - NYC
            'pct_days_nyc_diversion_shortage': pct_days_nyc_diversion_shortage,
            'total_nyc_diversion_shortage_mg': total_nyc_diversion_shortage_mg,
            'mean_annual_nyc_diversion_shortage_mg': mean_annual_nyc_diversion_shortage_mg,
            'max_daily_nyc_diversion_shortage_mg': max_daily_nyc_diversion_shortage_mg,
            'years_no_nyc_shortage': years_no_nyc_shortage,
            'years_minor_nyc_shortage': years_minor_nyc_shortage,

            # Drought Characteristics - Montague
            'max_consecutive_drought_days': max_consecutive_drought_days,
            'mean_drought_duration_days': mean_drought_duration_days,
            'n_drought_events': n_drought_events,
            'n_major_droughts': n_major_droughts,
            'n_severe_droughts': n_severe_droughts,
            'worst_drought_max_daily_shortage_mg': worst_drought_max_daily_shortage_mg,

            # Drought Characteristics - Trenton
            'max_consecutive_drought_days_trenton': max_consecutive_drought_days_trenton,
            'n_drought_events_trenton': n_drought_events_trenton,

            # System Stress
            'pct_days_combined_stress': pct_days_combined_stress,

            # Maximum Montague Shortage (rolling means)
            'max_1day_montague_shortage_mg': max_1day_montague_shortage_mg,
            'max_3day_montague_shortage_mg': max_3day_montague_shortage_mg,
            'max_7day_montague_shortage_mg': max_7day_montague_shortage_mg,

            # Drought Zone Classifications
            'years_drought_emergency': years_drought_emergency,
            'years_drought_watch': years_drought_watch,
            'years_drought_warning': years_drought_warning,
            'years_exactly_emergency': years_exactly_emergency,
            'years_exactly_watch': years_exactly_watch,
            'years_exactly_warning': years_exactly_warning,

            # NYC Contributions
            'mean_annual_nyc_contribution_mg': mean_annual_nyc_contribution_mg,
            'max_annual_nyc_contribution_mg': max_annual_nyc_contribution_mg,
            'min_annual_nyc_contribution_mg': min_annual_nyc_contribution_mg,
            'std_annual_nyc_contribution_mg': std_annual_nyc_contribution_mg,
            'total_nyc_contribution_mg': total_nyc_contribution_mg,
            'pct_days_nyc_contribution': pct_days_nyc_contribution,
            'n_days_high_nyc_contribution': n_days_high_nyc_contribution,

            # System Balance
            'nyc_contribution_to_shortage_ratio': nyc_contribution_to_shortage_ratio,
            'years_high_storage_and_reliable': years_high_storage_and_reliable,
            'years_vulnerable': years_vulnerable,

            # Legacy metric names (for backward compatibility)
            'years_reliable': years_reliable_montague,
            'years_high_storage': years_high_storage_june1,
            'years_trenton_reliable': years_reliable_trenton,
        }

    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.index.name = 'realization_id'

    print(f"  Calculated {len(metrics_df)} rows × {len(metrics_df.columns)} performance metrics")

    return metrics_df


def classify_years_by_min_zone(res_level_df):
    """
    Classify each year by the minimum drought zone reached.

    Parameters
    ----------
    res_level_df : pd.DataFrame
        Reservoir level DataFrame with 'nyc' column and datetime index

    Returns
    -------
    year_classifications : dict
        Dictionary mapping year -> {'min_zone': int, 'min_zone_date': pd.Timestamp}
    """
    # Add year column
    df = res_level_df.copy()
    df['year'] = df.index.year

    year_classifications = {}

    for year in df['year'].unique():
        year_data = df[df['year'] == year]

        # Find maximum zone value (higher zone = more severe drought)
        # Zone 6 is most severe drought, Zone 1 is flood
        max_zone = year_data['nyc'].max()

        # Find date when maximum zone occurred
        max_zone_date = year_data[year_data['nyc'] == max_zone].index[0]

        year_classifications[year] = {
            'min_zone': max_zone,
            'min_zone_date': max_zone_date
        }

    return year_classifications


def calculate_contribution_analysis_metrics(data, dataset_id, realizations,
                                            window_days=[30, 60, 90, 120, 150, 180, 270]):
    """
    Calculate year-level contribution analysis metrics for multiple aggregation windows.

    This function pre-computes metrics used by contribution analysis plotting scripts,
    eliminating the need to recalculate on-the-fly during figure generation.

    Parameters
    ----------
    data : pywrdrb.Data
        Data object with res_level, res_storage, contribution, inflow, ibt_diversions, ibt_demands
    dataset_id : str
        Dataset identifier
    realizations : list
        List of realization IDs
    window_days : list of int
        Window lengths in days to compute metrics for (default: [30, 60, 90, 120, 150, 180])

    Returns
    -------
    metrics_df : pd.DataFrame
        DataFrame with columns:
        - realization_id, year, min_zone, min_zone_date, min_storage_pct
        - For each window W in window_days:
          - contribution_total_{W}d: NYC→Montague contributions sum (MG)
          - contribution_ratio_{W}d: (contribution/inflow) × 100 (%)
          - inflow_total_{W}d: NYC reservoir inflow sum (MG)
          - demand_satisfaction_{W}d: volumetric diversion/demand ratio (≤1.0)
          - worst_1mo_demand_sat_{W}d: minimum 30-day rolling demand satisfaction (%)
    """
    print(f"  Calculating contribution analysis metrics for {len(realizations)} realizations...")

    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']
    records = []

    for r in realizations:
        if (r % 100 == 0) and (r > 0):
            print(f"    Processed {r}/{len(realizations)} realizations...")

        # Load timeseries (already in memory during postprocessing)
        res_level = data.res_level[dataset_id][r]

        # Calculate NYC combined storage percentage
        nyc_storage = data.res_storage[dataset_id][r][nyc_reservoirs].sum(axis=1)
        nyc_storage_pct = 100.0 * nyc_storage / NYC_TOTAL_CAPACITY

        # Get contribution and inflow data
        nyc_contributions = data.contribution[dataset_id][r]['mrf_montagueTrenton_nyc']
        nyc_inflow = data.inflow[dataset_id][r][nyc_reservoirs].sum(axis=1)

        # Get diversion and demand data
        nyc_diversion = data.ibt_diversions[dataset_id][r]['delivery_nyc']
        nyc_demand = data.ibt_demands[dataset_id][r]['demand_nyc']

        # Classify years by drought zone (once per realization)
        year_classifications = classify_years_by_min_zone(res_level)

        for year, info in year_classifications.items():
            # Calculate base metrics
            min_zone = info['min_zone']
            min_zone_date = info['min_zone_date']

            # Calculate minimum storage for the year
            year_mask = nyc_storage_pct.index.year == year
            min_storage = nyc_storage_pct[year_mask].min()

            record = {
                'realization_id': r,
                'year': year,
                'min_zone': min_zone,
                'min_zone_date': min_zone_date.isoformat(),  # Convert to string for CSV
                'min_storage_pct': min_storage
            }

            # Compute metrics for each window
            for W in window_days:
                start_date = min_zone_date - pd.Timedelta(days=W)
                mask = (nyc_contributions.index >= start_date) & (nyc_contributions.index <= min_zone_date)

                # Window-based calculations
                contrib_total = nyc_contributions[mask].sum()
                inflow_total = nyc_inflow[mask].sum()
                contrib_ratio = 100.0 * contrib_total / inflow_total if inflow_total > 0 else np.nan

                total_div = nyc_diversion[mask].sum()
                total_dem = nyc_demand[mask].sum()
                demand_sat = min(total_div / total_dem, 1.0) if total_dem > 0 else 1.0

                # Worst 1-month rolling demand satisfaction
                window_div = nyc_diversion[mask]
                window_dem = nyc_demand[mask]
                if len(window_div) >= 30:
                    rolling_div = window_div.rolling(30).sum()
                    rolling_dem = window_dem.rolling(30).sum()
                    rolling_sat = (rolling_div / rolling_dem).clip(upper=1.0)
                    worst_1mo = 100.0 * rolling_sat.min()
                else:
                    worst_1mo = 100.0 * demand_sat if len(window_div) > 0 else np.nan

                record.update({
                    f'contribution_total_{W}d': contrib_total,
                    f'contribution_ratio_{W}d': contrib_ratio,
                    f'inflow_total_{W}d': inflow_total,
                    f'demand_satisfaction_{W}d': demand_sat,
                    f'worst_1mo_demand_sat_{W}d': worst_1mo
                })

            records.append(record)

    # Convert to DataFrame
    metrics_df = pd.DataFrame(records)

    print(f"  Calculated {len(metrics_df)} year-realization pairs × {len(metrics_df.columns)} metrics")

    return metrics_df


def save_performance_metrics(metrics_df, dataset_id, output_dir):
    """
    Save performance metrics to CSV and print summary statistics.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Performance metrics
    dataset_id : str
        Dataset identifier
    output_dir : str
        Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/{dataset_id}_performance_metrics.csv"
    metrics_df.to_csv(fname)
    print(f"  Saved performance metrics: {fname}")

    # Print summary using centralized function
    print_performance_metrics_summary(metrics_df)


def calculate_and_save_performance_metrics(data, dataset_id, realizations, output_dir="./pywrdrb/performance_metrics"):
    """
    Calculate performance metrics and save to CSV (wrapper function).

    Parameters
    ----------
    data : pywrdrb.Data
        Data object with shortage, mrf_target, res_storage, ibt_diversions, ibt_demands, contribution
    dataset_id : str
        Dataset identifier
    realizations : list
        List of realization IDs
    output_dir : str
        Output directory path

    Returns
    -------
    metrics_df : pd.DataFrame
        DataFrame with performance metrics for all realizations
    """
    metrics_df = calculate_performance_metrics(data, dataset_id, realizations)
    save_performance_metrics(metrics_df, dataset_id, output_dir)
    return metrics_df


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
            shortage_series = target_series - flow_series
            shortage_series[shortage_series < 0] = 0
            shortage_series.iloc[:3] = 0.0

            # Ignore shortages with duration < 3 days
            shortage_durations = (shortage_series > 0).astype(int).groupby(
                (shortage_series > 0).astype(int).diff().ne(0).cumsum()
            ).cumsum()
            shortage_series[shortage_durations < 3] = 0.0

            node_shortages[node] = shortage_series

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
    ensemble_set_specs = [get_ensemble_set_spec(i, dataset_id) for i in range(N_ENSEMBLE_SETS)]

    # Setup pathnavigator for all ensemble sets
    pn_config = pywrdrb.get_pn_config()
    for spec in ensemble_set_specs:
        dataset_dir = spec.directory
        dataset_name = dataset_dir.split('/')[-1]
        pn_config[f"flows/{dataset_name}"] = os.path.abspath(dataset_dir)
    pywrdrb.load_pn_config(pn_config)

    # Load hydrologic flow data
    ensemble_set_names = [spec.directory.split('/')[-1] for spec in ensemble_set_specs]
    temp_data = pywrdrb.Data(results_sets=['major_flow'], print_status=False)
    temp_data.load_hydrologic_model_flow(ensemble_set_names)

    # Combine all sets into single dataset key with global realization IDs
    combined_gage_flow = {}
    for set_name in ensemble_set_names:
        if set_name not in temp_data.major_flow:
            continue
        set_data = temp_data.major_flow[set_name]
        set_idx = int(set_name.split('_set')[-1]) - 1

        local_ids = list(set_data.keys())
        min_local_id = min(local_ids) if local_ids else 0

        for local_id, df in set_data.items():
            local_id_normalized = local_id - min_local_id
            global_id = set_idx * N_REALIZATIONS_PER_ENSEMBLE_SET + local_id_normalized
            combined_gage_flow[global_id] = df

    data.gage_flow = {dataset_id: combined_gage_flow}
    print(f"      Loaded gage flow for {len(combined_gage_flow)} realizations")


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


def postprocess_dataset(dataset_id, recombine=True):
    """
    Complete postprocessing workflow for a dataset.

    This combines ensemble sets, calculates metrics, and exports results.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    recombine : bool
        If True, recombine ensemble sets from scratch

    Returns
    -------
    data : pywrdrb.Data
        Combined and postprocessed data object
    """
    print(f"\n{'='*80}")
    print(f"POSTPROCESSING: {dataset_id}")
    print(f"{'='*80}")

    # Combine ensemble sets
    data = combine_ensemble_sets(dataset_id, recombine=recombine)

    # Get list of realizations
    realizations = sorted(data.major_flow[dataset_id].keys())
    print(f"  Found {len(realizations)} realizations")

    # Calculate performance metrics
    metrics_df = calculate_performance_metrics(data, dataset_id, realizations)

    # Save performance metrics
    output_dir = "./pywrdrb/performance_metrics"
    save_performance_metrics(metrics_df, dataset_id, output_dir)

    # Export combined data
    if recombine:
        fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
        print(f"  Exporting combined data to {fname}...")
        data.export(fname)
        print(f"  Successfully exported!")

    print(f"\n{'='*80}")
    print(f"POSTPROCESSING COMPLETE: {dataset_id}")
    print(f"{'='*80}")

    return data


# =============================================================================
# EPISODE ANALYSIS PREPROCESSING FUNCTIONS
# =============================================================================

def preprocess_to_weekly(data, dataset_id, config=None):
    """
    Aggregate daily simulation outputs to weekly resolution for episode analysis.

    Creates a unified weekly time series DataFrame with all variables needed
    for episode identification and characterization.

    Parameters
    ----------
    data : pywrdrb.Data
        Data object with res_storage, res_level, major_flow, mrf_target,
        shortage, inflow, ibt_diversions, ibt_demands
    dataset_id : str
        Dataset identifier
    config : EpisodeAnalysisConfig, optional
        Configuration object. If None, uses default values.

    Returns
    -------
    weekly_ts : pd.DataFrame
        Weekly time series with columns:
        - realization_id, week, year, week_of_year, date
        - inflow_agg, storage_agg, storage_pct, ffmp_zone
        - nyc_demand, nyc_diversion, demand_satisfaction
        - montague_flow, montague_target, flow_satisfaction
    """
    from .utils import calculate_water_year_period_index

    # Get configuration defaults
    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']
    nyc_total_capacity = NYC_TOTAL_CAPACITY
    period_origin = 'june1'

    if config is not None:
        nyc_reservoirs = config.nyc_reservoirs
        nyc_total_capacity = config.nyc_total_capacity
        period_origin = config.period_origin

    realizations = sorted(data.res_storage[dataset_id].keys())
    print(f"  Preprocessing {len(realizations)} realizations to weekly resolution...")

    all_weekly = []

    for r in realizations:
        if (r % 100 == 0) and (r > 0):
            print(f"    Processed {r}/{len(realizations)} realizations...")

        # Extract daily data for this realization
        storage_daily = data.res_storage[dataset_id][r][nyc_reservoirs].sum(axis=1)
        inflow_daily = data.inflow[dataset_id][r]['nyc']

        # FFMP zone (if available)
        if hasattr(data, 'res_level') and dataset_id in data.res_level:
            ffmp_zone_daily = data.res_level[dataset_id][r]['nyc']
        else:
            ffmp_zone_daily = pd.Series(3, index=storage_daily.index)  # Default to Normal

        # NYC diversions and demands
        nyc_diversion_daily = data.ibt_diversions[dataset_id][r]['delivery_nyc']
        nyc_demand_daily = data.ibt_demands[dataset_id][r]['demand_nyc']

        # Montague flow and target
        montague_flow_daily = data.major_flow[dataset_id][r]['delMontague']
        montague_target_daily = data.mrf_target[dataset_id][r]['delMontague']

        # NYC contribution to Montague flow target (if available)
        # This represents the required releases NYC must make for downstream targets
        if hasattr(data, 'contribution') and dataset_id in data.contribution:
            if r in data.contribution[dataset_id]:
                nyc_montague_contrib_daily = data.contribution[dataset_id][r].get(
                    'mrf_montagueTrenton_nyc',
                    pd.Series(0.0, index=storage_daily.index)
                )
            else:
                nyc_montague_contrib_daily = pd.Series(0.0, index=storage_daily.index)
        else:
            nyc_montague_contrib_daily = pd.Series(0.0, index=storage_daily.index)

        # Create daily DataFrame
        daily_df = pd.DataFrame({
            'storage_agg': storage_daily,
            'inflow_agg': inflow_daily,
            'ffmp_zone': ffmp_zone_daily,
            'nyc_diversion': nyc_diversion_daily,
            'nyc_demand': nyc_demand_daily,
            'montague_flow': montague_flow_daily,
            'montague_target': montague_target_daily,
            'nyc_montague_contribution': nyc_montague_contrib_daily,
        })

        # Compute daily shortage indicators (True if shortage that day)
        daily_df['demand_shortage_day'] = (
            daily_df['nyc_diversion'] < daily_df['nyc_demand'] * 0.999
        ).astype(int)
        daily_df['flow_shortage_day'] = (
            daily_df['montague_flow'] < daily_df['montague_target'] * 0.999
        ).astype(int)

        # Function to compute max consecutive days within each weekly group
        def max_consecutive_days(series):
            """Count maximum consecutive 1s in a series."""
            if series.sum() == 0:
                return 0
            # Convert to string of 0s and 1s, split by 0s, find max length
            s = ''.join(series.astype(str).values)
            runs = s.split('0')
            return max(len(run) for run in runs)

        # Resample to weekly
        # Use week-ending (Sunday) to align with common conventions
        weekly_df = daily_df.resample('W').agg({
            'storage_agg': 'mean',           # Average storage over week
            'inflow_agg': 'sum',             # Total weekly inflow (MGD -> MG/week)
            'ffmp_zone': 'max',              # Worst zone during week (higher = worse)
            'nyc_diversion': 'sum',          # Total weekly diversion
            'nyc_demand': 'sum',             # Total weekly demand
            'montague_flow': 'sum',          # Total weekly flow
            'montague_target': 'sum',        # Total weekly target
            'nyc_montague_contribution': 'sum',  # Total weekly NYC contribution to Montague
            'demand_shortage_day': 'sum',    # Total days with demand shortage
            'flow_shortage_day': 'sum',      # Total days with flow shortage
        })

        # Also compute max consecutive shortage days per week
        weekly_consec_demand = daily_df['demand_shortage_day'].resample('W').apply(max_consecutive_days)
        weekly_consec_flow = daily_df['flow_shortage_day'].resample('W').apply(max_consecutive_days)

        weekly_df['demand_shortage_consec_days'] = weekly_consec_demand
        weekly_df['flow_shortage_consec_days'] = weekly_consec_flow

        # Add derived variables
        weekly_df['storage_pct'] = 100.0 * weekly_df['storage_agg'] / nyc_total_capacity
        weekly_df['demand_satisfaction'] = (
            weekly_df['nyc_diversion'] / weekly_df['nyc_demand']
        ).clip(upper=1.0)
        weekly_df['flow_satisfaction'] = (
            weekly_df['montague_flow'] / weekly_df['montague_target']
        ).clip(upper=1.0)

        # Add temporal indices
        weekly_df['realization_id'] = r
        weekly_df['date'] = weekly_df.index
        weekly_df['year'] = weekly_df.index.year
        weekly_df['week'] = range(len(weekly_df))  # Absolute week number

        # Week of year (for climatology)
        weekly_df['week_of_year'] = calculate_water_year_period_index(
            weekly_df.index, period='weekly', origin=period_origin
        )

        all_weekly.append(weekly_df)

    # Combine all realizations
    weekly_ts = pd.concat(all_weekly, ignore_index=True)

    # Reorder columns for clarity
    col_order = [
        'realization_id', 'week', 'year', 'week_of_year', 'date',
        'inflow_agg', 'storage_agg', 'storage_pct', 'ffmp_zone',
        'nyc_demand', 'nyc_diversion', 'demand_satisfaction',
        'demand_shortage_day', 'demand_shortage_consec_days',
        'montague_flow', 'montague_target', 'nyc_montague_contribution', 'flow_satisfaction',
        'flow_shortage_day', 'flow_shortage_consec_days',
    ]
    # Only include columns that exist (in case NYC contribution data not available)
    col_order = [c for c in col_order if c in weekly_ts.columns]
    weekly_ts = weekly_ts[col_order]

    print(f"  Created weekly time series: {len(weekly_ts)} rows")

    return weekly_ts


def compute_weekly_climatology(weekly_ts, variables=None):
    """
    Compute weekly climatology (mean and std) for specified variables.

    Climatology is computed across all realizations and years for each
    week-of-year, enabling standardization of variables.

    Parameters
    ----------
    weekly_ts : pd.DataFrame
        Weekly time series with 'week_of_year' column
    variables : list of str, optional
        Variables to compute climatology for.
        Default: ['inflow_agg', 'nyc_demand']

    Returns
    -------
    climatology : pd.DataFrame
        Climatology with columns for each variable's mean and std,
        indexed by week_of_year
    """
    if variables is None:
        variables = ['inflow_agg', 'nyc_demand']

    climatology_dict = {}

    for var in variables:
        if var not in weekly_ts.columns:
            print(f"  Warning: Variable '{var}' not found in weekly_ts, skipping")
            continue

        # Compute mean and std by week of year
        grouped = weekly_ts.groupby('week_of_year')[var]
        climatology_dict[f'{var}_mean'] = grouped.mean()
        climatology_dict[f'{var}_std'] = grouped.std()

    climatology = pd.DataFrame(climatology_dict)
    climatology.index.name = 'week_of_year'

    print(f"  Computed climatology for {len(variables)} variables, {len(climatology)} weeks")

    return climatology


def add_standardized_variables(weekly_ts, climatology):
    """
    Add standardized anomaly variables to weekly time series.

    Computes standardized anomalies relative to weekly climatology:
    - inflow_std: (inflow - mean) / std  [negative = deficit]
    - demand_std: (demand - mean) / std  [positive = high demand]
    - combined_stress_std: demand_std - inflow_std  [positive = stressful]
    - net_stress: demand - inflow  [positive = drawing storage]

    Parameters
    ----------
    weekly_ts : pd.DataFrame
        Weekly time series with week_of_year column
    climatology : pd.DataFrame
        Climatology from compute_weekly_climatology()

    Returns
    -------
    weekly_ts : pd.DataFrame
        Weekly time series with added standardized columns
    """
    weekly_ts = weekly_ts.copy()

    # Map climatology values to each row
    inflow_mean = weekly_ts['week_of_year'].map(climatology['inflow_agg_mean'])
    inflow_std = weekly_ts['week_of_year'].map(climatology['inflow_agg_std'])
    demand_mean = weekly_ts['week_of_year'].map(climatology['nyc_demand_mean'])
    demand_std = weekly_ts['week_of_year'].map(climatology['nyc_demand_std'])

    # Compute standardized variables
    # Avoid division by zero
    inflow_std_safe = inflow_std.replace(0, np.nan)
    demand_std_safe = demand_std.replace(0, np.nan)

    weekly_ts['inflow_std'] = (weekly_ts['inflow_agg'] - inflow_mean) / inflow_std_safe
    weekly_ts['demand_std'] = (weekly_ts['nyc_demand'] - demand_mean) / demand_std_safe

    # Combined stress (positive = stressful: high demand AND/OR low inflow)
    weekly_ts['combined_stress_std'] = weekly_ts['demand_std'] - weekly_ts['inflow_std']

    # Physical net stress (MGD, positive = net draw on storage)
    # Note: weekly values are in MG (sum of daily MGD), so divide by 7 for rate
    weekly_ts['net_stress'] = (weekly_ts['nyc_demand'] - weekly_ts['inflow_agg']) / 7.0

    print(f"  Added standardized variables: inflow_std, demand_std, combined_stress_std, net_stress")

    return weekly_ts


def load_episode_analysis_data(dataset_id):
    """
    Load required data for episode analysis.

    Loads postprocessed simulation data from HDF5 export file.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier

    Returns
    -------
    data : pywrdrb.Data
        Data object with required results sets loaded

    Raises
    ------
    FileNotFoundError
        If postprocessed data file does not exist
    """
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'

    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Postprocessed data not found: {fname}\n"
            "Run 04_postprocess_data.py first!"
        )

    print(f"Loading episode analysis data from {fname}...")
    data = pywrdrb.Data()
    data.load_from_export(
        fname,
        results_sets=[
            'res_storage',      # Storage levels
            'res_level',        # FFMP zone levels
            'major_flow',       # Montague flow
            'mrf_target',       # Montague targets
            'shortage',         # Pre-computed shortages
            'inflow',           # NYC inflows
            'ibt_diversions',   # NYC diversions
            'ibt_demands',      # NYC demands
            'contribution',     # NYC contribution to downstream targets
        ]
    )

    return data
