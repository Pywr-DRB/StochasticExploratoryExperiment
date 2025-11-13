"""
Core functions for postprocessing Pywr-DRB simulation outputs.

This module contains functions for:
- Loading and combining ensemble set outputs
- Calculating performance metrics
- Adding derived metrics (shortages, contributions)
- Exporting combined datasets
"""

import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from .metrics.shortfall import add_trenton_equiv_flow
from .config import (
    N_REALIZATIONS_PER_ENSEMBLE_SET,
    N_ENSEMBLE_SETS,
    NYC_TOTAL_CAPACITY,
    get_ensemble_set_spec
)


def calculate_performance_metrics(data, dataset_id, realizations):
    """
    Calculate comprehensive performance metrics for all realizations.

    This function calculates metrics across multiple categories:
    - Flow reliability (Montague, Trenton)
    - NYC reservoir storage (levels, frequencies, extremes)
    - Water supply reliability (diversions, shortages)
    - Drought characteristics (frequency, duration, severity)
    - System operations (releases, contributions, balances)

    Parameters
    ----------
    data : pywrdrb.Data
        Data object with shortage, mrf_target, res_storage, ibt_diversions, ibt_demands, contribution
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

        # =====================================================================
        # CATEGORY 1: FLOW RELIABILITY METRICS
        # =====================================================================

        # Montague reliability
        annual_shortage = montague_shortage.resample('YS').sum()
        annual_target = montague_target.resample('YS').sum()
        annual_reliability = (1 - (annual_shortage / annual_target)).clip(0, 1)

        years_reliable_montague = (annual_reliability > 0.90).sum()
        years_reliable_montague_95 = (annual_reliability > 0.95).sum()
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
        years_above_20pct = (min_annual_storage > 20).sum()
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

    # Calculate and print percentiles for key metrics
    print(f"\n  Key Performance Metrics Summary:")
    print(f"  {'='*60}")

    count_metrics = ['years_reliable', 'years_high_storage', 'years_above_20pct',
                     'years_low_carryover', 'years_trenton_reliable']
    for metric in count_metrics:
        if metric in metrics_df.columns:
            p5 = metrics_df[metric].quantile(0.05)
            p50 = metrics_df[metric].quantile(0.50)
            p95 = metrics_df[metric].quantile(0.95)
            print(f"    {metric:40s}: p5={p5:5.1f}, p50={p50:5.1f}, p95={p95:5.1f}")

    print(f"\n  Other Metrics Summary:")
    print(f"  {'='*60}")
    other_metrics = ['pct_days_nyc_diversion_shortage', 'max_consecutive_drought_days',
                     'mean_sept1_storage_pct', 'mean_annual_nyc_contribution_mg']
    for metric in other_metrics:
        if metric in metrics_df.columns:
            p5 = metrics_df[metric].quantile(0.05)
            p50 = metrics_df[metric].quantile(0.50)
            p95 = metrics_df[metric].quantile(0.95)
            if 'pct' in metric or 'storage' in metric:
                print(f"    {metric:40s}: p5={p5:5.1f}, p50={p50:5.1f}, p95={p95:5.1f}")
            else:
                print(f"    {metric:40s}: p5={p5:5.0f}, p50={p50:5.0f}, p95={p95:5.0f}")


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

    # Results to load
    results_sets = [
        'major_flow', 'mrf_target', 'res_storage',
        'res_release', 'shortage', 'inflow', 'contribution'
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

            # Load this ensemble set
            temp_data = pywrdrb.Data()
            temp_data.load_from_export(
                set_spec.output_file,
                results_sets=[results_set]
            )

            # Extract data for this set
            set_data = getattr(temp_data, results_set)[dataset_id]

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
