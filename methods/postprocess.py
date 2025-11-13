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
    Calculate performance metrics for all realizations.

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
    print(f"  Calculating performance metrics...")

    metrics = {}
    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']

    for r in realizations:
        if (r % 100 == 0) and (r > 0):
            print(f"    Processed {r}/{len(realizations)} realizations...")

        # Use pre-calculated shortage and target data
        montague_shortage = data.shortage[dataset_id][r]['delMontague']
        montague_target = data.mrf_target[dataset_id][r]['delMontague']

        # Metric 1: # years where Montague flow target met >90% of time
        annual_shortage = montague_shortage.resample('YS').sum()
        annual_target = montague_target.resample('YS').sum()
        annual_reliability = 1 - (annual_shortage / annual_target)
        annual_reliability = annual_reliability.clip(0, 1)
        n_years_reliable = (annual_reliability > 0.90).sum()

        # Metric 2: # years where NYC storage >95% on June 1
        nyc_storage = data.res_storage[dataset_id][r][nyc_reservoirs].sum(axis=1)
        nyc_storage_pct = 100.0 * nyc_storage / NYC_TOTAL_CAPACITY

        # Filter for June 1 dates
        june1_storage = nyc_storage_pct[(nyc_storage_pct.index.month == 6) &
                                        (nyc_storage_pct.index.day == 1)]
        n_years_high_storage = (june1_storage >= 95).sum()

        # Metric 3: Number of years where minimum NYC storage remains >20% throughout year
        min_annual_storage = nyc_storage_pct.resample('YS').min()
        n_years_above_20pct = (min_annual_storage > 20).sum()

        # Alternative threshold at 10%
        n_years_above_10pct = (min_annual_storage > 10).sum()

        # Metric 4: NYC Reservoir System Carryover Storage (September 1)
        sept1_storage = nyc_storage_pct[(nyc_storage_pct.index.month == 9) &
                                         (nyc_storage_pct.index.day == 1)]
        mean_sept1_storage_pct = sept1_storage.mean()
        n_years_low_carryover = (sept1_storage < 50).sum()

        # Metric 5: Trenton Flow Target Reliability
        trenton_shortage = data.shortage[dataset_id][r]['delTrenton']
        trenton_target = data.mrf_target[dataset_id][r]['delTrenton']
        annual_trenton_shortage = trenton_shortage.resample('YS').sum()
        annual_trenton_target = trenton_target.resample('YS').sum()
        trenton_reliability = 1 - (annual_trenton_shortage / annual_trenton_target)
        trenton_reliability = trenton_reliability.clip(0, 1)
        n_years_trenton_reliable = (trenton_reliability > 0.90).sum()

        # Metric 6: NYC Diversion Shortage Frequency
        nyc_diversion_actual = data.ibt_diversions[dataset_id][r]['delivery_nyc']
        nyc_diversion_demand = data.ibt_demands[dataset_id][r]['demand_nyc']
        nyc_diversion_shortage = nyc_diversion_demand - nyc_diversion_actual
        nyc_diversion_shortage[nyc_diversion_shortage < 0] = 0
        n_days_diversion_shortage = (nyc_diversion_shortage > 0).sum()
        pct_days_diversion_shortage = 100.0 * n_days_diversion_shortage / len(nyc_diversion_shortage)

        # Metric 7: Maximum Consecutive Days in Drought (Montague shortage)
        montague_shortage_binary = (montague_shortage > 0).astype(int)
        # Find consecutive stretches
        drought_events = montague_shortage_binary.groupby(
            (montague_shortage_binary != montague_shortage_binary.shift()).cumsum()
        ).sum()
        max_consecutive_shortage_days = drought_events.max() if len(drought_events[drought_events > 0]) > 0 else 0

        # Metric 8: Combined NYC Release for Downstream Targets (Mean Annual)
        total_nyc_contribution = data.contribution[dataset_id][r]['mrf_montagueTrenton_nyc']
        mean_annual_nyc_contribution_mg = total_nyc_contribution.resample('YS').sum().mean()
        max_annual_nyc_contribution_mg = total_nyc_contribution.resample('YS').sum().max()

        # Store metrics
        metrics[r] = {
            'years_reliable': n_years_reliable,
            'years_high_storage': n_years_high_storage,
            'years_above_20pct': n_years_above_20pct,
            'years_above_10pct': n_years_above_10pct,
            'mean_sept1_storage_pct': mean_sept1_storage_pct,
            'years_low_carryover': n_years_low_carryover,
            'years_trenton_reliable': n_years_trenton_reliable,
            'pct_days_nyc_diversion_shortage': pct_days_diversion_shortage,
            'max_consecutive_drought_days': max_consecutive_shortage_days,
            'mean_annual_nyc_contribution_mg': mean_annual_nyc_contribution_mg,
            'max_annual_nyc_contribution_mg': max_annual_nyc_contribution_mg
        }

    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.index.name = 'realization_id'

    print(f"  Calculated {len(metrics_df)} rows of performance metrics")

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
