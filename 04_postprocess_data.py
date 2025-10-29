"""
This script does the following (SERIAL VERSION - NO MPI):

- Load gauge flows from all ensemble set gage_flow_mgd files
- Load simulation results from all ensemble set outputs
- Combine the flows and simulation results from all sets into a single ensemble key
- Calculate additional metrics (shortages, contributions, performance metrics)
- Export combined data for analysis

Every dictionary in the final data object has format:

dict = {
    dataset_id: {
        realization_id : pd.DataFrame
    }
}

The pd.DataFrame has datetime index and node names as columns.

Usage:
    python 04_postprocess_data_serial.py <dataset_id>
"""

import sys
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from methods.metrics.shortfall import get_flow_and_target_values, add_trenton_equiv_flow
from config import *

# Output directory for performance metrics
PERFORMANCE_METRICS_DIR = f"{ROOT_DIR}/pywrdrb/performance_metrics"
os.makedirs(PERFORMANCE_METRICS_DIR, exist_ok=True)

# Storage capacities for NYC reservoirs (MG)
NYC_STORAGE_CAPACITIES = {
    'cannonsville': 95706,
    'pepacton': 140190,
    'neversink': 34941
}
NYC_TOTAL_CAPACITY = sum(NYC_STORAGE_CAPACITIES.values())


def calculate_and_save_performance_metrics(data, dataset_id, realizations):
    """
    Calculate performance metrics and save to CSV.

    Parameters
    ----------
    data : pywrdrb.Data
        Data object with shortage, mrf_target, and res_storage
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

        # Metric 2: # years where NYC storage >90% on June 1
        nyc_storage = data.res_storage[dataset_id][r][nyc_reservoirs].sum(axis=1)
        nyc_storage_pct = 100.0 * nyc_storage / NYC_TOTAL_CAPACITY

        # Filter for June 1 dates
        june1_storage = nyc_storage_pct[(nyc_storage_pct.index.month == 6) &
                                        (nyc_storage_pct.index.day == 1)]
        n_years_high_storage = (june1_storage > 90).sum()

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
        nyc_diversion_actual = data.ibt_diversions[dataset_id][r]['nyc']
        nyc_diversion_demand = data.ibt_demands[dataset_id][r]['nyc']
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

    # Save to CSV
    csv_file = f"{PERFORMANCE_METRICS_DIR}/{dataset_id}_performance_metrics.csv"
    metrics_df.to_csv(csv_file)
    print(f"  Saved performance metrics to: {csv_file}")

    # Calculate and print percentiles for key metrics
    print(f"\n  Key Performance Metrics Summary:")
    print(f"  {'='*60}")

    count_metrics = ['years_reliable', 'years_high_storage', 'years_above_20pct',
                     'years_low_carryover', 'years_trenton_reliable']
    for metric in count_metrics:
        p5 = metrics_df[metric].quantile(0.05)
        p50 = metrics_df[metric].quantile(0.50)
        p95 = metrics_df[metric].quantile(0.95)
        print(f"    {metric:40s}: p5={p5:5.1f}, p50={p50:5.1f}, p95={p95:5.1f}")

    print(f"\n  Other Metrics Summary:")
    print(f"  {'='*60}")
    other_metrics = ['pct_days_nyc_diversion_shortage', 'max_consecutive_drought_days',
                     'mean_sept1_storage_pct', 'mean_annual_nyc_contribution_mg']
    for metric in other_metrics:
        p5 = metrics_df[metric].quantile(0.05)
        p50 = metrics_df[metric].quantile(0.50)
        p95 = metrics_df[metric].quantile(0.95)
        if 'pct' in metric or 'storage' in metric:
            print(f"    {metric:40s}: p5={p5:5.1f}, p50={p50:5.1f}, p95={p95:5.1f}")
        else:
            print(f"    {metric:40s}: p5={p5:5.0f}, p50={p50:5.0f}, p95={p95:5.0f}")

    return metrics_df


def combine_ensemble_sets_and_calculate_metrics(dataset_id):
    """
    Combine all ensemble sets and calculate derived metrics (shortage, contributions).
    This is the most time-intensive part of postprocessing.

    Parameters:
    -----------
    dataset_id : str
        Dataset identifier to process

    Returns:
    --------
    keep_data : pywrdrb.Data
        Combined data object with all metrics
    """

    print(f"\n{'='*80}")
    print(f"COMBINING ENSEMBLE SETS AND CALCULATING METRICS: {dataset_id}")
    print(f"{'='*80}")

    dataset_config = DATASET_CONFIGS[dataset_id]
    ensemble_set_specs = ENSEMBLE_SETS[dataset_id]

    ### Load data through pywrdrb API #######################################
    print(f"Loading data for postprocessing...")

    ## Setup pathnavigator
    pn_config = pywrdrb.get_pn_config()
    for spec in ensemble_set_specs:
        dataset_dir = spec.directory
        dataset_name = spec.directory.split('/')[-1]
        pn_config[f"flows/{dataset_name}"] = os.path.abspath(dataset_dir)
    pywrdrb.load_pn_config(pn_config)

    ## Load hydrologic flow data
    print("  Loading hydrologic model flow...")

    ensemble_set_names = [spec.directory.split('/')[-1] for spec in ensemble_set_specs]
    results_sets = ['major_flow']
    data = pywrdrb.Data(results_sets=results_sets, print_status=False)
    data.load_hydrologic_model_flow(ensemble_set_names)

    # Combine all sets into single dataset key
    combined_gage_flow = {}
    for set_name in ensemble_set_names:
        set_data = data.major_flow[set_name]
        # Renumber realizations to be continuous across sets
        set_idx = int(set_name.split('_set')[-1]) - 1

        # Check if local IDs are 0-indexed or 1-indexed
        local_ids = list(set_data.keys())
        min_local_id = min(local_ids)

        for local_id, df in set_data.items():
            # Convert to 0-indexed if needed (local IDs might be 1-100 instead of 0-99)
            local_id_normalized = local_id - min_local_id
            global_id = set_idx * N_REALIZATIONS_PER_ENSEMBLE_SET + local_id_normalized
            combined_gage_flow[global_id] = df

    # Store combined gage flow
    gage_flow_dict = {dataset_id: combined_gage_flow}

    print("  Loading simulation outputs...")

    output_filenames = [spec.output_file for spec in ensemble_set_specs]
    output_filenames.append(RECONSTRUCTION_OUTPUT_FNAME)

    results_sets = [
        "major_flow",
        "inflow",
        "res_storage",
        "res_release",
        "mrf_target",
        "ibt_diversions",
        "ibt_demands",
        "nyc_release_components"
    ]

    data = pywrdrb.Data(results_sets=results_sets, print_status=False)
    data.load_output(output_filenames=output_filenames)
    data.load_observations(results_sets=['res_storage', 'major_flow', 'reservoir_downstream_gage'])
    data.res_release['obs'] = {}
    data.res_release['obs'][0] = data.reservoir_downstream_gage['obs'][0]

    # Combine all sets into single dataset key for each results_set
    for results_set in results_sets:
        combined_data = {}
        full_results_set_dict = getattr(data, results_set)

        for i, spec in enumerate(ensemble_set_specs):
            set_name = f"{dataset_id}_set{i+1}"
            if set_name not in full_results_set_dict:
                print(f"WARNING: {set_name} not found in {results_set}")
                continue

            set_data = full_results_set_dict[set_name]

            # Check if local IDs are 0-indexed or 1-indexed
            local_ids = list(set_data.keys())
            if local_ids:
                min_local_id = min(local_ids)
            else:
                min_local_id = 0

            # Renumber realizations to be continuous
            for local_id, df in set_data.items():
                # Convert to 0-indexed if needed
                local_id_normalized = local_id - min_local_id
                global_id = i * N_REALIZATIONS_PER_ENSEMBLE_SET + local_id_normalized
                combined_data[global_id] = df

        # Store combined data back
        full_results_set_dict[dataset_id] = combined_data
        setattr(data, results_set, full_results_set_dict)

    # Replace gage flow with combined version
    data.major_flow[dataset_id] = gage_flow_dict[dataset_id]

    # Add Trenton equivalent flow AFTER combining datasets
    # This ensures delTrenton_equiv is added to the combined dataset
    data = add_trenton_equiv_flow(data)

    print("  Data loading complete")

    ### Post-process data ##############################################
    print('Calculating shortages for different nodes...')

    all_shortage_dict = {}

    for model in ['reconstruction', dataset_id]:
        realizations = list(data.major_flow[model].keys())

        print(f"  Processing model: {model}")
        print(f"    Total realizations: {len(realizations)}")

        shortage_dict = {}

        # Initialize shortage dict for all realizations
        for r in realizations:
            shortage_dict[r] = {}

        # Process each node
        nodes = ['delMontague', 'delTrenton', 'nyc', 'nj']
        for node in nodes:
            print(f"    Processing {node}...")

            for i, r in enumerate(realizations):
                # Progress reporting (every 10%)
                if len(realizations) > 10 and (i + 1) % max(1, len(realizations) // 10) == 0:
                    progress = 100 * (i + 1) / len(realizations)
                    print(f"      Progress: {progress:.0f}% ({i+1}/{len(realizations)} realizations)")

                flow_series, target_series = get_flow_and_target_values(
                    data, node, model, r,
                    start_date=None, end_date=None
                )

                # Calculate shortages
                shortage_series = target_series - flow_series
                shortage_series[shortage_series < 0] = 0  # Set negative shortages (surplus) to zero
                shortage_series.iloc[:3] = 0.0  # Set first 3 days to 0.0 due to model warmup

                # Ignore shortages when duration of consecutive shortage>0 days is <3
                shortage_durations = (shortage_series > 0).astype(int).groupby(
                    (shortage_series > 0).astype(int).diff().ne(0).cumsum()
                ).cumsum()
                shortage_series[shortage_durations < 3] = 0.0

                shortage_dict[r][node] = shortage_series

        # Convert to DataFrames
        for r in realizations:
            shortage_dict[r] = pd.DataFrame(shortage_dict[r])

        all_shortage_dict[model] = shortage_dict
        print(f"    Completed shortage calculations for {model}")

    ## Calculate downstream contributions from NYC reservoirs
    print('Calculating total downstream contributions...')

    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']
    contribution_columns = [f'mrf_montagueTrenton_{res}' for res in nyc_reservoirs]
    all_contribution_dict = {}

    for model in ['reconstruction', dataset_id]:
        contribution_dict = {}
        realizations = list(data.major_flow[model].keys())

        for r in realizations:
            total_nyc_contribution = data.nyc_release_components[model][r].loc[:, contribution_columns].sum(axis=1)
            contribution_dict[r] = total_nyc_contribution.to_frame(name='mrf_montagueTrenton_nyc')

        all_contribution_dict[model] = contribution_dict

    ### Calculate aggregate NYC inflow
    print('Calculating aggregate NYC inflow...')

    for model in ['reconstruction', dataset_id]:
        realizations = list(data.inflow[model].keys())
        for r in realizations:
            data.inflow[model][r].loc[:, 'nyc'] = data.inflow[model][r].loc[:, nyc_reservoirs].sum(axis=1)

    ### Organize data to be kept for later
    print('Organizing final data structure...')

    keep_data = pywrdrb.Data()
    keep_data.gage_flow = gage_flow_dict
    keep_data.shortage = all_shortage_dict
    keep_data.contribution = all_contribution_dict

    # Make copies of output results_sets just for the combined dataset
    inflow_dict = {}
    major_flow_dict = {}
    res_storage_dict = {}
    ibt_diversions_dict = {}
    ibt_demands_dict = {}
    mrf_target_dict = {}

    for model in ['reconstruction', dataset_id]:
        if model in data.inflow:
            inflow_dict[model] = data.inflow[model]
            major_flow_dict[model] = data.major_flow[model]
            res_storage_dict[model] = data.res_storage[model]
            ibt_diversions_dict[model] = data.ibt_diversions[model]
            ibt_demands_dict[model] = data.ibt_demands[model]
            mrf_target_dict[model] = data.mrf_target[model]

    keep_data.inflow = inflow_dict
    keep_data.major_flow = major_flow_dict
    keep_data.res_storage = res_storage_dict
    keep_data.ibt_diversions = ibt_diversions_dict
    keep_data.ibt_demands = ibt_demands_dict
    keep_data.mrf_target = mrf_target_dict

    ### Export the new data object to HDF5
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
    print(f"Exporting combined data to {fname}...")
    keep_data.export(fname)
    print(f"Successfully combined and exported data for {dataset_id}!")

    return keep_data


def process_dataset(dataset_id, recombine_sets=False):
    """
    Process and combine all ensemble sets for a given dataset, then calculate performance metrics.

    Parameters:
    -----------
    dataset_id : str
        Dataset identifier to process
    recombine_sets : bool, optional
        If True, recombine all ensemble sets from scratch (time-intensive).
        If False, load existing combined data from HDF5 (much faster).
        Default: False

    Returns:
    --------
    success : bool
        True if processing completed successfully
    """

    print(f"\n{'='*80}")
    print(f"PROCESSING DATASET: {dataset_id}")
    print(f"{'='*80}")

    dataset_config = DATASET_CONFIGS[dataset_id]
    ensemble_set_specs = ENSEMBLE_SETS[dataset_id]

    # Check if all sets have been simulated
    missing_sets = []
    for spec in ensemble_set_specs:
        if not os.path.exists(spec.output_file):
            missing_sets.append(spec.set_id + 1)

    if missing_sets:
        print(f"WARNING: Missing output files for sets: {missing_sets}")
        print("Run simulations first!")
        return False

    # Determine whether to recombine or load existing data
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'

    if recombine_sets or not os.path.exists(fname):
        if not os.path.exists(fname):
            print(f"Combined data file not found. Will recombine ensemble sets.")
        else:
            print(f"recombine_sets=True. Will recombine ensemble sets from scratch.")

        # Recombine all ensemble sets (time-intensive)
        keep_data = combine_ensemble_sets_and_calculate_metrics(dataset_id)

    else:
        # Load existing combined data (fast)
        print(f"\nrecombine_sets=False. Loading existing combined data from:")
        print(f"  {fname}")
        keep_data = pywrdrb.Data()
        keep_data.load_from_export(fname)
        print(f"Successfully loaded combined data for {dataset_id}!")

    # Calculate and save performance metrics
    print(f"\nCalculating performance metrics for {dataset_id}...")
    realizations = list(keep_data.shortage[dataset_id].keys())
    calculate_and_save_performance_metrics(keep_data, dataset_id, realizations)

    # Also calculate historic (reconstruction) metrics for comparison
    print(f"\nCalculating historic (reconstruction) performance metrics...")
    reconstruction_realizations = list(keep_data.shortage['reconstruction'].keys())
    calculate_and_save_performance_metrics(keep_data, 'reconstruction', reconstruction_realizations)

    return True


def verify_postprocessing_output(dataset_id):
    """
    Verify that postprocessing output file exists and has reasonable size.

    NOTE: Does NOT load the file (which can take several minutes).
    Just checks existence and file size.

    Parameters:
    -----------
    dataset_id : str
        Dataset identifier to verify

    Returns:
    --------
    exists : bool
        True if file exists and has reasonable size
    """

    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'

    if not os.path.exists(fname):
        print(f"FAIL: Output file not found: {fname}")
        return False

    # Check file size
    file_size = os.path.getsize(fname)
    if file_size < 1024 * 1024:  # Less than 1MB is suspicious
        print(f"WARNING: Output file seems too small ({file_size} bytes)")
        return False

    # File exists and has reasonable size
    print(f"SUCCESS: Postprocessed data file exists ({file_size//1024//1024} MB)")
    return True


def main(dataset_id, recombine_sets=False):
    """
    Main function for postprocessing ensemble data.

    Parameters:
    -----------
    dataset_id : str
        Dataset identifier to process
    recombine_sets : bool, optional
        If True, recombine all ensemble sets from scratch (time-intensive).
        If False, load existing combined data from HDF5 (much faster).
        Default: False
    """

    print("=" * 80)
    print(f"POSTPROCESSING ENSEMBLE DATA: {dataset_id}")
    print("=" * 80)

    # Verify dataset
    verify_dataset_id(dataset_id)
    dataset_config = DATASET_CONFIGS[dataset_id]

    print(f"Dataset type: {dataset_config['type']}")
    print(f"Description: {dataset_config['description']}")
    print(f"Total realizations: {TOTAL_REALIZATIONS}")
    print(f"Ensemble sets: {N_ENSEMBLE_SETS}")
    print(f"Recombine sets: {recombine_sets}")
    print("=" * 80)

    # Process the dataset
    success = process_dataset(dataset_id, recombine_sets=recombine_sets)

    if success:
        # Verify output
        verify_postprocessing_output(dataset_id)

    print("=" * 80)
    print(f"Postprocessing {'completed successfully' if success else 'failed'}!")


if __name__ == "__main__":

    # Get the dataset_id and optional recombine_sets flag from command line arguments
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python 04_postprocess_data.py <dataset_id> [--recombine]")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        print()
        print("Options:")
        print("  --recombine    Recombine ensemble sets from scratch (slow, default: False)")
        print("                 If omitted, will load existing combined data (fast)")
        print()
        print("Examples:")
        print("  python 04_postprocess_data.py stationary_ensemble")
        print("  python 04_postprocess_data.py stationary_ensemble --recombine")
        sys.exit(1)

    dataset_id = sys.argv[1]
    verify_dataset_id(dataset_id)

    # Check for --recombine flag
    recombine_sets = False
    if len(sys.argv) == 3:
        if sys.argv[2] == '--recombine':
            recombine_sets = True
        else:
            print(f"ERROR: Unknown option '{sys.argv[2]}'")
            print("Use --recombine to recombine ensemble sets from scratch")
            sys.exit(1)

    main(dataset_id, recombine_sets=recombine_sets)
