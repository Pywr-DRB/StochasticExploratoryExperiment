"""
MPI-based parallel postprocessing for ensemble data.

NOTE: This script reimplements postprocessing logic (shortage calculation,
contribution calculation, historical model loading, gage flow loading) inline
for MPI distribution. The serial equivalent uses methods.postprocess.postprocess_dataset().
Any changes to the core postprocessing logic must be applied in BOTH places.
See also: serial_postprocessing.py (Step 4), methods/postprocess.py.

This script uses mpi4py to distribute shortage and contribution calculations
across multiple MPI ranks for faster processing of large ensembles.

REDESIGNED APPROACH - Split by Ensemble Sets (not realizations):
- Each rank processes one or more ENSEMBLE SETS (e.g., set1, set2, etc.)
- Each rank loads ALL realizations for its assigned ensemble sets
- This approach is more memory-efficient than loading partial data from many sets
- Preserves 0-1999 realization ID numbering in final output

Memory optimization:
- Each rank only loads the ensemble sets it's responsible for
- Data is processed and then discarded before gathering
- Only computed metrics (shortage, contribution) are gathered to rank 0

Usage:
    mpirun -np <N_RANKS> python 04_postprocess_data_mpi.py <dataset_id>

Example:
    mpirun -np 20 python 04_postprocess_data_mpi.py stationary_ensemble

Note: Optimal performance when N_RANKS = N_ENSEMBLE_SETS (e.g., 20 ranks for 20 sets)

Memory-efficient mode (--low-memory):
- Each rank writes its results to a temporary HDF5 file
- Rank 0 combines the temporary files sequentially
- Avoids MPI gather memory bottleneck
"""

import sys
import os
import gc
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# MPI CONFIGURATION
# =============================================================================
from methods.mpi_utils import get_comm, MPI_AVAILABLE, global_point_to_point_gather

import pywrdrb
from methods.metrics.shortfall import get_flow_and_target_values, add_trenton_equiv_flow, calculate_shortage_series
from methods.config import *
from methods.postprocess import (
    calculate_and_save_performance_metrics,
    calculate_contribution_analysis_metrics,
    calculate_and_save_zone_duration_events,
)

# Temporary directory for intermediate files
TEMP_DIR = f"{ROOT_DIR}/pywrdrb/outputs/temp_mpi"

# Output directory for performance metrics
PERFORMANCE_METRICS_DIR = f"{ROOT_DIR}/pywrdrb/performance_metrics"
os.makedirs(PERFORMANCE_METRICS_DIR, exist_ok=True)


def distribute_ensemble_sets(rank, size, n_sets):
    """
    Determine which ensemble sets this rank should process.

    Parameters
    ----------
    rank : int
        MPI rank (0 to size-1)
    size : int
        Total number of MPI ranks
    n_sets : int
        Total number of ensemble sets to distribute

    Returns
    -------
    rank_sets : list
        List of ensemble set indices (0-based) for this rank
    """
    sets_per_rank = n_sets // size
    remainder = n_sets % size

    start = rank * sets_per_rank + min(rank, remainder)
    count = sets_per_rank + (1 if rank < remainder else 0)

    return list(range(start, start + count))


def load_ensemble_set_data(dataset_id, set_idx, ensemble_set_specs):
    """
    Load data for a single ensemble set with all its realizations.

    This is memory-efficient because it only loads ONE ensemble set at a time.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    set_idx : int
        Ensemble set index (0-based)
    ensemble_set_specs : list
        List of EnsembleSetSpec objects

    Returns
    -------
    data : pywrdrb.Data
        Data object with this ensemble set loaded
    """
    spec = ensemble_set_specs[set_idx]

    # Setup pathnavigator for this specific set
    pn_config = pywrdrb.get_pn_config()
    dataset_dir = spec.directory
    dataset_name = spec.directory.split('/')[-1]
    pn_config[f"flows/{dataset_name}"] = os.path.abspath(dataset_dir)
    pywrdrb.load_pn_config(pn_config)

    # Load simulation outputs for this ensemble set
    output_filenames = [spec.output_file]

    results_sets = [
        "major_flow",
        "inflow",
        "res_storage",
        "res_release",
        "mrf_target",
        "ibt_diversions",
        "ibt_demands",
        "nyc_release_components",
        "res_level"
    ]

    data = pywrdrb.Data(results_sets=results_sets, print_status=False)
    data.load_output(output_filenames=output_filenames)

    return data


def compute_metrics_for_ensemble_set(data, dataset_id, set_idx, rank):
    """
    Compute shortage and contribution metrics for all realizations in one ensemble set.

    Parameters
    ----------
    data : pywrdrb.Data
        Data object with ensemble set loaded
    dataset_id : str
        Dataset identifier
    set_idx : int
        Ensemble set index (0-based)
    rank : int
        MPI rank

    Returns
    -------
    results : dict
        Dictionary with computed metrics and data for this ensemble set
    """
    set_name = f"{dataset_id}_set{set_idx + 1}"

    # Get realization IDs in this set
    if set_name not in data.major_flow:
        print(f"  Rank {rank}: WARNING - {set_name} not found in data.major_flow")
        return None

    local_ids = sorted(data.major_flow[set_name].keys())
    min_local_id = min(local_ids) if local_ids else 0

    print(f"  Rank {rank}: Processing {set_name} ({len(local_ids)} realizations)")

    # Add Trenton equivalent flow for this set
    data = add_trenton_equiv_flow(data)

    # Initialize storage
    results = {
        'set_idx': set_idx,
        'shortage': {},
        'contribution': {},
        'major_flow': {},
        'inflow': {},
        'res_storage': {},
        'ibt_diversions': {},
        'ibt_demands': {},
        'mrf_target': {},
        'res_level': {}
    }

    # Process each realization
    nodes = ['delMontague', 'delTrenton', 'nyc', 'nj']
    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']

    for i, local_id in enumerate(local_ids):
        # Calculate global realization ID
        local_id_normalized = local_id - min_local_id
        global_id = set_idx * N_REALIZATIONS_PER_ENSEMBLE_SET + local_id_normalized

        # Progress reporting (every 25%)
        if len(local_ids) > 4 and (i + 1) % max(1, len(local_ids) // 4) == 0:
            progress = 100 * (i + 1) / len(local_ids)
            print(f"    Rank {rank}: {progress:.0f}% complete ({i+1}/{len(local_ids)} realizations)")

        # Calculate shortages for each node
        node_shortages = {}
        for node in nodes:
            flow_series, target_series = get_flow_and_target_values(
                data, node, set_name, local_id, start_date=None, end_date=None
            )

            shortage_series = calculate_shortage_series(target_series, flow_series)

            node_shortages[node] = shortage_series

        results['shortage'][global_id] = pd.DataFrame(node_shortages)

        # Contribution calculations
        contribution_columns = [f'mrf_montagueTrenton_{res}' for res in nyc_reservoirs]
        total_nyc_contribution = data.nyc_release_components[set_name][local_id].loc[:, contribution_columns].sum(axis=1)
        results['contribution'][global_id] = total_nyc_contribution.to_frame(name='mrf_montagueTrenton_nyc')

        # Copy other results sets with global IDs
        # Add NYC aggregate inflow
        inflow_df = data.inflow[set_name][local_id].copy()
        inflow_df.loc[:, 'nyc'] = inflow_df.loc[:, nyc_reservoirs].sum(axis=1)
        results['inflow'][global_id] = inflow_df

        results['major_flow'][global_id] = data.major_flow[set_name][local_id]
        results['res_storage'][global_id] = data.res_storage[set_name][local_id]
        results['ibt_diversions'][global_id] = data.ibt_diversions[set_name][local_id]
        results['ibt_demands'][global_id] = data.ibt_demands[set_name][local_id]
        results['mrf_target'][global_id] = data.mrf_target[set_name][local_id]
        results['res_level'][global_id] = data.res_level[set_name][local_id]

    print(f"  Rank {rank}: Completed {set_name}")

    return results


def process_ensemble_sets_on_rank(dataset_id, ensemble_set_specs, rank_sets, rank):
    """
    Process all ensemble sets assigned to this rank.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    ensemble_set_specs : list
        List of EnsembleSetSpec objects
    rank_sets : list
        List of ensemble set indices assigned to this rank
    rank : int
        MPI rank

    Returns
    -------
    all_results : list
        List of results dictionaries for each ensemble set
    """
    all_results = []

    for set_idx in rank_sets:
        # Load data for this ensemble set
        print(f"  Rank {rank}: Loading ensemble set {set_idx + 1}...")
        data = load_ensemble_set_data(dataset_id, set_idx, ensemble_set_specs)

        # Compute metrics for this ensemble set
        results = compute_metrics_for_ensemble_set(data, dataset_id, set_idx, rank)

        if results is not None:
            all_results.append(results)

        # Free memory after processing each set
        del data
        gc.collect()

    return all_results


def save_results_to_temp_file(results_list, dataset_id, rank):
    """
    Save processed results to a temporary HDF5 file for this rank.

    This is used in low-memory mode to avoid gathering all data in memory.

    Parameters
    ----------
    results_list : list
        List of results dictionaries from this rank
    dataset_id : str
        Dataset identifier
    rank : int
        MPI rank

    Returns
    -------
    temp_fname : str
        Path to the temporary file
    """
    os.makedirs(TEMP_DIR, exist_ok=True)
    temp_fname = f"{TEMP_DIR}/{dataset_id}_rank{rank}_temp.hdf5"

    # Combine all results from this rank into single dictionaries
    combined = {
        'shortage': {},
        'contribution': {},
        'major_flow': {},
        'inflow': {},
        'res_storage': {},
        'ibt_diversions': {},
        'ibt_demands': {},
        'mrf_target': {},
        'res_level': {}
    }

    for set_results in results_list:
        if set_results is None:
            continue
        for key in combined.keys():
            if key in set_results:
                combined[key].update(set_results[key])

    # Create a temporary pywrdrb.Data object and export
    temp_data = pywrdrb.Data()
    temp_data.shortage = {dataset_id: combined['shortage']}
    temp_data.contribution = {dataset_id: combined['contribution']}
    temp_data.major_flow = {dataset_id: combined['major_flow']}
    temp_data.inflow = {dataset_id: combined['inflow']}
    temp_data.res_storage = {dataset_id: combined['res_storage']}
    temp_data.ibt_diversions = {dataset_id: combined['ibt_diversions']}
    temp_data.ibt_demands = {dataset_id: combined['ibt_demands']}
    temp_data.mrf_target = {dataset_id: combined['mrf_target']}
    temp_data.res_level = {dataset_id: combined['res_level']}

    temp_data.export(temp_fname)
    print(f"  Rank {rank}: Saved {len(combined['shortage'])} realizations to {temp_fname}")

    return temp_fname


def combine_temp_files_to_final(dataset_id, temp_files, ensemble_set_specs):
    """
    Combine temporary files from all ranks into final output.

    This is used in low-memory mode - loads and combines files sequentially
    to avoid holding all data in memory at once.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    temp_files : list
        List of temporary file paths from all ranks
    ensemble_set_specs : list
        List of EnsembleSetSpec objects

    Returns
    -------
    keep_data : pywrdrb.Data
        Combined data object
    """
    print(f"\nCombining {len(temp_files)} temporary files...")

    # Initialize combined dictionaries
    combined = {
        'shortage': {},
        'contribution': {},
        'major_flow': {},
        'inflow': {},
        'res_storage': {},
        'ibt_diversions': {},
        'ibt_demands': {},
        'mrf_target': {},
        'res_level': {}
    }

    results_sets = list(combined.keys())

    # Load each temp file sequentially and merge
    for i, temp_fname in enumerate(temp_files):
        if temp_fname is None or not os.path.exists(temp_fname):
            continue

        print(f"  Loading temp file {i+1}/{len(temp_files)}: {temp_fname}")

        temp_data = pywrdrb.Data()
        temp_data.load_from_export(temp_fname, results_sets=results_sets)

        # Merge data
        for key in combined.keys():
            data_dict = getattr(temp_data, key, {})
            if dataset_id in data_dict:
                combined[key].update(data_dict[dataset_id])

        # Free memory
        del temp_data
        gc.collect()

        # Optionally delete temp file after loading
        try:
            os.remove(temp_fname)
        except:
            pass

    # Verify we have all realizations
    n_realizations = len(combined['shortage'])
    print(f"  Combined {n_realizations} ensemble realizations")

    if n_realizations != TOTAL_REALIZATIONS:
        print(f"  WARNING: Expected {TOTAL_REALIZATIONS} realizations, got {n_realizations}")

    # Load and process historical models
    historical_data = load_and_process_historical_models(dataset_id)

    # Load gage flow data
    combined_gage_flow = load_gage_flow_data(dataset_id, ensemble_set_specs)

    # Create final data object with all models
    keep_data = pywrdrb.Data()

    # Combine ensemble and historical data
    keep_data.shortage = {dataset_id: combined['shortage']}
    keep_data.shortage.update(historical_data['shortage'])

    keep_data.contribution = {dataset_id: combined['contribution']}
    keep_data.contribution.update(historical_data['contribution'])

    keep_data.major_flow = {dataset_id: combined['major_flow']}
    keep_data.major_flow.update(historical_data['major_flow'])

    keep_data.inflow = {dataset_id: combined['inflow']}
    keep_data.inflow.update(historical_data['inflow'])

    keep_data.res_storage = {dataset_id: combined['res_storage']}
    keep_data.res_storage.update(historical_data['res_storage'])

    keep_data.ibt_diversions = {dataset_id: combined['ibt_diversions']}
    keep_data.ibt_diversions.update(historical_data['ibt_diversions'])

    keep_data.ibt_demands = {dataset_id: combined['ibt_demands']}
    keep_data.ibt_demands.update(historical_data['ibt_demands'])

    keep_data.mrf_target = {dataset_id: combined['mrf_target']}
    keep_data.mrf_target.update(historical_data['mrf_target'])

    keep_data.res_level = {dataset_id: combined['res_level']}
    keep_data.res_level.update(historical_data['res_level'])

    # Add gage flow
    keep_data.gage_flow = {dataset_id: combined_gage_flow}

    # Export final file
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
    print(f"Exporting combined data to {fname}...")
    keep_data.export(fname)
    print(f"Successfully combined and exported data for {dataset_id}!")

    # Clean up temp directory
    try:
        os.rmdir(TEMP_DIR)
    except:
        pass

    return keep_data


def load_and_process_historical_models(dataset_id):
    """
    Load and process historical/reference models (reconstruction, wrfaorc, wrf1960s).

    This is done on rank 0 only since these models are not distributed across ensemble sets.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier

    Returns
    -------
    historical_data : dict
        Dictionary containing processed data for historical models
    """
    print("\nLoading and processing historical models...")

    # Load historical model outputs
    output_filenames = [
        RECONSTRUCTION_OUTPUT_FNAME,
        WRFAORC_OUTPUT_FNAME,
        WRF1960s_OUTPUT_FNAME
    ]

    results_sets = [
        "major_flow",
        "inflow",
        "res_storage",
        "res_release",
        "mrf_target",
        "ibt_diversions",
        "ibt_demands",
        "nyc_release_components",
        "res_level"
    ]

    data = pywrdrb.Data(results_sets=results_sets, print_status=False)
    data.load_output(output_filenames=output_filenames)

    # Load observations
    data.load_observations(results_sets=['res_storage', 'major_flow', 'reservoir_downstream_gage'])
    data.res_release['obs'] = {}
    data.res_release['obs'][0] = data.reservoir_downstream_gage['obs'][0]

    # Add Trenton equivalent flow
    data = add_trenton_equiv_flow(data)

    # Process historical models
    historical_models = ['reconstruction', 'wrfaorc_withObsScaled', 'wrf1960s_calib_nlcd2016']
    nodes = ['delMontague', 'delTrenton', 'nyc', 'nj']
    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']

    historical_data = {
        'shortage': {},
        'contribution': {},
        'inflow': {},
        'major_flow': {},
        'res_storage': {},
        'ibt_diversions': {},
        'ibt_demands': {},
        'mrf_target': {},
        'res_level': {}
    }

    for model in historical_models:
        if model not in data.major_flow:
            print(f"  WARNING: {model} not found in loaded data")
            continue

        realizations = list(data.major_flow[model].keys())
        print(f"  Processing {model} ({len(realizations)} realizations)...")

        shortage_dict = {}
        contribution_dict = {}

        for r in realizations:
            # Calculate shortages for each node
            node_shortages = {}
            for node in nodes:
                flow_series, target_series = get_flow_and_target_values(
                    data, node, model, r, start_date=None, end_date=None
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

            # Contribution calculations
            contribution_columns = [f'mrf_montagueTrenton_{res}' for res in nyc_reservoirs]
            total_nyc_contribution = data.nyc_release_components[model][r].loc[:, contribution_columns].sum(axis=1)
            contribution_dict[r] = total_nyc_contribution.to_frame(name='mrf_montagueTrenton_nyc')

            # Add NYC aggregate inflow
            data.inflow[model][r].loc[:, 'nyc'] = data.inflow[model][r].loc[:, nyc_reservoirs].sum(axis=1)

        historical_data['shortage'][model] = shortage_dict
        historical_data['contribution'][model] = contribution_dict
        historical_data['inflow'][model] = data.inflow[model]
        historical_data['major_flow'][model] = data.major_flow[model]
        historical_data['res_storage'][model] = data.res_storage[model]
        historical_data['ibt_diversions'][model] = data.ibt_diversions[model]
        historical_data['ibt_demands'][model] = data.ibt_demands[model]
        historical_data['mrf_target'][model] = data.mrf_target[model]
        historical_data['res_level'][model] = data.res_level[model]

    print("  Historical model processing complete")
    return historical_data


def load_gage_flow_data(dataset_id, ensemble_set_specs):
    """
    Load and combine gage flow data from all ensemble sets.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    ensemble_set_specs : list
        List of EnsembleSetSpec objects

    Returns
    -------
    combined_gage_flow : dict
        Combined gage flow data with global realization IDs
    """
    print("\nLoading gage flow data...")

    # Setup pathnavigator
    pn_config = pywrdrb.get_pn_config()
    for spec in ensemble_set_specs:
        dataset_dir = spec.directory
        dataset_name = spec.directory.split('/')[-1]
        pn_config[f"flows/{dataset_name}"] = os.path.abspath(dataset_dir)
    pywrdrb.load_pn_config(pn_config)

    # Load hydrologic flow data
    ensemble_set_names = [spec.directory.split('/')[-1] for spec in ensemble_set_specs]
    results_sets = ['major_flow']
    data = pywrdrb.Data(results_sets=results_sets, print_status=False)
    data.load_hydrologic_model_flow(ensemble_set_names)

    # Combine all sets into single dataset key
    combined_gage_flow = {}
    for set_name in ensemble_set_names:
        if set_name not in data.major_flow:
            continue
        set_data = data.major_flow[set_name]
        set_idx = int(set_name.split('_set')[-1]) - 1

        local_ids = list(set_data.keys())
        min_local_id = min(local_ids) if local_ids else 0

        for local_id, df in set_data.items():
            local_id_normalized = local_id - min_local_id
            global_id = set_idx * N_REALIZATIONS_PER_ENSEMBLE_SET + local_id_normalized
            combined_gage_flow[global_id] = df

    print(f"  Loaded gage flow for {len(combined_gage_flow)} realizations")
    return combined_gage_flow


def combine_and_export_results(all_rank_results, dataset_id, ensemble_set_specs):
    """
    Combine results from all ranks and export to HDF5 (rank 0 only).

    Parameters
    ----------
    all_rank_results : list
        List of lists - each inner list contains results dicts from one rank
    dataset_id : str
        Dataset identifier
    ensemble_set_specs : list
        List of EnsembleSetSpec objects

    Returns
    -------
    keep_data : pywrdrb.Data
        Combined data object
    """
    print(f"\nCombining results from all ranks...")

    # Initialize combined dictionaries for ensemble data
    combined_shortage = {}
    combined_contribution = {}
    combined_major_flow = {}
    combined_inflow = {}
    combined_res_storage = {}
    combined_ibt_diversions = {}
    combined_ibt_demands = {}
    combined_mrf_target = {}
    combined_res_level = {}

    # Flatten and combine results from all ranks
    for rank_results in all_rank_results:
        if rank_results is None:
            continue

        for set_results in rank_results:
            if set_results is None:
                continue

            # Merge all dictionaries
            combined_shortage.update(set_results['shortage'])
            combined_contribution.update(set_results['contribution'])
            combined_major_flow.update(set_results['major_flow'])
            combined_inflow.update(set_results['inflow'])
            combined_res_storage.update(set_results['res_storage'])
            combined_ibt_diversions.update(set_results['ibt_diversions'])
            combined_ibt_demands.update(set_results['ibt_demands'])
            combined_mrf_target.update(set_results['mrf_target'])
            combined_res_level.update(set_results['res_level'])

    # Verify we have all realizations
    n_realizations = len(combined_shortage)
    print(f"  Combined {n_realizations} ensemble realizations")

    if n_realizations != TOTAL_REALIZATIONS:
        print(f"  WARNING: Expected {TOTAL_REALIZATIONS} realizations, got {n_realizations}")

    # Load and process historical models
    historical_data = load_and_process_historical_models(dataset_id)

    # Load gage flow data
    combined_gage_flow = load_gage_flow_data(dataset_id, ensemble_set_specs)

    # Create final data object with all models
    keep_data = pywrdrb.Data()

    # Combine ensemble and historical data
    keep_data.shortage = {dataset_id: combined_shortage}
    keep_data.shortage.update(historical_data['shortage'])

    keep_data.contribution = {dataset_id: combined_contribution}
    keep_data.contribution.update(historical_data['contribution'])

    keep_data.major_flow = {dataset_id: combined_major_flow}
    keep_data.major_flow.update(historical_data['major_flow'])

    keep_data.inflow = {dataset_id: combined_inflow}
    keep_data.inflow.update(historical_data['inflow'])

    keep_data.res_storage = {dataset_id: combined_res_storage}
    keep_data.res_storage.update(historical_data['res_storage'])

    keep_data.ibt_diversions = {dataset_id: combined_ibt_diversions}
    keep_data.ibt_diversions.update(historical_data['ibt_diversions'])

    keep_data.ibt_demands = {dataset_id: combined_ibt_demands}
    keep_data.ibt_demands.update(historical_data['ibt_demands'])

    keep_data.mrf_target = {dataset_id: combined_mrf_target}
    keep_data.mrf_target.update(historical_data['mrf_target'])

    keep_data.res_level = {dataset_id: combined_res_level}
    keep_data.res_level.update(historical_data['res_level'])

    # Add gage flow
    keep_data.gage_flow = {dataset_id: combined_gage_flow}

    # Export
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
    print(f"Exporting combined data to {fname}...")
    keep_data.export(fname)
    print(f"Successfully combined and exported data for {dataset_id}!")

    return keep_data


def combine_ensemble_sets_and_calculate_metrics_mpi(dataset_id, low_memory=False):
    """
    MPI-parallel version of combine_ensemble_sets_and_calculate_metrics.

    NEW APPROACH: Split by ensemble sets instead of realizations.
    Each rank processes complete ensemble sets, which is more memory-efficient.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    low_memory : bool
        If True, use file-based intermediate storage instead of MPI gather.
        This is slower but uses much less memory on rank 0.

    Returns
    -------
    keep_data : pywrdrb.Data
        Combined data object (only on rank 0, None on other ranks)
    """
    comm, rank, size = get_comm()

    if rank == 0:
        print(f"\n{'='*80}")
        print(f"MPI PARALLEL POSTPROCESSING: {dataset_id}")
        print(f"{'='*80}")
        print(f"MPI ranks: {size}")
        print(f"Ensemble sets: {N_ENSEMBLE_SETS}")
        print(f"Total realizations: {TOTAL_REALIZATIONS}")
        print(f"Realizations per set: {N_REALIZATIONS_PER_ENSEMBLE_SET}")
        print(f"Low-memory mode: {low_memory}")

        # Optimal configuration info
        if size == N_ENSEMBLE_SETS:
            print(f"Optimal configuration: 1 set per rank")
        elif size < N_ENSEMBLE_SETS:
            sets_per_rank = N_ENSEMBLE_SETS // size
            remainder = N_ENSEMBLE_SETS % size
            print(f"Sets per rank: {sets_per_rank}" + (f" to {sets_per_rank + 1}" if remainder > 0 else ""))
        else:
            print(f"WARNING: More ranks ({size}) than ensemble sets ({N_ENSEMBLE_SETS})")
            print(f"  Some ranks will be idle. Consider using fewer ranks.")

    # Get dataset configuration
    ensemble_set_specs = ENSEMBLE_SETS[dataset_id]

    # Distribute ensemble sets across ranks
    rank_sets = distribute_ensemble_sets(rank, size, N_ENSEMBLE_SETS)

    if len(rank_sets) > 0:
        print(f"Rank {rank}: Assigned ensemble sets {[s+1 for s in rank_sets]} ({len(rank_sets)} total)")
    else:
        print(f"Rank {rank}: No ensemble sets assigned (idle)")

    # Each rank processes its assigned ensemble sets independently (no barrier)
    if len(rank_sets) > 0:
        rank_results = process_ensemble_sets_on_rank(
            dataset_id, ensemble_set_specs, rank_sets, rank
        )
    else:
        rank_results = []

    keep_data = None

    if low_memory:
        # LOW-MEMORY MODE: Each rank writes to temp file, rank 0 combines files
        if len(rank_results) > 0:
            temp_fname = save_results_to_temp_file(rank_results, dataset_id, rank)
        else:
            temp_fname = None

        # Free memory after saving to file
        del rank_results
        gc.collect()

        # Gather file paths via point-to-point (no collective gather)
        all_temp_files = global_point_to_point_gather(
            comm, temp_fname, rank, size, tag=700
        )

        # Rank 0 combines temp files
        if rank == 0:
            # Filter out None entries
            valid_temp_files = [f for f in all_temp_files if f is not None]
            keep_data = combine_temp_files_to_final(dataset_id, valid_temp_files, ensemble_set_specs)

    else:
        # STANDARD MODE: Gather all results via point-to-point
        if rank == 0:
            print(f"\nGathering results from all ranks...")

        all_results = global_point_to_point_gather(
            comm, rank_results, rank, size, tag=701
        )

        # Rank 0 combines and exports
        if rank == 0:
            keep_data = combine_and_export_results(all_results, dataset_id, ensemble_set_specs)

    return keep_data


def process_dataset_mpi(dataset_id, recombine_sets=True, low_memory=False):
    """
    MPI-parallel version of process_dataset.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    recombine_sets : bool
        If True, recombine all ensemble sets (time-intensive)
        If False, load existing combined data (much faster)
    low_memory : bool
        If True, use file-based intermediate storage instead of MPI gather

    Returns
    -------
    success : bool
        True if processing completed successfully
    """
    comm, rank, size = get_comm()

    if rank == 0:
        print(f"\n{'='*80}")
        print(f"PROCESSING DATASET (MPI): {dataset_id}")
        print(f"{'='*80}")

    dataset_config = DATASET_CONFIGS[dataset_id]
    ensemble_set_specs = ENSEMBLE_SETS[dataset_id]

    # Check if all sets have been simulated (rank 0 only)
    if rank == 0:
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
        # Recombine all ensemble sets (MPI parallel)
        keep_data = combine_ensemble_sets_and_calculate_metrics_mpi(dataset_id, low_memory=low_memory)

    else:
        # Load existing combined data (rank 0 only)
        keep_data = None
        if rank == 0:
            print(f"\nrecombine_sets=False. Loading existing combined data from:")
            print(f"  {fname}")

            required_results_sets = ['shortage', 'mrf_target', 'res_storage',
                                     'ibt_diversions', 'ibt_demands', 'contribution',
                                     'res_level', 'inflow']

            keep_data = pywrdrb.Data()
            keep_data.load_from_export(fname, results_sets=required_results_sets)
            print(f"Successfully loaded combined data for {dataset_id}!")

    # Calculate performance metrics (rank 0 only)
    if rank == 0 and keep_data is not None:
        print(f"\nCalculating performance metrics for {dataset_id}...")
        try:
            realizations = list(keep_data.shortage[dataset_id].keys())
            print(f"  Found {len(realizations)} realizations in {dataset_id}")

            calculate_and_save_performance_metrics(
                keep_data, dataset_id, realizations, PERFORMANCE_METRICS_DIR
            )

            # NEW: Calculate contribution analysis metrics
            print(f"\nCalculating contribution analysis metrics for {dataset_id}...")
            contrib_metrics = calculate_contribution_analysis_metrics(
                keep_data, dataset_id, realizations
            )
            contrib_fname = f"{PERFORMANCE_METRICS_DIR}/{dataset_id}_contribution_metrics.csv"
            contrib_metrics.to_csv(contrib_fname, index=False)
            print(f"  Saved: {contrib_fname} ({len(contrib_metrics)} year-realization pairs)")

            # Calculate zone drought episode durations
            print(f"\nCalculating zone duration events for {dataset_id}...")
            calculate_and_save_zone_duration_events(
                keep_data, dataset_id, realizations, PERFORMANCE_METRICS_DIR
            )

        except Exception as e:
            print(f"ERROR calculating metrics for {dataset_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Also calculate historic (reconstruction) metrics for comparison
        print(f"\nCalculating historic (reconstruction) performance metrics...")
        try:
            reconstruction_realizations = list(keep_data.shortage['reconstruction'].keys())
            print(f"  Found {len(reconstruction_realizations)} realizations in reconstruction")
            calculate_and_save_performance_metrics(
                keep_data, 'reconstruction', reconstruction_realizations, PERFORMANCE_METRICS_DIR
            )
            calculate_and_save_zone_duration_events(
                keep_data, 'reconstruction', reconstruction_realizations, PERFORMANCE_METRICS_DIR
            )
        except KeyError:
            print(f"WARNING: Could not find 'reconstruction' data in loaded file.")
        except Exception as e:
            print(f"WARNING: Error calculating historic metrics: {e}")

    return True


def main_mpi(dataset_id, recombine_sets=True, low_memory=False):
    """
    Main MPI entry point for postprocessing.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    recombine_sets : bool
        If True, recombine ensemble sets from scratch
    low_memory : bool
        If True, use file-based intermediate storage instead of MPI gather
    """
    comm, rank, size = get_comm()

    if rank == 0:
        print("=" * 80)
        print(f"MPI POSTPROCESSING: {dataset_id}")
        print("=" * 80)

        verify_dataset_id(dataset_id)
        dataset_config = DATASET_CONFIGS[dataset_id]

        print(f"Dataset type: {dataset_config['type']}")
        print(f"Description: {dataset_config['description']}")
        print(f"Total realizations: {TOTAL_REALIZATIONS}")
        print(f"Ensemble sets: {N_ENSEMBLE_SETS}")
        print(f"Recombine sets: {recombine_sets}")
        print(f"Low-memory mode: {low_memory}")
        print("=" * 80)

    # Process the dataset
    success = process_dataset_mpi(dataset_id, recombine_sets=recombine_sets, low_memory=low_memory)

    if rank == 0:
        if success:
            # Verify output
            fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
            if os.path.exists(fname):
                file_size = os.path.getsize(fname)
                print(f"\nSUCCESS: Postprocessed data file exists ({file_size//1024//1024} MB)")
            else:
                print(f"\nFAIL: Output file not found: {fname}")

        print("=" * 80)
        print(f"Postprocessing {'completed successfully' if success else 'failed'}!")
        print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: mpirun -np <N_RANKS> python 04_postprocess_data_mpi.py <dataset_id> [options]")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        print()
        print("Options:")
        print("  --skip-recombine    Load existing combined data instead of recombining (fast)")
        print("  --low-memory        Use file-based intermediate storage (slower but less memory)")
        print()
        print("Recommended: Use N_RANKS equal to N_ENSEMBLE_SETS for optimal performance")
        print(f"             (currently N_ENSEMBLE_SETS = {N_ENSEMBLE_SETS})")
        print()
        print("Examples:")
        print(f"  mpirun -np {N_ENSEMBLE_SETS} python 04_postprocess_data_mpi.py stationary_ensemble")
        print(f"  mpirun -np {N_ENSEMBLE_SETS} python 04_postprocess_data_mpi.py stationary_ensemble --low-memory")
        print("  mpirun -np 16 python 04_postprocess_data_mpi.py stationary_ensemble --skip-recombine")
        sys.exit(1)

    dataset_id = sys.argv[1]
    verify_dataset_id(dataset_id)

    # Parse command line options
    recombine_sets = True
    low_memory = False

    for arg in sys.argv[2:]:
        if arg == '--skip-recombine':
            recombine_sets = False
        elif arg == '--low-memory':
            low_memory = True
        else:
            print(f"ERROR: Unknown option '{arg}'")
            print("Valid options: --skip-recombine, --low-memory")
            sys.exit(1)

    main_mpi(dataset_id, recombine_sets=recombine_sets, low_memory=low_memory)
