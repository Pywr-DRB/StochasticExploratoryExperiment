"""
MPI-based parallel postprocessing for ensemble data.

Shared logic (shortage calculation, data loading, historical model processing)
lives in methods/metrics/shortfall.py and methods/load.py.
The serial equivalent uses methods.postprocess.postprocess_dataset().
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
from methods.ensemble_utils import ENSEMBLE_SETS
from methods.load import (
    load_ensemble_set_data,
    load_and_process_historical_models,
    load_gage_flow_data,
)
# Temporary directory for intermediate files
TEMP_DIR = f"{OUTPUT_DIR}/temp_mpi"


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
        print(f"  Rank {rank} WARNING: {set_name} not found in data.major_flow")
        return None

    local_ids = sorted(data.major_flow[set_name].keys())
    min_local_id = min(local_ids) if local_ids else 0

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

    combined = {key: {} for key in RESULTS_SET_KEYS}
    for set_results in results_list:
        if set_results is None:
            continue
        for key in RESULTS_SET_KEYS:
            if key in set_results:
                combined[key].update(set_results[key])

    temp_data = pywrdrb.Data()
    for key in RESULTS_SET_KEYS:
        setattr(temp_data, key, {dataset_id: combined[key]})

    temp_data.export(temp_fname)
    return temp_fname


def combine_temp_files_to_final(dataset_id, temp_files, ensemble_set_specs):
    """
    Combine temporary files from all ranks into final output.

    Loads and combines files sequentially to avoid holding all data in memory
    at once during the MPI gather phase.

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
    combined = {key: {} for key in RESULTS_SET_KEYS}

    for i, temp_fname in enumerate(temp_files):
        if temp_fname is None or not os.path.exists(temp_fname):
            continue

        temp_data = pywrdrb.Data()
        temp_data.load_from_export(temp_fname, results_sets=RESULTS_SET_KEYS)

        for key in RESULTS_SET_KEYS:
            data_dict = getattr(temp_data, key, {})
            if dataset_id in data_dict:
                combined[key].update(data_dict[dataset_id])

        del temp_data
        gc.collect()

        try:
            os.remove(temp_fname)
        except:
            pass

    n_realizations = len(combined['shortage'])
    if n_realizations != TOTAL_REALIZATIONS:
        print(f"[POST] WARNING: Expected {TOTAL_REALIZATIONS} realizations, got {n_realizations}")

    historical_data = load_and_process_historical_models(dataset_id)
    keep_data = pywrdrb.Data()
    for key in RESULTS_SET_KEYS:
        ensemble_and_historical = {dataset_id: combined[key]}
        ensemble_and_historical.update(historical_data[key])
        setattr(keep_data, key, ensemble_and_historical)
    del combined, historical_data
    gc.collect()

    combined_gage_flow = load_gage_flow_data(dataset_id, ensemble_set_specs)
    keep_data.gage_flow = {dataset_id: combined_gage_flow}

    fname = f'{OUTPUT_DIR}/{dataset_id}_with_postprocessing.hdf5'
    keep_data.export(fname)
    print(f"[POST] {dataset_id}: Complete → {os.path.basename(fname)}")

    # Clean up temp directory
    try:
        os.rmdir(TEMP_DIR)
    except:
        pass

    return keep_data


RESULTS_SET_KEYS = [
    'shortage', 'contribution', 'major_flow', 'inflow',
    'res_storage', 'ibt_diversions', 'ibt_demands', 'mrf_target', 'res_level'
]


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
    combined = {key: {} for key in RESULTS_SET_KEYS}

    for rank_results in all_rank_results:
        if rank_results is None:
            continue
        for set_results in rank_results:
            if set_results is None:
                continue
            for key in RESULTS_SET_KEYS:
                combined[key].update(set_results[key])

    n_realizations = len(combined['shortage'])
    if n_realizations != TOTAL_REALIZATIONS:
        print(f"[POST] WARNING: Expected {TOTAL_REALIZATIONS} realizations, got {n_realizations}")

    historical_data = load_and_process_historical_models(dataset_id)
    keep_data = pywrdrb.Data()
    for key in RESULTS_SET_KEYS:
        ensemble_and_historical = {dataset_id: combined[key]}
        ensemble_and_historical.update(historical_data[key])
        setattr(keep_data, key, ensemble_and_historical)
    del combined, historical_data
    gc.collect()

    combined_gage_flow = load_gage_flow_data(dataset_id, ensemble_set_specs)
    keep_data.gage_flow = {dataset_id: combined_gage_flow}

    fname = f'{OUTPUT_DIR}/{dataset_id}_with_postprocessing.hdf5'
    keep_data.export(fname)
    print(f"[POST] {dataset_id}: Complete → {os.path.basename(fname)}")

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
        low_mem_tag = " [low-memory]" if low_memory else ""
        print(f"[POST] {dataset_id} | {N_ENSEMBLE_SETS} sets | {size} ranks{low_mem_tag}")
        if size > N_ENSEMBLE_SETS:
            print(f"[POST] WARNING: More ranks ({size}) than sets ({N_ENSEMBLE_SETS}) — some ranks idle.")

    # Get dataset configuration
    ensemble_set_specs = ENSEMBLE_SETS[dataset_id]

    # Distribute ensemble sets across ranks
    rank_sets = distribute_ensemble_sets(rank, size, N_ENSEMBLE_SETS)


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
    fname = f'{OUTPUT_DIR}/{dataset_id}_with_postprocessing.hdf5'

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
                                     'res_level', 'inflow', 'major_flow']

            keep_data = pywrdrb.Data()
            keep_data.load_from_export(fname, results_sets=required_results_sets)
            print(f"Successfully loaded combined data for {dataset_id}!")

    # Metric CSV calculation has been moved to 06_calculate_performance_metrics.py.
    # This script now only handles HDF5 postprocessing (shortage, contribution, export).

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

    verify_dataset_id(dataset_id)

    success = process_dataset_mpi(dataset_id, recombine_sets=recombine_sets, low_memory=low_memory)

    if rank == 0 and not success:
        print(f"[POST] ERROR: Postprocessing failed for {dataset_id}!")

    # Barrier so all ranks wait for rank 0 to finish before exiting.
    # Without this, non-rank-0 processes exit early and trigger
    # "exited without calling finalize" MPI errors.
    if comm is not None:
        comm.Barrier()


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
