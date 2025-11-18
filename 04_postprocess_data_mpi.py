"""
MPI-based parallel postprocessing for ensemble data.

This script uses mpi4py to distribute shortage and contribution calculations
across multiple MPI ranks for faster processing of large ensembles.

Key differences from serial version (04_postprocess_data.py):
- Uses MPI to parallelize shortage/contribution calculations
- Each rank processes a subset of realizations
- Rank 0 coordinates and gathers results
- Preserves 0-1999 realization ID numbering in final output

Usage:
    mpirun -np <N_RANKS> python 04_postprocess_data_mpi.py <dataset_id>

Example:
    mpirun -np 32 python 04_postprocess_data_mpi.py stationary_ensemble
"""

import sys
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from mpi4py import MPI

import pywrdrb
from methods.metrics.shortfall import get_flow_and_target_values, add_trenton_equiv_flow
from methods.config import *
from methods.postprocess import calculate_and_save_performance_metrics

# Output directory for performance metrics
PERFORMANCE_METRICS_DIR = f"{ROOT_DIR}/pywrdrb/performance_metrics"
os.makedirs(PERFORMANCE_METRICS_DIR, exist_ok=True)


def load_ensemble_sets_for_rank(dataset_id, ensemble_set_specs, rank_realizations):
    """
    Load only the ensemble sets needed for this rank's realizations.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    ensemble_set_specs : list
        List of EnsembleSetSpec objects
    rank_realizations : list
        List of global realization IDs this rank will process

    Returns
    -------
    data : pywrdrb.Data
        Data object with only needed ensemble sets loaded
    needed_set_indices : set
        Set of ensemble set indices that were loaded
    """
    # Determine which ensemble sets this rank needs
    needed_set_indices = set()
    for global_id in rank_realizations:
        set_idx = global_id // N_REALIZATIONS_PER_ENSEMBLE_SET
        needed_set_indices.add(set_idx)

    # Setup pathnavigator
    pn_config = pywrdrb.get_pn_config()
    for spec in ensemble_set_specs:
        dataset_dir = spec.directory
        dataset_name = spec.directory.split('/')[-1]
        pn_config[f"flows/{dataset_name}"] = os.path.abspath(dataset_dir)
    pywrdrb.load_pn_config(pn_config)

    # Load hydrologic flow data for needed sets only
    ensemble_set_names = [
        ensemble_set_specs[i].directory.split('/')[-1]
        for i in sorted(needed_set_indices)
    ]

    results_sets = ['major_flow']
    data = pywrdrb.Data(results_sets=results_sets, print_status=False)
    data.load_hydrologic_model_flow(ensemble_set_names)

    # Load simulation outputs (all sets, since file loading is cheap compared to computation)
    output_filenames = [spec.output_file for spec in ensemble_set_specs]
    output_filenames.append(RECONSTRUCTION_OUTPUT_FNAME)
    output_filenames.append(WRFAORC_OUTPUT_FNAME)
    output_filenames.append(WRF1960s_OUTPUT_FNAME)

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
    data.load_observations(results_sets=['res_storage', 'major_flow', 'reservoir_downstream_gage'])
    data.res_release['obs'] = {}
    data.res_release['obs'][0] = data.reservoir_downstream_gage['obs'][0]

    return data, needed_set_indices


def combine_ensemble_sets_data_only(dataset_id, ensemble_set_specs, data):
    """
    Combine ensemble sets into single dataset key (data loading/renumbering only).
    Does NOT compute metrics - that's done in parallel by ranks.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    ensemble_set_specs : list
        List of EnsembleSetSpec objects
    data : pywrdrb.Data
        Loaded data object

    Returns
    -------
    data : pywrdrb.Data
        Data object with combined ensemble sets
    """
    # Combine gage flow
    ensemble_set_names = [spec.directory.split('/')[-1] for spec in ensemble_set_specs]
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

    data.major_flow[dataset_id] = combined_gage_flow

    # Combine simulation output results_sets
    results_sets = [
        "major_flow", "inflow", "res_storage", "res_release",
        "mrf_target", "ibt_diversions", "ibt_demands",
        "nyc_release_components", "res_level"
    ]

    for results_set in results_sets:
        combined_data = {}
        full_results_set_dict = getattr(data, results_set)

        for i, spec in enumerate(ensemble_set_specs):
            set_name = f"{dataset_id}_set{i+1}"
            if set_name not in full_results_set_dict:
                continue

            set_data = full_results_set_dict[set_name]
            local_ids = list(set_data.keys())
            min_local_id = min(local_ids) if local_ids else 0

            for local_id, df in set_data.items():
                local_id_normalized = local_id - min_local_id
                global_id = i * N_REALIZATIONS_PER_ENSEMBLE_SET + local_id_normalized
                combined_data[global_id] = df

        full_results_set_dict[dataset_id] = combined_data
        setattr(data, results_set, full_results_set_dict)

    # Add Trenton equivalent flow
    data = add_trenton_equiv_flow(data)

    return data


def compute_metrics_for_rank(data, dataset_id, rank_realizations, rank, size):
    """
    Compute shortage and contribution metrics for this rank's realizations.

    Parameters
    ----------
    data : pywrdrb.Data
        Data object with ensemble data
    dataset_id : str
        Dataset identifier
    rank_realizations : list
        List of global realization IDs to process
    rank : int
        MPI rank
    size : int
        Total number of MPI ranks

    Returns
    -------
    results : dict
        Dictionary with shortage and contribution data for assigned realizations
    """
    results = {
        'shortage': {},
        'contribution': {},
        'inflow': {}
    }

    models = ['reconstruction', 'wrfaorc_withObsScaled', 'wrf1960s_calib_nlcd2016', dataset_id]

    for model in models:
        if model not in data.major_flow:
            continue

        # Filter to only realizations assigned to this rank
        model_realizations = [r for r in rank_realizations if r in data.major_flow[model]]

        if not model_realizations:
            continue

        if rank == 0:
            print(f"  Rank {rank}: Processing {model} ({len(model_realizations)} realizations)")

        # Initialize storage
        shortage_dict = {}
        contribution_dict = {}

        # Shortage calculations
        nodes = ['delMontague', 'delTrenton', 'nyc', 'nj']

        for i, r in enumerate(model_realizations):
            # Progress reporting (every 25% for this rank)
            if len(model_realizations) > 4 and (i + 1) % max(1, len(model_realizations) // 4) == 0:
                progress = 100 * (i + 1) / len(model_realizations)
                print(f"    Rank {rank}: {progress:.0f}% complete ({i+1}/{len(model_realizations)} realizations)")

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
            nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']
            contribution_columns = [f'mrf_montagueTrenton_{res}' for res in nyc_reservoirs]
            total_nyc_contribution = data.nyc_release_components[model][r].loc[:, contribution_columns].sum(axis=1)
            contribution_dict[r] = total_nyc_contribution.to_frame(name='mrf_montagueTrenton_nyc')

            # NYC aggregate inflow
            data.inflow[model][r].loc[:, 'nyc'] = data.inflow[model][r].loc[:, nyc_reservoirs].sum(axis=1)

        results['shortage'][model] = shortage_dict
        results['contribution'][model] = contribution_dict

    if rank == 0:
        print(f"  Rank {rank}: Metric calculations complete")

    return results


def distribute_realizations(rank, size, total_realizations):
    """
    Determine which realizations this rank should process.

    Parameters
    ----------
    rank : int
        MPI rank (0 to size-1)
    size : int
        Total number of MPI ranks
    total_realizations : int
        Total number of realizations to distribute

    Returns
    -------
    rank_realizations : list
        List of realization IDs for this rank
    """
    realizations_per_rank = total_realizations // size
    remainder = total_realizations % size

    start = rank * realizations_per_rank + min(rank, remainder)
    count = realizations_per_rank + (1 if rank < remainder else 0)

    return list(range(start, start + count))


def combine_and_export_results(all_results, data, dataset_id, ensemble_set_specs):
    """
    Combine results from all ranks and export to HDF5 (rank 0 only).

    Parameters
    ----------
    all_results : list
        List of results dicts from all ranks
    data : pywrdrb.Data
        Base data object
    dataset_id : str
        Dataset identifier
    ensemble_set_specs : list
        List of EnsembleSetSpec objects
    """
    print(f"\nCombining results from all ranks...")

    # Merge shortage and contribution dicts from all ranks
    combined_shortage = {}
    combined_contribution = {}

    models = ['reconstruction', 'wrfaorc_withObsScaled', 'wrf1960s_calib_nlcd2016', dataset_id]

    for model in models:
        combined_shortage[model] = {}
        combined_contribution[model] = {}

        for rank_results in all_results:
            if rank_results is None:
                continue

            if model in rank_results['shortage']:
                combined_shortage[model].update(rank_results['shortage'][model])

            if model in rank_results['contribution']:
                combined_contribution[model].update(rank_results['contribution'][model])

    # Create final data object
    keep_data = pywrdrb.Data()
    keep_data.shortage = combined_shortage
    keep_data.contribution = combined_contribution

    # Copy other results_sets from data object
    inflow_dict = {}
    major_flow_dict = {}
    res_storage_dict = {}
    ibt_diversions_dict = {}
    ibt_demands_dict = {}
    mrf_target_dict = {}
    res_level_dict = {}

    for model in models:
        if model in data.inflow:
            inflow_dict[model] = data.inflow[model]
            major_flow_dict[model] = data.major_flow[model]
            res_storage_dict[model] = data.res_storage[model]
            ibt_diversions_dict[model] = data.ibt_diversions[model]
            ibt_demands_dict[model] = data.ibt_demands[model]
            mrf_target_dict[model] = data.mrf_target[model]
            res_level_dict[model] = data.res_level[model]

    keep_data.inflow = inflow_dict
    keep_data.major_flow = major_flow_dict
    keep_data.res_storage = res_storage_dict
    keep_data.ibt_diversions = ibt_diversions_dict
    keep_data.ibt_demands = ibt_demands_dict
    keep_data.mrf_target = mrf_target_dict
    keep_data.res_level = res_level_dict

    # Add gage_flow (already combined in data loading)
    ensemble_set_names = [spec.directory.split('/')[-1] for spec in ensemble_set_specs]
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

    keep_data.gage_flow = {dataset_id: combined_gage_flow}

    # Export
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
    print(f"Exporting combined data to {fname}...")
    keep_data.export(fname)
    print(f"Successfully combined and exported data for {dataset_id}!")

    return keep_data


def combine_ensemble_sets_and_calculate_metrics_mpi(dataset_id):
    """
    MPI-parallel version of combine_ensemble_sets_and_calculate_metrics.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier

    Returns
    -------
    keep_data : pywrdrb.Data
        Combined data object (only on rank 0, None on other ranks)
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"\n{'='*80}")
        print(f"MPI PARALLEL POSTPROCESSING: {dataset_id}")
        print(f"{'='*80}")
        print(f"MPI ranks: {size}")
        print(f"Total realizations: {TOTAL_REALIZATIONS}")
        print(f"Realizations per rank: ~{TOTAL_REALIZATIONS // size}")

    # Get dataset configuration
    ensemble_set_specs = ENSEMBLE_SETS[dataset_id]

    # Distribute realizations across ranks
    rank_realizations = distribute_realizations(rank, size, TOTAL_REALIZATIONS)

    if rank == 0:
        print(f"\nRank {rank}: Assigned realizations {rank_realizations[0]}-{rank_realizations[-1]} ({len(rank_realizations)} total)")

    # Each rank loads data (only the sets it needs)
    if rank == 0:
        print(f"\nLoading data...")

    data, needed_sets = load_ensemble_sets_for_rank(dataset_id, ensemble_set_specs, rank_realizations)

    if rank == 0:
        print(f"  Rank {rank}: Loaded ensemble sets: {sorted(needed_sets)}")

    # Combine ensemble sets (renumber realizations)
    data = combine_ensemble_sets_data_only(dataset_id, ensemble_set_specs, data)

    # Each rank computes metrics for its realizations
    if rank == 0:
        print(f"\nComputing metrics in parallel...")

    rank_results = compute_metrics_for_rank(data, dataset_id, rank_realizations, rank, size)

    # Gather results to rank 0
    if rank == 0:
        print(f"\nGathering results from all ranks...")

    all_results = comm.gather(rank_results, root=0)

    # Rank 0 combines and exports
    keep_data = None
    if rank == 0:
        keep_data = combine_and_export_results(all_results, data, dataset_id, ensemble_set_specs)

    return keep_data


def process_dataset_mpi(dataset_id, recombine_sets=True):
    """
    MPI-parallel version of process_dataset.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    recombine_sets : bool
        If True, recombine all ensemble sets (time-intensive)
        If False, load existing combined data (much faster)

    Returns
    -------
    success : bool
        True if processing completed successfully
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

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
        keep_data = combine_ensemble_sets_and_calculate_metrics_mpi(dataset_id)

    else:
        # Load existing combined data (rank 0 only)
        keep_data = None
        if rank == 0:
            print(f"\nrecombine_sets=False. Loading existing combined data from:")
            print(f"  {fname}")

            required_results_sets = ['shortage', 'mrf_target', 'res_storage',
                                     'ibt_diversions', 'ibt_demands', 'contribution']

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
        except Exception as e:
            print(f"ERROR calculating metrics for {dataset_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Also calculate historic (reconstruction) metrics
        print(f"\nCalculating historic (reconstruction) performance metrics...")
        try:
            reconstruction_realizations = list(keep_data.shortage['reconstruction'].keys())
            print(f"  Found {len(reconstruction_realizations)} realizations in reconstruction")
            calculate_and_save_performance_metrics(
                keep_data, 'reconstruction', reconstruction_realizations, PERFORMANCE_METRICS_DIR
            )
        except KeyError:
            print(f"WARNING: Could not find 'reconstruction' data in loaded file.")
        except Exception as e:
            print(f"WARNING: Error calculating historic metrics: {e}")

    return True


def main_mpi(dataset_id, recombine_sets=True):
    """
    Main MPI entry point for postprocessing.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    recombine_sets : bool
        If True, recombine ensemble sets from scratch
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

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
        print("=" * 80)

    # Process the dataset
    success = process_dataset_mpi(dataset_id, recombine_sets=recombine_sets)

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
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: mpirun -np <N_RANKS> python 04_postprocess_data_mpi.py <dataset_id> [--skip-recombine]")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        print()
        print("Options:")
        print("  --skip-recombine    Load existing combined data instead of recombining (fast)")
        print()
        print("Examples:")
        print("  mpirun -np 32 python 04_postprocess_data_mpi.py stationary_ensemble")
        print("  mpirun -np 16 python 04_postprocess_data_mpi.py stationary_ensemble --skip-recombine")
        sys.exit(1)

    dataset_id = sys.argv[1]
    verify_dataset_id(dataset_id)

    # Check for --skip-recombine flag
    recombine_sets = True
    if len(sys.argv) == 3:
        if sys.argv[2] == '--skip-recombine':
            recombine_sets = False
        else:
            print(f"ERROR: Unknown option '{sys.argv[2]}'")
            print("Use --skip-recombine to skip recombining ensemble sets")
            sys.exit(1)

    main_mpi(dataset_id, recombine_sets=recombine_sets)
