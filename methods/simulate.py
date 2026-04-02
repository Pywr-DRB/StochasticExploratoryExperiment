"""
Core function for running Pywr-DRB simulations.
This module contains the simulation logic that can be used in both
serial and parallel modes.
"""

import os
import re
import glob
import math
import numpy as np

import pywrdrb
from pywrdrb.utils.hdf5 import get_hdf5_realization_numbers, combine_batched_hdf5_outputs

from methods.utils import get_parameter_subset_to_export
from methods.config import (
    N_REALIZATIONS_PER_PYWRDRB_BATCH,
    START_DATE,
    END_DATE,
    SALINITY_LSTM_PREDICTIONS,
    SALINITY_LSTM_OPTIONS,
    SAVE_RESULTS_SETS,
    FLOW_PREDICTION_MODE,
    MODEL_DIR,
)
from methods.ensemble_utils import get_ensemble_set_spec

from methods.mpi_utils import (
    MPI_AVAILABLE,
    point_to_point_barrier,
)

# Conditional MPI import
if MPI_AVAILABLE:
    from mpi4py import MPI


def run_ensemble_set_simulations(set_id, dataset_id, use_mpi=True,
                                  comm=None, local_rank=None, local_size=None,
                                  set_peer_ranks=None):
    """
    Run Pywr-DRB simulations for a single ensemble set

    Parameters:
    -----------
    set_id : int
        Ensemble set identifier (0-indexed)
    dataset_id : str
        Dataset identifier (e.g., 'stationary_ensemble', 'climate_adjusted_ssp245_min')
    use_mpi : bool
        If True, use MPI for parallel execution. If False, run serially.
    comm : MPI.Comm or None
        MPI communicator (COMM_WORLD). Required when local_rank/local_size are provided.
    local_rank : int or None
        This rank's position within its ensemble set group.
    local_size : int or None
        Number of ranks collaborating on this ensemble set.
    set_peer_ranks : list of int or None
        Global MPI ranks assigned to the same ensemble set.

    Returns:
    --------
    bool
        True if successful, False otherwise
    """

    # Determine MPI context
    if use_mpi and MPI_AVAILABLE and local_rank is not None:
        rank = local_rank
        size = local_size
    elif use_mpi and MPI_AVAILABLE:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        set_peer_ranks = None
    else:
        comm = None
        rank = 0
        size = 1
        set_peer_ranks = None

    # Get ensemble set specification
    set_spec = get_ensemble_set_spec(set_id, dataset_id)
    catchment_inflow_file = set_spec.files['catchment_inflow']
    ensemble_dir = set_spec.directory
    output_file = set_spec.output_file

    if rank == 0:
        print(f"Set {set_id + 1}: Running Pywr-DRB simulations for {dataset_id}")

    # Check if input file exists
    if not os.path.exists(catchment_inflow_file):
        print(f"Error: Input file not found: {catchment_inflow_file}")
        return False

    # Setup pathnavigator for this specific ensemble set
    pn_config = pywrdrb.get_pn_config()
    pn_config[f"flows/{dataset_id}_set{set_id + 1}"] = os.path.abspath(ensemble_dir)
    pywrdrb.load_pn_config(pn_config)

    try:
        # Clear old batched output files if they exist
        if rank == 0:
            batch_pattern = f"{os.path.dirname(output_file)}/{dataset_id}_set{set_id + 1}_rank*_batch*.hdf5"
            model_pattern = f"{os.path.dirname(output_file)}/../models/{dataset_id}_set{set_id + 1}_rank*_batch*.json"

            for pattern in [batch_pattern, model_pattern]:
                old_files = glob.glob(pattern)
                for file in old_files:
                    if os.path.exists(file):
                        os.remove(file)

        if use_mpi and comm:
            if set_peer_ranks is not None:
                point_to_point_barrier(comm, rank, size, set_peer_ranks)
            else:
                comm.Barrier()  # Wait for cleanup

        # Get realization IDs for this ensemble set
        # All ranks read independently to avoid broadcast
        realization_ids = get_hdf5_realization_numbers(catchment_inflow_file)
        realization_ids = [str(rid) for rid in realization_ids]

        # VALIDATION: Check if realization_ids match expected global IDs
        expected_realization_ids = set_spec.realizations
        if rank == 0:
            if type(realization_ids[0]) is str:
                expected_realization_ids = [str(r) for r in expected_realization_ids]

            if set(realization_ids) != set(expected_realization_ids):
                print(f"WARNING: Set {set_id + 1} - HDF5 realization IDs don't match expected global IDs")
                print(f"  Expected: {expected_realization_ids}")
                print(f"  Found in HDF5: {realization_ids}")

        # Optimized distribution: balance load across ranks
        # Use numpy array_split for better load balancing
        rank_realization_ids = np.array_split(realization_ids, size)[rank]
        rank_realization_ids = list(rank_realization_ids)
        n_rank_realizations = len(rank_realization_ids)

        # If this rank has no realizations, skip simulation but still
        # participate in synchronization below to avoid deadlock.
        if n_rank_realizations == 0:
            batch_filenames = []
            # Jump directly to the barrier/combine section
            if use_mpi and comm:
                if set_peer_ranks is not None:
                    point_to_point_barrier(comm, rank, size, set_peer_ranks)
                else:
                    comm.Barrier()
            return True

        # Split rank realizations into batches for memory management
        n_batches = math.ceil(n_rank_realizations / N_REALIZATIONS_PER_PYWRDRB_BATCH)
        batched_indices = {}

        for i in range(n_batches):
            batch_start = i * N_REALIZATIONS_PER_PYWRDRB_BATCH
            batch_end = min((i + 1) * N_REALIZATIONS_PER_PYWRDRB_BATCH, n_rank_realizations)
            batched_indices[i] = rank_realization_ids[batch_start:batch_end]
            batched_indices[i] = [str(rid) for rid in batched_indices[i]]  # force to be str

        # Run individual batches
        batch_filenames = []
        for batch, indices in batched_indices.items():

            # Model options for this batch
            model_options = {
                "inflow_ensemble_indices": indices,
                'nyc_nj_demand_source': 'custom',
                'flow_prediction_mode': FLOW_PREDICTION_MODE
            }

            # Add salinity LSTM options if enabled
            if SALINITY_LSTM_PREDICTIONS:
                model_options.update(SALINITY_LSTM_OPTIONS)

            # Build model
            mb = pywrdrb.ModelBuilder(
                inflow_type=f'{dataset_id}_set{set_id + 1}',
                start_date=START_DATE,
                end_date=END_DATE,
                options=model_options,
            )

            # Save model
            model_fname = f"{MODEL_DIR}/{dataset_id}_set{set_id + 1}_rank{rank}_batch{batch}.json"
            os.makedirs(MODEL_DIR, exist_ok=True)
            
            mb.make_model()
            mb.write_model(model_fname)

            # Load model
            model = pywrdrb.Model.load(model_fname)

            # Get list of parameters for specific results sets
            all_parameter_names = [p.name for p in model.parameters if p.name]
            subset_parameter_names = get_parameter_subset_to_export(
                all_parameter_names,
                results_set_subset=SAVE_RESULTS_SETS
            )
            export_parameters = [p for p in model.parameters if p.name in subset_parameter_names]

            # Setup output recorder
            batch_output_filename = f"{os.path.dirname(output_file)}/{dataset_id}_set{set_id + 1}_rank{rank}_batch{batch}.hdf5"
            recorder = pywrdrb.OutputRecorder(
                model=model,
                output_filename=batch_output_filename,
                parameters=export_parameters
            )

            # Run simulation
            model.run()

            batch_filenames.append(batch_output_filename)

            # Clean up model object to free memory (optimization)
            del model

        # Wait for all ranks to complete their batches
        if use_mpi and comm:
            if set_peer_ranks is not None:
                point_to_point_barrier(comm, rank, size, set_peer_ranks)
            else:
                comm.Barrier()

        # Combine all batched outputs for this ensemble set (rank 0 only)
        if rank == 0:
            print(f'Set {set_id + 1}: Combining batched outputs...')

            # Find all batch files for this set
            batch_pattern = f"{os.path.dirname(output_file)}/{dataset_id}_set{set_id + 1}_rank*_batch*.hdf5"
            all_batch_files = glob.glob(batch_pattern)

            if not all_batch_files:
                print(f"Set {set_id + 1}: No batch files found!")
                return False

            # IMPORTANT! Sort batch files by (rank, batch) to preserve realization order
            def sort_batch_files(filename):
                match = re.search(r'rank(\d+)_batch(\d+)', filename)
                if match:
                    rank_num = int(match.group(1))
                    batch_num = int(match.group(2))
                    return (rank_num, batch_num)
                return (0, 0)

            all_batch_files.sort(key=sort_batch_files)

            # If output file already exists, remove it
            if os.path.exists(output_file):
                print(f"Set {set_id + 1}: Removing existing output file: {output_file}")
                os.remove(output_file)

            print(f"Set {set_id + 1}: Found {len(all_batch_files)} batch files to combine")

            # Combine batch files
            combine_batched_hdf5_outputs(all_batch_files, output_file)

            # Cleanup batch files if requested
            if True:  # always clean up batch files after combining
                for file in all_batch_files:
                    if os.path.exists(file):
                        os.remove(file)

                # Also cleanup model files
                model_pattern = f"{os.path.dirname(output_file)}/../models/{dataset_id}_set{set_id + 1}_rank*_batch*.json"
                model_files = glob.glob(model_pattern)
                for file in model_files:
                    if os.path.exists(file):
                        os.remove(file)

                print(f"Set {set_id + 1}: Cleaned up {len(all_batch_files)} batch files")

            print(f"Set {set_id + 1}: Simulations completed successfully!")
            print(f"  Output file: {output_file}")

        return True

    except Exception as e:
        print(f"Error processing set {set_id + 1}: {str(e)}")
        return False
