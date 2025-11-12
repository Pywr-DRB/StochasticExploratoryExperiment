#!/usr/bin/env python3
"""
Run Pywr-DRB simulations for all ensemble sets in parallel using MPI rank distribution.
Automatically distributes ensemble sets across available MPI ranks.
"""

import os
import sys
from mpi4py import MPI

import pywrdrb

from methods.simulate import run_ensemble_set_simulations
from methods.config import (
    DATASET_CONFIGS,
    N_ENSEMBLE_SETS,
    N_REALIZATIONS_PER_ENSEMBLE_SET,
    verify_dataset_id,
    get_ensemble_set_spec,
    print_experiment_summary
)


def parallel_run_all_sets(dataset_id):
    """
    Distribute Pywr-DRB simulations across available MPI ranks
    (MPI-only function - always uses MPI)

    Parameters:
    -----------
    dataset_id : str
        Dataset identifier to simulate
    """

    # MPI setup (always required for this function)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Verify dataset
    verify_dataset_id(dataset_id)
    dataset_config = DATASET_CONFIGS[dataset_id]

    if rank == 0:
        print("=" * 60)
        print(f"PARALLEL PYWRDRB SIMULATIONS: {dataset_id}")
        print("=" * 60)
        print(f"Dataset type: {dataset_config['type']}")
        print(f"Description: {dataset_config['description']}")
        print(f"Total ensemble sets: {N_ENSEMBLE_SETS}")
        print(f"Realizations per set: {N_REALIZATIONS_PER_ENSEMBLE_SET}")
        print(f"Available MPI ranks: {size}")

        # Check which sets are ready for processing
        ready_sets = []
        for set_id in range(N_ENSEMBLE_SETS):
            set_spec = get_ensemble_set_spec(set_id, dataset_id)
            if os.path.exists(set_spec.files['catchment_inflow']):
                ready_sets.append(set_id)

        print(f"Ready ensemble sets: {len(ready_sets)}")

        if len(ready_sets) < N_ENSEMBLE_SETS:
            missing_sets = set(range(N_ENSEMBLE_SETS)) - set(ready_sets)
            print(f"Warning: Missing ensemble sets: {sorted(missing_sets)}")
            print("Run ensemble generation and prep first!")

        # Calculate optimal rank distribution
        if size >= N_ENSEMBLE_SETS:
            ranks_per_set = size // N_ENSEMBLE_SETS
            print(f"Ranks per ensemble set: {ranks_per_set}")
        else:
            print(f"Sets per rank: {N_ENSEMBLE_SETS // size + 1}")
        print("=" * 60)

    comm.Barrier()  # Wait for status messages

    # Track success/failure
    success_count = 0
    total_processed = 0

    if size >= N_ENSEMBLE_SETS:
        # More ranks than sets - distribute ranks across sets
        ranks_per_set = size // N_ENSEMBLE_SETS
        set_id = rank // ranks_per_set

        # Only process if we're within valid set range
        if set_id < N_ENSEMBLE_SETS:
            # Create sub-communicator for this ensemble set
            color = set_id
            local_comm = comm.Split(color, rank)

            # Store original communicator
            original_comm = MPI.COMM_WORLD

            # Temporarily replace global communicator for the simulation function
            MPI.COMM_WORLD = local_comm

            try:
                success = run_ensemble_set_simulations(set_id, dataset_id, use_mpi=True)
                total_processed = 1
                success_count = 1 if success else 0
            finally:
                # Restore original communicator
                MPI.COMM_WORLD = original_comm
                local_comm.Free()
    else:
        # More sets than ranks - each rank processes multiple sets sequentially
        sets_per_rank = N_ENSEMBLE_SETS // size
        extra_sets = N_ENSEMBLE_SETS % size

        if rank < extra_sets:
            my_sets = list(range(rank * (sets_per_rank + 1), (rank + 1) * (sets_per_rank + 1)))
        else:
            start = extra_sets * (sets_per_rank + 1) + (rank - extra_sets) * sets_per_rank
            my_sets = list(range(start, start + sets_per_rank))

        # Process each set sequentially on this rank
        for set_id in my_sets:
            # Create single-rank communicator
            local_comm = MPI.COMM_SELF
            original_comm = MPI.COMM_WORLD
            MPI.COMM_WORLD = local_comm

            try:
                success = run_ensemble_set_simulations(set_id, dataset_id, use_mpi=True)
                total_processed += 1
                success_count += 1 if success else 0
            finally:
                MPI.COMM_WORLD = original_comm

    # Collect results from all ranks
    comm.Barrier()

    # Sum up success counts across all ranks
    total_success = comm.reduce(success_count, op=MPI.SUM, root=0)
    total_attempts = comm.reduce(total_processed, op=MPI.SUM, root=0)

    if rank == 0:
        print("\n" + "=" * 60)
        print(f"PYWRDRB SIMULATIONS COMPLETED: {dataset_id}")
        print("=" * 60)
        print(f"Successfully processed: {total_success}/{total_attempts} sets")

        if total_success == N_ENSEMBLE_SETS:
            print("SUCCESS: All ensemble sets simulated successfully!")
        else:
            failed_count = N_ENSEMBLE_SETS - total_success
            print(f"WARNING: {failed_count} sets failed or were skipped")

            # Try to identify which sets might have failed
            failed_sets = []
            for set_id in range(N_ENSEMBLE_SETS):
                set_spec = get_ensemble_set_spec(set_id, dataset_id)
                if not os.path.exists(set_spec.output_file):
                    failed_sets.append(set_id + 1)

            if failed_sets:
                print(f"  Failed sets: {failed_sets}")

        print("=" * 60)
        print(f"Done with Pywr-DRB simulations for {dataset_id}!")


def verify_simulation_outputs(dataset_id):
    """
    Verify that all ensemble sets have been properly simulated

    Parameters:
    -----------
    dataset_id : str
        Dataset identifier to verify
    """

    verify_dataset_id(dataset_id)
    print(f"\nVerifying Pywr-DRB simulation outputs for {dataset_id}...")

    all_completed = True
    successful_sets = []
    failed_sets = []

    for set_id in range(N_ENSEMBLE_SETS):
        set_spec = get_ensemble_set_spec(set_id, dataset_id)

        if not os.path.exists(set_spec.output_file):
            print(f"FAIL:  Set {set_id + 1}: Output file not found")
            all_completed = False
            failed_sets.append(set_id + 1)
            continue

        # Check file size (basic validation)
        file_size = os.path.getsize(set_spec.output_file)
        if file_size < 1024:  # Less than 1KB is suspicious
            print(f"FAIL:  Set {set_id + 1}: Output file too small ({file_size} bytes)")
            all_completed = False
            failed_sets.append(set_id + 1)
            continue

        # Try to load with Pywr-DRB to verify format (optimization: only check first and last)
        if set_id == 0 or set_id == N_ENSEMBLE_SETS - 1:
            try:
                test_data = pywrdrb.Data(results_sets=["major_flow"])
                test_data.load_output(output_filenames=[set_spec.output_file])
                n_realizations = len(list(test_data.major_flow.values())[0])

                if n_realizations != N_REALIZATIONS_PER_ENSEMBLE_SET:
                    print(f"WARNING: Set {set_id + 1}: Expected {N_REALIZATIONS_PER_ENSEMBLE_SET} realizations, found {n_realizations}")
                    failed_sets.append(set_id + 1)
                else:
                    print(f"SUCCESS: Set {set_id + 1}: Valid output ({n_realizations} realizations, {file_size//1024//1024} MB)")
                    successful_sets.append(set_id + 1)

            except Exception as e:
                print(f"FAIL:  Set {set_id + 1}: Error loading output file - {str(e)}")
                all_completed = False
                failed_sets.append(set_id + 1)
        else:
            # For middle sets, just check file size
            print(f"SUCCESS: Set {set_id + 1}: Output exists ({file_size//1024//1024} MB)")
            successful_sets.append(set_id + 1)

    if all_completed:
        print(f"SUCCESS: All {N_ENSEMBLE_SETS} ensemble sets have valid simulation outputs!")
    else:
        print(f"WARNING: {len(failed_sets)} sets have invalid outputs: {failed_sets}")

    return all_completed


def main(dataset_id):
    """Main function (MPI-only)"""

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print(f"Starting Pywr-DRB simulations for {dataset_id}...")
        print_experiment_summary(dataset_id)

    # Run all ensemble set simulations in parallel
    parallel_run_all_sets(dataset_id)

    # Verify outputs (only on rank 0)
    if rank == 0:
        verify_simulation_outputs(dataset_id)
        print(f"\nPywr-DRB simulation workflow completed for {dataset_id}!")


if __name__ == "__main__":

    # Get the dataset_id from command line arguments
    if len(sys.argv) != 2:
        print("Usage: mpirun -n <N> python 03_run_pywrdrb_simulations.py <dataset_id>")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)

    dataset_id = sys.argv[1]
    verify_dataset_id(dataset_id)

    main(dataset_id)
