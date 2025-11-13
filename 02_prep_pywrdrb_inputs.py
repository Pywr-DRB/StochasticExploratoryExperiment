"""
Prepare Pywr-DRB inputs for all ensemble sets in parallel using MPI rank distribution.
Automatically distributes ensemble sets across available MPI ranks.
"""

import os
import sys
from mpi4py import MPI

from methods.prepare import prep_ensemble_set
from methods.config import (
    DATASET_CONFIGS,
    N_ENSEMBLE_SETS,
    verify_dataset_id,
    get_ensemble_set_spec,
    get_existing_ensemble_sets,
    print_experiment_summary
)


def parallel_prep_all_sets(dataset_id):
    """
    Distribute Pywr-DRB input preparation across available MPI ranks
    (MPI-only function - always uses MPI)

    Parameters:
    -----------
    dataset_id : str
        Dataset identifier to prepare
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
        print(f"PARALLEL PYWRDRB INPUT PREPARATION: {dataset_id}")
        print("=" * 60)
        print(f"Dataset type: {dataset_config['type']}")
        print(f"Description: {dataset_config['description']}")
        print(f"Total ensemble sets: {N_ENSEMBLE_SETS}")
        print(f"Available MPI ranks: {size}")

        # Check which sets exist
        existing_sets = get_existing_ensemble_sets(dataset_id)
        print(f"Found {len(existing_sets)} existing ensemble sets")

        if len(existing_sets) < N_ENSEMBLE_SETS:
            missing_sets = set(range(N_ENSEMBLE_SETS)) - set([s.set_id for s in existing_sets])
            print(f"Warning: Missing ensemble sets: {sorted(missing_sets)}")
            print("Run ensemble generation first!")

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
        # Strategy 1: More ranks than sets (preferred for large nodes)
        ranks_per_set = size // N_ENSEMBLE_SETS
        set_id = rank // ranks_per_set

        # Only process if we're within valid set range
        if set_id < N_ENSEMBLE_SETS:
            # Create sub-communicator for this ensemble set
            color = set_id
            local_comm = comm.Split(color, rank)

            # Store original communicator
            original_comm = MPI.COMM_WORLD

            # Temporarily replace global communicator for the prep function
            MPI.COMM_WORLD = local_comm

            try:
                success = prep_ensemble_set(set_id, dataset_id, use_mpi=True)
                total_processed = 1
                success_count = 1 if success else 0
            finally:
                # Restore original communicator
                MPI.COMM_WORLD = original_comm
                local_comm.Free()

    else:
        # Strategy 2: More sets than ranks (process multiple sets per rank)
        sets_per_rank = N_ENSEMBLE_SETS // size
        extra_sets = N_ENSEMBLE_SETS % size

        if rank < extra_sets:
            my_sets = list(range(rank * (sets_per_rank + 1), (rank + 1) * (sets_per_rank + 1)))
        else:
            start = extra_sets * (sets_per_rank + 1) + (rank - extra_sets) * sets_per_rank
            my_sets = list(range(start, start + sets_per_rank))

        # Process each set sequentially on this rank
        for set_id in my_sets:
            # Create single-rank communicator for sequential processing
            local_comm = MPI.COMM_SELF
            original_comm = MPI.COMM_WORLD
            MPI.COMM_WORLD = local_comm

            try:
                success = prep_ensemble_set(set_id, dataset_id, use_mpi=True)
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
        print(f"PYWRDRB INPUT PREPARATION COMPLETED: {dataset_id}")
        print("=" * 60)
        print(f"Successfully processed: {total_success}/{total_attempts} sets")

        if total_success == total_attempts and total_attempts == N_ENSEMBLE_SETS:
            print("SUCCESS: All ensemble sets prepared successfully!")
        else:
            failed_count = total_attempts - total_success if total_attempts else N_ENSEMBLE_SETS
            print(f"WARNING: {failed_count} sets failed or were skipped")

            # Try to identify which sets might have failed
            # by checking for expected output files
            failed_sets = []
            required_files = [
                'predicted_inflow',
                'diversion_nyc',
                'diversion_nj',
                'predicted_diversions'
            ]

            for set_id in range(N_ENSEMBLE_SETS):
                set_spec = get_ensemble_set_spec(set_id, dataset_id)
                # Check if any required file is missing
                for file_key in required_files:
                    if not os.path.exists(set_spec.files[file_key]):
                        if set_id + 1 not in failed_sets:
                            failed_sets.append(set_id + 1)
                        break

            if failed_sets:
                print(f"  Potentially failed sets: {failed_sets}")

        print("=" * 60)


def verify_prep_outputs(dataset_id):
    """
    Verify that all ensemble sets have been properly prepared for a dataset

    Parameters:
    -----------
    dataset_id : str
        Dataset identifier to verify
    """

    verify_dataset_id(dataset_id)
    print(f"\nVerifying Pywr-DRB input preparation for {dataset_id}...")

    all_prepared = True
    successful_sets = []
    failed_sets = []
    missing_files = {}

    # Required files to check
    required_files = [
        'predicted_inflow',
        'diversion_nyc',
        'diversion_nj',
        'predicted_diversions'
    ]

    for set_id in range(N_ENSEMBLE_SETS):
        set_spec = get_ensemble_set_spec(set_id, dataset_id)
        set_complete = True
        missing_in_set = []

        for file_key in required_files:
            fname = set_spec.files[file_key]
            if not os.path.exists(fname):
                set_complete = False
                missing_in_set.append(file_key)
                all_prepared = False

        if set_complete:
            successful_sets.append(set_id + 1)
        else:
            failed_sets.append(set_id + 1)
            missing_files[set_id + 1] = missing_in_set

    if all_prepared:
        print(f"SUCCESS: All {N_ENSEMBLE_SETS} ensemble sets properly prepared!")
        print(f"  All required files present:")
        for file_key in required_files:
            print(f"    - {file_key}")
    else:
        print(f"WARNING: {len(failed_sets)} sets not properly prepared: {failed_sets}")
        print(f"Successfully prepared: {len(successful_sets)} sets")
        print(f"\nMissing files by set:")
        for set_id, missing in missing_files.items():
            print(f"  Set {set_id}: {missing}")

    return all_prepared


def main(dataset_id):
    """Main function (MPI-only)"""

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print(f"Starting Pywr-DRB input preparation for {dataset_id}...")
        print_experiment_summary(dataset_id)

    # Prepare all ensemble sets in parallel
    parallel_prep_all_sets(dataset_id)

    # Verify outputs (only on rank 0)
    if rank == 0:
        verify_prep_outputs(dataset_id)
        print(f"\nPywr-DRB input preparation workflow completed for {dataset_id}!")


if __name__ == "__main__":

    # Get the dataset_id from command line arguments
    if len(sys.argv) != 2:
        print("Usage: mpirun -n <N> python 02_prep_pywrdrb_inputs.py <dataset_id>")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)

    dataset_id = sys.argv[1]
    verify_dataset_id(dataset_id)

    main(dataset_id)
