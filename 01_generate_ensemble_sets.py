"""
Generate all ensemble sets in parallel using MPI rank distribution.
Automatically distributes ensemble sets across available MPI ranks.
"""

import sys
from mpi4py import MPI

from methods.generate import generate_ensemble_set
from methods.config import (
    DATASET_CONFIGS,
    N_ENSEMBLE_SETS,
    N_REALIZATIONS_PER_ENSEMBLE_SET,
    N_YEARS,
    verify_dataset_id,
    ensure_ensemble_set_dirs,
    get_existing_ensemble_sets
)


def parallel_generate_all_sets(dataset_id):
    """
    Distribute ensemble set generation across available MPI ranks
    (MPI-only function - always uses MPI)

    Parameters:
    -----------
    dataset_id : str
        Dataset identifier to generate
    """

    # MPI setup (always required for this function)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Verify dataset
    verify_dataset_id(dataset_id)
    dataset_config = DATASET_CONFIGS[dataset_id]

    # Make sure more ranks than sets (preferred for large nodes)
    if size <= N_ENSEMBLE_SETS:
        if rank == 0:
            print(f"WARNING: Only {size} ranks for {N_ENSEMBLE_SETS} sets.")
            print(f"Ideally use > {N_ENSEMBLE_SETS} ranks for better performance.")

    if rank == 0:
        print("=" * 60)
        print(f"PARALLEL ENSEMBLE SET GENERATION: {dataset_id}")
        print("=" * 60)
        print(f"Dataset type: {dataset_config['type']}")
        print(f"Description: {dataset_config['description']}")
        print(f"Total ensemble sets: {N_ENSEMBLE_SETS}")
        print(f"Realizations per set: {N_REALIZATIONS_PER_ENSEMBLE_SET}")
        print(f"Available MPI ranks: {size}")
        print(f"Years per realization: {N_YEARS}")

        # Calculate optimal rank distribution
        if size >= N_ENSEMBLE_SETS:
            ranks_per_set = size // N_ENSEMBLE_SETS
            print(f"Ranks per ensemble set: {ranks_per_set}")
        else:
            print(f"More sets than ranks - will process sets sequentially")
        print("=" * 60)

    # Ensure all directories exist (all ranks call with exist_ok=True, no Barrier needed)
    ensure_ensemble_set_dirs(dataset_id)

    if size >= N_ENSEMBLE_SETS:
        # More ranks than sets - distribute ranks across sets
        ranks_per_set = size // N_ENSEMBLE_SETS
        set_id = rank // ranks_per_set

        # All ranks must participate in Split (it is a collective operation).
        # Fold leftover ranks into the last set to avoid MPI.UNDEFINED,
        # which can cause MPI_ERR_OTHER on some OpenMPI installations.
        if set_id >= N_ENSEMBLE_SETS:
            set_id = N_ENSEMBLE_SETS - 1
        color = set_id

        local_comm = comm.Split(color, rank)

        # Store original communicator
        original_comm = MPI.COMM_WORLD

        # Temporarily replace global communicator for the generation function
        MPI.COMM_WORLD = local_comm

        try:
            true_if_success = generate_ensemble_set(set_id, dataset_id, use_mpi=True)
            assert true_if_success, f"Set {set_id + 1} generation failed on rank {rank}"

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

        for set_id in my_sets:
            # Create single-rank communicator
            local_comm = MPI.COMM_SELF
            original_comm = MPI.COMM_WORLD
            MPI.COMM_WORLD = local_comm

            try:
                true_if_success = generate_ensemble_set(set_id, dataset_id, use_mpi=True)
                assert true_if_success, f"Set {set_id + 1} generation failed on rank {rank}"
            finally:
                MPI.COMM_WORLD = original_comm

    # Synchronize all ranks
    comm.Barrier()

    if rank == 0:
        print("\n" + "=" * 60)
        print(f"GENERATION COMPLETED: {dataset_id}")
        print("=" * 60)

        # Verify all sets were created
        existing_sets = get_existing_ensemble_sets(dataset_id)
        if len(existing_sets) == N_ENSEMBLE_SETS:
            print(f"SUCCESS: All {N_ENSEMBLE_SETS} ensemble sets verified")
        else:
            print(f"WARNING: Only {len(existing_sets)}/{N_ENSEMBLE_SETS} sets found")
            missing = set(range(N_ENSEMBLE_SETS)) - set([s.set_id for s in existing_sets])
            print(f"  Missing sets: {sorted(missing)}")


def main(dataset_id):
    """Main function (MPI-only)"""

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Generate all ensemble sets in parallel
    parallel_generate_all_sets(dataset_id)

    if rank == 0:
        print(f"\nEnsemble generation workflow completed for {dataset_id}!")


if __name__ == "__main__":

    # Get the dataset_id from command line arguments
    if len(sys.argv) != 2:
        print("Usage: mpirun -n <N> python 01_generate_ensemble_sets.py <dataset_id>")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)

    dataset_id = sys.argv[1]
    verify_dataset_id(dataset_id)

    try:
        main(dataset_id)
    except Exception as e:
        import traceback
        rank = MPI.COMM_WORLD.Get_rank()
        print(f"RANK {rank} FATAL ERROR: {e}", flush=True)
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
