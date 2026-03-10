"""
Prepare Pywr-DRB inputs for all ensemble sets in parallel using MPI rank distribution.

The pywrdrb preprocessors hardcode MPI.COMM_WORLD internally, so we cannot
run multiple preprocessors concurrently under the same MPI job. Instead,
only one rank per ensemble set (local_rank == 0) calls the preprocessor
in serial mode. With 20 sets and 240 ranks, 20 ranks work simultaneously
(one per set) while the rest idle for this step.

Usage:
  MPI:    mpirun -np N python 02_prep_pywrdrb_inputs.py <dataset_id>
  Serial: python 02_prep_pywrdrb_inputs.py <dataset_id>
"""

import os
import sys

from methods.mpi_utils import get_comm, get_set_assignments, MPI_AVAILABLE
from methods.prepare import prep_ensemble_set
from methods.config import (
    DATASET_CONFIGS,
    N_ENSEMBLE_SETS,
    verify_dataset_id,
    get_ensemble_set_spec,
    get_existing_ensemble_sets,
)
from methods.print_summary import print_prep_status

if MPI_AVAILABLE:
    from mpi4py import MPI


def parallel_prep_all_sets(dataset_id):
    """Distribute Pywr-DRB input preparation across available MPI ranks."""

    comm, rank, size = get_comm()

    verify_dataset_id(dataset_id)
    dataset_config = DATASET_CONFIGS[dataset_id]

    if rank == 0:
        print("=" * 60)
        print(f"PARALLEL PYWRDRB INPUT PREPARATION: {dataset_id}")
        print("=" * 60)
        print(f"Dataset type: {dataset_config['type']}")
        print(f"Description: {dataset_config['description']}")
        print(f"Total ensemble sets: {N_ENSEMBLE_SETS}")
        print(f"Available ranks: {size}")

        existing_sets = get_existing_ensemble_sets(dataset_id)
        print(f"Found {len(existing_sets)} existing ensemble sets")
        if len(existing_sets) < N_ENSEMBLE_SETS:
            missing_sets = set(range(N_ENSEMBLE_SETS)) - set([s.set_id for s in existing_sets])
            print(f"Warning: Missing ensemble sets: {sorted(missing_sets)}")
            print("Run ensemble generation first!")
        print("=" * 60)

    # Track success/failure
    success_count = 0
    total_processed = 0

    assignments = get_set_assignments(rank, size, N_ENSEMBLE_SETS)

    if size >= N_ENSEMBLE_SETS:
        # Multiple ranks per set — only local_rank 0 does the work
        set_id, local_rank, local_size = assignments[0]

        if local_rank == 0:
            success = prep_ensemble_set(set_id, dataset_id, use_mpi=False)
            total_processed = 1
            success_count = 1 if success else 0
    else:
        # More sets than ranks — each rank processes its assigned sets serially
        for set_id, local_rank, local_size in assignments:
            success = prep_ensemble_set(set_id, dataset_id, use_mpi=False)
            total_processed += 1
            success_count += 1 if success else 0

    # Collect results from all ranks
    if comm:
        comm.Barrier()
        total_success = comm.reduce(success_count, op=MPI.SUM, root=0)
        total_attempts = comm.reduce(total_processed, op=MPI.SUM, root=0)
    else:
        total_success = success_count
        total_attempts = total_processed

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

            failed_sets = []
            required_files = [
                'predicted_inflow',
                'diversion_nyc',
                'diversion_nj',
                'predicted_diversions'
            ]
            for set_id in range(N_ENSEMBLE_SETS):
                set_spec = get_ensemble_set_spec(set_id, dataset_id)
                for file_key in required_files:
                    if not os.path.exists(set_spec.files[file_key]):
                        if set_id + 1 not in failed_sets:
                            failed_sets.append(set_id + 1)
                        break
            if failed_sets:
                print(f"  Potentially failed sets: {failed_sets}")

        print("=" * 60)


def main(dataset_id):
    comm, rank, _ = get_comm()

    if rank == 0:
        print(f"Starting Pywr-DRB input preparation for {dataset_id}...")

    parallel_prep_all_sets(dataset_id)

    if rank == 0:
        print_prep_status(dataset_id)
        print(f"\nPywr-DRB input preparation workflow completed for {dataset_id}!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: mpirun -np N python 02_prep_pywrdrb_inputs.py <dataset_id>")
        print("       python 02_prep_pywrdrb_inputs.py <dataset_id>  (serial mode)")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)

    dataset_id = sys.argv[1]
    verify_dataset_id(dataset_id)

    try:
        main(dataset_id)
    except Exception as e:
        import traceback
        traceback.print_exc()
        comm, rank, _ = get_comm()
        print(f"RANK {rank} FATAL ERROR: {e}", flush=True)
        if comm:
            comm.Abort(1)
        else:
            sys.exit(1)
