"""
Prepare Pywr-DRB inputs for all ensemble sets in parallel using MPI rank distribution.

Uses comm.Split() to create per-set sub-communicators, then passes the
sub-communicator directly to the pywrdrb preprocessors via their `comm`
parameter. All ranks participate — with 240 ranks and 20 sets, each set
gets 12 ranks working in parallel.

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
)
from methods.ensemble_utils import get_ensemble_set_spec, get_existing_ensemble_sets
from methods.print_summary import print_prep_status

if MPI_AVAILABLE:
    from mpi4py import MPI


def parallel_prep_all_sets(dataset_id):
    """Distribute Pywr-DRB input preparation across available MPI ranks."""

    comm, rank, size = get_comm()

    verify_dataset_id(dataset_id)
    dataset_config = DATASET_CONFIGS[dataset_id]

    if rank == 0:
        print(f"[PREP] {dataset_id} | {N_ENSEMBLE_SETS} sets | {size} ranks")
        existing_sets = get_existing_ensemble_sets(dataset_id)
        if len(existing_sets) < N_ENSEMBLE_SETS:
            missing_sets = set(range(N_ENSEMBLE_SETS)) - set([s.set_id for s in existing_sets])
            print(f"[PREP] WARNING: Missing ensemble sets: {sorted(missing_sets)} — run generation first!")

    assignments = get_set_assignments(rank, size, N_ENSEMBLE_SETS)

    if size >= N_ENSEMBLE_SETS and comm is not None:
        set_id, local_rank, local_size = assignments[0]
        local_comm = comm.Split(color=set_id, key=local_rank)

        try:
            success = prep_ensemble_set(set_id, dataset_id, use_mpi=True, comm=local_comm)
        finally:
            local_comm.Free()

    else:
        # Serial or more sets than ranks — each rank processes sets serially
        for set_id, local_rank, local_size in assignments:
            success = prep_ensemble_set(set_id, dataset_id, use_mpi=False)

    # Wait for all sets to finish before verification
    if comm is not None:
        comm.Barrier()

    if rank == 0:
        required_files = ['predicted_inflow', 'diversion_nyc', 'diversion_nj', 'predicted_diversions']
        failed_sets = [
            sid + 1 for sid in range(N_ENSEMBLE_SETS)
            if not all(os.path.exists(get_ensemble_set_spec(sid, dataset_id).files[f]) for f in required_files)
        ]
        if not failed_sets:
            print(f"[PREP] {dataset_id}: {N_ENSEMBLE_SETS}/{N_ENSEMBLE_SETS} sets complete.")
        else:
            print(f"[PREP] WARNING: Missing files for sets: {failed_sets}")


def main(dataset_id):
    comm, rank, _ = get_comm()

    parallel_prep_all_sets(dataset_id)

    if rank == 0:
        print_prep_status(dataset_id)


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
