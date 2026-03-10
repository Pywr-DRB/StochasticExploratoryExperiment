"""
Prepare Pywr-DRB inputs for all ensemble sets in parallel using MPI rank distribution.

Uses comm.Split() to create per-set sub-communicators, then temporarily swaps
MPI.COMM_WORLD so the pywrdrb preprocessors (which hardcode COMM_WORLD) operate
within the sub-communicator scope. This allows all ranks to participate in
preprocessing, with each set's ranks collaborating via MPI internally.

With 240 ranks and 20 sets, each set gets 12 ranks working in parallel (~12x
speedup per set vs the serial-per-set approach).

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

    assignments = get_set_assignments(rank, size, N_ENSEMBLE_SETS)

    if size >= N_ENSEMBLE_SETS and comm is not None:
        # Multiple ranks per set — use comm.Split so ALL ranks participate
        set_id, local_rank, local_size = assignments[0]

        # Create sub-communicator scoped to this set
        local_comm = comm.Split(color=set_id, key=local_rank)

        # Temporarily swap MPI.COMM_WORLD so pywrdrb preprocessors
        # (which hardcode COMM_WORLD) operate within the sub-communicator
        saved_comm_world = MPI.COMM_WORLD
        MPI.COMM_WORLD = local_comm

        try:
            if local_rank == 0:
                print(f"Set {set_id+1}: {local_size} ranks collaborating via sub-communicator")
            success = prep_ensemble_set(set_id, dataset_id, use_mpi=True)
        finally:
            # Always restore original COMM_WORLD
            MPI.COMM_WORLD = saved_comm_world
            local_comm.Free()

    else:
        # Serial or more sets than ranks — each rank processes sets serially
        for set_id, local_rank, local_size in assignments:
            success = prep_ensemble_set(set_id, dataset_id, use_mpi=False)

    # Rank 0 verifies output files exist after global barrier
    if comm is not None:
        comm.Barrier()

    if rank == 0:
        print("\n" + "=" * 60)
        print(f"PYWRDRB INPUT PREPARATION COMPLETED: {dataset_id}")
        print("=" * 60)

        required_files = [
            'predicted_inflow',
            'diversion_nyc',
            'diversion_nj',
            'predicted_diversions'
        ]
        success_sets = []
        failed_sets = []
        for sid in range(N_ENSEMBLE_SETS):
            set_spec = get_ensemble_set_spec(sid, dataset_id)
            if all(os.path.exists(set_spec.files[f]) for f in required_files):
                success_sets.append(sid)
            else:
                failed_sets.append(sid + 1)

        print(f"Verified: {len(success_sets)}/{N_ENSEMBLE_SETS} sets have all required files")
        if not failed_sets:
            print("SUCCESS: All ensemble sets prepared successfully!")
        else:
            print(f"WARNING: Missing files for sets: {failed_sets}")

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
