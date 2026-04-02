#!/usr/bin/env python3
"""
Run Pywr-DRB simulations for all ensemble sets in parallel using MPI rank distribution.

Supports three execution modes:
  - MPI with more ranks than sets: multiple ranks collaborate per set
  - MPI with fewer ranks than sets: each rank processes sets serially
  - Serial (no mpirun): processes all sets sequentially on one process

Usage:
  MPI:    mpirun -np N python 03_run_pywrdrb_simulations.py <dataset_id>
  Serial: python 03_run_pywrdrb_simulations.py <dataset_id>
"""

import os
import sys

from methods.mpi_utils import get_comm, get_set_assignments, get_set_peer_ranks
from methods.simulate import run_ensemble_set_simulations
from methods.config import (
    DATASET_CONFIGS,
    N_ENSEMBLE_SETS,
    N_REALIZATIONS_PER_ENSEMBLE_SET,
    verify_dataset_id,
)
from methods.ensemble_utils import get_ensemble_set_spec
from methods.print_summary import print_simulation_status


def parallel_run_all_sets(dataset_id):
    """Distribute Pywr-DRB simulations across available MPI ranks."""

    comm, rank, size = get_comm()

    verify_dataset_id(dataset_id)
    dataset_config = DATASET_CONFIGS[dataset_id]

    if rank == 0:
        print(f"[SIM] {dataset_id} | {N_ENSEMBLE_SETS} sets × {N_REALIZATIONS_PER_ENSEMBLE_SET} realizations | {size} ranks")
        missing_sets = [
            set_id for set_id in range(N_ENSEMBLE_SETS)
            if not os.path.exists(get_ensemble_set_spec(set_id, dataset_id).files['catchment_inflow'])
        ]
        if missing_sets:
            print(f"[SIM] WARNING: Missing prepared sets: {missing_sets} — run prep first!")

    # Track success/failure
    success_count = 0
    total_processed = 0

    assignments = get_set_assignments(rank, size, N_ENSEMBLE_SETS)

    if size >= N_ENSEMBLE_SETS:
        # Multiple ranks per set — use point-to-point communication
        set_id, local_rank, local_size = assignments[0]
        peers = get_set_peer_ranks(rank, size, N_ENSEMBLE_SETS)

        success = run_ensemble_set_simulations(
            set_id, dataset_id, use_mpi=True,
            comm=comm,
            local_rank=local_rank, local_size=local_size,
            set_peer_ranks=peers,
        )
        total_processed = 1
        success_count = 1 if success else 0
    else:
        # More sets than ranks — each rank processes its assigned sets serially
        for set_id, local_rank, local_size in assignments:
            success = run_ensemble_set_simulations(set_id, dataset_id, use_mpi=False)
            total_processed += 1
            success_count += 1 if success else 0

    # Wait for all ranks to finish before verifying output files
    if comm is not None:
        comm.Barrier()

    if rank == 0:
        failed_sets = [
            sid + 1 for sid in range(N_ENSEMBLE_SETS)
            if not os.path.exists(get_ensemble_set_spec(sid, dataset_id).output_file)
        ]
        if not failed_sets:
            print(f"[SIM] {dataset_id}: {N_ENSEMBLE_SETS}/{N_ENSEMBLE_SETS} sets complete.")
        else:
            print(f"[SIM] WARNING: Missing output for sets: {failed_sets}")


def main(dataset_id):
    comm, rank, _ = get_comm()

    parallel_run_all_sets(dataset_id)

    if rank == 0:
        print_simulation_status(dataset_id)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: mpirun -np N python 03_run_pywrdrb_simulations.py <dataset_id>")
        print("       python 03_run_pywrdrb_simulations.py <dataset_id>  (serial mode)")
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
