"""
Generate all ensemble sets in parallel using MPI rank distribution.

IMPORTANT: This script avoids ALL collective MPI operations (bcast, barrier,
gather, reduce) on the global communicator. The OpenMPI 4.0.5 + libfabric
1.12.1 stack on Hopper HPC segfaults during global collectives with 240 ranks
across 8 nodes. Instead, each rank loads data independently from disk, and
only point-to-point send/recv is used within small set peer groups.

Supports three execution modes:
  - MPI with more ranks than sets: multiple ranks collaborate per set
  - MPI with fewer ranks than sets: each rank processes sets serially
  - Serial (no mpirun): processes all sets sequentially on one process

Usage:
  MPI:    mpirun -np N python 01_generate_ensemble_sets.py <dataset_id>
  Serial: python 01_generate_ensemble_sets.py <dataset_id>
"""

import sys
import numpy as np

from methods.mpi_utils import get_comm, get_set_assignments, get_set_peer_ranks
from methods.generate import generate_ensemble_set
from methods.load import load_baseline_historical_flow
from methods.config import (
    DATASET_CONFIGS,
    N_ENSEMBLE_SETS,
    N_REALIZATIONS_PER_ENSEMBLE_SET,
    N_YEARS,
    BASELINE_DATASET,
    pywrdrb_nodes_to_generate,
    verify_dataset_id,
)
from methods.ensemble_utils import ensure_ensemble_set_dirs, get_existing_ensemble_sets


def _load_and_prepare_data(rank):
    """Load baseline historical flow data independently on this rank.

    Every rank reads from disk to avoid global MPI collectives (bcast)
    which crash on the Hopper HPC libfabric/RDMA stack at scale.
    """
    Q = load_baseline_historical_flow(gage_flow=True, period='full',
                                      flowtype=BASELINE_DATASET)
    Q_baseline = load_baseline_historical_flow(gage_flow=True, period='baseline',
                                               flowtype=BASELINE_DATASET)
    Q_inflow = load_baseline_historical_flow(gage_flow=False,
                                             period='full',
                                             flowtype=BASELINE_DATASET)
    Q = Q.loc[:, pywrdrb_nodes_to_generate]
    Q_baseline = Q_baseline.loc[:, pywrdrb_nodes_to_generate]

    # Replace zeros with NaN (physically unrealistic)
    n_zeros = (Q == 0.0).sum().sum()
    if n_zeros > 0:
        if rank == 0:
            print(f"Replacing {n_zeros} zero values with NaN")
        Q.replace(0, np.nan, inplace=True)
        Q_inflow.replace(0, np.nan, inplace=True)
        Q_baseline.replace(0, np.nan, inplace=True)

    return Q, Q_baseline, Q_inflow


def parallel_generate_all_sets(dataset_id):
    """Distribute ensemble set generation across available MPI ranks."""

    comm, rank, size = get_comm()

    # Verify dataset
    verify_dataset_id(dataset_id)
    dataset_config = DATASET_CONFIGS[dataset_id]

    if rank == 0:
        print(f"[GENERATE] {dataset_id} | {N_ENSEMBLE_SETS} sets × {N_REALIZATIONS_PER_ENSEMBLE_SET} realizations | {size} ranks")

    # Ensure all directories exist
    ensure_ensemble_set_dirs(dataset_id)

    if size >= N_ENSEMBLE_SETS:
        # ------------------------------------------------------------------
        # Multi-rank-per-set regime: each rank loads data independently,
        # then works on its assigned set using point-to-point communication.
        # NO global collectives (bcast/barrier) — they crash on Hopper.
        # ------------------------------------------------------------------

        # Each rank loads data from disk independently
        Q, Q_baseline, Q_inflow = _load_and_prepare_data(rank)

        # Determine this rank's set assignment
        assignments = get_set_assignments(rank, size, N_ENSEMBLE_SETS)
        set_id, local_rank, local_size = assignments[0]
        peers = get_set_peer_ranks(rank, size, N_ENSEMBLE_SETS)

        success = generate_ensemble_set(
            set_id, dataset_id, use_mpi=True,
            comm=comm,
            local_rank=local_rank, local_size=local_size,
            set_peer_ranks=peers,
            preloaded_data=(Q, Q_baseline, Q_inflow),
        )
        assert success, f"Set {set_id + 1} generation failed on rank {rank}"

    else:
        # ------------------------------------------------------------------
        # More sets than ranks: each rank processes its assigned sets serially.
        # ------------------------------------------------------------------
        assignments = get_set_assignments(rank, size, N_ENSEMBLE_SETS)
        for set_id, local_rank, local_size in assignments:
            success = generate_ensemble_set(set_id, dataset_id, use_mpi=False)
            assert success, f"Set {set_id + 1} generation failed on rank {rank}"

    # No global barrier — each set completes independently.
    # Rank 0 verifies output files exist after its own set finishes.
    if rank == 0:
        existing_sets = get_existing_ensemble_sets(dataset_id)
        if len(existing_sets) == N_ENSEMBLE_SETS:
            print(f"[GENERATE] {dataset_id}: {N_ENSEMBLE_SETS}/{N_ENSEMBLE_SETS} sets complete.")
        else:
            missing = set(range(N_ENSEMBLE_SETS)) - set([s.set_id for s in existing_sets])
            print(f"[GENERATE] WARNING: {len(existing_sets)}/{N_ENSEMBLE_SETS} sets found. Missing: {sorted(missing)}")


def main(dataset_id):
    _, rank, _ = get_comm()

    parallel_generate_all_sets(dataset_id)



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: mpirun -np N python 01_generate_ensemble_sets.py <dataset_id>")
        print("       python 01_generate_ensemble_sets.py <dataset_id>  (serial mode)")
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
