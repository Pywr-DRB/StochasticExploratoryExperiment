"""
Calculate satisficing conditions during drought vs non-drought years.

Uses MPI parallelization to distribute realizations across ranks.

This script addresses the question: Is there a meaningful difference between
performance outcomes during years with droughts vs years without droughts?

Analysis includes:
1. Satisficing during ALL simulation years (baseline)
2. Satisficing during years with some drought events
3. Satisficing during years with no drought events

Satisficing conditions:
- NYC storage >= 20% throughout evaluation period
- Montague flow target violations <= 3 consecutive days

Usage:
    mpirun -np N python 06_calculate_satisficing_by_drought.py <dataset_id> [ssi_windows...]
    mpirun -np N python 06_calculate_satisficing_by_drought.py <dataset_id> --all

Examples:
    mpirun -np 80 python 06_calculate_satisficing_by_drought.py stationary_ensemble --all
    python 06_calculate_satisficing_by_drought.py stationary_ensemble --all  # serial fallback
"""

import sys
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from mpi4py import MPI

import pywrdrb
from methods.config import *
from methods.load import load_drought_events
from methods.utils import distribute_realizations_across_ranks
from methods.verification import verify_postprocessing_output
from methods.print_summary import print_satisficing_summary
from methods.drought_analysis import calculate_statistical_significance
from methods.save import save_satisficing_results, SATISFICING_ANALYSIS_DIR
from methods.metrics.satisficing import (
    calculate_satisficing_conditions,
    calculate_satisficing_during_droughts,
    calculate_satisficing_non_drought_periods
)


class _LocalData:
    """Lightweight data wrapper holding a realization subset.

    Mimics the pywrdrb.Data interface used by the satisficing functions:
        data.res_storage[dataset_id][realization_id]
        data.shortage[dataset_id][realization_id]
        data.inflow[dataset_id][realization_id]
        data.contribution[dataset_id][realization_id]
    """

    def __init__(self, dataset_id, res_storage, shortage, inflow, contribution):
        self.res_storage = {dataset_id: res_storage}
        self.shortage = {dataset_id: shortage}
        self.inflow = {dataset_id: inflow}
        self.contribution = {dataset_id: contribution}


def _build_local_data(full_data, dataset_id, realization_ids):
    """Extract a subset of realizations from the full data object."""
    return _LocalData(
        dataset_id,
        res_storage={r: full_data.res_storage[dataset_id][r] for r in realization_ids},
        shortage={r: full_data.shortage[dataset_id][r] for r in realization_ids},
        inflow={r: full_data.inflow[dataset_id][r] for r in realization_ids},
        contribution={r: full_data.contribution[dataset_id][r] for r in realization_ids},
    )


def process_ssi_window_mpi(local_data, dataset_id, ssi_window,
                           drought_events_df, my_realizations,
                           all_years_results=None):
    """Process a single SSI window on the local rank's realizations.

    Parameters
    ----------
    local_data : _LocalData
        Data object containing only this rank's realizations.
    dataset_id : str
        Dataset identifier.
    ssi_window : int
        SSI window (3, 6, or 12).
    drought_events_df : pd.DataFrame
        Full drought events (will be filtered to local realizations).
    my_realizations : list
        Realization IDs assigned to this rank.
    all_years_results : pd.DataFrame or None
        Pre-computed all-years results for this rank's realizations.

    Returns
    -------
    all_years_results, drought_results, non_drought_results : pd.DataFrame
    """
    # Filter drought events to local realizations only
    local_droughts = drought_events_df[
        drought_events_df['realization_id'].isin(my_realizations)
    ].copy()

    # 1. All years (reuse if already computed for first SSI window)
    if all_years_results is None:
        all_years_results = calculate_satisficing_conditions(
            local_data, dataset_id,
            period_type='year',
            evaluate_all_years=True,
            storage_threshold=20.0,
            violation_days=3
        )

    # 2. Years with drought events
    drought_results = calculate_satisficing_during_droughts(
        local_data, dataset_id, local_droughts,
        storage_threshold=20.0,
        violation_days=3
    )

    # 3. Years without drought events
    non_drought_results = calculate_satisficing_non_drought_periods(
        local_data, dataset_id, local_droughts,
        storage_threshold=20.0,
        violation_days=3
    )

    return all_years_results, drought_results, non_drought_results


def main(dataset_id, ssi_windows):
    """MPI-parallelized satisficing analysis."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("=" * 80)
        print(f"SATISFICING ANALYSIS (MPI): {dataset_id}")
        print(f"SSI Windows: {ssi_windows}")
        print(f"Using {size} MPI ranks")
        print("=" * 80)

    # Verify postprocessed data exists (rank 0 only, broadcast result)
    verification_ok = False
    if rank == 0:
        try:
            verify_postprocessing_output(dataset_id)
            verification_ok = True
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
    verification_ok = comm.bcast(verification_ok, root=0)
    if not verification_ok:
        return

    # --- Rank 0 loads full data and distributes subsets ---
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'

    if rank == 0:
        print(f"\nLoading postprocessed data from: {fname}")
        data = pywrdrb.Data()
        data.load_from_export(
            fname,
            results_sets=['res_storage', 'inflow', 'shortage', 'contribution']
        )
        realization_ids = sorted(data.shortage[dataset_id].keys())
        n_realizations = len(realization_ids)
        print(f"  Loaded {n_realizations} realizations")
    else:
        realization_ids = None
        n_realizations = None

    # Broadcast metadata
    n_realizations = comm.bcast(n_realizations, root=0)
    realization_ids = comm.bcast(realization_ids, root=0)

    # Determine each rank's assigned realizations
    my_realizations = distribute_realizations_across_ranks(realization_ids, rank, size)

    # Distribute per-rank data subsets via send/recv
    if rank == 0:
        for r in range(size):
            r_ids = distribute_realizations_across_ranks(realization_ids, r, size)
            subset = _build_local_data(data, dataset_id, r_ids)
            if r == 0:
                local_data = subset
            else:
                comm.send(subset, dest=r, tag=200)
        del data  # free full data on rank 0
        print(f"  Distributed data to {size} ranks")
    else:
        local_data = comm.recv(source=0, tag=200)

    print(f"Rank {rank}: received {len(my_realizations)} realizations")

    # --- Process each SSI window ---
    local_all_years = None  # reuse across SSI windows

    for ssi_window in ssi_windows:
        if rank == 0:
            print(f"\n{'=' * 60}")
            print(f"PROCESSING SSI WINDOW: {ssi_window}")
            print(f"{'=' * 60}")

        # Load drought events (small CSV, all ranks read independently)
        drought_events_df = load_drought_events(dataset_id, ssi_window)

        # Process on local realizations
        local_all_years, local_drought, local_non_drought = process_ssi_window_mpi(
            local_data, dataset_id, ssi_window,
            drought_events_df, my_realizations,
            all_years_results=local_all_years,
        )

        # Gather results to rank 0
        gathered_all = comm.gather(local_all_years, root=0)
        gathered_drought = comm.gather(local_drought, root=0)
        gathered_non_drought = comm.gather(local_non_drought, root=0)

        # Rank 0: combine, summarize, save
        if rank == 0:
            all_years_results = pd.concat(gathered_all, ignore_index=True)
            drought_results = pd.concat(gathered_drought, ignore_index=True)
            non_drought_results = pd.concat(gathered_non_drought, ignore_index=True)

            print(f"\n  All years: {len(all_years_results)} year-realization pairs")
            print(f"  Drought years: {len(drought_results)} year-realization pairs")
            print(f"  Non-drought years: {len(non_drought_results)} year-realization pairs")

            # Summary statistics
            print_satisficing_summary(
                all_years_results, drought_results, non_drought_results,
                dataset_id, ssi_window
            )

            # Statistical significance tests
            if len(drought_results) > 0 and len(non_drought_results) > 0:
                calculate_statistical_significance(drought_results, non_drought_results)

            # Save results
            save_satisficing_results(
                all_years_results, drought_results, non_drought_results,
                dataset_id, ssi_window
            )

    if rank == 0:
        print("\n" + "=" * 80)
        print("ALL ANALYSES COMPLETE!")
        print("=" * 80)
        print(f"\nResults saved to: {SATISFICING_ANALYSIS_DIR}/")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        print(f"\nAvailable datasets: {list(DATASET_CONFIGS.keys())}")
        print(f"Available SSI windows: {SSI_WINDOWS}")
        sys.exit(1)

    dataset_id = sys.argv[1]

    # Parse SSI windows
    if sys.argv[2] == '--all':
        ssi_windows = list(SSI_WINDOWS)
    else:
        ssi_windows = []
        for arg in sys.argv[2:]:
            try:
                ssi_window = int(arg)
                if ssi_window not in SSI_WINDOWS:
                    print(f"ERROR: Invalid SSI window {ssi_window}. Must be one of {SSI_WINDOWS}")
                    sys.exit(1)
                ssi_windows.append(ssi_window)
            except ValueError:
                print(f"ERROR: Invalid SSI window '{arg}'. Must be an integer or '--all'.")
                sys.exit(1)

    # Validate dataset
    verify_dataset_id(dataset_id)

    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        print(f"\nProcessing dataset: {dataset_id}")
        print(f"SSI windows: {ssi_windows}")

    main(dataset_id, ssi_windows)
