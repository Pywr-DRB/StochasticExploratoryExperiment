"""
07_calculate_event_metrics.py

Calculate per-drought-event metrics from raw HDF5 timeseries for the
Sankey-Parallel Coordinate figure.

Uses MPI parallelization to distribute realizations across ranks,
following the same pattern as 06_calculate_satisficing_by_drought.py.

Each SSI-defined drought event becomes one sample with hazard characteristics,
system action metrics, and outcome metrics computed over the exact event window.

Usage:
    mpirun -np N python 07_calculate_event_metrics.py <dataset_id> [--all | ssi_windows...]
    mpirun -np N python 07_calculate_event_metrics.py stationary_ensemble --all
    python 07_calculate_event_metrics.py stationary_ensemble --all   # serial fallback

Output:
    pywrdrb/event_metrics/{dataset_id}_ssi{window}_event_metrics.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from mpi4py import MPI

import pywrdrb
from methods.config import *
from methods.load import load_drought_events
from methods.utils import distribute_realizations_across_ranks
from methods.verification import verify_postprocessing_output
from methods.metrics.event_metrics import calculate_all_event_metrics


# Output directory
EVENT_METRICS_DIR = os.path.join(ROOT_DIR, 'pywrdrb', 'event_metrics')


class _LocalData:
    """Lightweight data wrapper holding a realization subset.

    Mimics the pywrdrb.Data interface used by event_metrics functions:
        data.res_storage[dataset_id][realization_id]
        data.shortage[dataset_id][realization_id]
        data.inflow[dataset_id][realization_id]
        data.contribution[dataset_id][realization_id]
        data.ibt_diversions[dataset_id][realization_id]
        data.ibt_demands[dataset_id][realization_id]
    """

    def __init__(self, dataset_id, res_storage, shortage, inflow,
                 contribution, ibt_diversions, ibt_demands):
        self.res_storage = {dataset_id: res_storage}
        self.shortage = {dataset_id: shortage}
        self.inflow = {dataset_id: inflow}
        self.contribution = {dataset_id: contribution}
        self.ibt_diversions = {dataset_id: ibt_diversions}
        self.ibt_demands = {dataset_id: ibt_demands}


def _build_local_data(full_data, dataset_id, realization_ids):
    """Extract a subset of realizations from the full data object."""
    return _LocalData(
        dataset_id,
        res_storage={r: full_data.res_storage[dataset_id][r]
                     for r in realization_ids},
        shortage={r: full_data.shortage[dataset_id][r]
                  for r in realization_ids},
        inflow={r: full_data.inflow[dataset_id][r]
                for r in realization_ids},
        contribution={r: full_data.contribution[dataset_id][r]
                      for r in realization_ids},
        ibt_diversions={r: full_data.ibt_diversions[dataset_id][r]
                        for r in realization_ids},
        ibt_demands={r: full_data.ibt_demands[dataset_id][r]
                     for r in realization_ids},
    )


def process_ssi_window_mpi(local_data, dataset_id, ssi_window,
                            drought_events_df, my_realizations):
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

    Returns
    -------
    pd.DataFrame
        Event metrics for this rank's realizations.
    """
    # Filter drought events to local realizations only
    local_droughts = drought_events_df[
        drought_events_df['realization_id'].isin(my_realizations)
    ].copy()

    if len(local_droughts) == 0:
        return pd.DataFrame()

    # Calculate event metrics over exact drought windows
    metrics_df = calculate_all_event_metrics(
        local_data, dataset_id, local_droughts,
        storage_threshold=20.0, violation_days=3
    )

    return metrics_df


def save_metrics(metrics_df, dataset_id, ssi_window):
    """Save metrics DataFrame to CSV."""
    os.makedirs(EVENT_METRICS_DIR, exist_ok=True)
    fname = os.path.join(EVENT_METRICS_DIR,
                         f'{dataset_id}_ssi{ssi_window}_event_metrics.csv')
    metrics_df.to_csv(fname, index=False)
    print(f"  Saved {len(metrics_df)} events to: {fname}")
    return fname


def main(dataset_id, ssi_windows):
    """MPI-parallelized event metric calculation."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("=" * 80)
        print(f"EVENT METRICS CALCULATION (MPI): {dataset_id}")
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
            results_sets=['res_storage', 'inflow', 'shortage',
                          'contribution', 'ibt_diversions', 'ibt_demands']
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
    my_realizations = distribute_realizations_across_ranks(
        realization_ids, rank, size
    )

    # Distribute per-rank data subsets via send/recv
    if rank == 0:
        for r in range(size):
            r_ids = distribute_realizations_across_ranks(
                realization_ids, r, size
            )
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
    for ssi_window in ssi_windows:
        if rank == 0:
            print(f"\n{'=' * 60}")
            print(f"PROCESSING SSI WINDOW: {ssi_window}")
            print(f"{'=' * 60}")

        # Load drought events (small CSV, all ranks read independently)
        drought_events_df = load_drought_events(dataset_id, ssi_window)

        # Process on local realizations
        local_metrics = process_ssi_window_mpi(
            local_data, dataset_id, ssi_window,
            drought_events_df, my_realizations,
        )

        # Gather results to rank 0
        gathered = comm.gather(local_metrics, root=0)

        # Rank 0: combine and save
        if rank == 0:
            all_metrics = pd.concat(
                [df for df in gathered if len(df) > 0],
                ignore_index=True
            )
            print(f"\n  Combined: {len(all_metrics)} events across "
                  f"{n_realizations} realizations")

            # Classification summary
            if len(all_metrics) > 0:
                for cat in ['all_pass', 'storage_fail',
                            'montague_fail', 'both_fail']:
                    n = (all_metrics['classification'] == cat).sum()
                    if n > 0:
                        print(f"    {cat}: {n}")

                save_metrics(all_metrics, dataset_id, ssi_window)
            else:
                print("  No events computed. Skipping save.")

    if rank == 0:
        print("\n" + "=" * 80)
        print("EVENT METRICS COMPLETE!")
        print("=" * 80)
        print(f"\nResults saved to: {EVENT_METRICS_DIR}/")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        print(f"\nAvailable datasets: {list(DATASET_CONFIGS.keys())}")
        print(f"Available SSI windows: {SSI_WINDOWS}")
        sys.exit(1)

    dataset_id = sys.argv[1]

    # Parse SSI windows (same pattern as 06)
    if sys.argv[2] == '--all':
        ssi_windows = list(SSI_WINDOWS)
    else:
        ssi_windows = []
        for arg in sys.argv[2:]:
            try:
                ssi_window = int(arg)
                if ssi_window not in SSI_WINDOWS:
                    print(f"ERROR: Invalid SSI window {ssi_window}. "
                          f"Must be one of {SSI_WINDOWS}")
                    sys.exit(1)
                ssi_windows.append(ssi_window)
            except ValueError:
                print(f"ERROR: Invalid SSI window '{arg}'. "
                      "Must be an integer or '--all'.")
                sys.exit(1)

    # Validate dataset
    verify_dataset_id(dataset_id)

    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        print(f"\nProcessing dataset: {dataset_id}")
        print(f"SSI windows: {ssi_windows}")

    main(dataset_id, ssi_windows)
