"""
07_calculate_event_metrics.py

Calculate per-drought-event metrics from raw HDF5 timeseries for the
Sankey-Parallel Coordinate figure.

Each rank independently:
  1. Reads realization IDs from HDF5 metadata (no data loaded)
  2. Loads ONLY its assigned realizations (selective HDF5 reading)
  3. Computes event metrics over drought windows
  4. Sends results to rank 0 via point-to-point send/recv

No global MPI collectives (bcast, gather, barrier, reduce) are used.

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

from methods.mpi_utils import get_comm, global_point_to_point_gather
from methods.utils import distribute_realizations_across_ranks
from methods.load import (
    get_realization_ids_from_export,
    load_rank_subset_from_export,
    load_drought_events,
)
from methods.config import *
from methods.verification import verify_postprocessing_output
from methods.metrics.event_metrics import calculate_all_event_metrics


# Output directory
EVENT_METRICS_DIR = os.path.join(ROOT_DIR, 'pywrdrb', 'event_metrics')


def process_ssi_window(local_data, dataset_id, ssi_window,
                        drought_events_df, my_realizations):
    """Process a single SSI window on the local rank's realizations.

    Parameters
    ----------
    local_data : pywrdrb.Data
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
    """MPI-parallelized event metric calculation with rank-specific loading."""
    comm, rank, size = get_comm()

    if rank == 0:
        print("=" * 80)
        print(f"EVENT METRICS CALCULATION (MPI): {dataset_id}")
        print(f"SSI Windows: {ssi_windows}")
        print(f"Using {size} MPI rank(s)")
        print("=" * 80)

    # Each rank checks file existence independently
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
    if not os.path.exists(fname):
        print(f"Rank {rank} ERROR: Postprocessed data not found: {fname}")
        return

    # Each rank reads realization IDs from HDF5 metadata (fast, no data loaded)
    realization_ids = get_realization_ids_from_export(fname, dataset_id)
    n_realizations = len(realization_ids)

    # Determine this rank's assigned realizations
    my_realizations = distribute_realizations_across_ranks(
        realization_ids, rank, size
    )

    if rank == 0:
        print(f"  Total realizations: {n_realizations}")
        print(f"  Realizations per rank: ~{n_realizations // size}")

    # Each rank loads ONLY its assigned realizations (staggered I/O)
    results_sets = ['res_storage', 'inflow', 'shortage', 'contribution',
                    'ibt_diversions', 'ibt_demands']
    print(f"Rank {rank}: loading {len(my_realizations)} realizations from HDF5...")
    local_data = load_rank_subset_from_export(
        fname, my_realizations, results_sets, rank, size
    )
    print(f"Rank {rank}: loaded {len(my_realizations)} realizations")

    # --- Process each SSI window ---
    for ssi_window in ssi_windows:
        if rank == 0:
            print(f"\n{'=' * 60}")
            print(f"PROCESSING SSI WINDOW: {ssi_window}")
            print(f"{'=' * 60}")

        # Load drought events (small CSV, all ranks read independently)
        drought_events_df = load_drought_events(dataset_id, ssi_window)

        # Process on local realizations
        local_metrics = process_ssi_window(
            local_data, dataset_id, ssi_window,
            drought_events_df, my_realizations,
        )

        # Gather results to rank 0 via point-to-point (no collective gather)
        gathered = global_point_to_point_gather(
            comm, local_metrics, rank, size, tag=620
        )

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

    comm, rank, _ = get_comm()
    if rank == 0:
        print(f"\nProcessing dataset: {dataset_id}")
        print(f"SSI windows: {ssi_windows}")

    main(dataset_id, ssi_windows)
