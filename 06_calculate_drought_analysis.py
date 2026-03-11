"""
06_calculate_drought_analysis.py

Combined MPI-parallel drought analysis: annual satisficing + per-event metrics.

Each rank independently:
  1. Reads realization IDs from HDF5 metadata (no data loaded)
  2. Loads ONLY its assigned realizations (selective HDF5 reading)
  3. Computes annual satisficing (with drought overlap annotation)
  4. Computes per-event metrics over exact drought windows
  5. Sends results to rank 0 via point-to-point send/recv

No global MPI collectives (bcast, gather, barrier, reduce) are used.

Usage:
    mpirun -np N python 06_calculate_drought_analysis.py <dataset_id> --all
    mpirun -np N python 06_calculate_drought_analysis.py <dataset_id> 3 6 12
    python 06_calculate_drought_analysis.py <dataset_id> --all   # serial fallback

Output:
    pywrdrb/satisficing_analysis/{dataset_id}_ssi{window}_annual_satisficing.csv
    pywrdrb/event_metrics/{dataset_id}_ssi{window}_event_metrics.csv
"""

import os
import sys
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
from methods.metrics.satisficing import calculate_annual_satisficing
from methods.metrics.event_metrics import calculate_all_event_metrics
from methods.print_summary import print_satisficing_summary
from methods.drought_analysis import calculate_statistical_significance
from methods.save import save_annual_satisficing, SATISFICING_ANALYSIS_DIR

# Output directory for event metrics
EVENT_METRICS_DIR = os.path.join(ROOT_DIR, 'pywrdrb', 'event_metrics')


def main(dataset_id, ssi_windows):
    """MPI-parallelized drought analysis: annual satisficing + event metrics."""
    comm, rank, size = get_comm()

    if rank == 0:
        print("=" * 80)
        print(f"DROUGHT ANALYSIS (MPI): {dataset_id}")
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

    # Load data once — superset of what both analyses need
    results_sets = ['res_storage', 'inflow', 'shortage', 'contribution',
                    'ibt_diversions', 'ibt_demands']
    print(f"Rank {rank}: loading {len(my_realizations)} realizations from HDF5...")
    local_data = load_rank_subset_from_export(
        fname, my_realizations, results_sets, rank, size
    )
    print(f"Rank {rank}: loaded {len(my_realizations)} realizations")

    # --- Process each SSI window ---
    # Annual satisficing without drought annotation is the same across SSI windows.
    # We compute it once (first window) then just update n_droughts_in_year per window.
    local_base_satisficing = None

    for ssi_window in ssi_windows:
        if rank == 0:
            print(f"\n{'=' * 60}")
            print(f"PROCESSING SSI WINDOW: {ssi_window}")
            print(f"{'=' * 60}")

        # Load drought events (small CSV, all ranks read independently)
        drought_events_df = load_drought_events(dataset_id, ssi_window)

        # --- Part 1: Annual satisficing ---
        local_satisficing = calculate_annual_satisficing(
            local_data, dataset_id, drought_events_df,
            storage_threshold=20.0, violation_days=3
        )

        # Gather to rank 0
        gathered_satisficing = global_point_to_point_gather(
            comm, local_satisficing, rank, size, tag=610
        )

        # --- Part 2: Per-event metrics ---
        local_droughts = drought_events_df[
            drought_events_df['realization_id'].isin(my_realizations)
        ].copy()

        if len(local_droughts) > 0:
            local_event_metrics = calculate_all_event_metrics(
                local_data, dataset_id, local_droughts,
                storage_threshold=20.0, violation_days=3
            )
        else:
            local_event_metrics = pd.DataFrame()

        gathered_events = global_point_to_point_gather(
            comm, local_event_metrics, rank, size, tag=620
        )

        # --- Rank 0: combine, summarize, save ---
        if rank == 0:
            # Combine annual satisficing
            combined_satisficing = pd.concat(gathered_satisficing, ignore_index=True)

            print(f"\n  Annual satisficing: {len(combined_satisficing)} year-realization pairs")
            n_drought = (combined_satisficing['n_droughts_in_year'] > 0).sum()
            n_non_drought = (combined_satisficing['n_droughts_in_year'] == 0).sum()
            print(f"    Drought years: {n_drought}, Non-drought years: {n_non_drought}")

            # Print summary
            print_satisficing_summary(combined_satisficing, dataset_id, ssi_window)

            # Statistical significance
            drought_subset = combined_satisficing[combined_satisficing['n_droughts_in_year'] > 0]
            non_drought_subset = combined_satisficing[combined_satisficing['n_droughts_in_year'] == 0]
            if len(drought_subset) > 0 and len(non_drought_subset) > 0:
                calculate_statistical_significance(drought_subset, non_drought_subset)

            # Save annual satisficing
            save_annual_satisficing(combined_satisficing, dataset_id, ssi_window)

            # Combine event metrics
            all_events = pd.concat(
                [df for df in gathered_events if len(df) > 0],
                ignore_index=True
            )
            print(f"\n  Event metrics: {len(all_events)} events across "
                  f"{n_realizations} realizations")

            if len(all_events) > 0:
                for cat in ['all_pass', 'storage_fail', 'montague_fail', 'both_fail']:
                    n = (all_events['classification'] == cat).sum()
                    if n > 0:
                        print(f"    {cat}: {n}")

                # Save event metrics
                os.makedirs(EVENT_METRICS_DIR, exist_ok=True)
                event_fname = os.path.join(EVENT_METRICS_DIR,
                                           f'{dataset_id}_ssi{ssi_window}_event_metrics.csv')
                all_events.to_csv(event_fname, index=False)
                print(f"  Saved {len(all_events)} events to: {event_fname}")
            else:
                print("  No events computed. Skipping save.")

    if rank == 0:
        print("\n" + "=" * 80)
        print("DROUGHT ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"\nSatisficing results: {SATISFICING_ANALYSIS_DIR}/")
        print(f"Event metrics: {EVENT_METRICS_DIR}/")


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
                    print(f"ERROR: Invalid SSI window {ssi_window}. "
                          f"Must be one of {SSI_WINDOWS}")
                    sys.exit(1)
                ssi_windows.append(ssi_window)
            except ValueError:
                print(f"ERROR: Invalid SSI window '{arg}'. "
                      "Must be an integer or '--all'.")
                sys.exit(1)

    verify_dataset_id(dataset_id)

    comm, rank, _ = get_comm()
    if rank == 0:
        print(f"\nProcessing dataset: {dataset_id}")
        print(f"SSI windows: {ssi_windows}")

    main(dataset_id, ssi_windows)
