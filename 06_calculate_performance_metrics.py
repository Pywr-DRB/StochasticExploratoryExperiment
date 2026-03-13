"""
06_calculate_performance_metrics.py

MPI-parallel calculation of all performance metric CSVs.

Each rank independently:
  1. Reads realization IDs from HDF5 metadata (no data loaded)
  2. Loads ONLY its assigned realizations (selective HDF5 reading)
  3. Computes annual metrics (per water-year × period), Hashimoto RRV,
     contribution analysis, zone duration events, and per-SSI-event metrics
  4. Sends results to rank 0 via point-to-point send/recv

Requires step 04 (SSI drought identification) to have completed first.
Requires step 05 (HDF5 postprocessing) to have completed first.

No global MPI collectives (bcast, gather, barrier, reduce) are used.

Usage:
    mpirun -np N python 06_calculate_performance_metrics.py <dataset_id> --all
    mpirun -np N python 06_calculate_performance_metrics.py <dataset_id> 3 6 12
    python 06_calculate_performance_metrics.py <dataset_id> --all   # serial fallback

Output:
    pywrdrb/performance_metrics/{dataset_id}_annual_metrics.csv
    pywrdrb/performance_metrics/{dataset_id}_hashimoto_metrics.csv
    pywrdrb/performance_metrics/{dataset_id}_hashimoto_shortage_events.csv
    pywrdrb/performance_metrics/{dataset_id}_contribution_metrics.csv
    pywrdrb/performance_metrics/{dataset_id}_zone_duration_events.csv
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
from methods.postprocess import (
    calculate_annual_metrics,
    calculate_hashimoto_all,
    calculate_contribution_analysis_metrics,
    calculate_and_save_zone_duration_events,
    save_metrics_csv,
)
from methods.metrics.event_metrics import calculate_all_event_metrics

# Output directories
PERFORMANCE_METRICS_DIR = os.path.join(ROOT_DIR, 'pywrdrb', 'performance_metrics')
EVENT_METRICS_DIR = os.path.join(ROOT_DIR, 'pywrdrb', 'event_metrics')


def main(dataset_id, ssi_windows):
    """MPI-parallelized performance metric calculation."""
    comm, rank, size = get_comm()

    if rank == 0:
        print("=" * 80)
        print(f"PERFORMANCE METRICS (MPI): {dataset_id}")
        print(f"SSI Windows: {ssi_windows}")
        print(f"Using {size} MPI rank(s)")
        print("=" * 80)

    # Each rank checks file existence independently
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
    if not os.path.exists(fname):
        print(f"Rank {rank} ERROR: Postprocessed data not found: {fname}")
        return

    # Each rank reads realization IDs from HDF5 metadata
    realization_ids = get_realization_ids_from_export(fname, dataset_id)
    n_realizations = len(realization_ids)

    # Determine this rank's assigned realizations
    my_realizations = distribute_realizations_across_ranks(
        realization_ids, rank, size
    )

    if rank == 0:
        print(f"  Total realizations: {n_realizations}")
        print(f"  Realizations per rank: ~{n_realizations // size}")

    # Load data — superset of what all analyses need
    results_sets = [
        'major_flow', 'mrf_target', 'res_storage', 'res_level',
        'inflow', 'ibt_diversions', 'ibt_demands',
        'contribution', 'shortage', 'nyc_release_components',
    ]
    print(f"Rank {rank}: loading {len(my_realizations)} realizations from HDF5...")
    local_data = load_rank_subset_from_export(
        fname, my_realizations, results_sets, rank, size
    )
    print(f"Rank {rank}: loaded {len(my_realizations)} realizations")

    # =========================================================================
    # Use first SSI window for the main annual metrics drought conditioning.
    # Annual metrics are computed once (drought/nondrought split from primary SSI).
    # Per-SSI-window event metrics are computed separately for each window.
    # =========================================================================
    primary_ssi_window = ssi_windows[0]
    primary_drought_events = load_drought_events(dataset_id, primary_ssi_window)

    if primary_drought_events is None or len(primary_drought_events) == 0:
        print(f"Rank {rank} ERROR: No SSI-{primary_ssi_window} drought events found. "
              f"Run 04_calculate_ssi_drought_metrics.py first.")
        return

    # =========================================================================
    # Part 1: Annual metrics (realization × water_year × period)
    # =========================================================================
    print(f"Rank {rank}: calculating annual metrics...")
    local_annual = calculate_annual_metrics(
        local_data, dataset_id, my_realizations, primary_drought_events
    )
    gathered_annual = global_point_to_point_gather(
        comm, local_annual, rank, size, tag=700
    )

    # =========================================================================
    # Part 2: Hashimoto RRV metrics (simulation-level + per-event)
    # =========================================================================
    print(f"Rank {rank}: calculating Hashimoto metrics...")
    local_hashimoto, local_hashimoto_events = calculate_hashimoto_all(
        local_data, dataset_id, my_realizations
    )
    gathered_hashimoto = global_point_to_point_gather(
        comm, local_hashimoto, rank, size, tag=710
    )
    gathered_hashimoto_events = global_point_to_point_gather(
        comm, local_hashimoto_events, rank, size, tag=711
    )

    # =========================================================================
    # Part 3: Contribution analysis metrics
    # =========================================================================
    print(f"Rank {rank}: calculating contribution metrics...")
    local_contrib = calculate_contribution_analysis_metrics(
        local_data, dataset_id, my_realizations
    )
    gathered_contrib = global_point_to_point_gather(
        comm, local_contrib, rank, size, tag=720
    )

    # =========================================================================
    # Part 4: Zone duration events
    # =========================================================================
    print(f"Rank {rank}: calculating zone duration events...")
    local_zone_records = []
    from methods.zone_duration_metrics import calculate_drought_zone_events
    for r in my_realizations:
        zone_series = local_data.res_level[dataset_id][r]['nyc']
        episodes = calculate_drought_zone_events(zone_series, min_end_days=7)
        for ep in episodes:
            local_zone_records.append({
                'realization_id': r,
                'start_date': ep['start_date'].isoformat(),
                'end_date': ep['end_date'].isoformat(),
                'duration_days': ep['duration_days'],
                'max_zone': ep['max_zone'],
            })
    local_zone_df = pd.DataFrame(local_zone_records)
    gathered_zone = global_point_to_point_gather(
        comm, local_zone_df, rank, size, tag=730
    )

    # =========================================================================
    # Part 5: Per-SSI-event metrics (for each SSI window)
    # =========================================================================
    gathered_event_metrics = {}
    for ssi_window in ssi_windows:
        drought_events_df = load_drought_events(dataset_id, ssi_window)
        if drought_events_df is None or len(drought_events_df) == 0:
            print(f"Rank {rank}: WARNING - No SSI-{ssi_window} drought events, skipping.")
            continue

        local_droughts = drought_events_df[
            drought_events_df['realization_id'].isin(my_realizations)
        ].copy()

        if len(local_droughts) > 0:
            local_event_metrics = calculate_all_event_metrics(
                local_data, dataset_id, local_droughts
            )
        else:
            local_event_metrics = pd.DataFrame()

        gathered = global_point_to_point_gather(
            comm, local_event_metrics, rank, size, tag=740 + ssi_window
        )
        gathered_event_metrics[ssi_window] = gathered

    # =========================================================================
    # Rank 0: combine and save all CSVs
    # =========================================================================
    if rank == 0:
        os.makedirs(PERFORMANCE_METRICS_DIR, exist_ok=True)
        os.makedirs(EVENT_METRICS_DIR, exist_ok=True)

        # Annual metrics
        combined_annual = pd.concat(gathered_annual, ignore_index=True)
        combined_annual.sort_values(['realization_id', 'water_year', 'period'],
                                    inplace=True)
        save_metrics_csv(combined_annual, dataset_id, 'annual_metrics',
                         PERFORMANCE_METRICS_DIR)
        print(f"  Annual metrics: {len(combined_annual)} rows")

        # Hashimoto simulation-level
        combined_hashimoto = pd.concat(gathered_hashimoto, ignore_index=True)
        combined_hashimoto.sort_values('realization_id', inplace=True)
        save_metrics_csv(combined_hashimoto, dataset_id, 'hashimoto_metrics',
                         PERFORMANCE_METRICS_DIR)

        # Hashimoto shortage events
        combined_hash_events = pd.concat(
            [df for df in gathered_hashimoto_events if len(df) > 0],
            ignore_index=True
        )
        if len(combined_hash_events) > 0:
            save_metrics_csv(combined_hash_events, dataset_id,
                             'hashimoto_shortage_events', PERFORMANCE_METRICS_DIR)

        # Contribution metrics
        combined_contrib = pd.concat(gathered_contrib, ignore_index=True)
        combined_contrib.sort_values(['realization_id', 'year'], inplace=True)
        save_metrics_csv(combined_contrib, dataset_id, 'contribution_metrics',
                         PERFORMANCE_METRICS_DIR)

        # Zone duration events
        combined_zone = pd.concat(
            [df for df in gathered_zone if len(df) > 0],
            ignore_index=True
        )
        if len(combined_zone) > 0:
            save_metrics_csv(combined_zone, dataset_id, 'zone_duration_events',
                             PERFORMANCE_METRICS_DIR)

        # Per-SSI-event metrics
        for ssi_window, gathered in gathered_event_metrics.items():
            all_events = pd.concat(
                [df for df in gathered if len(df) > 0],
                ignore_index=True
            )
            if len(all_events) > 0:
                event_fname = os.path.join(
                    EVENT_METRICS_DIR,
                    f'{dataset_id}_ssi{ssi_window}_event_metrics.csv'
                )
                all_events.to_csv(event_fname, index=False)
                print(f"  Saved {len(all_events)} SSI-{ssi_window} events: {event_fname}")

        print("\n" + "=" * 80)
        print("PERFORMANCE METRICS COMPLETE!")
        print("=" * 80)
        print(f"\nMetrics: {PERFORMANCE_METRICS_DIR}/")
        print(f"Events:  {EVENT_METRICS_DIR}/")


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
                    print(f"WARNING: SSI window {ssi_window} not in configured windows {SSI_WINDOWS}")
                ssi_windows.append(ssi_window)
            except ValueError:
                print(f"ERROR: Invalid SSI window '{arg}' — must be an integer")
                sys.exit(1)

    if not ssi_windows:
        print("ERROR: No valid SSI windows specified.")
        sys.exit(1)

    main(dataset_id, ssi_windows)
