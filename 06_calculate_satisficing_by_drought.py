"""
Calculate satisficing conditions during drought vs non-drought years.

Each rank independently:
  1. Reads realization IDs from HDF5 metadata (no data loaded)
  2. Loads ONLY its assigned realizations (selective HDF5 reading)
  3. Computes satisficing metrics
  4. Sends results to rank 0 via point-to-point send/recv

No global MPI collectives (bcast, gather, barrier, reduce) are used.

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
    python 06_calculate_satisficing_by_drought.py <dataset_id> --all  (serial fallback)
"""

import sys
import os
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
from methods.print_summary import print_satisficing_summary
from methods.drought_analysis import calculate_statistical_significance
from methods.save import save_satisficing_results, SATISFICING_ANALYSIS_DIR
from methods.metrics.satisficing import (
    calculate_satisficing_conditions,
    calculate_satisficing_during_droughts,
    calculate_satisficing_non_drought_periods
)


def process_ssi_window(local_data, dataset_id, ssi_window,
                        drought_events_df, my_realizations,
                        all_years_results=None):
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
    """MPI-parallelized satisficing analysis with rank-specific loading."""
    comm, rank, size = get_comm()

    if rank == 0:
        print("=" * 80)
        print(f"SATISFICING ANALYSIS (MPI): {dataset_id}")
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
    results_sets = ['res_storage', 'inflow', 'shortage', 'contribution']
    print(f"Rank {rank}: loading {len(my_realizations)} realizations from HDF5...")
    local_data = load_rank_subset_from_export(
        fname, my_realizations, results_sets, rank, size
    )
    print(f"Rank {rank}: loaded {len(my_realizations)} realizations")

    # --- Process each SSI window ---
    local_all_years = None  # reuse across SSI windows
    combined_all_years = None  # cache on rank 0 after first gather

    for ssi_window in ssi_windows:
        if rank == 0:
            print(f"\n{'=' * 60}")
            print(f"PROCESSING SSI WINDOW: {ssi_window}")
            print(f"{'=' * 60}")

        # Load drought events (small CSV, all ranks read independently)
        drought_events_df = load_drought_events(dataset_id, ssi_window)

        # Process on local realizations
        local_all_years, local_drought, local_non_drought = process_ssi_window(
            local_data, dataset_id, ssi_window,
            drought_events_df, my_realizations,
            all_years_results=local_all_years,
        )

        # Gather all-years results only on first SSI window (drought-independent)
        if combined_all_years is None:
            gathered_all = global_point_to_point_gather(
                comm, local_all_years, rank, size, tag=610
            )
        gathered_drought = global_point_to_point_gather(
            comm, local_drought, rank, size, tag=611
        )
        gathered_non_drought = global_point_to_point_gather(
            comm, local_non_drought, rank, size, tag=612
        )

        # Rank 0: combine, summarize, save
        if rank == 0:
            if combined_all_years is None:
                combined_all_years = pd.concat(gathered_all, ignore_index=True)
            all_years_results = combined_all_years
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

    comm, rank, _ = get_comm()
    if rank == 0:
        print(f"\nProcessing dataset: {dataset_id}")
        print(f"SSI windows: {ssi_windows}")

    main(dataset_id, ssi_windows)
