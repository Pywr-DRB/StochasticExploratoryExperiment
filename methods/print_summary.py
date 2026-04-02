"""
Print summary utilities for experiment configuration and results.

This module provides functions for printing formatted summaries of
experiment configurations, ensemble sets, and analysis results.

All print_*() summary functions should be imported from this module.
"""


def print_satisficing_summary(df, dataset_id, ssi_window):
    """
    Print comprehensive summary statistics comparing satisficing across conditions.

    Parameters
    ----------
    df : pd.DataFrame
        Annual satisficing DataFrame with n_droughts_in_year column.
    dataset_id : str
        Dataset identifier
    ssi_window : int
        SSI window
    """
    import pandas as pd

    drought_years = df[df['n_droughts_in_year'] > 0]
    non_drought_years = df[df['n_droughts_in_year'] == 0]

    print("\n" + "=" * 80)
    print(f"SATISFICING ANALYSIS SUMMARY: {dataset_id}, SSI-{ssi_window}")
    print("=" * 80)

    subsets = [
        (df, "All Years"),
        (drought_years, "Years with Droughts"),
        (non_drought_years, "Years without Droughts"),
    ]

    # Satisficing rates
    print("\nOVERALL SATISFICING RATES:")
    print("-" * 80)
    print(f"{'Condition':<30} {'Total Years':>15} {'Satisficing':>15} {'%':>10}")
    print("-" * 80)

    pcts = {}
    for subset, label in subsets:
        n = len(subset)
        n_sat = subset['satisficing'].sum() if n > 0 else 0
        pct = 100 * n_sat / n if n > 0 else 0
        pcts[label] = pct
        print(f"{label:<30} {n:>15,} {n_sat:>15,} {pct:>9.1f}%")
    print("-" * 80)

    # Comparisons
    print("\nCOMPARISONS:")
    print("-" * 80)
    print(f"Years with Droughts vs All Years:       {pcts['Years with Droughts'] - pcts['All Years']:+.1f} percentage points")
    print(f"Years without Droughts vs All Years:    {pcts['Years without Droughts'] - pcts['All Years']:+.1f} percentage points")
    print(f"Years without vs with Droughts:         {pcts['Years without Droughts'] - pcts['Years with Droughts']:+.1f} percentage points")
    print("-" * 80)

    # Failure breakdown
    print("\nFAILURE BREAKDOWN BY CONDITION:")
    print("-" * 80)

    for subset, label in subsets:
        if len(subset) == 0:
            continue

        n_total = len(subset)
        storage_fail = subset['nyc_min_storage_pct'] < 20
        montague_fail = subset['montague_max_consec_shortage_days'] > 3

        print(f"\n{label}:")
        print(f"  Storage < 20% only:        {(storage_fail & ~montague_fail).sum():>6,} "
              f"({100 * (storage_fail & ~montague_fail).sum() / n_total:>5.1f}%)")
        print(f"  Montague fail only:        {(~storage_fail & montague_fail).sum():>6,} "
              f"({100 * (~storage_fail & montague_fail).sum() / n_total:>5.1f}%)")
        print(f"  Both failures:             {(storage_fail & montague_fail).sum():>6,} "
              f"({100 * (storage_fail & montague_fail).sum() / n_total:>5.1f}%)")
        print(f"  Satisficing:               {(~storage_fail & ~montague_fail).sum():>6,} "
              f"({100 * (~storage_fail & ~montague_fail).sum() / n_total:>5.1f}%)")

    # Detailed metrics
    print("\n" + "=" * 80)
    print("DETAILED METRICS:")
    print("=" * 80)

    metrics_summary = []
    for subset, label in subsets:
        if len(subset) == 0:
            continue
        record = {
            'Condition': label,
            'Mean Min Storage (%)': subset['nyc_min_storage_pct'].mean(),
            'Median Min Storage (%)': subset['nyc_min_storage_pct'].median(),
            'Mean Max Shortage (days)': subset['montague_max_consec_shortage_days'].mean(),
            'Median Max Shortage (days)': subset['montague_max_consec_shortage_days'].median(),
        }
        metrics_summary.append(record)

    metrics_df = pd.DataFrame(metrics_summary)
    print("\n" + metrics_df.to_string(index=False))

    print("\n" + "=" * 80)


def print_prep_status(dataset_id):
    """
    Print detailed status of Pywr-DRB input preparation for all ensemble sets.

    Unlike verify_prep_outputs() in methods.verification (which just checks existence),
    this function provides detailed per-set status reporting after the prep step.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier to report on

    Returns
    -------
    bool
        True if all sets are properly prepared
    """
    import os
    from methods.config import N_ENSEMBLE_SETS
    from methods.ensemble_utils import get_ensemble_set_spec
    from methods.verification import verify_dataset_id

    verify_dataset_id(dataset_id)
    print(f"\nVerifying Pywr-DRB input preparation for {dataset_id}...")

    all_prepared = True
    successful_sets = []
    failed_sets = []
    missing_files = {}

    # Required files to check
    required_files = [
        'predicted_inflow',
        'diversion_nyc',
        'diversion_nj',
        'predicted_diversions'
    ]

    for set_id in range(N_ENSEMBLE_SETS):
        set_spec = get_ensemble_set_spec(set_id, dataset_id)
        set_complete = True
        missing_in_set = []

        for file_key in required_files:
            fname = set_spec.files[file_key]
            if not os.path.exists(fname):
                set_complete = False
                missing_in_set.append(file_key)
                all_prepared = False

        if set_complete:
            successful_sets.append(set_id + 1)
        else:
            failed_sets.append(set_id + 1)
            missing_files[set_id + 1] = missing_in_set

    if all_prepared:
        print(f"SUCCESS: All {N_ENSEMBLE_SETS} ensemble sets properly prepared!")
        print(f"  All required files present:")
        for file_key in required_files:
            print(f"    - {file_key}")
    else:
        print(f"WARNING: {len(failed_sets)} sets not properly prepared: {failed_sets}")
        print(f"Successfully prepared: {len(successful_sets)} sets")
        print(f"\nMissing files by set:")
        for set_id, missing in missing_files.items():
            print(f"  Set {set_id}: {missing}")

    return all_prepared


def print_simulation_status(dataset_id):
    """
    Print detailed status of Pywr-DRB simulations for all ensemble sets.

    Unlike verify_simulation_outputs() in methods.verification (which just checks existence),
    this function provides detailed per-set validation including file size and content checks.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier to report on

    Returns
    -------
    bool
        True if all sets have valid simulation outputs
    """
    import os
    import pywrdrb
    from methods.config import N_ENSEMBLE_SETS, N_REALIZATIONS_PER_ENSEMBLE_SET
    from methods.ensemble_utils import get_ensemble_set_spec
    from methods.verification import verify_dataset_id

    verify_dataset_id(dataset_id)
    print(f"\nVerifying Pywr-DRB simulation outputs for {dataset_id}...")

    all_completed = True
    successful_sets = []
    failed_sets = []

    for set_id in range(N_ENSEMBLE_SETS):
        set_spec = get_ensemble_set_spec(set_id, dataset_id)

        if not os.path.exists(set_spec.output_file):
            print(f"FAIL:  Set {set_id + 1}: Output file not found")
            all_completed = False
            failed_sets.append(set_id + 1)
            continue

        # Check file size (basic validation)
        file_size = os.path.getsize(set_spec.output_file)
        if file_size < 1024:  # Less than 1KB is suspicious
            print(f"FAIL:  Set {set_id + 1}: Output file too small ({file_size} bytes)")
            all_completed = False
            failed_sets.append(set_id + 1)
            continue

        # Try to load with Pywr-DRB to verify format (optimization: only check first and last)
        if set_id == 0 or set_id == N_ENSEMBLE_SETS - 1:
            try:
                test_data = pywrdrb.Data(results_sets=["major_flow"])
                test_data.load_output(output_filenames=[set_spec.output_file])
                n_realizations = len(list(test_data.major_flow.values())[0])

                if n_realizations != N_REALIZATIONS_PER_ENSEMBLE_SET:
                    print(f"WARNING: Set {set_id + 1}: Expected {N_REALIZATIONS_PER_ENSEMBLE_SET} realizations, found {n_realizations}")
                    failed_sets.append(set_id + 1)
                else:
                    print(f"SUCCESS: Set {set_id + 1}: Valid output ({n_realizations} realizations, {file_size//1024//1024} MB)")
                    successful_sets.append(set_id + 1)

            except Exception as e:
                print(f"FAIL:  Set {set_id + 1}: Error loading output file - {str(e)}")
                all_completed = False
                failed_sets.append(set_id + 1)
        else:
            # For middle sets, just check file size
            print(f"SUCCESS: Set {set_id + 1}: Output exists ({file_size//1024//1024} MB)")
            successful_sets.append(set_id + 1)

    if all_completed:
        print(f"SUCCESS: All {N_ENSEMBLE_SETS} ensemble sets have valid simulation outputs!")
    else:
        print(f"WARNING: {len(failed_sets)} sets have invalid outputs: {failed_sets}")

    return all_completed
