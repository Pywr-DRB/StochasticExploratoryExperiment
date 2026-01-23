"""
Print summary utilities for experiment configuration and results.

This module provides functions for printing formatted summaries of
experiment configurations, ensemble sets, and analysis results.

All print_*() summary functions should be imported from this module.
"""


def print_experiment_summary(dataset_id):
    """
    Print comprehensive experiment configuration summary.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    """
    # Import here to avoid circular imports
    from methods.config import (
        DATASET_CONFIGS, TOTAL_REALIZATIONS, N_ENSEMBLE_SETS,
        N_REALIZATIONS_PER_ENSEMBLE_SET, N_YEARS, START_DATE, END_DATE,
        N_PYWRDRB_BATCHES_PER_SET, N_REALIZATIONS_PER_PYWRDRB_BATCH,
        pywrdrb_nodes_to_generate, pywrdrb_nodes_to_regress,
        get_existing_ensemble_sets
    )
    from methods.verification import verify_dataset_id

    verify_dataset_id(dataset_id)
    dataset_config = DATASET_CONFIGS[dataset_id]
    generated_sets = get_existing_ensemble_sets(dataset_id)

    print("=" * 80)
    print("ENSEMBLE EXPERIMENT CONFIGURATION")
    print("=" * 80)
    print(f"Dataset ID: {dataset_id}")
    print(f"Dataset Type: {dataset_config['type']}")
    print(f"Description: {dataset_config['description']}")
    if dataset_config['type'] == 'climate_adjusted':
        print(f"Monthly % Changes: {dataset_config['monthly_prc_change']}")
    print()
    print(f"Total Realizations: {TOTAL_REALIZATIONS:,}")
    print(f"Ensemble Sets: {N_ENSEMBLE_SETS}")
    print(f"Realizations per Set: {N_REALIZATIONS_PER_ENSEMBLE_SET}")
    print(f"Years per Realization: {N_YEARS}")
    print(f"Simulation Period: {START_DATE} to {END_DATE}")
    print()
    print("Pywr-DRB Batching:")
    print(f"  Batches per Set: {N_PYWRDRB_BATCHES_PER_SET}")
    print(f"  Realizations per Batch: {N_REALIZATIONS_PER_PYWRDRB_BATCH}")
    print()
    print("Node Configuration:")
    print(f"  Nodes to Generate (KN): {len(pywrdrb_nodes_to_generate)}")
    print(f"  Nodes to Regress: {len(pywrdrb_nodes_to_regress)}")
    print()
    print("File Structure:")
    for i, spec in enumerate(generated_sets):
        print(f"  Set {i+1}: {spec.directory}")
        if i >= 2:  # Limit output for large experiments
            print(f"  ... (and {len(generated_sets)-3} more sets)")
            break
    print("=" * 80)


def print_ensemble_set_summary(set_id, dataset_id):
    """
    Print summary for a specific ensemble set.

    Parameters
    ----------
    set_id : int
        Ensemble set ID (0-indexed)
    dataset_id : str
        Dataset identifier
    """
    from methods.config import get_ensemble_set_spec

    spec = get_ensemble_set_spec(set_id, dataset_id)
    print(f"\n{dataset_id} Ensemble Set {set_id + 1} Summary:")
    print(f"  Dataset Type: {spec.ensemble_type}")
    print(f"  Global Realizations: {spec.start_realization}-{spec.end_realization-1}")
    print(f"  Directory: {spec.directory}")
    print(f"  Pywr-DRB Batches: {len(spec.pywrdrb_batches)}")
    print(f"  Output File: {spec.output_file}")


def print_satisficing_summary(all_years_results, drought_results, non_drought_results,
                              dataset_id, ssi_window):
    """
    Print comprehensive summary statistics comparing satisficing across conditions.

    Parameters
    ----------
    all_years_results : pd.DataFrame
        Satisficing results for all years
    drought_results : pd.DataFrame
        Satisficing results for years with drought events
    non_drought_results : pd.DataFrame
        Satisficing results for years without drought events
    dataset_id : str
        Dataset identifier
    ssi_window : int
        SSI window
    """
    print("\n" + "=" * 80)
    print(f"SATISFICING ANALYSIS SUMMARY: {dataset_id}, SSI-{ssi_window}")
    print("=" * 80)

    # Calculate satisficing percentages
    n_all = len(all_years_results)
    n_sat_all = all_years_results['satisficing'].sum()
    pct_sat_all = 100 * n_sat_all / n_all if n_all > 0 else 0

    n_drought = len(drought_results)
    n_sat_drought = drought_results['satisficing'].sum()
    pct_sat_drought = 100 * n_sat_drought / n_drought if n_drought > 0 else 0

    n_non_drought = len(non_drought_results)
    n_sat_non_drought = non_drought_results['satisficing'].sum()
    pct_sat_non_drought = 100 * n_sat_non_drought / n_non_drought if n_non_drought > 0 else 0

    print("\nOVERALL SATISFICING RATES:")
    print("-" * 80)
    print(f"{'Condition':<30} {'Total Years':>15} {'Satisficing':>15} {'%':>10}")
    print("-" * 80)
    print(f"{'All Years (Jun-Dec)':<30} {n_all:>15,} {n_sat_all:>15,} {pct_sat_all:>9.1f}%")
    print(f"{'Years with Droughts':<30} {n_drought:>15,} {n_sat_drought:>15,} {pct_sat_drought:>9.1f}%")
    print(f"{'Years without Droughts':<30} {n_non_drought:>15,} {n_sat_non_drought:>15,} {pct_sat_non_drought:>9.1f}%")
    print("-" * 80)

    # Calculate difference
    diff_drought_vs_all = pct_sat_drought - pct_sat_all
    diff_non_vs_all = pct_sat_non_drought - pct_sat_all
    diff_non_vs_drought = pct_sat_non_drought - pct_sat_drought

    print("\nCOMPARISONS:")
    print("-" * 80)
    print(f"Years with Droughts vs All Years:       {diff_drought_vs_all:+.1f} percentage points")
    print(f"Years without Droughts vs All Years:    {diff_non_vs_all:+.1f} percentage points")
    print(f"Years without vs with Droughts:         {diff_non_vs_drought:+.1f} percentage points")
    print("-" * 80)

    # Failure breakdown
    print("\nFAILURE BREAKDOWN BY CONDITION:")
    print("-" * 80)

    for results, label in [
        (all_years_results, "All Years"),
        (drought_results, "Years with Droughts"),
        (non_drought_results, "Years without Droughts")
    ]:
        if len(results) == 0:
            continue

        n_total = len(results)
        storage_fail = results['min_storage_pct'] < 20
        montague_fail = results['max_violation_days'] > 3

        print(f"\n{label}:")
        print(f"  Storage < 20% only:        {(storage_fail & ~montague_fail).sum():>6,} "
              f"({100 * (storage_fail & ~montague_fail).sum() / n_total:>5.1f}%)")
        print(f"  Montague fail only:        {(~storage_fail & montague_fail).sum():>6,} "
              f"({100 * (~storage_fail & montague_fail).sum() / n_total:>5.1f}%)")
        print(f"  Both failures:             {(storage_fail & montague_fail).sum():>6,} "
              f"({100 * (storage_fail & montague_fail).sum() / n_total:>5.1f}%)")
        print(f"  Satisficing:               {(~storage_fail & ~montague_fail).sum():>6,} "
              f"({100 * (~storage_fail & ~montague_fail).sum() / n_total:>5.1f}%)")

    # Detailed metrics section
    import pandas as pd
    print("\n" + "=" * 80)
    print("DETAILED METRICS:")
    print("=" * 80)

    metrics_summary = []
    for results, label in [
        (all_years_results, "All Years"),
        (drought_results, "Years with Droughts"),
        (non_drought_results, "Years without Droughts")
    ]:
        if len(results) == 0:
            continue

        metrics_summary.append({
            'Condition': label,
            'Mean Min Storage (%)': results['min_storage_pct'].mean(),
            'Median Min Storage (%)': results['min_storage_pct'].median(),
            'Mean Max Violations (days)': results['max_violation_days'].mean(),
            'Median Max Violations (days)': results['max_violation_days'].median(),
            'Mean NYC Inflow (MG)': results['nyc_inflow'].mean(),
            'Mean Montague Contrib (MG)': results['montague_contrib'].mean()
        })

    metrics_df = pd.DataFrame(metrics_summary)
    print("\n" + metrics_df.to_string(index=False))

    print("\n" + "=" * 80)


def print_performance_metrics_summary(metrics_df):
    """
    Print summary statistics for performance metrics.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Performance metrics DataFrame
    """
    print(f"\n  Key Performance Metrics Summary:")
    print(f"  {'='*60}")

    count_metrics = ['years_reliable', 'years_high_storage', 'years_above_20pct',
                     'years_low_carryover', 'years_trenton_reliable']
    for metric in count_metrics:
        if metric in metrics_df.columns:
            p5 = metrics_df[metric].quantile(0.05)
            p50 = metrics_df[metric].quantile(0.50)
            p95 = metrics_df[metric].quantile(0.95)
            print(f"    {metric:40s}: p5={p5:5.1f}, p50={p50:5.1f}, p95={p95:5.1f}")

    print(f"\n  Other Metrics Summary:")
    print(f"  {'='*60}")
    other_metrics = ['pct_days_nyc_diversion_shortage', 'max_consecutive_drought_days',
                     'mean_sept1_storage_pct', 'mean_annual_nyc_contribution_mg']
    for metric in other_metrics:
        if metric in metrics_df.columns:
            p5 = metrics_df[metric].quantile(0.05)
            p50 = metrics_df[metric].quantile(0.50)
            p95 = metrics_df[metric].quantile(0.95)
            if 'pct' in metric or 'storage' in metric:
                print(f"    {metric:40s}: p5={p5:5.1f}, p50={p50:5.1f}, p95={p95:5.1f}")
            else:
                print(f"    {metric:40s}: p5={p5:5.0f}, p50={p50:5.0f}, p95={p95:5.0f}")

    print(f"\n  Drought Zone & Shortage Metrics Summary:")
    print(f"  {'='*60}")
    zone_metrics = ['years_drought_emergency', 'years_drought_watch', 'years_drought_warning',
                    'max_1day_montague_shortage_mg', 'max_3day_montague_shortage_mg', 'max_7day_montague_shortage_mg']
    for metric in zone_metrics:
        if metric in metrics_df.columns:
            p5 = metrics_df[metric].quantile(0.05)
            p50 = metrics_df[metric].quantile(0.50)
            p95 = metrics_df[metric].quantile(0.95)
            print(f"    {metric:40s}: p5={p5:5.1f}, p50={p50:5.1f}, p95={p95:5.1f}")


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
    from methods.config import N_ENSEMBLE_SETS, get_ensemble_set_spec
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
    from methods.config import N_ENSEMBLE_SETS, N_REALIZATIONS_PER_ENSEMBLE_SET, get_ensemble_set_spec
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
