"""
Calculate satisficing conditions during drought vs non-drought periods.

This script addresses the question: Is there a meaningful difference between
performance outcomes during drought periods vs non-drought periods?

Analysis includes:
1. Satisficing during ALL simulation years (baseline)
2. Satisficing during SSI-identified drought periods only
3. Satisficing during non-drought periods only

Satisficing conditions:
- NYC storage >= 20% throughout evaluation period
- Montague flow target violations <= 3 consecutive days

The results are saved in a format suitable for comparative plotting and analysis.

Usage:
    python 06_calculate_satisficing_by_drought.py <dataset_id> <ssi_window>

Example:
    python 06_calculate_satisficing_by_drought.py stationary_ensemble 6
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from config import *
from methods.load import load_drought_events
from methods.verification import verify_postprocessing_output
from methods.metrics.satisficing import (
    calculate_satisficing_conditions,
    calculate_satisficing_during_droughts,
    calculate_satisficing_non_drought_periods
)

# Output directory
SATISFICING_ANALYSIS_DIR = f"{ROOT_DIR}/pywrdrb/satisficing_analysis"
os.makedirs(SATISFICING_ANALYSIS_DIR, exist_ok=True)


def print_summary_statistics(all_years_results, drought_results, non_drought_results,
                            dataset_id, ssi_window):
    """
    Print comprehensive summary statistics comparing satisficing across conditions.

    Parameters
    ----------
    all_years_results : pd.DataFrame
        Satisficing results for all years
    drought_results : pd.DataFrame
        Satisficing results during droughts
    non_drought_results : pd.DataFrame
        Satisficing results during non-drought periods
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
    print(f"{'Condition':<30} {'Total Periods':>15} {'Satisficing':>15} {'%':>10}")
    print("-" * 80)
    print(f"{'All Years (Jun-Dec)':<30} {n_all:>15,} {n_sat_all:>15,} {pct_sat_all:>9.1f}%")
    print(f"{'During Droughts':<30} {n_drought:>15,} {n_sat_drought:>15,} {pct_sat_drought:>9.1f}%")
    print(f"{'Non-Drought Years':<30} {n_non_drought:>15,} {n_sat_non_drought:>15,} {pct_sat_non_drought:>9.1f}%")
    print("-" * 80)

    # Calculate difference
    diff_drought_vs_all = pct_sat_drought - pct_sat_all
    diff_non_vs_all = pct_sat_non_drought - pct_sat_all
    diff_non_vs_drought = pct_sat_non_drought - pct_sat_drought

    print("\nCOMPARISONS:")
    print("-" * 80)
    print(f"Drought vs All Years:       {diff_drought_vs_all:+.1f} percentage points")
    print(f"Non-Drought vs All Years:   {diff_non_vs_all:+.1f} percentage points")
    print(f"Non-Drought vs Drought:     {diff_non_vs_drought:+.1f} percentage points")
    print("-" * 80)

    # Failure breakdown
    print("\nFAILURE BREAKDOWN BY CONDITION:")
    print("-" * 80)

    for results, label in [
        (all_years_results, "All Years"),
        (drought_results, "During Droughts"),
        (non_drought_results, "Non-Drought Years")
    ]:
        if len(results) == 0:
            continue

        n_total = len(results)
        storage_fail = results['min_storage_pct'] < 20
        montague_fail = results['max_violation_days'] > 3
        both_fail = storage_fail & montague_fail

        print(f"\n{label}:")
        print(f"  Storage < 20% only:        {(storage_fail & ~montague_fail).sum():>6,} "
              f"({100*(storage_fail & ~montague_fail).sum()/n_total:>5.1f}%)")
        print(f"  Montague > 3 days only:    {(montague_fail & ~storage_fail).sum():>6,} "
              f"({100*(montague_fail & ~storage_fail).sum()/n_total:>5.1f}%)")
        print(f"  Both failures:             {both_fail.sum():>6,} "
              f"({100*both_fail.sum()/n_total:>5.1f}%)")
        print(f"  Total non-satisficing:     {(~results['satisficing']).sum():>6,} "
              f"({100*(~results['satisficing']).sum()/n_total:>5.1f}%)")

    # Storage and violation statistics
    print("\n" + "=" * 80)
    print("DETAILED METRICS:")
    print("=" * 80)

    metrics_summary = []
    for results, label in [
        (all_years_results, "All Years"),
        (drought_results, "During Droughts"),
        (non_drought_results, "Non-Drought Years")
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


def calculate_statistical_significance(drought_results, non_drought_results):
    """
    Perform statistical tests to determine if differences are significant.

    Parameters
    ----------
    drought_results : pd.DataFrame
        Satisficing results during droughts
    non_drought_results : pd.DataFrame
        Satisficing results during non-drought periods

    Returns
    -------
    dict
        Statistical test results
    """
    from scipy.stats import chi2_contingency, mannwhitneyu

    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)

    results = {}

    # Chi-square test for satisficing rates
    satisficing_contingency = np.array([
        [drought_results['satisficing'].sum(), (~drought_results['satisficing']).sum()],
        [non_drought_results['satisficing'].sum(), (~non_drought_results['satisficing']).sum()]
    ])

    chi2, p_value, dof, expected = chi2_contingency(satisficing_contingency)
    results['chi2_satisficing'] = {
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof
    }

    print("\nChi-Square Test: Satisficing Rates (Drought vs Non-Drought)")
    print("-" * 80)
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"p-value: {p_value:.4e}")
    print(f"Degrees of freedom: {dof}")
    if p_value < 0.001:
        print("Result: HIGHLY SIGNIFICANT (p < 0.001)")
    elif p_value < 0.01:
        print("Result: VERY SIGNIFICANT (p < 0.01)")
    elif p_value < 0.05:
        print("Result: SIGNIFICANT (p < 0.05)")
    else:
        print("Result: NOT SIGNIFICANT (p >= 0.05)")

    # Mann-Whitney U test for storage levels
    u_stat_storage, p_value_storage = mannwhitneyu(
        drought_results['min_storage_pct'],
        non_drought_results['min_storage_pct'],
        alternative='two-sided'
    )

    results['mannwhitney_storage'] = {
        'u_statistic': u_stat_storage,
        'p_value': p_value_storage
    }

    print("\nMann-Whitney U Test: Minimum Storage Levels")
    print("-" * 80)
    print(f"U statistic: {u_stat_storage:.4f}")
    print(f"p-value: {p_value_storage:.4e}")
    if p_value_storage < 0.001:
        print("Result: HIGHLY SIGNIFICANT (p < 0.001)")
    elif p_value_storage < 0.01:
        print("Result: VERY SIGNIFICANT (p < 0.01)")
    elif p_value_storage < 0.05:
        print("Result: SIGNIFICANT (p < 0.05)")
    else:
        print("Result: NOT SIGNIFICANT (p >= 0.05)")

    # Mann-Whitney U test for violation days
    u_stat_violations, p_value_violations = mannwhitneyu(
        drought_results['max_violation_days'],
        non_drought_results['max_violation_days'],
        alternative='two-sided'
    )

    results['mannwhitney_violations'] = {
        'u_statistic': u_stat_violations,
        'p_value': p_value_violations
    }

    print("\nMann-Whitney U Test: Maximum Violation Days")
    print("-" * 80)
    print(f"U statistic: {u_stat_violations:.4f}")
    print(f"p-value: {p_value_violations:.4e}")
    if p_value_violations < 0.001:
        print("Result: HIGHLY SIGNIFICANT (p < 0.001)")
    elif p_value_violations < 0.01:
        print("Result: VERY SIGNIFICANT (p < 0.01)")
    elif p_value_violations < 0.05:
        print("Result: SIGNIFICANT (p < 0.05)")
    else:
        print("Result: NOT SIGNIFICANT (p >= 0.05)")

    print("=" * 80)

    return results


def save_results(all_years_results, drought_results, non_drought_results,
                dataset_id, ssi_window):
    """
    Save results to CSV files.

    Parameters
    ----------
    all_years_results : pd.DataFrame
        All years results
    drought_results : pd.DataFrame
        Drought period results
    non_drought_results : pd.DataFrame
        Non-drought period results
    dataset_id : str
        Dataset identifier
    ssi_window : int
        SSI window
    """
    print("\n" + "=" * 80)
    print("SAVING RESULTS TO CSV")
    print("=" * 80)

    # Save individual results
    fnames = []

    fname = f"{SATISFICING_ANALYSIS_DIR}/{dataset_id}_ssi{ssi_window}_all_years.csv"
    all_years_results.to_csv(fname, index=False)
    fnames.append(fname)
    print(f"Saved: {fname}")

    fname = f"{SATISFICING_ANALYSIS_DIR}/{dataset_id}_ssi{ssi_window}_during_droughts.csv"
    drought_results.to_csv(fname, index=False)
    fnames.append(fname)
    print(f"Saved: {fname}")

    fname = f"{SATISFICING_ANALYSIS_DIR}/{dataset_id}_ssi{ssi_window}_non_drought.csv"
    non_drought_results.to_csv(fname, index=False)
    fnames.append(fname)
    print(f"Saved: {fname}")

    # Create combined dataset with condition label
    all_years_labeled = all_years_results.copy()
    all_years_labeled['condition'] = 'all_years'

    drought_labeled = drought_results.copy()
    drought_labeled['condition'] = 'drought'

    non_drought_labeled = non_drought_results.copy()
    non_drought_labeled['condition'] = 'non_drought'

    combined = pd.concat([all_years_labeled, drought_labeled, non_drought_labeled],
                        ignore_index=True)

    fname = f"{SATISFICING_ANALYSIS_DIR}/{dataset_id}_ssi{ssi_window}_combined.csv"
    combined.to_csv(fname, index=False)
    fnames.append(fname)
    print(f"Saved: {fname}")

    print("=" * 80)

    return fnames


def main(dataset_id, ssi_window):
    """
    Main function to calculate satisficing by drought condition.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    ssi_window : int
        SSI window (3, 6, or 12 months)
    """
    print("=" * 80)
    print(f"SATISFICING ANALYSIS: {dataset_id}, SSI-{ssi_window}")
    print("=" * 80)

    # Verify postprocessed data exists
    verify_postprocessing_output(dataset_id)

    # Load postprocessed simulation data
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
    print(f"\nLoading postprocessed data from: {fname}")
    data = pywrdrb.Data()
    data.load_from_export(fname, results_sets=['res_storage', 'inflow', 'shortage', 'contribution'])
    print("  Data loaded successfully")

    # Load drought events
    drought_events_df = load_drought_events(dataset_id, ssi_window)

    # 1. Calculate satisficing for ALL simulation years (baseline)
    print("\n" + "=" * 80)
    print("CALCULATING: All Years (Jun-Dec)")
    print("=" * 80)
    all_years_results = calculate_satisficing_conditions(
        data, dataset_id,
        period_type='year',
        evaluate_all_years=True,
        storage_threshold=20.0,
        violation_days=3
    )
    print(f"  Evaluated {len(all_years_results)} year-realization pairs")

    # 2. Calculate satisficing during drought periods
    print("\n" + "=" * 80)
    print("CALCULATING: During Drought Periods")
    print("=" * 80)
    drought_results = calculate_satisficing_during_droughts(
        data, dataset_id, drought_events_df,
        storage_threshold=20.0,
        violation_days=3
    )
    print(f"  Evaluated {len(drought_results)} drought events")

    # 3. Calculate satisficing during non-drought periods
    print("\n" + "=" * 80)
    print("CALCULATING: Non-Drought Periods")
    print("=" * 80)
    non_drought_results = calculate_satisficing_non_drought_periods(
        data, dataset_id, drought_events_df,
        storage_threshold=20.0,
        violation_days=3
    )
    print(f"  Evaluated {len(non_drought_results)} non-drought periods")

    # Print summary statistics
    print_summary_statistics(all_years_results, drought_results, non_drought_results,
                            dataset_id, ssi_window)

    # Statistical significance tests
    if len(drought_results) > 0 and len(non_drought_results) > 0:
        stat_results = calculate_statistical_significance(drought_results, non_drought_results)

    # Save results
    save_results(all_years_results, drought_results, non_drought_results,
                dataset_id, ssi_window)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {SATISFICING_ANALYSIS_DIR}/")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        print(f"\nAvailable datasets: {list(DATASET_CONFIGS.keys())}")
        print(f"Available SSI windows: {SSI_WINDOWS}")
        sys.exit(1)

    dataset_id = sys.argv[1]
    ssi_window = int(sys.argv[2])

    # Validate inputs
    verify_dataset_id(dataset_id)
    if ssi_window not in SSI_WINDOWS:
        print(f"ERROR: Invalid SSI window. Must be one of {SSI_WINDOWS}")
        sys.exit(1)

    main(dataset_id, ssi_window)
