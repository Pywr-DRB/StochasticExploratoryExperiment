"""
Calculate satisficing conditions during drought vs non-drought years.

This script addresses the question: Is there a meaningful difference between
performance outcomes during years with droughts vs years without droughts?

Analysis includes:
1. Satisficing during ALL simulation years (baseline)
2. Satisficing during years with some drought events
3. Satisficing during years with no drought events

Satisficing conditions:
- NYC storage >= 20% throughout evaluation period (Jun-Dec)
- Montague flow target violations <= 3 consecutive days

The results are saved in a format suitable for comparative plotting and analysis.

Usage:
    python 06_calculate_satisficing_by_drought.py <dataset_id> [ssi_windows...]
    python 06_calculate_satisficing_by_drought.py <dataset_id> --all

Examples:
    python 06_calculate_satisficing_by_drought.py stationary_ensemble 6
    python 06_calculate_satisficing_by_drought.py stationary_ensemble 3 6 12
    python 06_calculate_satisficing_by_drought.py stationary_ensemble --all
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from methods.config import *
from methods.load import load_drought_events
from methods.verification import verify_postprocessing_output
from methods.print_summary import print_satisficing_summary
from methods.save import save_satisficing_results, SATISFICING_ANALYSIS_DIR
from methods.metrics.satisficing import (
    calculate_satisficing_conditions,
    calculate_satisficing_during_droughts,
    calculate_satisficing_non_drought_periods
)


def calculate_statistical_significance(drought_results, non_drought_results):
    """
    Perform statistical tests to determine if differences are significant.

    Parameters
    ----------
    drought_results : pd.DataFrame
        Satisficing results for years with droughts
    non_drought_results : pd.DataFrame
        Satisficing results for years without droughts

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

    print("\nChi-Square Test: Satisficing Rates (Years with vs without Droughts)")
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


def process_ssi_window(data, dataset_id, ssi_window, all_years_results=None):
    """
    Process a single SSI window for the given dataset.

    Parameters
    ----------
    data : pywrdrb.Data
        Pre-loaded data object
    dataset_id : str
        Dataset identifier
    ssi_window : int
        SSI window (3, 6, or 12 months)
    all_years_results : pd.DataFrame, optional
        Pre-computed all years results (to avoid recomputation)
    """
    print("\n" + "=" * 80)
    print(f"PROCESSING SSI WINDOW: {ssi_window}")
    print("=" * 80)

    # Load drought events for this SSI window
    drought_events_df = load_drought_events(dataset_id, ssi_window)

    # 1. Calculate satisficing for ALL simulation years (only once if not provided)
    if all_years_results is None:
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
    else:
        print("\n" + "=" * 80)
        print("REUSING: All Years (Jun-Dec) results")
        print("=" * 80)
        print(f"  Using {len(all_years_results)} pre-computed year-realization pairs")

    # 2. Calculate satisficing for years with drought events
    print("\n" + "=" * 80)
    print("CALCULATING: Years with Drought Events")
    print("=" * 80)
    drought_results = calculate_satisficing_during_droughts(
        data, dataset_id, drought_events_df,
        storage_threshold=20.0,
        violation_days=3
    )
    print(f"  Evaluated {len(drought_results)} year-realization pairs with droughts")

    # 3. Calculate satisficing for years without drought events
    print("\n" + "=" * 80)
    print("CALCULATING: Years without Drought Events")
    print("=" * 80)
    non_drought_results = calculate_satisficing_non_drought_periods(
        data, dataset_id, drought_events_df,
        storage_threshold=20.0,
        violation_days=3
    )
    print(f"  Evaluated {len(non_drought_results)} year-realization pairs without droughts")

    # Print summary statistics
    print_satisficing_summary(all_years_results, drought_results, non_drought_results,
                              dataset_id, ssi_window)

    # Statistical significance tests
    if len(drought_results) > 0 and len(non_drought_results) > 0:
        stat_results = calculate_statistical_significance(drought_results, non_drought_results)

    # Save results
    save_satisficing_results(all_years_results, drought_results, non_drought_results,
                             dataset_id, ssi_window)

    return all_years_results


def main(dataset_id, ssi_windows):
    """
    Main function to calculate satisficing by drought condition for multiple SSI windows.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    ssi_windows : list of int
        SSI windows to process (e.g., [3, 6, 12])
    """
    print("=" * 80)
    print(f"SATISFICING ANALYSIS: {dataset_id}")
    print(f"SSI Windows: {ssi_windows}")
    print("=" * 80)

    # Verify postprocessed data exists
    verify_postprocessing_output(dataset_id)

    # Load postprocessed simulation data ONCE for all SSI windows
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
    print(f"\nLoading postprocessed data from: {fname}")
    print("  (This data will be reused for all SSI windows)")
    data = pywrdrb.Data()
    data.load_from_export(fname, results_sets=['res_storage', 'inflow', 'shortage', 'contribution'])
    print("  Data loaded successfully")

    # Process each SSI window
    all_years_results = None
    for ssi_window in ssi_windows:
        try:
            all_years_results = process_ssi_window(data, dataset_id, ssi_window, all_years_results)
        except Exception as e:
            print(f"\nERROR processing SSI window {ssi_window}: {e}")
            import traceback
            traceback.print_exc()
            continue

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
        # Parse individual SSI windows from command line
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

    print(f"\nProcessing dataset: {dataset_id}")
    print(f"SSI windows: {ssi_windows}")

    main(dataset_id, ssi_windows)
