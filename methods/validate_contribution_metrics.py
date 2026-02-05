"""
Validation script for pre-computed contribution analysis metrics.

This script compares pre-computed metrics (from CSV) against on-the-fly calculations
(from HDF5) to ensure accuracy and consistency.

Usage:
    python methods/validate_contribution_metrics.py [dataset_id] [--sample-size N]

Examples:
    python methods/validate_contribution_metrics.py stationary_ensemble
    python methods/validate_contribution_metrics.py --all
    python methods/validate_contribution_metrics.py stationary_ensemble --sample-size 5
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pywrdrb
from methods.load_contribution_metrics import load_contribution_metrics
from methods.postprocess import calculate_contribution_analysis_metrics
from methods.config import DATASET_CONFIGS


def validate_metrics(dataset_id, sample_size=10, tolerance=1e-6, verbose=True):
    """
    Compare cached metrics with on-the-fly calculation for random sample.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    sample_size : int
        Number of realizations to validate (default: 10)
    tolerance : float
        Numerical tolerance for comparisons (default: 1e-6)
    verbose : bool
        Print detailed comparison output (default: True)

    Returns
    -------
    bool
        True if all metrics match within tolerance, False otherwise
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"VALIDATING: {dataset_id}")
        print(f"{'='*80}")

    # Load cached metrics
    try:
        cached_df = load_contribution_metrics(dataset_id)
        if verbose:
            print(f"✓ Loaded cached metrics: {len(cached_df)} year-realization pairs")
    except FileNotFoundError as e:
        print(f"✗ ERROR: {e}")
        return False

    # Load HDF5 data for recalculation
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
    if not os.path.exists(fname):
        print(f"✗ ERROR: HDF5 file not found: {fname}")
        return False

    if verbose:
        print(f"✓ Loading HDF5 data from: {fname}")

    data = pywrdrb.Data()
    data.load_from_export(
        fname,
        results_sets=['res_level', 'res_storage', 'contribution', 'inflow',
                     'ibt_diversions', 'ibt_demands']
    )

    # Get available realizations
    all_realizations = list(data.res_level[dataset_id].keys())

    # Sample realizations for validation
    if sample_size > len(all_realizations):
        sample_size = len(all_realizations)
        if verbose:
            print(f"  Reduced sample size to {sample_size} (all available realizations)")

    sample_realizations = sorted(np.random.choice(
        all_realizations, sample_size, replace=False
    ))

    if verbose:
        print(f"✓ Sampled {sample_size} realizations for validation: {sample_realizations}")
        print(f"\nRecalculating metrics for sample...")

    # Recalculate metrics for sample
    recalc_df = calculate_contribution_analysis_metrics(data, dataset_id, sample_realizations)

    if verbose:
        print(f"✓ Recalculated {len(recalc_df)} year-realization pairs\n")
        print(f"Comparing metrics...")

    # Compare metrics
    passed = True
    total_comparisons = 0
    failed_comparisons = 0

    for r in sample_realizations:
        cached_subset = cached_df[cached_df['realization_id'] == r].copy()
        recalc_subset = recalc_df[recalc_df['realization_id'] == r].copy()

        if len(cached_subset) == 0:
            print(f"  ✗ MISSING: Realization {r} not found in cached data")
            passed = False
            continue

        if len(cached_subset) != len(recalc_subset):
            print(f"  ✗ LENGTH MISMATCH: Realization {r} has {len(cached_subset)} cached rows vs {len(recalc_subset)} recalculated")
            passed = False
            continue

        # Sort both by year for proper comparison
        cached_subset = cached_subset.sort_values('year').reset_index(drop=True)
        recalc_subset = recalc_subset.sort_values('year').reset_index(drop=True)

        # Compare each column
        for col in recalc_df.columns:
            if col == 'min_zone_date':
                # Skip datetime comparison (string formatting differences are ok)
                continue

            if col not in cached_subset.columns:
                print(f"  ✗ MISSING COLUMN: {col} not in cached data")
                passed = False
                continue

            cached_vals = cached_subset[col].values
            recalc_vals = recalc_subset[col].values

            total_comparisons += len(cached_vals)

            # Compare with tolerance
            if np.issubdtype(cached_vals.dtype, np.number):
                matches = np.allclose(cached_vals, recalc_vals, atol=tolerance, equal_nan=True)
                if not matches:
                    # Find specific mismatches
                    mismatch_mask = ~np.isclose(cached_vals, recalc_vals, atol=tolerance, equal_nan=True)
                    n_mismatches = mismatch_mask.sum()
                    failed_comparisons += n_mismatches

                    if verbose:
                        print(f"  ✗ MISMATCH: r={r}, col={col}, {n_mismatches}/{len(cached_vals)} values differ")
                        # Show first few mismatches
                        mismatch_indices = np.where(mismatch_mask)[0][:3]
                        for idx in mismatch_indices:
                            print(f"      Year {cached_subset.loc[idx, 'year']}: cached={cached_vals[idx]:.6f}, recalc={recalc_vals[idx]:.6f}")
                    passed = False
            else:
                # For non-numeric columns
                matches = (cached_vals == recalc_vals).all()
                if not matches:
                    n_mismatches = (cached_vals != recalc_vals).sum()
                    failed_comparisons += n_mismatches

                    if verbose:
                        print(f"  ✗ MISMATCH: r={r}, col={col}, {n_mismatches}/{len(cached_vals)} values differ")
                    passed = False

    # Summary
    if verbose:
        print(f"\n{'='*80}")
        if passed:
            print(f"✓ VALIDATION PASSED")
            print(f"  All {total_comparisons} comparisons matched within tolerance ({tolerance})")
        else:
            print(f"✗ VALIDATION FAILED")
            print(f"  {failed_comparisons}/{total_comparisons} comparisons failed")
            print(f"  Tolerance: {tolerance}")
        print(f"{'='*80}\n")

    return passed


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'dataset_id',
        type=str,
        nargs='?',
        default=None,
        help='Dataset identifier to validate (or use --all)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Validate all datasets'
    )

    parser.add_argument(
        '--sample-size',
        type=int,
        default=10,
        help='Number of realizations to validate per dataset (default: 10)'
    )

    parser.add_argument(
        '--tolerance',
        type=float,
        default=1e-6,
        help='Numerical tolerance for comparisons (default: 1e-6)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed output (only show pass/fail)'
    )

    args = parser.parse_args()

    verbose = not args.quiet

    # Determine which datasets to validate
    if args.all:
        datasets = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']
    elif args.dataset_id:
        datasets = [args.dataset_id]
    else:
        parser.print_help()
        print(f"\nAvailable datasets: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)

    # Validate each dataset
    results = {}
    for dataset_id in datasets:
        try:
            passed = validate_metrics(
                dataset_id,
                sample_size=args.sample_size,
                tolerance=args.tolerance,
                verbose=verbose
            )
            results[dataset_id] = passed
        except Exception as e:
            print(f"\n✗ ERROR validating {dataset_id}: {e}")
            import traceback
            traceback.print_exc()
            results[dataset_id] = False

    # Print summary
    if len(results) > 1:
        print(f"\n{'='*80}")
        print("VALIDATION SUMMARY")
        print(f"{'='*80}")
        for dataset_id, passed in results.items():
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"  {dataset_id:30s} {status}")
        print(f"{'='*80}\n")

        # Exit with error code if any failed
        if not all(results.values()):
            sys.exit(1)


if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)
    main()
