"""
Compare copula parameters across datasets to verify methodology.

This script fits copulas to all datasets and generates a comparison table
showing how marginal distributions, copula correlations, and interarrival
times differ across climate scenarios.

This validates the methodological approach used in 09_plot_drought_frequency.py,
which fits separate copulas for each dataset rather than using a single
stationary copula for all scenarios.

Usage:
------
python 07_compare_copula_parameters.py [ssi_window]

Arguments:
    ssi_window : int, optional (default=12)
        SSI window size in months

Example:
    python 07_compare_copula_parameters.py 12
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from methods.config import *
from methods.plotting.styles import DATASET_ORDER
from methods.copula import (
    load_drought_events,
    fit_marginal_distributions,
    fit_gaussian_copula,
    check_tail_dependence,
    calculate_interarrival_time,
)


def compare_copula_parameters(ssi_window=12, output_file=None):
    """
    Compare copula parameters across all datasets.

    Parameters
    ----------
    ssi_window : int
        SSI window size in months
    output_file : str, optional
        Path to save comparison table (CSV)

    Returns
    -------
    pd.DataFrame
        Comparison table with copula parameters
    """
    print("=" * 80)
    print(f"COPULA PARAMETER COMPARISON (SSI-{ssi_window})")
    print("=" * 80)
    print()

    # Results storage
    results = []

    # Analyze each dataset
    for dataset_id in DATASET_ORDER:
        print(f"Analyzing {dataset_id}...")

        try:
            # Load drought events
            df = load_drought_events(dataset_id, ssi_window)

            # Fit marginals
            marginals = fit_marginal_distributions(df)

            # Fit copula
            copula = fit_gaussian_copula(df, marginals)

            # Check tail dependence
            tail = check_tail_dependence(copula['U'], df)

            # Calculate interarrival time
            interarrival = calculate_interarrival_time(df, n_years=N_YEARS)

            # Extract parameters
            severity_params = marginals['severity_params']
            magnitude_params = marginals['magnitude_params']

            # Store results
            result = {
                'dataset_id': dataset_id,
                'n_events': len(df),
                'interarrival_years': interarrival,
                'copula_rho': copula['rho'],
                'kendalls_tau': tail['tau'],
                'tail_dep_lower': tail['lambda_L'],
                'tail_dep_upper': tail['lambda_U'],
                'magnitude_mean': magnitude_params[0],
                'magnitude_std': magnitude_params[1],
                'severity_a': severity_params[0] if len(severity_params) > 0 else np.nan,
                'severity_c': severity_params[1] if len(severity_params) > 1 else np.nan,
                'severity_loc': severity_params[2] if len(severity_params) > 2 else np.nan,
                'severity_scale': severity_params[3] if len(severity_params) > 3 else np.nan,
            }

            results.append(result)

            print(f"  ✓ Events: {len(df):,}")
            print(f"  ✓ Interarrival: {interarrival:.2f} years")
            print(f"  ✓ Copula ρ: {copula['rho']:.4f}")
            print(f"  ✓ Kendall's τ: {tail['tau']:.4f}")
            print()

        except FileNotFoundError as e:
            print(f"  ✗ Skipping: {e}")
            print()
            continue
        except Exception as e:
            print(f"  ✗ Error: {e}")
            print()
            continue

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Print comparison table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE: KEY PARAMETERS")
    print("=" * 80)
    print()

    if len(df_results) > 0:
        # Select key columns for display
        display_cols = [
            'dataset_id', 'n_events', 'interarrival_years',
            'copula_rho', 'kendalls_tau',
            'magnitude_mean', 'magnitude_std',
            'tail_dep_lower', 'tail_dep_upper'
        ]

        # Format for display
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.precision', 4)

        print(df_results[display_cols].to_string(index=False))
        print()

        # Calculate percentage changes from stationary
        if 'stationary_ensemble' in df_results['dataset_id'].values:
            print("=" * 80)
            print("PERCENTAGE CHANGE FROM STATIONARY")
            print("=" * 80)
            print()

            stat_row = df_results[df_results['dataset_id'] == 'stationary_ensemble'].iloc[0]

            pct_changes = []
            for _, row in df_results.iterrows():
                if row['dataset_id'] == 'stationary_ensemble':
                    continue

                pct_change = {
                    'dataset_id': row['dataset_id'],
                    'n_events_pct': 100 * (row['n_events'] - stat_row['n_events']) / stat_row['n_events'],
                    'interarrival_pct': 100 * (row['interarrival_years'] - stat_row['interarrival_years']) / stat_row['interarrival_years'],
                    'copula_rho_pct': 100 * (row['copula_rho'] - stat_row['copula_rho']) / stat_row['copula_rho'],
                    'magnitude_mean_pct': 100 * (row['magnitude_mean'] - stat_row['magnitude_mean']) / stat_row['magnitude_mean'],
                    'magnitude_std_pct': 100 * (row['magnitude_std'] - stat_row['magnitude_std']) / stat_row['magnitude_std'],
                }
                pct_changes.append(pct_change)

            if pct_changes:
                df_pct = pd.DataFrame(pct_changes)
                print(df_pct.to_string(index=False))
                print()

        # Save to file if requested
        if output_file:
            df_results.to_csv(output_file, index=False)
            print(f"Saved full results to: {output_file}")
            print()

    else:
        print("No results to display.")
        print()

    print("=" * 80)
    print()

    return df_results


def print_interpretation_guide():
    """Print guide for interpreting the comparison table."""
    print("=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    print()
    print("Key Parameters:")
    print("  - n_events: Total number of drought events detected")
    print("  - interarrival_years: Expected time between drought events (E[L])")
    print("  - copula_rho: Gaussian copula correlation parameter")
    print("  - kendalls_tau: Kendall's rank correlation (alternative measure)")
    print("  - magnitude_mean/std: Normal distribution parameters for log(magnitude)")
    print("  - tail_dep_lower/upper: Empirical tail dependence at 5th/95th percentiles")
    print()
    print("Expected Climate Scenario Effects:")
    print()
    print("  Climate Low (Dry):")
    print("    • MORE events (n_events ↑)")
    print("    • SHORTER interarrival (E[L] ↓)")
    print("    • LARGER magnitude mean (μ ↑)")
    print("    • Potentially STRONGER correlation (ρ ↑) if compound extremes")
    print()
    print("  Climate High (Wet):")
    print("    • FEWER events (n_events ↓)")
    print("    • LONGER interarrival (E[L] ↑)")
    print("    • SMALLER magnitude mean (μ ↓)")
    print("    • Potentially WEAKER correlation (ρ ↓) if decoupling")
    print()
    print("Methodological Implications:")
    print("  If parameters differ significantly across datasets, this validates")
    print("  the approach of fitting separate copulas for each climate scenario")
    print("  rather than using a single stationary copula for all scenarios.")
    print()
    print("=" * 80)
    print()


def main(ssi_window=12):
    """
    Main entry point.

    Parameters
    ----------
    ssi_window : int
        SSI window size in months
    """
    # Compare parameters
    output_dir = f"{ROOT_DIR}/pywrdrb/drought_metrics"
    output_file = f"{output_dir}/copula_parameter_comparison_ssi{ssi_window}.csv"

    df_results = compare_copula_parameters(ssi_window, output_file)

    # Print interpretation guide
    if len(df_results) > 0:
        print_interpretation_guide()

    print("Done!")


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 2:
        print("Usage: python 07_compare_copula_parameters.py [ssi_window]")
        print("Example: python 07_compare_copula_parameters.py 12")
        sys.exit(1)

    ssi_window = int(sys.argv[1]) if len(sys.argv) == 2 else 12

    main(ssi_window)
