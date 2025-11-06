"""
Copula diagnostic plots for drought severity and magnitude relationships.

This script generates comprehensive diagnostic figures for the copula-based
joint probability model used in drought frequency analysis. The diagnostics
validate the choice of marginal distributions and copula structure.

The methodology matches exactly what is used in 09_plot_drought_frequency.py:
- Marginal distributions configured in config.DROUGHT_MARGINAL_DISTRIBUTIONS
- Copula type configured in config.DROUGHT_COPULA_TYPE ('gaussian' or 't_copula')

Diagnostic plots include:
1. Joint scatter plots with marginal distributions
2. Fitted marginal PDF comparisons
3. Empirical copula visualization (uniform marginals)
4. Q-Q plots for marginal distributions
5. Empirical vs. simulated copula data
6. Tail dependence analysis
7. Kendall's tau comparison

These diagnostics are generated for all SSI windows configured in config.SSI_WINDOWS.

Usage:
------
python 06_ssi_copula_diagnostics.py [dataset_id]

Arguments:
    dataset_id : str, optional (default='stationary_ensemble')
        Dataset identifier to analyze

Example:
    python 06_ssi_copula_diagnostics.py stationary_ensemble
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

from methods.config import *
from methods.copula import (
    load_drought_events,
    fit_marginal_distributions,
    fit_gaussian_copula,
    fit_t_copula,
    transform_to_uniform,
    check_tail_dependence,
    calculate_tail_dependence_curves,
    calculate_interarrival_time,
)
from methods.plotting.copula_diagnostics import (
    plot_joint_scatter,
    plot_marginal_fits,
    plot_empirical_copula,
    plot_copula_diagnostics,
    plot_tail_dependence,
    generate_diagnostics_summary,
)


def run_copula_diagnostics(dataset_id, ssi_window):
    """
    Run complete copula diagnostics for a dataset and SSI window.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    ssi_window : int
        SSI window size (months)
    """
    print(f"\n{'='*80}")
    print(f"Running copula diagnostics: {dataset_id} (SSI-{ssi_window})")
    print(f"{'='*80}")

    # Create output directory
    output_dir = f"{FIG_DIR}/copula_diagnostics/{dataset_id}/ssi{ssi_window}"
    os.makedirs(output_dir, exist_ok=True)

    # Load drought events
    print("Loading drought events...")
    df = load_drought_events(dataset_id, ssi_window)
    print(f"  Loaded {len(df):,} drought events")

    # Fit marginal distributions (using config)
    print("Fitting marginal distributions...")
    marginals = fit_marginal_distributions(df)
    # Frozen distributions have .dist.name, not .name
    sev_dist_name = marginals['severity_dist'].dist.name if hasattr(marginals['severity_dist'], 'dist') else 'truncnorm'
    mag_dist_name = marginals['magnitude_dist'].dist.name if hasattr(marginals['magnitude_dist'], 'dist') else 'truncnorm'
    print(f"  Severity: {sev_dist_name}")
    print(f"  Magnitude: {mag_dist_name}")

    # Transform to uniform marginals
    print("Transforming to uniform marginals...")
    U = transform_to_uniform(df, marginals)

    # Fit copula based on config
    print(f"Fitting {DROUGHT_COPULA_TYPE} copula...")
    if DROUGHT_COPULA_TYPE == 't_copula':
        copula_params = fit_t_copula(U)
        copula = {
            'rho': copula_params['rho'],
            'nu': copula_params['nu'],
            'loglik': copula_params['loglik'],
            'U': U,
            'type': 't_copula'
        }
        print(f"  Correlation (ρ): {copula['rho']:.4f}")
        print(f"  Degrees of freedom (ν): {copula['nu']:.2f}")
        print(f"  Log-likelihood: {copula['loglik']:.2f}")
    else:
        copula = fit_gaussian_copula(df, marginals)
        copula['type'] = 'gaussian'
        print(f"  Correlation (ρ): {copula['rho']:.4f}")
        print(f"  Log-likelihood: {copula['loglik']:.2f}")

    # Check tail dependence
    print("Checking tail dependence...")
    tail = check_tail_dependence(copula['U'], df)
    print(f"  Kendall's tau: {tail['tau']:.4f}")
    print(f"  Empirical λ_L: {tail['lambda_L']:.3f}")
    print(f"  Empirical λ_U: {tail['lambda_U']:.3f}")
    print(f"  Tail dependence detected: {tail['has_tail_dependence']}")

    if tail['has_tail_dependence']:
        print(f"  t-copula ρ: {tail['t_copula_rho']:.4f}, ν: {tail['t_copula_nu']:.2f}")

    # Calculate tail dependence curves
    print("Calculating tail dependence curves...")
    tail_curves = calculate_tail_dependence_curves(copula['U'])

    # Calculate interarrival time
    print("Calculating interarrival time...")
    interarrival = calculate_interarrival_time(df, n_years=N_YEARS)
    print(f"  Expected interarrival: {interarrival:.2f} years")

    # Generate plots
    print("\nGenerating diagnostic plots...")

    print("  Plot 1: Joint scatter...")
    plot_joint_scatter(df, ssi_window, dataset_id, output_dir)

    print("  Plot 2: Marginal fits...")
    plot_marginal_fits(df, marginals, ssi_window, dataset_id, output_dir)

    print("  Plot 3: Empirical copula...")
    plot_empirical_copula(copula['U'], ssi_window, dataset_id, output_dir)

    print("  Plot 4: Copula diagnostics...")
    plot_copula_diagnostics(df, marginals, copula, tail, ssi_window, dataset_id, output_dir)

    print("  Plot 5: Tail dependence...")
    plot_tail_dependence(copula['U'], tail, tail_curves, ssi_window, dataset_id, output_dir)

    # Generate summary
    print("\nGenerating diagnostics summary...")
    generate_diagnostics_summary(df, marginals, copula, tail, interarrival,
                                ssi_window, dataset_id, output_dir)

    print(f"\n{'='*80}")
    print(f"Diagnostics complete for {dataset_id} (SSI-{ssi_window})")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}")


def main(dataset_id='stationary_ensemble'):
    """
    Main function to run copula diagnostics for all SSI windows.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier (default: 'stationary_ensemble')
    """
    # Verify dataset
    verify_dataset_id(dataset_id)

    print("=" * 80)
    print("SSI COPULA DIAGNOSTICS")
    print("=" * 80)
    print(f"Dataset: {dataset_id}")
    print(f"SSI windows: {SSI_WINDOWS}")
    print("=" * 80)

    # Run diagnostics for each SSI window
    for ssi_window in SSI_WINDOWS:
        try:
            run_copula_diagnostics(dataset_id, ssi_window)
        except FileNotFoundError as e:
            print(f"\nWARNING: {e}")
            print(f"Skipping SSI-{ssi_window}...")
            continue
        except Exception as e:
            print(f"\nERROR in SSI-{ssi_window}: {e}")
            import traceback
            traceback.print_exc()
            print(f"Skipping SSI-{ssi_window}...")
            continue

    print("\n" + "=" * 80)
    print("ALL DIAGNOSTICS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 2:
        print("Usage: python 06_ssi_copula_diagnostics.py [dataset_id]")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        print("Default: stationary_ensemble")
        sys.exit(1)

    dataset_id = sys.argv[1] if len(sys.argv) == 2 else 'stationary_ensemble'

    # Verify dataset
    verify_dataset_id(dataset_id)

    main(dataset_id)
