"""
Core functions for drought analysis using SSI-based metrics.

This module contains functions for:
- Calculating SSI-based drought metrics for historical and ensemble data
- Processing drought events across multiple SSI windows
- Calculating satisficing conditions during drought/non-drought years
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from synhydro.droughts.ssi import SSIDroughtMetrics, SSI

from scipy.stats import chi2_contingency, mannwhitneyu

from .config import BASELINE_DATASET, NYC_RESERVOIRS
from .load import load_baseline_historical_flow, load_wrf1960s_historical_flow
from .metrics.satisficing import calculate_annual_satisficing
from .print_summary import print_satisficing_summary
from .save import save_annual_satisficing


def fit_ssi_calculator(ssi_window, node='nyc_aggregate'):
    """
    Fit an SSI calculator on the baseline period (1980-2019) historical flow.

    This is the single source of truth for SSI fitting across the codebase.
    All drought analysis functions (historic, serial ensemble, MPI ensemble)
    should use this helper to ensure consistent SSI parameterization.

    Parameters
    ----------
    ssi_window : int
        SSI timescale in months (e.g., 3, 6, 12).
    node : str
        Column name to fit SSI on (default: 'nyc_aggregate').

    Returns
    -------
    ssi_calculator : SSI
        Fitted SSI calculator ready for .transform() calls.
    """
    Q_baseline = load_baseline_historical_flow(
        gage_flow=False, period='full', flowtype=BASELINE_DATASET
    )
    Q_baseline.replace(0, np.nan, inplace=True)
    Q_baseline.drop(columns=['delTrenton'], inplace=True)

    Q_baseline[node] = Q_baseline[NYC_RESERVOIRS].sum(axis=1)

    # Extract only the target node series before resampling/cleanup
    # to avoid dropna removing months due to NaN in unrelated columns
    node_daily = Q_baseline[node].copy()
    node_monthly = node_daily.resample('MS').sum()
    node_monthly.replace(0, np.nan, inplace=True)
    node_monthly.dropna(inplace=True)

    ssi_calculator = SSI(normal_scores_transform=False, timescale=ssi_window)
    ssi_calculator.fit(node_monthly)
    return ssi_calculator


def calculate_historic_observed_droughts(ssi_windows, output_dir):
    """
    Calculate SSI-based drought metrics for historic observed data.

    Parameters
    ----------
    ssi_windows : list of int
        SSI window sizes in months (e.g., [3, 6, 12])
    output_dir : str
        Directory to save drought metrics

    Returns
    -------
    bool
        True if successful
    """
    print("=" * 60)
    print("CALCULATING HISTORIC OBSERVED DROUGHTS")
    print("=" * 60)

    # Load FULL historic record for transform (to find all droughts)
    Q = load_baseline_historical_flow(gage_flow=False, period='full', flowtype=BASELINE_DATASET)
    Q.replace(0, np.nan, inplace=True)
    Q.drop(columns=['delTrenton'], inplace=True)

    if BASELINE_DATASET == 'wrfaorc_withObsScaled':
        Q_1960s = load_wrf1960s_historical_flow(gage_flow=False)
        Q_1960s.replace(0, np.nan, inplace=True)
        Q_1960s.drop(columns=['delTrenton'], inplace=True)
        Q_full = pd.concat([Q_1960s, Q], axis=0).sort_index()
        Q_full.replace(0, np.nan, inplace=True)
        Q_full.dropna(axis=0, how='any', inplace=True)
    else:
        Q_full = Q.copy()

    node = 'nyc_aggregate'
    Q_full[node] = Q_full[NYC_RESERVOIRS].sum(axis=1)

    # Extract only the target node series before resampling/cleanup
    # to avoid dropna removing months due to NaN in unrelated columns
    node_daily = Q_full[node].copy()
    node_monthly = node_daily.resample('MS').sum()
    node_monthly.replace(0, np.nan, inplace=True)
    node_monthly.dropna(inplace=True)

    print(f"Loaded historic data with {Q_full.shape[0] // 365} years of daily data for {Q_full.shape[1]} sites.")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each SSI window
    for ssi_window in ssi_windows:
        print(f"\nProcessing SSI window: {ssi_window} months")

        # Fit SSI on baseline period (1980-2019)
        ssi_calculator = fit_ssi_calculator(ssi_window, node=node)
        drought_calculator = SSIDroughtMetrics()

        # Transform the FULL historic record
        ssi_obs = ssi_calculator.transform(node_monthly)
        obs_droughts = drought_calculator.calculate_drought_metrics(ssi_obs)

        # Save observed drought metrics
        obs_droughts.reset_index(inplace=True, drop=True)

        # If baseline dataset is wrfaorc_withObsScaled, drop droughts spanning 1960s-1970s gap
        if BASELINE_DATASET == 'wrfaorc_withObsScaled':
            obs_droughts = obs_droughts[~((obs_droughts['start'] < pd.to_datetime('1970-01-01')) &
                                           (obs_droughts['end'] > pd.to_datetime('1979-01-01')))]

        obs_fname = f"{output_dir}/observed_ssi{ssi_window}_drought_events.csv"
        obs_droughts.to_csv(obs_fname, index=False)
        print(f"  Saved observed drought metrics: {obs_fname}")

    print("\n" + "=" * 60)
    print("Historic observed droughts calculation completed!")
    print("=" * 60)
    return True


def calculate_statistical_significance(drought_results, non_drought_results):
    """
    Perform statistical tests comparing satisficing during drought vs non-drought years.

    Tests:
    - Chi-square test on satisficing rates
    - Mann-Whitney U test on minimum storage levels
    - Mann-Whitney U test on maximum violation days

    Parameters
    ----------
    drought_results : pd.DataFrame
        Satisficing results for years with droughts
    non_drought_results : pd.DataFrame
        Satisficing results for years without droughts

    Returns
    -------
    dict
        Statistical test results with keys: chi2_satisficing,
        mannwhitney_storage, mannwhitney_violations
    """
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
    results['chi2_satisficing'] = {'chi2': chi2, 'p_value': p_value, 'dof': dof}

    print("\nChi-Square Test: Satisficing Rates (Years with vs without Droughts)")
    print("-" * 80)
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"p-value: {p_value:.4e}")
    print(f"Degrees of freedom: {dof}")
    _print_significance(p_value)

    # Mann-Whitney U test for storage levels
    u_stat_storage, p_value_storage = mannwhitneyu(
        drought_results['nyc_min_storage_pct'],
        non_drought_results['nyc_min_storage_pct'],
        alternative='two-sided'
    )
    results['mannwhitney_storage'] = {'u_statistic': u_stat_storage, 'p_value': p_value_storage}

    print("\nMann-Whitney U Test: Minimum Storage Levels")
    print("-" * 80)
    print(f"U statistic: {u_stat_storage:.4f}")
    print(f"p-value: {p_value_storage:.4e}")
    _print_significance(p_value_storage)

    # Mann-Whitney U test for violation days
    u_stat_violations, p_value_violations = mannwhitneyu(
        drought_results['max_violation_days'],
        non_drought_results['max_violation_days'],
        alternative='two-sided'
    )
    results['mannwhitney_violations'] = {'u_statistic': u_stat_violations, 'p_value': p_value_violations}

    print("\nMann-Whitney U Test: Maximum Violation Days")
    print("-" * 80)
    print(f"U statistic: {u_stat_violations:.4f}")
    print(f"p-value: {p_value_violations:.4e}")
    _print_significance(p_value_violations)

    print("=" * 80)
    return results


def _print_significance(p_value):
    """Print significance interpretation for a p-value."""
    if p_value < 0.001:
        print("Result: HIGHLY SIGNIFICANT (p < 0.001)")
    elif p_value < 0.01:
        print("Result: VERY SIGNIFICANT (p < 0.01)")
    elif p_value < 0.05:
        print("Result: SIGNIFICANT (p < 0.05)")
    else:
        print("Result: NOT SIGNIFICANT (p >= 0.05)")


