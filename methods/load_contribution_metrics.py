"""
Utilities for loading and processing pre-computed contribution analysis metrics.

This module provides a clean API for plotting scripts to load pre-computed
contribution metrics, eliminating the need to recalculate them on-the-fly.
"""

import pandas as pd
import numpy as np
import os


def load_contribution_metrics(dataset_id):
    """
    Load pre-computed contribution metrics CSV.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier (e.g., 'stationary_ensemble', 'climate_adjusted_low')

    Returns
    -------
    pd.DataFrame
        Contribution metrics with columns:
        - realization_id, year, min_zone, min_zone_date, min_storage_pct
        - contribution_total_{W}d, contribution_ratio_{W}d, inflow_total_{W}d,
          demand_satisfaction_{W}d, worst_1mo_demand_sat_{W}d
          for W in [30, 60, 90, 120, 150, 180, 270]

    Raises
    ------
    FileNotFoundError
        If pre-computed metrics file does not exist (triggers fallback in plotting scripts)
    """
    fname = f'./pywrdrb/performance_metrics/{dataset_id}_contribution_metrics.csv'

    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Pre-computed metrics not found: {fname}\n"
            "Run postprocessing (04_postprocess_data_mpi.py) to generate these files."
        )

    df = pd.read_csv(fname)

    # Convert min_zone_date back to datetime if needed for compatibility
    if 'min_zone_date' in df.columns:
        df['min_zone_date'] = pd.to_datetime(df['min_zone_date'])

    return df


def get_metrics_for_window(df, window_days, metrics=None):
    """
    Extract metrics for a specific window length.

    Parameters
    ----------
    df : pd.DataFrame
        Contribution metrics dataframe from load_contribution_metrics()
    window_days : int
        Window length (30, 60, 90, 120, 150, or 180 days)
    metrics : list of str, optional
        Specific metrics to extract. If None, extracts all window-specific metrics.
        Available: ['contribution_total', 'contribution_ratio', 'inflow_total',
                   'demand_satisfaction', 'worst_1mo_demand_sat']

    Returns
    -------
    pd.DataFrame
        DataFrame with base columns + selected window metrics

    Raises
    ------
    ValueError
        If window_days not in pre-computed window lengths

    Examples
    --------
    >>> df = load_contribution_metrics('stationary_ensemble')
    >>> df_90d = get_metrics_for_window(df, 90)
    >>> # Returns columns: realization_id, year, min_zone, min_zone_date, min_storage_pct,
    >>> #                  contribution_total_90d, contribution_ratio_90d, etc.
    """
    available_windows = [30, 60, 90, 120, 150, 180, 270]
    if window_days not in available_windows:
        raise ValueError(
            f"Window {window_days} days not pre-computed. Available: {available_windows}\n"
            f"Note: 3 months ≈ 90 days, 6 months ≈ 180 days, 9 months ≈ 270 days"
        )

    # Base columns always included
    base_cols = ['realization_id', 'year', 'min_zone', 'min_zone_date', 'min_storage_pct']

    # Window-specific metrics
    if metrics is None:
        metrics = ['contribution_total', 'contribution_ratio', 'inflow_total',
                   'demand_satisfaction', 'worst_1mo_demand_sat']

    window_cols = [f'{metric}_{window_days}d' for metric in metrics]

    # Verify all columns exist
    missing_cols = [col for col in window_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataframe: {missing_cols}")

    return df[base_cols + window_cols].copy()


def categorize_by_zone(df, zone_categories=None):
    """
    Group metrics by drought zone categories.

    Parameters
    ----------
    df : pd.DataFrame
        Contribution metrics dataframe
    zone_categories : dict, optional
        Dictionary mapping category name -> list of zone values
        Default: {'Normal': [0], 'Watch': [1], 'Warning': [2], 'Emergency': [3, 4]}

    Returns
    -------
    dict
        Dictionary mapping category name -> DataFrame subset

    Examples
    --------
    >>> df = load_contribution_metrics('stationary_ensemble')
    >>> categorized = categorize_by_zone(df)
    >>> emergency_years = categorized['Emergency']
    """
    if zone_categories is None:
        # Default FFMP zone categories
        zone_categories = {
            'Normal': [0, 1, 2, 3],      # Zones 0-3 (Normal and above)
            'Watch': [4, 5],             # Drought Watch
            'Warning': [2],              # Drought Warning
            'Emergency': [6]             # Drought Emergency
        }

    categorized = {}
    for category, zones in zone_categories.items():
        categorized[category] = df[df['min_zone'].isin(zones)].copy()

    return categorized


def find_optimal_window_for_correlation(dataset_id, target_metric='min_storage_pct',
                                       source_metric='contribution_ratio',
                                       window_range=(30, 180, 10)):
    """
    Find optimal window length maximizing correlation magnitude WITHOUT recalculation.

    This function replaces the expensive optimization loop in SI6 by using pre-computed
    metrics to quickly test correlations across different window lengths.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    target_metric : str
        Target metric for correlation (default: 'min_storage_pct')
    source_metric : str
        Source metric for correlation (default: 'contribution_ratio')
        Will be tested as '{source_metric}_{W}d' for each window W
    window_range : tuple of (min, max, step)
        Range of window days to test (default: 30 to 180 in steps of 10)

    Returns
    -------
    dict
        Dictionary with keys:
        - 'optimal_window': int, window length with maximum correlation magnitude
        - 'optimal_correlation': float, correlation value at optimal window
        - 'all_correlations': dict, mapping window -> correlation for all tested windows

    Examples
    --------
    >>> result = find_optimal_window_for_correlation('stationary_ensemble')
    >>> print(f"Optimal window: {result['optimal_window']} days")
    >>> print(f"Correlation: {result['optimal_correlation']:.4f}")
    """
    df = load_contribution_metrics(dataset_id)
    available_windows = [30, 60, 90, 120, 150, 180, 270]

    min_w, max_w, step = window_range
    # Find available windows within the requested range
    test_windows = [w for w in available_windows if min_w <= w <= max_w]

    if not test_windows:
        raise ValueError(
            f"No pre-computed windows in range {min_w}-{max_w}. "
            f"Available windows: {available_windows}"
        )

    correlations = {}

    for w in test_windows:
        source_col = f'{source_metric}_{w}d'

        if source_col not in df.columns:
            print(f"  Warning: Column {source_col} not found, skipping")
            continue

        # Calculate correlation for this window
        df_clean = df[[source_col, target_metric]].dropna()

        if len(df_clean) > 10:  # Require minimum sample size
            corr = df_clean.corr().iloc[0, 1]
            correlations[w] = corr
            print(f"  Window {w:3d} days: r = {corr:7.4f}")
        else:
            print(f"  Window {w:3d} days: insufficient data")

    if not correlations:
        raise ValueError("Could not calculate correlations for any window in range")

    # Find window with maximum correlation magnitude (most negative for inverse correlation)
    optimal_window = min(correlations.keys(), key=lambda k: correlations[k])
    optimal_corr = correlations[optimal_window]

    print(f"\n  Optimal window: {optimal_window} days (r = {optimal_corr:.4f})")

    return {
        'optimal_window': optimal_window,
        'optimal_correlation': optimal_corr,
        'all_correlations': correlations
    }


def convert_months_to_days(n_months):
    """
    Convert month window to approximate days for compatibility.

    Parameters
    ----------
    n_months : int
        Number of months

    Returns
    -------
    int
        Approximate number of days (n_months * 30)

    Examples
    --------
    >>> convert_months_to_days(3)  # 3 months
    90
    >>> convert_months_to_days(6)  # 6 months
    180
    """
    return n_months * 30


def get_available_windows():
    """
    Get list of available pre-computed window lengths.

    Returns
    -------
    list of int
        Available window lengths in days
    """
    return [30, 60, 90, 120, 150, 180, 270]


def print_metrics_summary(df, dataset_id=None):
    """
    Print summary statistics for contribution metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Contribution metrics dataframe
    dataset_id : str, optional
        Dataset identifier for display
    """
    if dataset_id:
        print(f"\nContribution Metrics Summary: {dataset_id}")
    else:
        print("\nContribution Metrics Summary")
    print("=" * 80)

    n_realizations = df['realization_id'].nunique()
    n_years = df['year'].nunique()
    n_total = len(df)

    print(f"  Realizations: {n_realizations}")
    print(f"  Years: {n_years}")
    print(f"  Total year-realization pairs: {n_total}")
    print(f"\nDrought Zone Distribution:")

    zone_counts = df['min_zone'].value_counts().sort_index()
    for zone, count in zone_counts.items():
        pct = 100.0 * count / n_total
        print(f"    Zone {zone}: {count:6d} ({pct:5.1f}%)")

    # Show sample statistics for one window (90 days)
    if 'contribution_ratio_90d' in df.columns:
        print(f"\nContribution Ratio (90-day window) Statistics:")
        contrib_ratio = df['contribution_ratio_90d'].dropna()
        print(f"    Mean:   {contrib_ratio.mean():6.2f}%")
        print(f"    Median: {contrib_ratio.median():6.2f}%")
        print(f"    Std:    {contrib_ratio.std():6.2f}%")
        print(f"    Min:    {contrib_ratio.min():6.2f}%")
        print(f"    Max:    {contrib_ratio.max():6.2f}%")

    print("=" * 80)
