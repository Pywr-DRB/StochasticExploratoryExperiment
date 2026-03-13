"""
Contribution analysis utilities for pre-computed contribution metrics.

Provides functions to filter, categorize, and analyze contribution metrics
by window length and drought zone.
"""

import pandas as pd

from methods.load import load_contribution_metrics


def get_metrics_for_window(df, window_days, metrics=None):
    """
    Extract metrics for a specific window length from contribution metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Contribution metrics dataframe from load_contribution_metrics()
    window_days : int
        Window length (30, 60, 90, 120, 150, 180, or 270 days)
    metrics : list of str, optional
        Specific metrics to extract. If None, extracts all window-specific metrics.

    Returns
    -------
    pd.DataFrame
        DataFrame with base columns + selected window metrics

    Raises
    ------
    ValueError
        If window_days not in pre-computed window lengths.
    """
    available_windows = [30, 60, 90, 120, 150, 180, 270]
    if window_days not in available_windows:
        raise ValueError(
            f"Window {window_days} days not pre-computed. Available: {available_windows}\n"
            f"Note: 3 months = 90 days, 6 months = 180 days, 9 months = 270 days"
        )

    base_cols = ['realization_id', 'year', 'annual_max_zone', 'annual_max_zone_date', 'annual_min_storage_pct']

    if metrics is None:
        metrics = ['contribution_total', 'contribution_ratio', 'inflow_total',
                   'demand_satisfaction', 'worst_1mo_demand_sat']

    window_cols = [f'{metric}_{window_days}d' for metric in metrics]

    missing_cols = [col for col in window_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataframe: {missing_cols}")

    return df[base_cols + window_cols].copy()


def categorize_by_zone(df, zone_categories=None):
    """
    Group contribution metrics by drought zone categories.

    Parameters
    ----------
    df : pd.DataFrame
        Contribution metrics dataframe
    zone_categories : dict, optional
        Dictionary mapping category name -> list of zone values.
        Default: Normal=[0,1,2,3], Watch=[4,5], Warning=[2], Emergency=[6]

    Returns
    -------
    dict
        Dictionary mapping category name -> DataFrame subset
    """
    if zone_categories is None:
        zone_categories = {
            'Normal': [0, 1, 2, 3],
            'Watch': [4, 5],
            'Warning': [2],
            'Emergency': [6]
        }

    categorized = {}
    for category, zones in zone_categories.items():
        categorized[category] = df[df['annual_max_zone'].isin(zones)].copy()

    return categorized


def find_optimal_window_for_correlation(dataset_id, target_metric='annual_min_storage_pct',
                                       source_metric='contribution_ratio',
                                       window_range=(30, 180, 10)):
    """
    Find optimal window length maximizing correlation magnitude using pre-computed metrics.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    target_metric : str
        Target metric for correlation (default: 'annual_min_storage_pct')
    source_metric : str
        Source metric for correlation (default: 'contribution_ratio')
    window_range : tuple of (min, max, step)
        Range of window days to test

    Returns
    -------
    dict
        Keys: 'optimal_window', 'optimal_correlation', 'all_correlations'
    """
    df = load_contribution_metrics(dataset_id)
    available_windows = [30, 60, 90, 120, 150, 180, 270]

    min_w, max_w, step = window_range
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

        df_clean = df[[source_col, target_metric]].dropna()

        if len(df_clean) > 10:
            corr = df_clean.corr().iloc[0, 1]
            correlations[w] = corr
            print(f"  Window {w:3d} days: r = {corr:7.4f}")
        else:
            print(f"  Window {w:3d} days: insufficient data")

    if not correlations:
        raise ValueError("Could not calculate correlations for any window in range")

    optimal_window = min(correlations.keys(), key=lambda k: correlations[k])
    optimal_corr = correlations[optimal_window]

    print(f"\n  Optimal window: {optimal_window} days (r = {optimal_corr:.4f})")

    return {
        'optimal_window': optimal_window,
        'optimal_correlation': optimal_corr,
        'all_correlations': correlations
    }
