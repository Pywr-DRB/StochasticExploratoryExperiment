"""
Contribution analysis utilities for pre-computed contribution metrics.

Provides functions to filter, categorize, and analyze contribution metrics
by window length and drought zone.
"""


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

    base_cols = ['realization_id', 'year', 'annual_max_zone', 'annual_max_zone_date',
                 'annual_min_storage_pct', 'annual_min_storage_date']

    if metrics is None:
        metrics = ['contribution_total', 'contribution_ratio', 'inflow_total',
                   'diversion_total', 'diversion_ratio',
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


