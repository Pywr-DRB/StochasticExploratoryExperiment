"""
Plotting functions for ensemble summary figures (main manuscript).

This module provides functions for creating publication-quality summary
figures comparing synthetic ensemble and historical streamflow data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import ranksums, levene

from .styles import (
    HISTORIC_COLOR, HISTORIC_LABEL,
    DATASET_COLORS, DATASET_LABELS,
    ALPHA_FILL, LINEWIDTH_MEDIUM, LINEWIDTH_THICK,
    DPI_PRINT, apply_publication_style
)

# NYC reservoirs for aggregate flow calculation
NYC_RESERVOIRS = ['cannonsville', 'pepacton', 'neversink']

# Month labels for week-of-year x-axis
MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# Approximate week numbers for start of each month
MONTH_WEEK_STARTS = [1, 5, 9, 14, 18, 22, 27, 31, 35, 40, 44, 48]


def _get_aggregate_flow(df: pd.DataFrame, sites: list = None) -> pd.Series:
    """
    Get aggregate flow as sum of specified sites.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with sites as columns
    sites : list, optional
        List of site names to aggregate. Defaults to NYC_RESERVOIRS.

    Returns
    -------
    pd.Series
        Summed flow across sites
    """
    if sites is None:
        sites = NYC_RESERVOIRS

    # Filter to available sites
    available = [s for s in sites if s in df.columns]
    if not available:
        raise ValueError(f"None of the requested sites {sites} found in DataFrame columns: {df.columns.tolist()}")

    return df[available].sum(axis=1)


def _pre_aggregate_synthetic(Q_synthetic: dict, sites: list = None) -> dict:
    """
    Pre-aggregate synthetic ensemble flows across sites.

    Computes site aggregation once so panel functions don't repeat it.

    Parameters
    ----------
    Q_synthetic : dict
        Dictionary of synthetic flow DataFrames keyed by realization ID
    sites : list, optional
        Sites to aggregate. Defaults to NYC_RESERVOIRS.

    Returns
    -------
    dict
        Dictionary mapping realization ID to aggregated pd.Series
    """
    if sites is None:
        sites = NYC_RESERVOIRS

    result = {}
    for real_id, real_df in Q_synthetic.items():
        if isinstance(real_df, pd.DataFrame):
            result[real_id] = _get_aggregate_flow(real_df, sites)
        else:
            result[real_id] = real_df
    return result


def _vectorized_autocorr(series: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Compute autocorrelation for lags 1..max_lag using vectorized numpy.

    Parameters
    ----------
    series : np.ndarray
        1D array of values (no NaNs)
    max_lag : int
        Maximum lag

    Returns
    -------
    np.ndarray
        Autocorrelation values for lags 1..max_lag
    """
    n = len(series)
    mean = series.mean()
    var = np.sum((series - mean) ** 2)
    if var == 0:
        return np.zeros(max_lag)

    autocorr = np.empty(max_lag)
    centered = series - mean
    for lag in range(1, max_lag + 1):
        if n > lag:
            autocorr[lag - 1] = np.sum(centered[:n - lag] * centered[lag:]) / var
        else:
            autocorr[lag - 1] = np.nan
    return autocorr


def plot_weekly_streamflow_percentiles(
    Q_historic: pd.DataFrame,
    Q_synthetic: dict,
    sites: list = None,
    ax=None,
    ylabel: str = 'Streamflow (MGD)',
    xlabel: str = 'Month',
    percentiles: tuple = (5, 95),
    show_legend: bool = False,
    synthetic_color: str = None,
    synthetic_label: str = 'Synthetic',
    _hist_agg: pd.Series = None,
    _syn_agg: dict = None,
):
    """
    Plot weekly streamflow percentile bands for synthetic vs historic data.

    Parameters
    ----------
    Q_historic : pd.DataFrame
        Historical daily flow data with DatetimeIndex
    Q_synthetic : dict
        Dictionary of synthetic flow DataFrames keyed by realization ID
    sites : list, optional
        List of sites to aggregate. Defaults to NYC_RESERVOIRS.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    ylabel, xlabel : str
        Axis labels
    percentiles : tuple
        Lower and upper percentiles for range (default 5, 95)
    show_legend : bool
        Whether to show legend on this axis
    synthetic_color : str, optional
        Color for synthetic data.
    synthetic_label : str
        Label for synthetic data in legend
    _hist_agg : pd.Series, optional
        Pre-aggregated historic flow (internal optimization).
    _syn_agg : dict, optional
        Pre-aggregated synthetic flows (internal optimization).

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    if synthetic_color is None:
        synthetic_color = DATASET_COLORS['stationary_ensemble']

    # Use pre-aggregated data if available, otherwise compute
    if _hist_agg is None:
        _hist_agg = _get_aggregate_flow(Q_historic, sites)
    if _syn_agg is None:
        _syn_agg = _pre_aggregate_synthetic(Q_synthetic, sites)

    # Process historic data to weekly
    Q_hist_weekly = _hist_agg.resample('W').mean()
    hist_weeks = np.array(Q_hist_weekly.index.isocalendar().week, dtype=int)
    hist_values = Q_hist_weekly.values

    # Compute weekly climatology for historic using numpy
    weeks = np.arange(1, 53)
    hist_median = np.empty(52)
    hist_p_low = np.empty(52)
    hist_p_high = np.empty(52)
    for w in weeks:
        mask = hist_weeks == w
        vals = hist_values[mask]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            hist_median[w - 1] = np.median(vals)
            hist_p_low[w - 1] = np.percentile(vals, percentiles[0])
            hist_p_high[w - 1] = np.percentile(vals, percentiles[1])
        else:
            hist_median[w - 1] = hist_p_low[w - 1] = hist_p_high[w - 1] = np.nan

    # Process synthetic data: collect all weekly values by week-of-year
    # Stack into a single array for vectorized percentile computation
    syn_weekly_by_week = {w: [] for w in range(1, 53)}
    for real_id, flow_series in _syn_agg.items():
        weekly = flow_series.resample('W').mean()
        w_arr = np.array(weekly.index.isocalendar().week, dtype=int)
        v_arr = weekly.values
        for w in weeks:
            mask = w_arr == w
            vals = v_arr[mask]
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                syn_weekly_by_week[w].append(vals)

    syn_median = np.empty(52)
    syn_p_low = np.empty(52)
    syn_p_high = np.empty(52)
    for w in weeks:
        if syn_weekly_by_week[w]:
            all_vals = np.concatenate(syn_weekly_by_week[w])
            syn_median[w - 1] = np.median(all_vals)
            syn_p_low[w - 1] = np.percentile(all_vals, percentiles[0])
            syn_p_high[w - 1] = np.percentile(all_vals, percentiles[1])
        else:
            syn_median[w - 1] = syn_p_low[w - 1] = syn_p_high[w - 1] = np.nan

    # Plot synthetic range and median
    ax.fill_between(
        weeks, syn_p_low, syn_p_high,
        alpha=ALPHA_FILL, color=synthetic_color,
        label=f'{synthetic_label} ({percentiles[0]}-{percentiles[1]}%)'
    )
    ax.plot(
        weeks, syn_median,
        color=synthetic_color, linewidth=LINEWIDTH_MEDIUM, linestyle='-',
        label=f'{synthetic_label} (median)'
    )

    # Plot historic range and median
    ax.fill_between(
        weeks, hist_p_low, hist_p_high,
        alpha=ALPHA_FILL * 0.7, color=HISTORIC_COLOR,
        label=f'{HISTORIC_LABEL} ({percentiles[0]}-{percentiles[1]}%)'
    )
    ax.plot(
        weeks, hist_median,
        color=HISTORIC_COLOR, linewidth=LINEWIDTH_THICK, linestyle='--',
        label=f'{HISTORIC_LABEL} (median)'
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 52)
    ax.set_ylim(bottom=0)
    ax.set_xticks(MONTH_WEEK_STARTS)
    ax.set_xticklabels(MONTH_LABELS)

    if show_legend:
        ax.legend(loc='upper right', frameon=True)
    ax.grid(False)

    return ax


def plot_fdc_percentile_comparison(
    Q_historic: pd.DataFrame,
    Q_synthetic: dict,
    sites: list = None,
    ax=None,
    ylabel: str = 'Streamflow (MGD)',
    xlabel: str = 'Exceedance Probability',
    percentiles: tuple = (5, 95),
    show_legend: bool = False,
    synthetic_color: str = None,
    synthetic_label: str = 'Synthetic',
    log_scale: bool = True,
    _hist_agg: pd.Series = None,
    _syn_agg: dict = None,
):
    """
    Plot flow duration curve comparison showing percentile range across years.

    Parameters
    ----------
    Q_historic : pd.DataFrame
        Historical daily flow data with DatetimeIndex
    Q_synthetic : dict
        Dictionary of synthetic flow DataFrames keyed by realization ID
    sites : list, optional
        List of sites to aggregate. Defaults to NYC_RESERVOIRS.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    ylabel, xlabel : str
        Axis labels
    percentiles : tuple
        Lower and upper percentiles for range (default 5, 95)
    show_legend : bool
        Whether to show legend
    synthetic_color : str, optional
        Color for synthetic data.
    synthetic_label : str
        Label for synthetic data in legend
    log_scale : bool
        Whether to use log scale for y-axis
    _hist_agg : pd.Series, optional
        Pre-aggregated historic flow.
    _syn_agg : dict, optional
        Pre-aggregated synthetic flows.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if synthetic_color is None:
        synthetic_color = DATASET_COLORS['stationary_ensemble']

    # Use pre-aggregated data if available
    if _hist_agg is None:
        _hist_agg = _get_aggregate_flow(Q_historic, sites)
    if _syn_agg is None:
        _syn_agg = _pre_aggregate_synthetic(Q_synthetic, sites)

    # Standard exceedance probabilities
    n_points = 100
    exceedance_probs = np.linspace(0.01, 0.99, n_points)

    def compute_annual_fdcs(flow_series):
        """Compute FDCs for each year using vectorized operations."""
        values = flow_series.dropna().values
        years = flow_series.dropna().index.year

        annual_fdcs = []
        for year in np.unique(years):
            mask = years == year
            year_vals = values[mask]
            if len(year_vals) < 300:
                continue
            sorted_flows = np.sort(year_vals)[::-1]
            n = len(sorted_flows)
            probs = np.arange(1, n + 1) / (n + 1)
            fdc_values = np.interp(exceedance_probs, probs, sorted_flows)
            annual_fdcs.append(fdc_values)

        return np.array(annual_fdcs)

    # Compute historic FDCs
    hist_fdcs = compute_annual_fdcs(_hist_agg)
    hist_median = np.median(hist_fdcs, axis=0)
    hist_p_low = np.percentile(hist_fdcs, percentiles[0], axis=0)
    hist_p_high = np.percentile(hist_fdcs, percentiles[1], axis=0)

    # Compute synthetic FDCs (all realizations combined)
    all_syn_fdcs = []
    for real_id, flow_series in _syn_agg.items():
        fdcs = compute_annual_fdcs(flow_series)
        if len(fdcs) > 0:
            all_syn_fdcs.append(fdcs)

    all_syn_fdcs = np.vstack(all_syn_fdcs)
    syn_median = np.median(all_syn_fdcs, axis=0)
    syn_p_low = np.percentile(all_syn_fdcs, percentiles[0], axis=0)
    syn_p_high = np.percentile(all_syn_fdcs, percentiles[1], axis=0)

    # Plot synthetic range and median
    ax.fill_between(
        exceedance_probs, syn_p_low, syn_p_high,
        alpha=ALPHA_FILL, color=synthetic_color,
        label=f'{synthetic_label} ({percentiles[0]}-{percentiles[1]}%)'
    )
    ax.plot(
        exceedance_probs, syn_median,
        color=synthetic_color, linewidth=LINEWIDTH_MEDIUM, linestyle='-',
        label=f'{synthetic_label} (median)'
    )

    # Plot historic range and median
    ax.fill_between(
        exceedance_probs, hist_p_low, hist_p_high,
        alpha=ALPHA_FILL * 0.7, color=HISTORIC_COLOR,
        label=f'{HISTORIC_LABEL} ({percentiles[0]}-{percentiles[1]}%)'
    )
    ax.plot(
        exceedance_probs, hist_median,
        color=HISTORIC_COLOR, linewidth=LINEWIDTH_THICK, linestyle='--',
        label=f'{HISTORIC_LABEL} (median)'
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)

    if log_scale:
        ax.set_yscale('log')

    if show_legend:
        ax.legend(loc='upper right', frameon=True)
    ax.grid(False)

    return ax


def plot_autocorrelation_comparison(
    Q_historic: pd.DataFrame,
    Q_synthetic: dict,
    sites: list = None,
    ax=None,
    max_lag: int = 30,
    ylabel: str = 'Autocorrelation',
    xlabel: str = 'Lag (days)',
    show_legend: bool = False,
    synthetic_color: str = None,
    synthetic_label: str = 'Synthetic',
    _hist_agg: pd.Series = None,
    _syn_agg: dict = None,
):
    """
    Plot autocorrelation comparison for synthetic vs historic data.

    Parameters
    ----------
    Q_historic : pd.DataFrame
        Historical daily flow data with DatetimeIndex
    Q_synthetic : dict
        Dictionary of synthetic flow DataFrames keyed by realization ID
    sites : list, optional
        List of sites to aggregate. Defaults to NYC_RESERVOIRS.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    max_lag : int
        Maximum lag for autocorrelation (default 30 days)
    ylabel, xlabel : str
        Axis labels
    show_legend : bool
        Whether to show legend
    synthetic_color : str, optional
        Color for synthetic data.
    synthetic_label : str
        Label for synthetic data in legend
    _hist_agg : pd.Series, optional
        Pre-aggregated historic flow.
    _syn_agg : dict, optional
        Pre-aggregated synthetic flows.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if synthetic_color is None:
        synthetic_color = DATASET_COLORS['stationary_ensemble']

    # Use pre-aggregated data if available
    if _hist_agg is None:
        _hist_agg = _get_aggregate_flow(Q_historic, sites)
    if _syn_agg is None:
        _syn_agg = _pre_aggregate_synthetic(Q_synthetic, sites)

    lag_range = np.arange(1, max_lag + 1)

    # Vectorized autocorrelation for historic data
    hist_series = _hist_agg.dropna().values
    hist_autocorr = _vectorized_autocorr(hist_series, max_lag)

    # Vectorized autocorrelation for each synthetic realization
    n_realizations = len(_syn_agg)
    syn_autocorr = np.zeros((n_realizations, max_lag))

    for i, (real_id, flow_series) in enumerate(_syn_agg.items()):
        series = flow_series.dropna().values
        syn_autocorr[i, :] = _vectorized_autocorr(series, max_lag)

    # Plot synthetic range and median
    ax.fill_between(
        lag_range,
        np.nanmin(syn_autocorr, axis=0),
        np.nanmax(syn_autocorr, axis=0),
        alpha=ALPHA_FILL, color=synthetic_color,
        label=f'{synthetic_label} (range)'
    )
    ax.plot(
        lag_range, np.nanmedian(syn_autocorr, axis=0),
        color=synthetic_color, linewidth=LINEWIDTH_MEDIUM, linestyle='-',
        label=f'{synthetic_label} (median)'
    )

    # Plot historic autocorrelation
    ax.plot(
        lag_range, hist_autocorr,
        color=HISTORIC_COLOR, linewidth=LINEWIDTH_THICK, linestyle='--',
        label=f'{HISTORIC_LABEL}'
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, max_lag + 1)
    ax.set_ylim(-0.2, 1.0)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    if show_legend:
        ax.legend(loc='upper right', frameon=True)
    ax.grid(False)

    return ax


def plot_monthly_streamflow_percentiles(
    Q_historic: pd.DataFrame,
    Q_synthetic: dict,
    sites: list = None,
    ax=None,
    ylabel: str = 'Streamflow (MGD)',
    xlabel: str = 'Month',
    percentiles: tuple = (5, 95),
    show_legend: bool = False,
    synthetic_color: str = None,
    synthetic_label: str = 'Synthetic',
    _hist_agg: pd.Series = None,
    _syn_agg: dict = None,
):
    """
    Plot monthly streamflow percentile bands for synthetic vs historic data.

    Parameters
    ----------
    Q_historic : pd.DataFrame
        Historical daily flow data with DatetimeIndex
    Q_synthetic : dict
        Dictionary of synthetic flow DataFrames keyed by realization ID
    sites : list, optional
        List of sites to aggregate. Defaults to NYC_RESERVOIRS.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    ylabel, xlabel : str
        Axis labels
    percentiles : tuple
        Lower and upper percentiles for range (default 5, 95)
    show_legend : bool
        Whether to show legend
    synthetic_color : str, optional
        Color for synthetic data.
    synthetic_label : str
        Label for synthetic data in legend
    _hist_agg : pd.Series, optional
        Pre-aggregated historic flow.
    _syn_agg : dict, optional
        Pre-aggregated synthetic flows.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    if synthetic_color is None:
        synthetic_color = DATASET_COLORS['stationary_ensemble']

    # Use pre-aggregated data if available
    if _hist_agg is None:
        _hist_agg = _get_aggregate_flow(Q_historic, sites)
    if _syn_agg is None:
        _syn_agg = _pre_aggregate_synthetic(Q_synthetic, sites)

    # Process historic data to monthly
    Q_hist_monthly = _hist_agg.resample('M').mean()
    hist_months = Q_hist_monthly.index.month
    hist_values = Q_hist_monthly.values

    months = np.arange(1, 13)
    hist_median = np.empty(12)
    hist_p_low = np.empty(12)
    hist_p_high = np.empty(12)
    for m in months:
        vals = hist_values[hist_months == m]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            hist_median[m - 1] = np.median(vals)
            hist_p_low[m - 1] = np.percentile(vals, percentiles[0])
            hist_p_high[m - 1] = np.percentile(vals, percentiles[1])
        else:
            hist_median[m - 1] = hist_p_low[m - 1] = hist_p_high[m - 1] = np.nan

    # Process synthetic data: collect all monthly values by month
    syn_monthly_by_month = {m: [] for m in range(1, 13)}
    for real_id, flow_series in _syn_agg.items():
        monthly = flow_series.resample('M').mean()
        m_arr = monthly.index.month
        v_arr = monthly.values
        for m in months:
            vals = v_arr[m_arr == m]
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                syn_monthly_by_month[m].append(vals)

    syn_median = np.empty(12)
    syn_p_low = np.empty(12)
    syn_p_high = np.empty(12)
    for m in months:
        if syn_monthly_by_month[m]:
            all_vals = np.concatenate(syn_monthly_by_month[m])
            syn_median[m - 1] = np.median(all_vals)
            syn_p_low[m - 1] = np.percentile(all_vals, percentiles[0])
            syn_p_high[m - 1] = np.percentile(all_vals, percentiles[1])
        else:
            syn_median[m - 1] = syn_p_low[m - 1] = syn_p_high[m - 1] = np.nan

    # Plot synthetic range and median
    ax.fill_between(
        months, syn_p_low, syn_p_high,
        alpha=ALPHA_FILL, color=synthetic_color,
        label=f'{synthetic_label} ({percentiles[0]}-{percentiles[1]}%)'
    )
    ax.plot(
        months, syn_median,
        color=synthetic_color, linewidth=LINEWIDTH_MEDIUM, linestyle='-',
        label=f'{synthetic_label} (median)'
    )

    # Plot historic range and median
    ax.fill_between(
        months, hist_p_low, hist_p_high,
        alpha=ALPHA_FILL * 0.7, color=HISTORIC_COLOR,
        label=f'{HISTORIC_LABEL} ({percentiles[0]}-{percentiles[1]}%)'
    )
    ax.plot(
        months, hist_median,
        color=HISTORIC_COLOR, linewidth=LINEWIDTH_THICK, linestyle='--',
        label=f'{HISTORIC_LABEL} (median)'
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(bottom=0)
    ax.set_xticks(months)
    ax.set_xticklabels(MONTH_LABELS)

    if show_legend:
        ax.legend(loc='upper right', frameon=True)
    ax.grid(False)

    return ax


def plot_pvalue_comparison(
    Q_historic: pd.DataFrame,
    Q_synthetic: dict,
    sites: list = None,
    ax=None,
    ylabel: str = 'p-value',
    xlabel: str = 'Month',
    significance_threshold: float = 0.05,
    wilcoxon_color: str = '#648FFF',
    levene_color: str = '#DC267F',
    show_legend: bool = False,
    _hist_agg: pd.Series = None,
    _syn_agg: dict = None,
):
    """
    Plot Levene and Wilcoxon test p-values comparing historic vs synthetic by month.

    Parameters
    ----------
    Q_historic : pd.DataFrame
        Historical daily flow data with DatetimeIndex
    Q_synthetic : dict
        Dictionary of synthetic flow DataFrames keyed by realization ID
    sites : list, optional
        List of sites to aggregate. Defaults to NYC_RESERVOIRS.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    ylabel, xlabel : str
        Axis labels
    significance_threshold : float
        Threshold for significance line (default 0.05)
    wilcoxon_color : str
        Color for Wilcoxon test bars
    levene_color : str
        Color for Levene test bars
    show_legend : bool
        Whether to show legend
    _hist_agg : pd.Series, optional
        Pre-aggregated historic flow.
    _syn_agg : dict, optional
        Pre-aggregated synthetic flows.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    # Use pre-aggregated data if available
    if _hist_agg is None:
        _hist_agg = _get_aggregate_flow(Q_historic, sites)
    if _syn_agg is None:
        _syn_agg = _pre_aggregate_synthetic(Q_synthetic, sites)

    # Process historic data to monthly
    Q_hist_monthly = _hist_agg.resample('M').mean()
    H_df = Q_hist_monthly.to_frame(name='flow')
    H_pivot = H_df.pivot_table(
        index=H_df.index.year, columns=H_df.index.month, values='flow'
    )
    H_pivot = H_pivot.reindex(columns=range(1, 13))
    H_proc = H_pivot.values

    # Process synthetic data to monthly (vectorized pivot per realization)
    syn_monthly_list = []
    for real_id, flow_series in _syn_agg.items():
        monthly = flow_series.resample('M').mean()
        monthly_df = monthly.to_frame(name='flow')
        monthly_pivot = monthly_df.pivot_table(
            index=monthly_df.index.year, columns=monthly_df.index.month, values='flow'
        )
        monthly_pivot = monthly_pivot.reindex(columns=range(1, 13))
        syn_monthly_list.append(monthly_pivot.values)

    S_proc = np.vstack(syn_monthly_list)

    # Compute p-values for each month
    n_months = 12
    wilcoxon_pvals = np.zeros(n_months)
    levene_pvals = np.zeros(n_months)

    for i in range(n_months):
        try:
            h_vals = H_proc[:, i]
            s_vals = S_proc[:, i]
            h_valid = h_vals[~np.isnan(h_vals)]
            s_valid = s_vals[~np.isnan(s_vals)]

            if len(h_valid) > 0 and len(s_valid) > 0:
                wilcoxon_pvals[i] = ranksums(h_valid, s_valid)[1]
                levene_pvals[i] = levene(h_valid, s_valid)[1]
            else:
                wilcoxon_pvals[i] = np.nan
                levene_pvals[i] = np.nan
        except Exception:
            wilcoxon_pvals[i] = np.nan
            levene_pvals[i] = np.nan

    # Create grouped bar chart
    months = np.arange(1, 13)
    bar_width = 0.35

    ax.bar(months - bar_width/2, wilcoxon_pvals, bar_width,
           label='Wilcoxon', color=wilcoxon_color, edgecolor='k')
    ax.bar(months + bar_width/2, levene_pvals, bar_width,
           label='Levene', color=levene_color, edgecolor='k')

    ax.axhline(significance_threshold, color='k', linewidth=1, linestyle='--',
               label=f'p={significance_threshold}')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(months)
    ax.set_xticklabels(MONTH_LABELS)

    if show_legend:
        ax.legend(loc='upper right', frameon=True)
    ax.grid(False)

    return ax


def plot_ensemble_summary_figure(
    Q_historic: pd.DataFrame,
    Q_synthetic: dict,
    sites: list = None,
    dataset_id: str = 'stationary_ensemble',
    fname: str = None,
    figsize: tuple = (12, 14),
    percentiles: tuple = (0, 100),
    max_lag: int = 30,
):
    """
    Create 4-panel summary figure for manuscript with autocorrelation, FDC,
    monthly flow ranges, and statistical test p-values.

    Layout (2, 1, 1):
    - Top row: Autocorrelation (left), FDC ranges (right)
    - Middle row: Monthly flow ranges (full width)
    - Bottom row: Levene & Wilcoxon p-values bar chart (full width)

    Uses aggregate flow from NYC reservoirs (sum of cannonsville, pepacton, neversink).

    Parameters
    ----------
    Q_historic : pd.DataFrame
        Historical daily flow data with DatetimeIndex
    Q_synthetic : dict
        Dictionary of synthetic flow DataFrames keyed by realization ID
    sites : list, optional
        List of sites to aggregate. Defaults to NYC_RESERVOIRS.
    dataset_id : str
        Dataset identifier for color scheme
    fname : str, optional
        Filename to save figure
    figsize : tuple
        Figure size (width, height)
    percentiles : tuple
        Lower and upper percentiles for range (default 5, 95)
    max_lag : int
        Maximum lag for autocorrelation plot (default 30 days)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    apply_publication_style()

    if sites is None:
        sites = NYC_RESERVOIRS

    synthetic_color = DATASET_COLORS.get(dataset_id, DATASET_COLORS['stationary_ensemble'])
    synthetic_label = DATASET_LABELS.get(dataset_id, 'Synthetic')

    wilcoxon_color = "#2E2E2E"
    levene_color = "#B0B0B0"

    # Pre-aggregate flows ONCE for all panels
    hist_agg = _get_aggregate_flow(Q_historic, sites)
    syn_agg = _pre_aggregate_synthetic(Q_synthetic, sites)

    # Create figure with GridSpec layout
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.3])

    ax_autocorr = fig.add_subplot(gs[0, 0])
    ax_fdc = fig.add_subplot(gs[0, 1])
    ax_monthly = fig.add_subplot(gs[1, :])
    ax_pvalues = fig.add_subplot(gs[2, :])

    # Panel A: Autocorrelation comparison
    plot_autocorrelation_comparison(
        Q_historic, Q_synthetic,
        ax=ax_autocorr, max_lag=max_lag,
        synthetic_color=synthetic_color, synthetic_label=synthetic_label,
        show_legend=False,
        _hist_agg=hist_agg, _syn_agg=syn_agg,
    )

    # Panel B: FDC percentile comparison
    plot_fdc_percentile_comparison(
        Q_historic, Q_synthetic,
        ax=ax_fdc, percentiles=percentiles,
        synthetic_color=synthetic_color, synthetic_label=synthetic_label,
        show_legend=False,
        _hist_agg=hist_agg, _syn_agg=syn_agg,
    )

    # Panel C: Monthly streamflow percentiles
    plot_monthly_streamflow_percentiles(
        Q_historic, Q_synthetic,
        ax=ax_monthly, percentiles=percentiles,
        synthetic_color=synthetic_color, synthetic_label=synthetic_label,
        show_legend=False,
        _hist_agg=hist_agg, _syn_agg=syn_agg,
    )

    # Panel D: Levene & Wilcoxon p-values
    plot_pvalue_comparison(
        Q_historic, Q_synthetic,
        ax=ax_pvalues,
        wilcoxon_color=wilcoxon_color, levene_color=levene_color,
        show_legend=False,
        _hist_agg=hist_agg, _syn_agg=syn_agg,
    )

    # Shared legend
    legend_handles = [
        Patch(facecolor=synthetic_color, alpha=ALPHA_FILL,
              label=f'{synthetic_label} (range)'),
        Line2D([0], [0], color=synthetic_color, linewidth=LINEWIDTH_MEDIUM,
               linestyle='-', label=f'{synthetic_label} (median)'),
        Patch(facecolor=HISTORIC_COLOR, alpha=ALPHA_FILL * 0.7,
              label=f'{HISTORIC_LABEL} (range)'),
        Line2D([0], [0], color=HISTORIC_COLOR, linewidth=LINEWIDTH_THICK,
               linestyle='--', label=f'{HISTORIC_LABEL} (median)'),
        Patch(facecolor=wilcoxon_color, label='Wilcoxon p'),
        Patch(facecolor=levene_color, label='Levene p'),
        Line2D([0], [0], color='k', linestyle='--', linewidth=1, label='p=0.05'),
    ]

    fig.legend(
        handles=legend_handles,
        loc='lower center', ncol=7, frameon=False,
        bbox_to_anchor=(0.5, -0.02), fontsize=9,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    if fname:
        plt.savefig(fname, dpi=DPI_PRINT, bbox_inches='tight')
        print(f"Saved figure to {fname}")
        plt.close(fig)

    return fig


def plot_ensemble_convergence(
    Q_syn_site: pd.DataFrame,
    realization_ids: list,
    site: str = 'delMontague',
    axes=None,
    n_bootstrap_samples: int = 50,
    step_size: int = None,
    n_steps: int = 40,
    synthetic_color: str = None,
    fname: str = None,
    figsize: tuple = (12, 5),
):
    """
    Plot convergence diagnostics for ensemble mean and variance of annual flow.

    Uses bootstrap resampling to show how the range of ensemble statistics
    narrows as the number of realizations increases.

    Parameters
    ----------
    Q_syn_site : pd.DataFrame
        Synthetic flow data for a single site, with columns as realization IDs
        and a DatetimeIndex.
    realization_ids : list
        List of realization IDs (must match columns in Q_syn_site).
    site : str
        Site name (used for axis labels).
    axes : tuple of (ax_mean, ax_var), optional
        Two matplotlib axes to plot on.
    n_bootstrap_samples : int
        Number of bootstrap resamples per subset size (default 50).
    step_size : int, optional
        Step size for the number-of-realizations sequence.
        If None, automatically computed from n_steps.
    n_steps : int
        Approximate number of evaluation points along the x-axis (default 40).
        Ignored if step_size is provided.
    synthetic_color : str, optional
        Color for the fill and line.
    fname : str, optional
        If provided, save figure to this path.
    figsize : tuple
        Figure size if creating new axes (default (12, 5)).

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure object (only if axes were not provided).
    """
    if synthetic_color is None:
        synthetic_color = DATASET_COLORS['stationary_ensemble']

    n_realizations = len(realization_ids)

    # Auto-compute step_size to get ~n_steps evaluation points
    if step_size is None:
        step_size = max(1, n_realizations // n_steps)

    # Pre-compute annual sums once
    annual_sums = Q_syn_site[realization_ids].resample('YE').sum()
    realization_means = annual_sums.mean(axis=0).values
    realization_vars = annual_sums.var(axis=0).values

    # Subset sizes to evaluate
    n_subset_sizes = list(range(1, n_realizations + 1, step_size))
    if n_subset_sizes[-1] != n_realizations:
        n_subset_sizes.append(n_realizations)
    n_subset_sizes = np.array(n_subset_sizes)

    # Bootstrap resampling using integer indices
    mean_ranges = np.empty((len(n_subset_sizes), 2))
    var_ranges = np.empty((len(n_subset_sizes), 2))

    rng = np.random.default_rng(42)
    indices = np.arange(n_realizations)

    for i, n_real in enumerate(n_subset_sizes):
        bootstrap_idx = np.array([
            rng.choice(indices, size=n_real, replace=False)
            for _ in range(n_bootstrap_samples)
        ])

        boot_means = realization_means[bootstrap_idx].mean(axis=1)
        boot_vars = realization_vars[bootstrap_idx].mean(axis=1)

        mean_ranges[i] = [boot_means.min(), boot_means.max()]
        var_ranges[i] = [boot_vars.min(), boot_vars.max()]

    # Plotting
    created_fig = False
    if axes is None:
        fig, (ax_mean, ax_var) = plt.subplots(1, 2, figsize=figsize)
        created_fig = True
    else:
        ax_mean, ax_var = axes
        fig = ax_mean.get_figure()

    ax_mean.fill_between(
        n_subset_sizes, mean_ranges[:, 0], mean_ranges[:, 1],
        alpha=ALPHA_FILL, color=synthetic_color, label='Bootstrap range',
    )
    ax_mean.plot(
        n_subset_sizes, mean_ranges.mean(axis=1),
        color=synthetic_color, linewidth=LINEWIDTH_MEDIUM, linestyle='-',
        label='Midpoint',
    )
    ax_mean.set_xlabel('Number of Realizations')
    ax_mean.set_ylabel('Mean Annual Flow (MG)')
    ax_mean.set_title(f'Mean Convergence ({site})')
    ax_mean.set_xlim(0, n_realizations)
    ax_mean.legend(loc='upper right', frameon=True)
    ax_mean.grid(False)

    ax_var.fill_between(
        n_subset_sizes, var_ranges[:, 0], var_ranges[:, 1],
        alpha=ALPHA_FILL, color=synthetic_color, label='Bootstrap range',
    )
    ax_var.plot(
        n_subset_sizes, var_ranges.mean(axis=1),
        color=synthetic_color, linewidth=LINEWIDTH_MEDIUM, linestyle='-',
        label='Midpoint',
    )
    ax_var.set_xlabel('Number of Realizations')
    ax_var.set_ylabel('Variance of Annual Flow (MG$^2$)')
    ax_var.set_title(f'Variance Convergence ({site})')
    ax_var.set_xlim(0, n_realizations)
    ax_var.legend(loc='upper right', frameon=True)
    ax_var.grid(False)

    if created_fig:
        plt.tight_layout()
        if fname:
            plt.savefig(fname, dpi=DPI_PRINT, bbox_inches='tight')
            print(f"Saved convergence figure to {fname}")
            plt.close(fig)
        return fig

    return None
