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
from scipy.stats import pearsonr, ranksums, levene

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
):
    """
    Plot weekly streamflow percentile bands for synthetic vs historic data.

    Shows the median and percentile range (default 5th-95th) for both
    synthetic ensemble and historical data at weekly timescale.

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
    ylabel : str
        Y-axis label
    xlabel : str
        X-axis label
    percentiles : tuple
        Lower and upper percentiles for range (default 5, 95)
    show_legend : bool
        Whether to show legend on this axis
    synthetic_color : str, optional
        Color for synthetic data. Defaults to stationary_ensemble color.
    synthetic_label : str
        Label for synthetic data in legend

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    if synthetic_color is None:
        synthetic_color = DATASET_COLORS['stationary_ensemble']

    if sites is None:
        sites = NYC_RESERVOIRS

    # Process historic data to weekly (aggregate across sites)
    Q_hist_agg = _get_aggregate_flow(Q_historic, sites)
    Q_hist_weekly_df = pd.DataFrame({'flow': Q_hist_agg.resample('W').mean()})
    Q_hist_weekly_df['week_of_year'] = Q_hist_weekly_df.index.isocalendar().week

    # Compute weekly climatology for historic (5th-95th percentile)
    hist_stats = Q_hist_weekly_df.groupby('week_of_year')['flow'].agg(
        median='median',
        p_low=lambda x: np.percentile(x.dropna(), percentiles[0]),
        p_high=lambda x: np.percentile(x.dropna(), percentiles[1])
    )

    # Process synthetic data
    syn_flows = []
    for real_id, real_df in Q_synthetic.items():
        if isinstance(real_df, pd.DataFrame):
            flow_series = _get_aggregate_flow(real_df, sites)
        else:
            flow_series = real_df
        weekly = flow_series.resample('W').mean()
        weekly_df = pd.DataFrame({'flow': weekly})
        weekly_df['week_of_year'] = weekly_df.index.isocalendar().week
        weekly_df['realization'] = real_id
        syn_flows.append(weekly_df)
    syn_weekly_all = pd.concat(syn_flows, ignore_index=True)

    syn_stats = syn_weekly_all.groupby('week_of_year')['flow'].agg(
        median='median',
        p_low=lambda x: np.percentile(x.dropna(), percentiles[0]),
        p_high=lambda x: np.percentile(x.dropna(), percentiles[1])
    )

    # Ensure weeks 1-52 are present
    weeks = np.arange(1, 53)
    hist_stats = hist_stats.reindex(weeks)
    syn_stats = syn_stats.reindex(weeks)

    # Plot synthetic range and median
    ax.fill_between(
        weeks,
        syn_stats['p_low'],
        syn_stats['p_high'],
        alpha=ALPHA_FILL,
        color=synthetic_color,
        label=f'{synthetic_label} ({percentiles[0]}-{percentiles[1]}%)'
    )
    ax.plot(
        weeks,
        syn_stats['median'],
        color=synthetic_color,
        linewidth=LINEWIDTH_MEDIUM,
        linestyle='-',
        label=f'{synthetic_label} (median)'
    )

    # Plot historic range and median (dashed median line)
    ax.fill_between(
        weeks,
        hist_stats['p_low'],
        hist_stats['p_high'],
        alpha=ALPHA_FILL * 0.7,
        color=HISTORIC_COLOR,
        label=f'{HISTORIC_LABEL} ({percentiles[0]}-{percentiles[1]}%)'
    )
    ax.plot(
        weeks,
        hist_stats['median'],
        color=HISTORIC_COLOR,
        linewidth=LINEWIDTH_THICK,
        linestyle='--',
        label=f'{HISTORIC_LABEL} (median)'
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 52)
    ax.set_ylim(bottom=0)

    # Set month labels on x-axis
    ax.set_xticks(MONTH_WEEK_STARTS)
    ax.set_xticklabels(MONTH_LABELS)

    if show_legend:
        ax.legend(loc='upper right', frameon=True)

    # No grid
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
):
    """
    Plot flow duration curve comparison showing percentile range across years.

    Shows annual FDC percentile range for both synthetic ensemble and historical data.

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
    ylabel : str
        Y-axis label
    xlabel : str
        X-axis label
    percentiles : tuple
        Lower and upper percentiles for range (default 5, 95)
    show_legend : bool
        Whether to show legend on this axis
    synthetic_color : str, optional
        Color for synthetic data. Defaults to stationary_ensemble color.
    synthetic_label : str
        Label for synthetic data in legend
    log_scale : bool
        Whether to use log scale for y-axis

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if synthetic_color is None:
        synthetic_color = DATASET_COLORS['stationary_ensemble']

    if sites is None:
        sites = NYC_RESERVOIRS

    def compute_annual_fdcs(flow_series):
        """Compute FDCs for each year."""
        flow_df = pd.DataFrame({'flow': flow_series})
        flow_df['year'] = flow_df.index.year

        # Standard exceedance probabilities
        n_points = 100
        exceedance_probs = np.linspace(0.01, 0.99, n_points)

        annual_fdcs = []
        for year, group in flow_df.groupby('year'):
            if len(group) < 300:  # Skip incomplete years
                continue
            sorted_flows = np.sort(group['flow'].dropna().values)[::-1]
            n = len(sorted_flows)
            probs = np.arange(1, n + 1) / (n + 1)
            # Interpolate to standard probabilities
            fdc_values = np.interp(exceedance_probs, probs, sorted_flows)
            annual_fdcs.append(fdc_values)

        return exceedance_probs, np.array(annual_fdcs)

    # Compute historic FDCs (aggregate across sites)
    Q_hist_agg = _get_aggregate_flow(Q_historic, sites)
    hist_probs, hist_fdcs = compute_annual_fdcs(Q_hist_agg)

    hist_median = np.median(hist_fdcs, axis=0)
    hist_p_low = np.percentile(hist_fdcs, percentiles[0], axis=0)
    hist_p_high = np.percentile(hist_fdcs, percentiles[1], axis=0)

    # Compute synthetic FDCs (all realizations combined)
    all_syn_fdcs = []
    for real_id, real_df in Q_synthetic.items():
        if isinstance(real_df, pd.DataFrame):
            flow_series = _get_aggregate_flow(real_df, sites)
        else:
            flow_series = real_df
        _, fdcs = compute_annual_fdcs(flow_series)
        all_syn_fdcs.extend(fdcs)

    all_syn_fdcs = np.array(all_syn_fdcs)
    syn_median = np.median(all_syn_fdcs, axis=0)
    syn_p_low = np.percentile(all_syn_fdcs, percentiles[0], axis=0)
    syn_p_high = np.percentile(all_syn_fdcs, percentiles[1], axis=0)

    # Plot synthetic range and median
    ax.fill_between(
        hist_probs,
        syn_p_low,
        syn_p_high,
        alpha=ALPHA_FILL,
        color=synthetic_color,
        label=f'{synthetic_label} ({percentiles[0]}-{percentiles[1]}%)'
    )
    ax.plot(
        hist_probs,
        syn_median,
        color=synthetic_color,
        linewidth=LINEWIDTH_MEDIUM,
        linestyle='-',
        label=f'{synthetic_label} (median)'
    )

    # Plot historic range and median (dashed median line)
    ax.fill_between(
        hist_probs,
        hist_p_low,
        hist_p_high,
        alpha=ALPHA_FILL * 0.7,
        color=HISTORIC_COLOR,
        label=f'{HISTORIC_LABEL} ({percentiles[0]}-{percentiles[1]}%)'
    )
    ax.plot(
        hist_probs,
        hist_median,
        color=HISTORIC_COLOR,
        linewidth=LINEWIDTH_THICK,
        linestyle='--',
        label=f'{HISTORIC_LABEL} (median)'
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)

    if log_scale:
        ax.set_yscale('log')

    if show_legend:
        ax.legend(loc='upper right', frameon=True)

    # No grid
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
):
    """
    Plot autocorrelation comparison for synthetic vs historic data.

    Shows the autocorrelation function (ACF) for both synthetic ensemble
    and historical data, with ensemble range shown as fill.

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
    max_lag : int
        Maximum lag for autocorrelation (default 30 days)
    ylabel : str
        Y-axis label
    xlabel : str
        X-axis label
    show_legend : bool
        Whether to show legend on this axis
    synthetic_color : str, optional
        Color for synthetic data. Defaults to stationary_ensemble color.
    synthetic_label : str
        Label for synthetic data in legend

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if synthetic_color is None:
        synthetic_color = DATASET_COLORS['stationary_ensemble']

    if sites is None:
        sites = NYC_RESERVOIRS

    # Aggregate flows
    Q_hist_agg = _get_aggregate_flow(Q_historic, sites)

    # Calculate autocorrelation for historic data
    lag_range = np.arange(1, max_lag + 1)
    hist_series = Q_hist_agg.dropna().values
    hist_autocorr = np.zeros(len(lag_range))

    for j, lag in enumerate(lag_range):
        if len(hist_series) > lag:
            hist_autocorr[j] = pearsonr(hist_series[:-lag], hist_series[lag:])[0]
        else:
            hist_autocorr[j] = np.nan

    # Calculate autocorrelation for each synthetic realization
    n_realizations = len(Q_synthetic)
    syn_autocorr = np.zeros((n_realizations, len(lag_range)))

    for i, (real_id, real_df) in enumerate(Q_synthetic.items()):
        if isinstance(real_df, pd.DataFrame):
            flow_series = _get_aggregate_flow(real_df, sites)
        else:
            flow_series = real_df

        series = flow_series.dropna().values
        for j, lag in enumerate(lag_range):
            if len(series) > lag:
                syn_autocorr[i, j] = pearsonr(series[:-lag], series[lag:])[0]
            else:
                syn_autocorr[i, j] = np.nan

    # Plot synthetic range and median
    ax.fill_between(
        lag_range,
        np.nanmin(syn_autocorr, axis=0),
        np.nanmax(syn_autocorr, axis=0),
        alpha=ALPHA_FILL,
        color=synthetic_color,
        label=f'{synthetic_label} (range)'
    )
    ax.plot(
        lag_range,
        np.nanmedian(syn_autocorr, axis=0),
        color=synthetic_color,
        linewidth=LINEWIDTH_MEDIUM,
        linestyle='-',
        label=f'{synthetic_label} (median)'
    )

    # Plot historic autocorrelation with markers
    ax.plot(
        lag_range,
        hist_autocorr,
        color=HISTORIC_COLOR,
        linewidth=LINEWIDTH_THICK,
        linestyle='--',
        label=f'{HISTORIC_LABEL}'
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, max_lag + 1)
    ax.set_ylim(-0.2, 1.0)

    # Add horizontal line at 0
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
):
    """
    Plot monthly streamflow percentile bands for synthetic vs historic data.

    Shows the median and percentile range (default 5th-95th) for both
    synthetic ensemble and historical data at monthly timescale.

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
    ylabel : str
        Y-axis label
    xlabel : str
        X-axis label
    percentiles : tuple
        Lower and upper percentiles for range (default 5, 95)
    show_legend : bool
        Whether to show legend on this axis
    synthetic_color : str, optional
        Color for synthetic data. Defaults to stationary_ensemble color.
    synthetic_label : str
        Label for synthetic data in legend

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    if synthetic_color is None:
        synthetic_color = DATASET_COLORS['stationary_ensemble']

    if sites is None:
        sites = NYC_RESERVOIRS

    # Process historic data to monthly (aggregate across sites)
    Q_hist_agg = _get_aggregate_flow(Q_historic, sites)
    Q_hist_monthly_df = pd.DataFrame({'flow': Q_hist_agg.resample('M').mean()})
    Q_hist_monthly_df['month'] = Q_hist_monthly_df.index.month

    # Compute monthly climatology for historic
    hist_stats = Q_hist_monthly_df.groupby('month')['flow'].agg(
        median='median',
        p_low=lambda x: np.percentile(x.dropna(), percentiles[0]),
        p_high=lambda x: np.percentile(x.dropna(), percentiles[1])
    )

    # Process synthetic data
    syn_flows = []
    for real_id, real_df in Q_synthetic.items():
        if isinstance(real_df, pd.DataFrame):
            flow_series = _get_aggregate_flow(real_df, sites)
        else:
            flow_series = real_df
        monthly = flow_series.resample('M').mean()
        monthly_df = pd.DataFrame({'flow': monthly})
        monthly_df['month'] = monthly_df.index.month
        monthly_df['realization'] = real_id
        syn_flows.append(monthly_df)
    syn_monthly_all = pd.concat(syn_flows, ignore_index=True)

    syn_stats = syn_monthly_all.groupby('month')['flow'].agg(
        median='median',
        p_low=lambda x: np.percentile(x.dropna(), percentiles[0]),
        p_high=lambda x: np.percentile(x.dropna(), percentiles[1])
    )

    # Ensure months 1-12 are present
    months = np.arange(1, 13)
    hist_stats = hist_stats.reindex(months)
    syn_stats = syn_stats.reindex(months)

    # Plot synthetic range and median
    ax.fill_between(
        months,
        syn_stats['p_low'],
        syn_stats['p_high'],
        alpha=ALPHA_FILL,
        color=synthetic_color,
        label=f'{synthetic_label} ({percentiles[0]}-{percentiles[1]}%)'
    )
    ax.plot(
        months,
        syn_stats['median'],
        color=synthetic_color,
        linewidth=LINEWIDTH_MEDIUM,
        linestyle='-',
        label=f'{synthetic_label} (median)'
    )

    # Plot historic range and median (dashed median line)
    ax.fill_between(
        months,
        hist_stats['p_low'],
        hist_stats['p_high'],
        alpha=ALPHA_FILL * 0.7,
        color=HISTORIC_COLOR,
        label=f'{HISTORIC_LABEL} ({percentiles[0]}-{percentiles[1]}%)'
    )
    ax.plot(
        months,
        hist_stats['median'],
        color=HISTORIC_COLOR,
        linewidth=LINEWIDTH_THICK,
        linestyle='--',
        label=f'{HISTORIC_LABEL} (median)'
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(bottom=0)

    # Set month labels on x-axis
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
):
    """
    Plot Levene and Wilcoxon test p-values comparing historic vs synthetic by month.

    Creates a grouped bar chart showing p-values for both statistical tests
    for each month, with a significance threshold line.

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
    ylabel : str
        Y-axis label
    xlabel : str
        X-axis label
    significance_threshold : float
        Threshold for significance line (default 0.05)
    wilcoxon_color : str
        Color for Wilcoxon test bars
    levene_color : str
        Color for Levene test bars
    show_legend : bool
        Whether to show legend on this axis

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    if sites is None:
        sites = NYC_RESERVOIRS

    # Process historic data to monthly
    Q_hist_agg = _get_aggregate_flow(Q_historic, sites)
    Q_hist_monthly = Q_hist_agg.resample('M').mean()

    # Pivot historic data: rows = years, columns = months
    H_df = Q_hist_monthly.to_frame(name='flow')
    H_pivot = H_df.pivot_table(
        index=H_df.index.year,
        columns=H_df.index.month,
        values='flow'
    )
    # Ensure all 12 months are present
    H_pivot = H_pivot.reindex(columns=range(1, 13))
    H_proc = H_pivot.values  # Shape: (n_years, 12)

    # Process synthetic data to monthly and combine all realizations
    syn_monthly_list = []
    for real_id, real_df in Q_synthetic.items():
        if isinstance(real_df, pd.DataFrame):
            flow_series = _get_aggregate_flow(real_df, sites)
        else:
            flow_series = real_df

        monthly = flow_series.resample('M').mean()
        monthly_df = monthly.to_frame(name='flow')
        monthly_pivot = monthly_df.pivot_table(
            index=monthly_df.index.year,
            columns=monthly_df.index.month,
            values='flow'
        )
        monthly_pivot = monthly_pivot.reindex(columns=range(1, 13))
        syn_monthly_list.append(monthly_pivot.values)

    # Stack all realizations: shape (n_realizations * n_years, 12)
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
           label='Wilcoxon', color=wilcoxon_color, edgecolor='none')
    ax.bar(months + bar_width/2, levene_pvals, bar_width,
           label='Levene', color=levene_color, edgecolor='none')

    # Add significance threshold line
    ax.axhline(significance_threshold, color='k', linewidth=1, linestyle='--',
               label=f'p={significance_threshold}')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(0, 1.05)

    # Set month labels on x-axis
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
    percentiles: tuple = (5, 95),
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

    # P-value bar colors (from DATASET_COLORS_ALT for contrast)
    wilcoxon_color = '#648FFF'  # Blue
    levene_color = '#DC267F'    # Magenta

    # Create figure with GridSpec layout (2, 1, 1)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.6])

    ax_autocorr = fig.add_subplot(gs[0, 0])   # Top-left: autocorrelation
    ax_fdc = fig.add_subplot(gs[0, 1])        # Top-right: FDC
    ax_monthly = fig.add_subplot(gs[1, :])    # Middle: monthly flow (full width)
    ax_pvalues = fig.add_subplot(gs[2, :])    # Bottom: p-values (full width)

    # Panel A (top-left): Autocorrelation comparison
    plot_autocorrelation_comparison(
        Q_historic,
        Q_synthetic,
        sites=sites,
        ax=ax_autocorr,
        max_lag=max_lag,
        synthetic_color=synthetic_color,
        synthetic_label=synthetic_label,
        show_legend=False,
    )

    # Panel B (top-right): FDC percentile comparison
    plot_fdc_percentile_comparison(
        Q_historic,
        Q_synthetic,
        sites=sites,
        ax=ax_fdc,
        percentiles=percentiles,
        synthetic_color=synthetic_color,
        synthetic_label=synthetic_label,
        show_legend=False,
    )

    # Panel C (middle): Monthly streamflow percentiles
    plot_monthly_streamflow_percentiles(
        Q_historic,
        Q_synthetic,
        sites=sites,
        ax=ax_monthly,
        percentiles=percentiles,
        synthetic_color=synthetic_color,
        synthetic_label=synthetic_label,
        show_legend=False,
    )

    # Panel D (bottom): Levene & Wilcoxon p-values
    plot_pvalue_comparison(
        Q_historic,
        Q_synthetic,
        sites=sites,
        ax=ax_pvalues,
        wilcoxon_color=wilcoxon_color,
        levene_color=levene_color,
        show_legend=False,
    )

    # Create shared legend below the figure
    # Build legend handles manually for consistent appearance
    legend_handles = [
        # Flow-based panels (autocorr, FDC, monthly)
        Patch(facecolor=synthetic_color, alpha=ALPHA_FILL,
              label=f'{synthetic_label} (range)'),
        Line2D([0], [0], color=synthetic_color, linewidth=LINEWIDTH_MEDIUM,
               linestyle='-', label=f'{synthetic_label} (median)'),
        Patch(facecolor=HISTORIC_COLOR, alpha=ALPHA_FILL * 0.7,
              label=f'{HISTORIC_LABEL} (range)'),
        Line2D([0], [0], color=HISTORIC_COLOR, linewidth=LINEWIDTH_THICK,
               linestyle='--', label=f'{HISTORIC_LABEL} (median)'),
        # P-value panel
        Patch(facecolor=wilcoxon_color, label='Wilcoxon p'),
        Patch(facecolor=levene_color, label='Levene p'),
        Line2D([0], [0], color='k', linestyle='--', linewidth=1, label='p=0.05'),
    ]

    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)  # Make room for legend

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
    step_size: int = 20,
    synthetic_color: str = None,
    fname: str = None,
    figsize: tuple = (12, 5),
):
    """
    Plot convergence diagnostics for ensemble mean and variance of annual flow.

    Uses bootstrap resampling to show how the range of ensemble statistics
    narrows as the number of realizations increases.

    Efficiency notes:
    - Pre-computes per-realization annual mean and variance as 1D arrays.
    - Bootstrap sampling operates on integer indices into these arrays,
      avoiding repeated DataFrame indexing and resampling.

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
        Two matplotlib axes to plot on. If None, creates a new 1x2 figure.
    n_bootstrap_samples : int
        Number of bootstrap resamples per subset size (default 50).
    step_size : int
        Step size for the number-of-realizations sequence (default 20).
    synthetic_color : str, optional
        Color for the fill and line. Defaults to stationary_ensemble color.
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

    # Pre-compute annual sums once: shape (n_years, n_realizations)
    annual_sums = Q_syn_site[realization_ids].resample('YE').sum()

    # Pre-compute per-realization statistics as 1D arrays
    # This avoids repeated DataFrame operations during bootstrap
    realization_means = annual_sums.mean(axis=0).values  # (n_realizations,)
    realization_vars = annual_sums.var(axis=0).values     # (n_realizations,)

    # Subset sizes to evaluate
    n_subset_sizes = list(range(1, n_realizations + 1, step_size))
    if n_subset_sizes[-1] != n_realizations:
        n_subset_sizes.append(n_realizations)
    n_subset_sizes = np.array(n_subset_sizes)

    # Bootstrap resampling using integer indices for speed
    mean_ranges = np.empty((len(n_subset_sizes), 2))
    var_ranges = np.empty((len(n_subset_sizes), 2))

    rng = np.random.default_rng(42)
    indices = np.arange(n_realizations)

    for i, n_real in enumerate(n_subset_sizes):
        # Generate all bootstrap index sets at once: (n_bootstrap, n_real)
        bootstrap_idx = np.array([
            rng.choice(indices, size=n_real, replace=False)
            for _ in range(n_bootstrap_samples)
        ])

        # Vectorized stat computation across all bootstrap samples
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

    # Mean annual flow convergence
    ax_mean.fill_between(
        n_subset_sizes,
        mean_ranges[:, 0],
        mean_ranges[:, 1],
        alpha=ALPHA_FILL,
        color=synthetic_color,
        label='Bootstrap range',
    )
    ax_mean.plot(
        n_subset_sizes,
        mean_ranges.mean(axis=1),
        color=synthetic_color,
        linewidth=LINEWIDTH_MEDIUM,
        linestyle='-',
        label='Midpoint',
    )
    ax_mean.set_xlabel('Number of Realizations')
    ax_mean.set_ylabel('Mean Annual Flow (MG)')
    ax_mean.set_title(f'Mean Convergence ({site})')
    ax_mean.set_yscale('log')
    ax_mean.legend(loc='upper right', frameon=True)
    ax_mean.grid(False)

    # Variance of annual flow convergence
    ax_var.fill_between(
        n_subset_sizes,
        var_ranges[:, 0],
        var_ranges[:, 1],
        alpha=ALPHA_FILL,
        color=synthetic_color,
        label='Bootstrap range',
    )
    ax_var.plot(
        n_subset_sizes,
        var_ranges.mean(axis=1),
        color=synthetic_color,
        linewidth=LINEWIDTH_MEDIUM,
        linestyle='-',
        label='Midpoint',
    )
    ax_var.set_xlabel('Number of Realizations')
    ax_var.set_ylabel('Variance of Annual Flow (MG$^2$)')
    ax_var.set_title(f'Variance Convergence ({site})')
    ax_var.set_yscale('log')
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


# Keep old function names as aliases for backwards compatibility
plot_weekly_streamflow_range = plot_weekly_streamflow_percentiles
plot_fdc_range_comparison = plot_fdc_percentile_comparison
