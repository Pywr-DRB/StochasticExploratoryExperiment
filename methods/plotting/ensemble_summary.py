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

# Unit conversion: 1 MGD = 0.003785411784 MCM/day
MGD_TO_MCM = 3.785411784e-3


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


def plot_fdc_percentile_comparison(
    Q_historic: pd.DataFrame,
    Q_synthetic: dict,
    sites: list = None,
    ax=None,
    ylabel: str = 'Streamflow (MCM/day)',
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
    syn_median = np.median(all_syn_fdcs, axis=0) * MGD_TO_MCM
    syn_p_low = np.percentile(all_syn_fdcs, percentiles[0], axis=0) * MGD_TO_MCM
    syn_p_high = np.percentile(all_syn_fdcs, percentiles[1], axis=0) * MGD_TO_MCM
    hist_median = hist_median * MGD_TO_MCM
    hist_p_low = hist_p_low * MGD_TO_MCM
    hist_p_high = hist_p_high * MGD_TO_MCM

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
    percentiles: tuple = (1, 99),
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
        np.nanpercentile(syn_autocorr, percentiles[0], axis=0),
        np.nanpercentile(syn_autocorr, percentiles[1], axis=0),
        alpha=ALPHA_FILL, color=synthetic_color,
        label=f'{synthetic_label} ({percentiles[0]}-{percentiles[1]}%)'
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


def plot_weekly_streamflow_percentiles(
    Q_historic: pd.DataFrame,
    Q_synthetic: dict,
    sites: list = None,
    ax=None,
    ylabel: str = 'Streamflow (MCM/day)',
    xlabel: str = 'Week of Year',
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

    n_weeks = 52
    weeks = np.arange(1, n_weeks + 1)

    # Process historic data to weekly
    Q_hist_weekly = _hist_agg.resample('W').mean()
    hist_weeks = Q_hist_weekly.index.isocalendar().week.values.astype(int)
    hist_values = Q_hist_weekly.values

    hist_median = np.empty(n_weeks)
    hist_p_low = np.empty(n_weeks)
    hist_p_high = np.empty(n_weeks)
    for w in weeks:
        vals = hist_values[hist_weeks == w]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            hist_median[w - 1] = np.median(vals)
            hist_p_low[w - 1] = np.percentile(vals, percentiles[0])
            hist_p_high[w - 1] = np.percentile(vals, percentiles[1])
        else:
            hist_median[w - 1] = hist_p_low[w - 1] = hist_p_high[w - 1] = np.nan

    # Process synthetic data: collect all weekly values by week
    syn_weekly_by_week = {w: [] for w in range(1, n_weeks + 1)}
    for real_id, flow_series in _syn_agg.items():
        weekly = flow_series.resample('W').mean()
        w_arr = weekly.index.isocalendar().week.values.astype(int)
        v_arr = weekly.values
        for w in weeks:
            vals = v_arr[w_arr == w]
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                syn_weekly_by_week[w].append(vals)

    syn_median = np.empty(n_weeks)
    syn_p_low = np.empty(n_weeks)
    syn_p_high = np.empty(n_weeks)
    for w in weeks:
        if syn_weekly_by_week[w]:
            all_vals = np.concatenate(syn_weekly_by_week[w])
            syn_median[w - 1] = np.median(all_vals)
            syn_p_low[w - 1] = np.percentile(all_vals, percentiles[0])
            syn_p_high[w - 1] = np.percentile(all_vals, percentiles[1])
        else:
            syn_median[w - 1] = syn_p_low[w - 1] = syn_p_high[w - 1] = np.nan

    # Convert MGD to MCM/day before plotting
    syn_p_low   = syn_p_low   * MGD_TO_MCM
    syn_p_high  = syn_p_high  * MGD_TO_MCM
    syn_median  = syn_median  * MGD_TO_MCM
    hist_p_low  = hist_p_low  * MGD_TO_MCM
    hist_p_high = hist_p_high * MGD_TO_MCM
    hist_median = hist_median * MGD_TO_MCM

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
    ax.set_xlim(1, 52.85)
    ax.set_yscale('log')
    ax.set_xticks(MONTH_WEEK_STARTS)
    ax.set_xticklabels(MONTH_LABELS)

    if show_legend:
        ax.legend(loc='upper right', frameon=True)
    ax.grid(False)

    return ax


def _compute_weekly_pvalues(
    hist_agg: pd.Series,
    syn_agg: dict,
    significance_threshold: float = 0.05,
) -> tuple:
    """
    Compute per-realization Wilcoxon and Levene pass fractions for each of 52 weeks.

    For each week, each realization is tested independently against the full
    historic record.  The returned arrays contain the *fraction of realizations*
    whose p-value exceeds ``significance_threshold`` — i.e. values close to 1.0
    indicate that nearly all realizations are statistically indistinguishable
    from the historic for that week.

    Returns
    -------
    wilcoxon_pass_frac, levene_pass_frac : np.ndarray of shape (52,)
    """
    n_weeks = 52
    weeks = np.arange(1, n_weeks + 1)

    Q_hist_weekly = hist_agg.resample('W').mean()
    hist_week_nums = Q_hist_weekly.index.isocalendar().week.values.astype(int)
    hist_vals_arr = Q_hist_weekly.values

    # Pre-compute weekly values per realization
    syn_by_real_week = {}
    for real_id, flow_series in syn_agg.items():
        weekly = flow_series.resample('W').mean()
        w_arr = weekly.index.isocalendar().week.values.astype(int)
        v_arr = weekly.values
        real_weeks = {}
        for w in weeks:
            vals = v_arr[w_arr == w]
            vals = vals[~np.isnan(vals)]
            real_weeks[w] = vals
        syn_by_real_week[real_id] = real_weeks

    realization_ids = list(syn_by_real_week.keys())
    n_real = len(realization_ids)

    wilcoxon_pass_frac = np.full(n_weeks, np.nan)
    levene_pass_frac = np.full(n_weeks, np.nan)

    for w in weeks:
        h_valid = hist_vals_arr[hist_week_nums == w]
        h_valid = h_valid[~np.isnan(h_valid)]
        if len(h_valid) < 2:
            continue

        wilcoxon_passes = 0
        levene_passes = 0
        n_tested = 0

        for real_id in realization_ids:
            s_valid = syn_by_real_week[real_id][w]
            if len(s_valid) < 2:
                continue
            try:
                if ranksums(h_valid, s_valid)[1] > significance_threshold:
                    wilcoxon_passes += 1
                if levene(h_valid, s_valid)[1] > significance_threshold:
                    levene_passes += 1
                n_tested += 1
            except Exception:
                pass

        if n_tested > 0:
            wilcoxon_pass_frac[w - 1] = wilcoxon_passes / n_tested
            levene_pass_frac[w - 1] = levene_passes / n_tested

    return wilcoxon_pass_frac, levene_pass_frac


def plot_pvalue_comparison(
    Q_historic: pd.DataFrame,
    Q_synthetic: dict,
    sites: list = None,
    ax=None,
    which: str = 'wilcoxon',
    ylabel: str = None,
    xlabel: str = None,
    significance_threshold: float = 0.05,
    pass_threshold: float = 0.9,
    show_xticklabels: bool = True,
    show_legend: bool = False,
    _hist_agg: pd.Series = None,
    _syn_agg: dict = None,
):
    """
    Plot per-realization Wilcoxon or Levene pass fraction by week.

    Each bar shows the fraction of realizations whose p-value vs the historic
    record exceeds ``significance_threshold`` (default 0.05).  Values near 1.0
    indicate most realizations are statistically indistinguishable from the
    historic for that week.  A reference line is drawn at ``pass_threshold``
    (default 0.9).

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
    which : str
        Which test to plot: 'wilcoxon' or 'levene'.
    ylabel, xlabel : str, optional
        Axis labels. Auto-set from `which` if None.
    significance_threshold : float
        p-value cutoff for pass/fail per realization (default 0.05).
    pass_threshold : float
        Reference line for fraction-passing (default 0.9).
    show_xticklabels : bool
        Whether to show x-axis tick labels.
    show_legend : bool
        Whether to show legend.
    _hist_agg : pd.Series, optional
        Pre-aggregated historic flow.
    _syn_agg : dict, optional
        Pre-aggregated synthetic flows.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 2))

    if _hist_agg is None:
        _hist_agg = _get_aggregate_flow(Q_historic, sites)
    if _syn_agg is None:
        _syn_agg = _pre_aggregate_synthetic(Q_synthetic, sites)

    wilcoxon_pass_frac, levene_pass_frac = _compute_weekly_pvalues(
        _hist_agg, _syn_agg, significance_threshold=significance_threshold,
    )

    pass_frac = wilcoxon_pass_frac if which == 'wilcoxon' else levene_pass_frac

    if ylabel is None:
        ylabel = 'Wilcoxon\nfrac. pass' if which == 'wilcoxon' else 'Levene\nfrac. pass'
    if xlabel is None:
        xlabel = 'Week of Year' if show_xticklabels else ''

    weeks = np.arange(1, 53)
    bar_width = 0.75

    ax.bar(weeks, pass_frac, bar_width, align='edge',
           facecolor='white', edgecolor='black', linewidth=0.5)

    ax.axhline(pass_threshold, color='k', linewidth=1, linestyle='--',
               label=f'{int(pass_threshold * 100)}% pass')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(1, 52 + bar_width + 0.1)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(MONTH_WEEK_STARTS)
    if show_xticklabels:
        ax.set_xticklabels(MONTH_LABELS)
    else:
        ax.set_xticklabels([])

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
    weekly flow ranges, and statistical test p-values.

    Layout (2, 1, 1):
    - Top row: Autocorrelation (left), FDC ranges (right)
    - Middle row: Weekly flow ranges (full width)
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

    # Pre-aggregate flows ONCE for all panels
    hist_agg = _get_aggregate_flow(Q_historic, sites)
    syn_agg = _pre_aggregate_synthetic(Q_synthetic, sites)

    # Create figure with GridSpec layout: 2 content rows + 2 thin p-value rows
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 0.22, 0.22])

    ax_autocorr = fig.add_subplot(gs[0, 0])
    ax_fdc      = fig.add_subplot(gs[0, 1])
    ax_monthly  = fig.add_subplot(gs[1, :])
    ax_wilcoxon = fig.add_subplot(gs[2, :])
    ax_levene   = fig.add_subplot(gs[3, :])

    # Panel A: Autocorrelation comparison
    plot_autocorrelation_comparison(
        Q_historic, Q_synthetic,
        ax=ax_autocorr, max_lag=max_lag, percentiles=percentiles,
        synthetic_color=synthetic_color, synthetic_label=synthetic_label,
        show_legend=False,
        _hist_agg=hist_agg, _syn_agg=syn_agg,
    )
    ax_autocorr.text(0.02, 0.97, 'a)', transform=ax_autocorr.transAxes,
                     fontsize=12, va='top', ha='left')

    # Panel B: FDC percentile comparison
    plot_fdc_percentile_comparison(
        Q_historic, Q_synthetic,
        ax=ax_fdc, percentiles=percentiles,
        synthetic_color=synthetic_color, synthetic_label=synthetic_label,
        show_legend=False,
        _hist_agg=hist_agg, _syn_agg=syn_agg,
    )
    ax_fdc.text(0.02, 0.97, 'b)', transform=ax_fdc.transAxes,
                fontsize=12, va='top', ha='left')

    # Panel C: Weekly streamflow percentiles
    plot_weekly_streamflow_percentiles(
        Q_historic, Q_synthetic,
        ax=ax_monthly, percentiles=percentiles,
        synthetic_color=synthetic_color, synthetic_label=synthetic_label,
        show_legend=False,
        _hist_agg=hist_agg, _syn_agg=syn_agg,
    )
    ax_monthly.text(0.01, 0.97, 'c)', transform=ax_monthly.transAxes,
                    fontsize=12, va='top', ha='left')

    # Panel D: Wilcoxon rank-sum p-values (no x-tick labels — shared with panel E)
    plot_pvalue_comparison(
        Q_historic, Q_synthetic,
        ax=ax_wilcoxon, which='wilcoxon',
        show_xticklabels=False,
        show_legend=False,
        _hist_agg=hist_agg, _syn_agg=syn_agg,
    )
    ax_wilcoxon.text(0.01, 0.85, 'd)', transform=ax_wilcoxon.transAxes,
                     fontsize=12, va='top', ha='left')

    # Panel E: Levene p-values (x-tick labels shown here)
    plot_pvalue_comparison(
        Q_historic, Q_synthetic,
        ax=ax_levene, which='levene',
        show_xticklabels=True,
        show_legend=False,
        _hist_agg=hist_agg, _syn_agg=syn_agg,
    )
    ax_levene.text(0.01, 0.85, 'e)', transform=ax_levene.transAxes,
                   fontsize=12, va='top', ha='left')

    # Shared legend (flow panels only; p-value panels are self-labeled via ylabel)
    legend_handles = [
        Patch(facecolor=synthetic_color, alpha=ALPHA_FILL,
              label=f'{synthetic_label} ({percentiles[0]}-{percentiles[1]}%)'),
        Line2D([0], [0], color=synthetic_color, linewidth=LINEWIDTH_MEDIUM,
               linestyle='-', label=f'{synthetic_label} (median)'),
        Patch(facecolor=HISTORIC_COLOR, alpha=ALPHA_FILL * 0.7,
              label=f'{HISTORIC_LABEL} ({percentiles[0]}-{percentiles[1]}%)'),
        Line2D([0], [0], color=HISTORIC_COLOR, linewidth=LINEWIDTH_THICK,
               linestyle='--', label=f'{HISTORIC_LABEL} (median)'),
        Line2D([0], [0], color='k', linestyle='--', linewidth=1, label='p=0.05'),
    ]

    fig.legend(
        handles=legend_handles,
        loc='lower center', ncol=5, frameon=False,
        bbox_to_anchor=(0.5, -0.02), fontsize=9,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.07)

    # Align y-axis labels across the three full-width panels
    fig.align_ylabels([ax_autocorr, ax_monthly, ax_wilcoxon, ax_levene])

    # Reduce blank space between panels d and e
    pos_w = ax_wilcoxon.get_position()
    pos_l = ax_levene.get_position()
    gap = pos_w.y0 - (pos_l.y0 + pos_l.height)
    shift = gap * 0.6
    ax_levene.set_position([pos_l.x0, pos_l.y0 + shift, pos_l.width, pos_l.height])

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
