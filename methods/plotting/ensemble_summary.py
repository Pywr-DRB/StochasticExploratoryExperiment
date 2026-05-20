"""
Plotting functions for ensemble summary figures (main manuscript).

This module provides functions for creating publication-quality summary
figures comparing synthetic ensemble and historical streamflow data.
"""

import warnings

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
    ALPHA_FILL, ALPHA_BAND_OUTER, ALPHA_BAND_INNER,
    LINEWIDTH_THIN, LINEWIDTH_MEDIUM, LINEWIDTH_THICK,
    DPI_PRINT, apply_publication_style
)

# NYC reservoirs for aggregate flow calculation
NYC_RESERVOIRS = ['cannonsville', 'pepacton', 'neversink']

# Month labels for week-of-year x-axis
MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# Approximate week numbers for start of each month
MONTH_WEEK_STARTS = [1, 5, 9, 14, 18, 22, 27, 31, 35, 40, 44, 48]

# Re-exported for backward compatibility; canonical definition lives in methods.config.
from methods.config import MGD_TO_MCM


def _assign_usgs_water_year(dates: pd.DatetimeIndex) -> np.ndarray:
    """Return array of USGS water years (Oct 1 - Sep 30) for a DatetimeIndex.

    WY N spans Oct 1 of year N-1 through Sep 30 of year N (USGS convention),
    i.e. months Oct/Nov/Dec belong to the next year's WY. This is distinct
    from the FFMP June 1 water year used by methods.water_year.
    """
    months = dates.month.values
    years = dates.year.values
    return np.where(months >= 10, years + 1, years)


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
    ylabel: str = 'Combined NYC Reservoir Inflow\nAnnual FDCs (MCM/day)',
    xlabel: str = 'Exceedance Probability',
    percentiles: tuple = (0.5, 99.5),
    inner_percentiles: tuple = (25, 75),
    show_legend: bool = False,
    synthetic_color: str = None,
    synthetic_label: str = 'Synthetic',
    log_scale: bool = True,
    show_inner_band: bool = True,
    year_basis: str = 'calendar',
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
    year_basis : str
        Year boundary for grouping daily flows into annual FDCs.
        'calendar' (default): calendar year, Jan 1 - Dec 31.
        'usgs_water_year': USGS hydrologic water year, Oct 1 - Sep 30.
        Distinct from the FFMP June 1 water year in methods.water_year.
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
        if year_basis == 'usgs_water_year':
            years = _assign_usgs_water_year(flow_series.dropna().index)
        else:
            years = flow_series.dropna().index.year.values

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
    syn_iq_low = np.percentile(all_syn_fdcs, inner_percentiles[0], axis=0) * MGD_TO_MCM
    syn_iq_high = np.percentile(all_syn_fdcs, inner_percentiles[1], axis=0) * MGD_TO_MCM
    hist_median = hist_median * MGD_TO_MCM
    hist_p_low = hist_p_low * MGD_TO_MCM
    hist_p_high = hist_p_high * MGD_TO_MCM
    hist_iq_low = np.percentile(hist_fdcs, inner_percentiles[0], axis=0) * MGD_TO_MCM
    hist_iq_high = np.percentile(hist_fdcs, inner_percentiles[1], axis=0) * MGD_TO_MCM

    # Layered from bottom: syn 99% → hist 99% → syn 50% → hist 50% → syn median → hist median
    ax.fill_between(
        exceedance_probs, syn_p_low, syn_p_high,
        alpha=ALPHA_BAND_OUTER, color=synthetic_color, linewidth=0,
        zorder=1, label=f'{synthetic_label} 99% IQR'
    )
    ax.plot(
        exceedance_probs, hist_p_low,
        color=HISTORIC_COLOR, linewidth=LINEWIDTH_THIN, linestyle='-',
        zorder=2, label=f'{HISTORIC_LABEL} 99% IQR (lower)'
    )
    ax.plot(
        exceedance_probs, hist_p_high,
        color=HISTORIC_COLOR, linewidth=LINEWIDTH_THIN, linestyle='-',
        zorder=2, label=f'{HISTORIC_LABEL} 99% IQR (upper)'
    )
    if show_inner_band:
        ax.fill_between(
            exceedance_probs, syn_iq_low, syn_iq_high,
            alpha=ALPHA_BAND_INNER, color=synthetic_color, linewidth=0,
            zorder=3, label=f'{synthetic_label} 50% IQR'
        )
        ax.fill_between(
            exceedance_probs, hist_iq_low, hist_iq_high,
            alpha=ALPHA_BAND_INNER, color=HISTORIC_COLOR, linewidth=0,
            zorder=4, label=f'{HISTORIC_LABEL} 50% IQR'
        )
    ax.plot(
        exceedance_probs, syn_median,
        color=synthetic_color, linewidth=LINEWIDTH_MEDIUM, linestyle='-',
        zorder=5, label=f'{synthetic_label} (median)'
    )
    ax.plot(
        exceedance_probs, hist_median,
        color=HISTORIC_COLOR, linewidth=LINEWIDTH_THICK, linestyle='--',
        zorder=6, label=f'{HISTORIC_LABEL} (median)'
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
    ylabel: str = 'Daily Streamflow\nAutocorrelation',
    xlabel: str = 'Lag (days)',
    percentiles: tuple = (0.5, 99.5),
    inner_percentiles: tuple = (25, 75),
    show_legend: bool = False,
    synthetic_color: str = None,
    synthetic_label: str = 'Synthetic',
    show_inner_band: bool = True,
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

    # Plot synthetic: outer band → inner band → median
    ax.fill_between(
        lag_range,
        np.nanpercentile(syn_autocorr, percentiles[0], axis=0),
        np.nanpercentile(syn_autocorr, percentiles[1], axis=0),
        alpha=ALPHA_BAND_OUTER, color=synthetic_color, linewidth=0,
        label=f'{synthetic_label} 99% IQR'
    )
    if show_inner_band:
        ax.fill_between(
            lag_range,
            np.nanpercentile(syn_autocorr, inner_percentiles[0], axis=0),
            np.nanpercentile(syn_autocorr, inner_percentiles[1], axis=0),
            alpha=ALPHA_BAND_INNER, color=synthetic_color, linewidth=0,
            label=f'{synthetic_label} 50% IQR'
        )
    ax.plot(
        lag_range, np.nanmedian(syn_autocorr, axis=0),
        color=synthetic_color, linewidth=LINEWIDTH_MEDIUM, linestyle='-',
        label=f'{synthetic_label} (median)'
    )

    # Historic: single line (only one series — no uncertainty band)
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
    timescale: str = 'weekly',
    ylabel: str = 'Combined NYC Reservoir\nWeekly Inflow (MCM)',
    xlabel: str = None,
    percentiles: tuple = (0.5, 99.5),
    inner_percentiles: tuple = (25, 75),
    show_legend: bool = False,
    synthetic_color: str = None,
    synthetic_label: str = 'Synthetic',
    show_inner_band: bool = True,
    _hist_agg: pd.Series = None,
    _syn_agg: dict = None,
):
    """
    Plot streamflow percentile bands for synthetic vs historic data.

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
    timescale : str
        'weekly' (52 periods) or 'monthly' (12 periods).
    ylabel, xlabel : str
        Axis labels. xlabel auto-set from timescale if None.
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

    if timescale == 'monthly':
        n_periods = 12
        periods = np.arange(1, n_periods + 1)

        Q_hist_res = _hist_agg.resample('ME').sum()
        hist_period_nums = Q_hist_res.index.month.values
        hist_values = Q_hist_res.values

        hist_median = np.empty(n_periods)
        hist_p_low = np.empty(n_periods)
        hist_p_high = np.empty(n_periods)
        hist_iq_low = np.empty(n_periods)
        hist_iq_high = np.empty(n_periods)
        for p in periods:
            vals = hist_values[hist_period_nums == p]
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                hist_median[p - 1] = np.median(vals)
                hist_p_low[p - 1] = np.percentile(vals, percentiles[0])
                hist_p_high[p - 1] = np.percentile(vals, percentiles[1])
                hist_iq_low[p - 1] = np.percentile(vals, inner_percentiles[0])
                hist_iq_high[p - 1] = np.percentile(vals, inner_percentiles[1])
            else:
                hist_median[p-1] = hist_p_low[p-1] = hist_p_high[p-1] = np.nan
                hist_iq_low[p-1] = hist_iq_high[p-1] = np.nan

        syn_by_period = {p: [] for p in periods}
        for flow_series in _syn_agg.values():
            res = flow_series.resample('ME').sum()
            p_arr = res.index.month.values
            v_arr = res.values
            for p in periods:
                vals = v_arr[p_arr == p]
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0:
                    syn_by_period[p].extend(vals.tolist())

        syn_median = np.empty(n_periods)
        syn_p_low = np.empty(n_periods)
        syn_p_high = np.empty(n_periods)
        syn_iq_low = np.empty(n_periods)
        syn_iq_high = np.empty(n_periods)
        for p in periods:
            if syn_by_period[p]:
                all_vals = np.array(syn_by_period[p])
                syn_median[p - 1] = np.median(all_vals)
                syn_p_low[p - 1] = np.percentile(all_vals, percentiles[0])
                syn_p_high[p - 1] = np.percentile(all_vals, percentiles[1])
                syn_iq_low[p - 1] = np.percentile(all_vals, inner_percentiles[0])
                syn_iq_high[p - 1] = np.percentile(all_vals, inner_percentiles[1])
            else:
                syn_median[p-1] = syn_p_low[p-1] = syn_p_high[p-1] = np.nan
                syn_iq_low[p-1] = syn_iq_high[p-1] = np.nan

        xtick_positions = periods
        xtick_labels = MONTH_LABELS
        xlim = (0.5, n_periods + 0.5)
        xlabel = xlabel or 'Month'

    else:  # weekly
        n_periods = 52
        periods = np.arange(1, n_periods + 1)

        Q_hist_weekly = _hist_agg.resample('W').sum()
        hist_weeks = Q_hist_weekly.index.isocalendar().week.values.astype(int)
        hist_values = Q_hist_weekly.values

        hist_median = np.empty(n_periods)
        hist_p_low = np.empty(n_periods)
        hist_p_high = np.empty(n_periods)
        hist_iq_low = np.empty(n_periods)
        hist_iq_high = np.empty(n_periods)
        for w in periods:
            vals = hist_values[hist_weeks == w]
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                hist_median[w - 1] = np.median(vals)
                hist_p_low[w - 1] = np.percentile(vals, percentiles[0])
                hist_p_high[w - 1] = np.percentile(vals, percentiles[1])
                hist_iq_low[w - 1] = np.percentile(vals, inner_percentiles[0])
                hist_iq_high[w - 1] = np.percentile(vals, inner_percentiles[1])
            else:
                hist_median[w-1] = hist_p_low[w-1] = hist_p_high[w-1] = np.nan
                hist_iq_low[w-1] = hist_iq_high[w-1] = np.nan

        syn_weekly_by_week = {w: [] for w in periods}
        for flow_series in _syn_agg.values():
            weekly = flow_series.resample('W').sum()
            w_arr = weekly.index.isocalendar().week.values.astype(int)
            v_arr = weekly.values
            for w in periods:
                vals = v_arr[w_arr == w]
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0:
                    syn_weekly_by_week[w].append(vals)

        syn_median = np.empty(n_periods)
        syn_p_low = np.empty(n_periods)
        syn_p_high = np.empty(n_periods)
        syn_iq_low = np.empty(n_periods)
        syn_iq_high = np.empty(n_periods)
        for w in periods:
            if syn_weekly_by_week[w]:
                all_vals = np.concatenate(syn_weekly_by_week[w])
                syn_median[w - 1] = np.median(all_vals)
                syn_p_low[w - 1] = np.percentile(all_vals, percentiles[0])
                syn_p_high[w - 1] = np.percentile(all_vals, percentiles[1])
                syn_iq_low[w - 1] = np.percentile(all_vals, inner_percentiles[0])
                syn_iq_high[w - 1] = np.percentile(all_vals, inner_percentiles[1])
            else:
                syn_median[w-1] = syn_p_low[w-1] = syn_p_high[w-1] = np.nan
                syn_iq_low[w-1] = syn_iq_high[w-1] = np.nan

        xtick_positions = MONTH_WEEK_STARTS
        xtick_labels = MONTH_LABELS
        xlim = (1, 52.85)
        xlabel = xlabel or 'Week of Year'

    # Convert MGD to MCM (weekly/monthly total)
    syn_p_low   = syn_p_low   * MGD_TO_MCM
    syn_p_high  = syn_p_high  * MGD_TO_MCM
    syn_iq_low  = syn_iq_low  * MGD_TO_MCM
    syn_iq_high = syn_iq_high * MGD_TO_MCM
    syn_median  = syn_median  * MGD_TO_MCM
    hist_p_low  = hist_p_low  * MGD_TO_MCM
    hist_p_high = hist_p_high * MGD_TO_MCM
    hist_iq_low  = hist_iq_low  * MGD_TO_MCM
    hist_iq_high = hist_iq_high * MGD_TO_MCM
    hist_median = hist_median * MGD_TO_MCM

    # Layered from bottom: syn 99% → hist 99% → syn 50% → hist 50% → syn median → hist median
    ax.fill_between(
        periods, syn_p_low, syn_p_high,
        alpha=ALPHA_BAND_OUTER, color=synthetic_color, linewidth=0,
        zorder=1, label=f'{synthetic_label} 99% IQR'
    )
    ax.plot(
        periods, hist_p_low,
        color=HISTORIC_COLOR, linewidth=LINEWIDTH_THIN, linestyle='-',
        zorder=2, label=f'{HISTORIC_LABEL} 99% IQR (lower)'
    )
    ax.plot(
        periods, hist_p_high,
        color=HISTORIC_COLOR, linewidth=LINEWIDTH_THIN, linestyle='-',
        zorder=2, label=f'{HISTORIC_LABEL} 99% IQR (upper)'
    )
    if show_inner_band:
        ax.fill_between(
            periods, syn_iq_low, syn_iq_high,
            alpha=ALPHA_BAND_INNER, color=synthetic_color, linewidth=0,
            zorder=3, label=f'{synthetic_label} 50% IQR'
        )
        ax.fill_between(
            periods, hist_iq_low, hist_iq_high,
            alpha=ALPHA_BAND_INNER, color=HISTORIC_COLOR, linewidth=0,
            zorder=4, label=f'{HISTORIC_LABEL} 50% IQR'
        )
    ax.plot(
        periods, syn_median,
        color=synthetic_color, linewidth=LINEWIDTH_MEDIUM, linestyle='-',
        zorder=5, label=f'{synthetic_label} (median)'
    )
    ax.plot(
        periods, hist_median,
        color=HISTORIC_COLOR, linewidth=LINEWIDTH_THICK, linestyle='--',
        zorder=6, label=f'{HISTORIC_LABEL} (median)'
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_yscale('log')
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels)

    if show_legend:
        ax.legend(loc='upper right', frameon=True)
    ax.grid(False)

    return ax


def _compute_pooled_pvalues(
    hist_agg: pd.Series,
    syn_agg: dict,
    timescale: str = 'weekly',
) -> tuple:
    """
    Compute Wilcoxon rank-sum and Levene p-values by pooling all synthetic
    realizations and comparing directly to historic data.

    For each time period (week or month), all synthetic values from all
    realizations are pooled into a single sample and tested against the
    historic values for that period.  The size difference between the pooled
    synthetic and historic samples is accepted by design.

    Parameters
    ----------
    hist_agg : pd.Series
        Aggregated historic flow.
    syn_agg : dict
        Aggregated synthetic flows keyed by realization ID.
    timescale : str
        'weekly' (52 periods, grouped by ISO week) or
        'monthly' (12 periods, grouped by calendar month).

    Returns
    -------
    wilcoxon_pvals, levene_pvals : np.ndarray
        P-values for each period; NaN where data are insufficient.
    """
    if timescale == 'weekly':
        n_periods = 52
        Q_hist_res = hist_agg.resample('W').mean()
        hist_period_nums = Q_hist_res.index.isocalendar().week.values.astype(int)
        hist_vals_arr = Q_hist_res.values
        periods = np.arange(1, n_periods + 1)

        syn_by_period = {p: [] for p in periods}
        for flow_series in syn_agg.values():
            res = flow_series.resample('W').mean()
            p_arr = res.index.isocalendar().week.values.astype(int)
            v_arr = res.values
            for p in periods:
                vals = v_arr[p_arr == p]
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0:
                    syn_by_period[p].extend(vals.tolist())
    else:  # monthly
        n_periods = 12
        Q_hist_res = hist_agg.resample('ME').mean()
        hist_period_nums = Q_hist_res.index.month.values
        hist_vals_arr = Q_hist_res.values
        periods = np.arange(1, n_periods + 1)

        syn_by_period = {p: [] for p in periods}
        for flow_series in syn_agg.values():
            res = flow_series.resample('ME').mean()
            p_arr = res.index.month.values
            v_arr = res.values
            for p in periods:
                vals = v_arr[p_arr == p]
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0:
                    syn_by_period[p].extend(vals.tolist())

    wilcoxon_pvals = np.full(n_periods, np.nan)
    levene_pvals = np.full(n_periods, np.nan)

    for p in periods:
        h_valid = hist_vals_arr[hist_period_nums == p]
        h_valid = h_valid[~np.isnan(h_valid)]
        s_valid = np.array(syn_by_period[p])
        if len(h_valid) < 2 or len(s_valid) < 2:
            continue
        try:
            wilcoxon_pvals[p - 1] = ranksums(h_valid, s_valid)[1]
            levene_pvals[p - 1] = levene(h_valid, s_valid)[1]
        except Exception:
            pass

    return wilcoxon_pvals, levene_pvals


def plot_pvalue_comparison(
    Q_historic: pd.DataFrame,
    Q_synthetic: dict,
    sites: list = None,
    ax=None,
    which: str = 'wilcoxon',
    timescale: str = 'weekly',
    ylabel: str = None,
    xlabel: str = None,
    significance_threshold: float = 0.05,
    show_xticklabels: bool = True,
    show_legend: bool = False,
    _hist_agg: pd.Series = None,
    _syn_agg: dict = None,
):
    """
    Plot Wilcoxon rank-sum or Levene p-values by month or week.

    All synthetic realizations are pooled for each period and compared
    directly to the historic values.  Each bar shows the p-value for that
    period; bars below ``significance_threshold`` are filled to indicate
    a statistically significant difference.  Y-axis is linear (0–1).

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
    timescale : str
        'weekly' (52 bars) or 'monthly' (12 bars).
    ylabel, xlabel : str, optional
        Axis labels. Auto-set if None.
    significance_threshold : float
        Reference line for statistical significance (default 0.05).
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

    wilcoxon_pvals, levene_pvals = _compute_pooled_pvalues(
        _hist_agg, _syn_agg, timescale=timescale,
    )

    pvals = wilcoxon_pvals if which == 'wilcoxon' else levene_pvals

    test_label = 'Wilcoxon' if which == 'wilcoxon' else 'Levene'
    period_label = 'Month' if timescale == 'monthly' else 'Week'

    if ylabel is None:
        ylabel = f'{test_label}\np-value'
    if xlabel is None:
        xlabel = period_label if show_xticklabels else ''

    if timescale == 'monthly':
        n_periods = 12
        positions = np.arange(1, n_periods + 1)
        bar_width = 0.75
        xtick_positions = positions
        xtick_labels = MONTH_LABELS
        xlim = (0.5, n_periods + bar_width + 0.1)
    else:
        n_periods = 52
        positions = np.arange(1, n_periods + 1)
        bar_width = 0.75
        xtick_positions = MONTH_WEEK_STARTS
        xtick_labels = MONTH_LABELS
        xlim = (1, n_periods + bar_width + 0.1)

    # Color bars by significance
    colors = ['#d73027' if (not np.isnan(p) and p < significance_threshold) else 'white'
              for p in pvals]
    ax.bar(positions, pvals, bar_width, align='edge',
           color=colors, edgecolor='black', linewidth=0.5)

    ax.axhline(significance_threshold, color='k', linewidth=1, linestyle='--',
               label=f'p = {significance_threshold}')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(xtick_positions)
    if show_xticklabels:
        ax.set_xticklabels(xtick_labels)
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
    timescale: str = 'weekly',
    stats_timescale: str = 'monthly',
):
    """
    Create 4-panel summary figure for manuscript with autocorrelation, FDC,
    periodic flow ranges, and statistical test p-values.

    Layout (2, 1, 1):
    - Top row: Autocorrelation (left), FDC ranges (right)
    - Middle row: Flow percentile bands by month or week (full width)
    - Bottom row: Wilcoxon & Levene p-values bar chart (full width)

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
        Lower and upper percentiles for range (default 0, 100)
    max_lag : int
        Maximum lag for autocorrelation plot (default 30 days)
    timescale : str
        Time-period aggregation for panel C (flow percentiles): 'monthly'
        (12 periods) or 'weekly' (52 periods). Default 'weekly'.
    stats_timescale : str or None
        Time-period aggregation for panels D–E (statistical tests).
        If None, uses the same value as ``timescale``. Default 'monthly'.

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

    if stats_timescale is None:
        stats_timescale = timescale

    # Pre-aggregate flows ONCE for all panels
    hist_agg = _get_aggregate_flow(Q_historic, sites)
    syn_agg = _pre_aggregate_synthetic(Q_synthetic, sites)

    # Create figure with GridSpec layout: 2 content rows + 2 thin p-value rows
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 0.22, 0.22])

    ax_autocorr = fig.add_subplot(gs[0, 0])
    ax_fdc      = fig.add_subplot(gs[0, 1])
    ax_periodic = fig.add_subplot(gs[1, :])
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

    # Panel C: Streamflow percentiles by timescale
    plot_weekly_streamflow_percentiles(
        Q_historic, Q_synthetic,
        ax=ax_periodic, timescale=timescale, percentiles=percentiles,
        synthetic_color=synthetic_color, synthetic_label=synthetic_label,
        show_legend=False,
        _hist_agg=hist_agg, _syn_agg=syn_agg,
    )
    ax_periodic.text(0.01, 0.97, 'c)', transform=ax_periodic.transAxes,
                     fontsize=12, va='top', ha='left')

    # Panel D: Wilcoxon rank-sum p-values (no x-tick labels — shared with panel E)
    plot_pvalue_comparison(
        Q_historic, Q_synthetic,
        ax=ax_wilcoxon, which='wilcoxon', timescale=stats_timescale,
        show_xticklabels=False,
        show_legend=False,
        _hist_agg=hist_agg, _syn_agg=syn_agg,
    )
    ax_wilcoxon.text(0.01, 0.85, 'd)', transform=ax_wilcoxon.transAxes,
                     fontsize=12, va='top', ha='left')

    # Panel E: Levene p-values (x-tick labels shown here)
    plot_pvalue_comparison(
        Q_historic, Q_synthetic,
        ax=ax_levene, which='levene', timescale=stats_timescale,
        show_xticklabels=True,
        show_legend=False,
        _hist_agg=hist_agg, _syn_agg=syn_agg,
    )
    ax_levene.text(0.01, 0.85, 'e)', transform=ax_levene.transAxes,
                   fontsize=12, va='top', ha='left')

    # Shared legend (flow panels only; p-value panels are self-labeled via ylabel)
    period_label = 'month' if timescale == 'monthly' else 'week'
    legend_handles = [
        Patch(facecolor=synthetic_color, alpha=ALPHA_FILL,
              label=f'{synthetic_label} ({percentiles[0]}-{percentiles[1]}%)'),
        Line2D([0], [0], color=synthetic_color, linewidth=LINEWIDTH_MEDIUM,
               linestyle='-', label=f'{synthetic_label} (median)'),
        Patch(facecolor=HISTORIC_COLOR, alpha=ALPHA_FILL * 0.7,
              label=f'{HISTORIC_LABEL} ({percentiles[0]}-{percentiles[1]}%)'),
        Line2D([0], [0], color=HISTORIC_COLOR, linewidth=LINEWIDTH_THICK,
               linestyle='--', label=f'{HISTORIC_LABEL} (median)'),
        Line2D([0], [0], color='k', linestyle='--', linewidth=1, label='p = 0.05'),
    ]

    fig.legend(
        handles=legend_handles,
        loc='lower center', ncol=3, frameon=False,
        bbox_to_anchor=(0.5, -0.04), fontsize=9,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.09)

    # Align y-axis labels across the three full-width panels
    fig.align_ylabels([ax_autocorr, ax_periodic, ax_wilcoxon, ax_levene])

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


def plot_low_flow_convergence(
    Q_syn_site: pd.DataFrame,
    realization_ids: list,
    site_label: str,
    Q_obs: pd.Series = None,
    n_bootstrap_samples: int = 200,
    n_steps: int = 30,
    synthetic_color: str = None,
    fname: str = None,
    figsize: tuple = (14, 6),
):
    """
    Plot ensemble convergence of LOW-FLOW EXTREMES.

    The diagnostic answers: "Does adding more realizations still reveal more
    extreme low-flow events, or has the ensemble's low-flow tail been
    characterized?" For each subset size n, the ensemble extreme (MIN over
    realizations and water years) is computed across many bootstrap subsamples.
    The bootstrap median is expected to MONOTONICALLY DECREASE with n and
    plateau when the tail is well-sampled. There is no averaging across
    realizations -- only min and bootstrap percentiles.

    Two panels:

    1. Minimum 7-day mean flow (acute drought extreme).
       For each realization, the 7-day moving-average daily flow is computed;
       the per-realization minimum is taken across all water years; the
       ensemble extreme is the smallest of these per-realization minima
       across the subset of n realizations.

    2. Minimum annual Q95 (chronic-low-flow-year extreme).
       Q95 = flow exceeded 95% of days within a water year (i.e., the 5th
       percentile of daily flow within that water year). For each
       (realization, water year) pair we have one annual Q95. The ensemble
       extreme is the smallest annual Q95 across the n realizations x all
       water years.

    Parameters
    ----------
    Q_syn_site : pd.DataFrame
        Daily synthetic flow (MGD), columns = realization IDs, DatetimeIndex.
    realization_ids : list
        Realization IDs to include.
    site_label : str
        Site name shown in titles.
    Q_obs : pd.Series, optional
        Historical daily flow at the same site. When provided, a horizontal
        dashed line marks the historical-record value of each metric.
    n_bootstrap_samples : int
        Bootstrap subsamples per subset size (default 200).
    n_steps : int
        Number of subset sizes along the x-axis (default 30).
    synthetic_color : str, optional
        Color for the ensemble band/line.
    fname : str, optional
        If provided, save the figure to this path.
    figsize : tuple
        Figure size (default (14, 6)).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if synthetic_color is None:
        synthetic_color = DATASET_COLORS['stationary_ensemble']

    n_realizations = len(realization_ids)
    Q_syn_site = Q_syn_site[realization_ids]

    # Per-realization extrema (compute once; bootstrap then just min over subset)
    rolling7 = Q_syn_site.rolling(window=7, min_periods=7).mean()
    min_7day_per_real = np.nanmin(rolling7.values, axis=0)  # (N,)

    wy = _assign_usgs_water_year(Q_syn_site.index)
    annual_q95 = Q_syn_site.groupby(wy).quantile(0.05)  # (n_years, N)
    min_q95_per_real = np.nanmin(annual_q95.values, axis=0)  # (N,)

    n_subset_sizes = np.unique(
        np.linspace(1, n_realizations, n_steps).round().astype(int)
    )

    am7_bands = np.empty((len(n_subset_sizes), 3))
    q95_bands = np.empty((len(n_subset_sizes), 3))

    rng = np.random.default_rng(42)
    indices = np.arange(n_realizations)

    for i, n in enumerate(n_subset_sizes):
        boot_am7 = np.empty(n_bootstrap_samples)
        boot_q95 = np.empty(n_bootstrap_samples)
        for b in range(n_bootstrap_samples):
            idx = rng.choice(indices, size=n, replace=False)
            boot_am7[b] = np.min(min_7day_per_real[idx])
            boot_q95[b] = np.min(min_q95_per_real[idx])
        am7_bands[i] = np.percentile(boot_am7, [5, 50, 95])
        q95_bands[i] = np.percentile(boot_q95, [5, 50, 95])

    obs_min_7day = obs_min_q95 = None
    if Q_obs is not None:
        obs_rolling7 = Q_obs.rolling(window=7, min_periods=7).mean()
        obs_min_7day = float(np.nanmin(obs_rolling7.values))

        obs_wy = _assign_usgs_water_year(Q_obs.index)
        obs_annual_q95 = Q_obs.groupby(obs_wy).quantile(0.05)
        obs_min_q95 = float(np.nanmin(obs_annual_q95.values))

    fig, (ax_am7, ax_q95) = plt.subplots(1, 2, figsize=figsize)

    panel_specs = [
        (
            ax_am7, am7_bands, obs_min_7day,
            f'Minimum 7-day mean flow — {site_label}',
            'Ensemble min of 7-day mean flow (MGD)\n[over N realizations × all water years]',
        ),
        (
            ax_q95, q95_bands, obs_min_q95,
            f'Minimum annual Q95 — {site_label}',
            'Ensemble min of annual Q95 (MGD)\n[over N realizations × all water years]',
        ),
    ]

    for ax, bands, obs_val, title, ylabel in panel_specs:
        ax.fill_between(
            n_subset_sizes, bands[:, 0], bands[:, 2],
            alpha=ALPHA_FILL, color=synthetic_color,
            label=f'Synthetic: bootstrap 5–95% ({n_bootstrap_samples} subsamples)',
        )
        ax.plot(
            n_subset_sizes, bands[:, 1],
            color=synthetic_color, linewidth=LINEWIDTH_MEDIUM,
            label='Synthetic: bootstrap median of ensemble min',
        )
        if obs_val is not None and np.isfinite(obs_val):
            ax.axhline(
                obs_val, color=HISTORIC_COLOR, linestyle='--',
                linewidth=LINEWIDTH_MEDIUM,
                label='Historical-record min',
            )
        ax.set_xlabel('Number of realizations N\n(subsampled without replacement)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(0, n_realizations)
        ax.legend(loc='upper right', frameon=True, fontsize='small')
        ax.grid(False)

    fig.text(
        0.5, -0.04,
        'Each panel shows the ENSEMBLE MINIMUM of a low-flow metric vs N — no averaging across realizations. '
        'The median is expected to decrease monotonically with N and plateau once the low-flow tail is well sampled. '
        '7-day mean = 7-day moving-average daily flow; annual Q95 = 5th percentile of daily flow within each USGS water year.',
        ha='center', va='top', fontsize='small', wrap=True,
    )
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=DPI_PRINT, bbox_inches='tight')
        print(f"Saved low-flow convergence figure to {fname}")
        plt.close(fig)
    return fig


_DROUGHT_METRIC_DEFAULTS = {
    'duration':        ('event duration',          'months'),
    'magnitude':       ('event magnitude',         'SSI·months'),
    'severity':        ('event severity (peak)',   'SSI std. dev.'),
    'avg_severity':    ('event avg severity',      'SSI std. dev.'),
    'recovery_period': ('recovery period',         'months'),
}


def plot_drought_metric_convergence(
    droughts: pd.DataFrame,
    realization_ids: list,
    ssi_window: int,
    obs_droughts: pd.DataFrame = None,
    metrics: list = None,
    metric_labels: dict = None,
    include_extremes: bool = True,
    n_bootstrap_samples: int = 200,
    n_steps: int = 30,
    synthetic_color: str = None,
    fname: str = None,
    figsize: tuple = None,
):
    """
    Convergence of drought-event metrics vs N realizations.

    Two complementary views, stacked as rows in one figure:

    1. POOLED MEAN (top row): for each subset of n realizations, pool all
       drought events from those realizations and compute the equal-weight
       mean of each metric:

           pooled_mean = Σ metric / count(events)

       Stability question: "Is the central tendency of drought events stable?"

    2. ENSEMBLE MAX (bottom row, when include_extremes=True): for each
       subset, the maximum across (realizations × events) of each metric.

           ensemble_max = max(metric over events in n realizations)

       Stability question: "Have we sampled the worst drought the generator
       can produce?" Expected to monotonically increase with N and plateau.

    For both, the 5–50–95 percentiles across n_bootstrap_samples without-
    replacement subsamples define the band and median line. Horizontal
    dashed lines mark the historical-record mean (top row) and max (bottom
    row) when `obs_droughts` is provided.

    Parameters
    ----------
    droughts : pd.DataFrame
        Long-format drought events; must contain `realization_id` plus the
        metric columns. Output of `load_drought_events(..., observed=False)`.
    realization_ids : list
        All N realization IDs in the ensemble (zero-event realizations are
        included in N but contribute 0 to the pool and NaN to the max).
    ssi_window : int
        SSI window (3, 6, or 12) — used for titles and footer.
    obs_droughts : pd.DataFrame, optional
        Observed drought events for the same SSI window. Historical mean
        and max of each metric are drawn as dashed reference lines.
    metrics : list of str, optional
        Columns to plot. Defaults to ['duration', 'magnitude', 'severity'].
    metric_labels : dict, optional
        Override the (display_label, units) tuple for any metric.
    include_extremes : bool
        If True (default), produce a second row of ensemble-max panels.
    n_bootstrap_samples : int
        Bootstrap subsamples per subset size (default 200).
    n_steps : int
        Number of subset sizes along the x-axis (default 30).
    synthetic_color : str, optional
        Color for ensemble bands and lines.
    fname : str, optional
        If provided, save the figure to this path.
    figsize : tuple, optional
        Defaults to (4.8 · n_metrics + 0.8, 5.5 · n_rows).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if synthetic_color is None:
        synthetic_color = DATASET_COLORS['stationary_ensemble']

    if metrics is None:
        metrics = ['duration', 'magnitude', 'severity']

    labels = dict(_DROUGHT_METRIC_DEFAULTS)
    if metric_labels:
        labels.update(metric_labels)

    n_realizations = len(realization_ids)
    n_metrics = len(metrics)

    # Per-realization aggregates (zero-event realizations: sum=0, count=0, max=NaN)
    per_real_sum = (
        droughts.groupby('realization_id')[metrics]
        .sum()
        .reindex(realization_ids, fill_value=0)
    )
    per_real_count = (
        droughts.groupby('realization_id').size()
        .reindex(realization_ids, fill_value=0)
    )
    per_real_max = (
        droughts.groupby('realization_id')[metrics]
        .max()
        .reindex(realization_ids)
    )
    sums_arr = per_real_sum.values.astype(float)
    counts_arr = per_real_count.values.astype(float)
    maxes_arr = per_real_max.values.astype(float)

    n_subset_sizes = np.unique(
        np.linspace(1, n_realizations, n_steps).round().astype(int)
    )

    mean_bands = np.full((len(n_subset_sizes), 3, n_metrics), np.nan)
    max_bands = np.full((len(n_subset_sizes), 3, n_metrics), np.nan)

    rng = np.random.default_rng(42)
    indices = np.arange(n_realizations)

    for i, n in enumerate(n_subset_sizes):
        boot_mean = np.full((n_bootstrap_samples, n_metrics), np.nan)
        boot_max = np.full((n_bootstrap_samples, n_metrics), np.nan)
        for b in range(n_bootstrap_samples):
            idx = rng.choice(indices, size=n, replace=False)
            total_count = counts_arr[idx].sum()
            if total_count > 0:
                boot_mean[b] = sums_arr[idx].sum(axis=0) / total_count
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                boot_max[b] = np.nanmax(maxes_arr[idx], axis=0)
        mean_bands[i] = np.nanpercentile(boot_mean, [5, 50, 95], axis=0)
        max_bands[i] = np.nanpercentile(boot_max, [5, 50, 95], axis=0)

    obs_means = {}
    obs_maxes = {}
    if obs_droughts is not None and len(obs_droughts) > 0:
        for m in metrics:
            if m in obs_droughts.columns:
                obs_means[m] = float(obs_droughts[m].mean())
                obs_maxes[m] = float(obs_droughts[m].max())

    n_rows = 2 if include_extremes else 1
    if figsize is None:
        figsize = (4.8 * n_metrics + 0.8, 5.5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_metrics, figsize=figsize, squeeze=False)

    row_specs = [
        ('Pooled mean', 'Mean', mean_bands, obs_means,
         'Synthetic: bootstrap median of pooled mean'),
    ]
    if include_extremes:
        row_specs.append(
            ('Ensemble max', 'Max', max_bands, obs_maxes,
             'Synthetic: bootstrap median of ensemble max')
        )

    for r, (row_name, stat_prefix, bands, obs_vals, median_label) in enumerate(row_specs):
        for k, metric in enumerate(metrics):
            ax = axes[r, k]
            display_label, units = labels.get(metric, (metric, ''))

            ax.fill_between(
                n_subset_sizes, bands[:, 0, k], bands[:, 2, k],
                alpha=ALPHA_FILL, color=synthetic_color,
                label=f'Synthetic: bootstrap 5–95% ({n_bootstrap_samples} subsamples)',
            )
            ax.plot(
                n_subset_sizes, bands[:, 1, k],
                color=synthetic_color, linewidth=LINEWIDTH_MEDIUM,
                label=median_label,
            )
            if metric in obs_vals and np.isfinite(obs_vals[metric]):
                ax.axhline(
                    obs_vals[metric], color=HISTORIC_COLOR, linestyle='--',
                    linewidth=LINEWIDTH_MEDIUM,
                    label=f'Historical {stat_prefix.lower()}',
                )
            ax.set_xlabel('Number of realizations N\n(subsampled without replacement)')
            ax.set_ylabel(f'{stat_prefix} {display_label} ({units})')
            ax.set_title(f'{row_name} — {display_label}')
            ax.set_xlim(0, n_realizations)
            ax.legend(loc='best', frameon=True, fontsize='small')
            ax.grid(False)

    fig.suptitle(
        f'Drought event metric convergence — SSI-{ssi_window} '
        f'(NYC aggregate inflow, {n_realizations} realizations)',
        fontsize='medium', y=1.0,
    )

    footer = (
        'Top row: pooled mean = Σ metric ÷ count(events) over the subsampled events; each event has equal weight. '
        + (
            'Bottom row: ensemble max = max metric over all events in the subsample; expected to increase monotonically and plateau as N grows. '
            if include_extremes else ''
        )
        + f'SSI-{ssi_window} = standardized streamflow index on a {ssi_window}-month window over NYC aggregate inflow. '
        + 'magnitude = Σ monthly SSI deficits over the event; severity = peak (most negative) SSI value, absolute.'
    )
    fig.text(0.5, -0.03, footer, ha='center', va='top', fontsize='small', wrap=True)

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=DPI_PRINT, bbox_inches='tight')
        print(f"Saved drought metric convergence figure to {fname}")
        plt.close(fig)
    return fig
