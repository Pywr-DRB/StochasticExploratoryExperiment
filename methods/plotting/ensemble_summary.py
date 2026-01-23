"""
Plotting functions for ensemble summary figures (main manuscript).

This module provides functions for creating publication-quality summary
figures comparing synthetic ensemble and historical streamflow data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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


def plot_ensemble_summary_figure(
    Q_historic: pd.DataFrame,
    Q_synthetic: dict,
    sites: list = None,
    dataset_id: str = 'stationary_ensemble',
    fname: str = None,
    figsize: tuple = (12, 5),
    percentiles: tuple = (5, 95),
):
    """
    Create 2x1 summary figure for manuscript with weekly percentiles and FDC comparison.

    Uses aggregate flow from NYC reservoirs (sum of cannonsville, pepacton, neversink).
    Weekly flow panel is 1.67x the width of the FDC panel.

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

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    apply_publication_style()

    if sites is None:
        sites = NYC_RESERVOIRS

    # Create figure with weekly flow panel 1.67x wider than FDC panel
    fig, axes = plt.subplots(1, 2, figsize=figsize,
                              gridspec_kw={'width_ratios': [1.67, 1]})

    synthetic_color = DATASET_COLORS.get(dataset_id, DATASET_COLORS['stationary_ensemble'])
    synthetic_label = DATASET_LABELS.get(dataset_id, 'Synthetic')

    # Panel A: Weekly streamflow percentiles (no individual legend, no title)
    plot_weekly_streamflow_percentiles(
        Q_historic,
        Q_synthetic,
        sites=sites,
        ax=axes[0],
        percentiles=percentiles,
        synthetic_color=synthetic_color,
        synthetic_label=synthetic_label,
        show_legend=False,
    )

    # Panel B: FDC percentile comparison (no individual legend, no title)
    plot_fdc_percentile_comparison(
        Q_historic,
        Q_synthetic,
        sites=sites,
        ax=axes[1],
        percentiles=percentiles,
        synthetic_color=synthetic_color,
        synthetic_label=synthetic_label,
        show_legend=False,
    )

    # Create shared legend below the figure
    # Build legend handles manually for consistent appearance
    legend_handles = [
        Patch(facecolor=synthetic_color, alpha=ALPHA_FILL,
              label=f'{synthetic_label} ({percentiles[0]}-{percentiles[1]}%)'),
        Line2D([0], [0], color=synthetic_color, linewidth=LINEWIDTH_MEDIUM,
               linestyle='-', label=f'{synthetic_label} (median)'),
        Patch(facecolor=HISTORIC_COLOR, alpha=ALPHA_FILL * 0.7,
              label=f'{HISTORIC_LABEL} ({percentiles[0]}-{percentiles[1]}%)'),
        Line2D([0], [0], color=HISTORIC_COLOR, linewidth=LINEWIDTH_THICK,
               linestyle='--', label=f'{HISTORIC_LABEL} (median)'),
    ]

    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=10,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend

    if fname:
        plt.savefig(fname, dpi=DPI_PRINT, bbox_inches='tight')
        print(f"Saved figure to {fname}")
        plt.close(fig)

    return fig


# Keep old function names as aliases for backwards compatibility
plot_weekly_streamflow_range = plot_weekly_streamflow_percentiles
plot_fdc_range_comparison = plot_fdc_percentile_comparison
