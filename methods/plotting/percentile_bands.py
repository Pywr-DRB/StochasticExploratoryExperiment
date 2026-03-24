"""
Percentile band and difference band plotting for water-year timeseries.

Shared by F4 contribution figures and potentially other timeseries analyses.
"""

import numpy as np
import pandas as pd

from methods.water_year import MONTH_STARTS_WY, MONTH_LABELS_WY
from methods.plotting.styles import FONTSIZE_MEDIUM


def calculate_percentiles(df):
    """Calculate 1/25/50/75/99 percentiles for each row (day-of-year)."""
    return df.T.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T


def calculate_pairwise_difference_percentiles(baseline_df, comparison_df):
    """
    Quantile-matched difference percentiles (comparison − baseline).

    For each day: quantile-match the two distributions, compute differences,
    then report 1/25/50/75/99 percentiles of those differences.
    """
    common_idx = baseline_df.index.intersection(comparison_df.index)
    baseline_arr = baseline_df.loc[common_idx].values
    comparison_arr = comparison_df.loc[common_idx].values

    quantile_levels = np.linspace(0, 1, 101)
    output_pcts = np.array([0.01, 0.25, 0.50, 0.75, 0.99])

    baseline_q = np.nanquantile(baseline_arr, quantile_levels, axis=1)
    comparison_q = np.nanquantile(comparison_arr, quantile_levels, axis=1)
    diffs = comparison_q - baseline_q
    diff_pcts = np.percentile(diffs, output_pcts * 100, axis=0)

    return pd.DataFrame(
        {'1%': diff_pcts[0], '25%': diff_pcts[1], '50%': diff_pcts[2],
         '75%': diff_pcts[3], '99%': diff_pcts[4]},
        index=common_idx,
    )


def format_xaxis_water_year(ax, fontsize=None, show_labels=True):
    """Set month-boundary ticks for a water-year x-axis (1-366)."""
    if fontsize is None:
        fontsize = FONTSIZE_MEDIUM
    ax.set_xticks(MONTH_STARTS_WY)
    ax.set_xticklabels(MONTH_LABELS_WY if show_labels else [], fontsize=fontsize)
    ax.set_xlim(1, 366)


def plot_bands(ax, percentiles, color, alpha_outer=0.15, alpha_inner=0.35,
               label_prefix='', representative_year=None, linewidth=1.8):
    """
    Plot fill-between percentile bands: 1-99% (outer), 25-75% (inner), median.

    Parameters
    ----------
    label_prefix : str
        If non-empty, legend labels are attached to each element.
    representative_year : dict, optional
        If provided and contains 'contribution_trace', overlays the trace.
    """
    doy = percentiles.index.values

    ax.fill_between(doy, percentiles['1%'], percentiles['99%'],
                    color=color, alpha=alpha_outer, linewidth=0,
                    label=f'{label_prefix}1st-99th %ile' if label_prefix else None)
    ax.fill_between(doy, percentiles['25%'], percentiles['75%'],
                    color=color, alpha=alpha_inner, linewidth=0,
                    label=f'{label_prefix}25th-75th %ile' if label_prefix else None)
    ax.plot(doy, percentiles['50%'], color=color, linewidth=linewidth,
            label=f'{label_prefix}Median' if label_prefix else None)

    if representative_year is not None and 'contribution_trace' in representative_year:
        trace = representative_year['contribution_trace']
        year = representative_year['year']
        ax.plot(trace.index, trace.values, color=color, linewidth=1.5,
                linestyle='--', alpha=0.8, label=f'Rep. Year ({year})')


def plot_difference_bands(ax, diff_percentiles, color, alpha_outer=0.15,
                          alpha_inner=0.35, label_prefix='', linewidth=1.8):
    """Plot fill-between bands for a difference timeseries."""
    doy = diff_percentiles.index.values

    ax.fill_between(doy, diff_percentiles['1%'], diff_percentiles['99%'],
                    color=color, alpha=alpha_outer, linewidth=0,
                    label=f'{label_prefix}1st-99th %ile' if label_prefix else None)
    ax.fill_between(doy, diff_percentiles['25%'], diff_percentiles['75%'],
                    color=color, alpha=alpha_inner, linewidth=0,
                    label=f'{label_prefix}25th-75th %ile' if label_prefix else None)
    ax.plot(doy, diff_percentiles['50%'], color=color, linewidth=linewidth,
            label=f'{label_prefix}Median' if label_prefix else None)
