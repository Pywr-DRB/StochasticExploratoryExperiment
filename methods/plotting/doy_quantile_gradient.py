"""
Day-of-water-year quantile gradient plot.

Fills the region between successive quantile levels of a collection of
water-year traces with a sequential/diverging colormap, producing a
"% of traces below y-axis value" gradient against a day-of-water-year axis.

Designed as a background on which specific trajectories (e.g. worst-case
drought events) can be overlaid.
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def compute_doy_quantile_grid(aligned_traces_df, n_levels=21):
    """Compute quantile thresholds across traces at each day-of-water-year.

    Parameters
    ----------
    aligned_traces_df : pd.DataFrame
        Rows indexed by day-of-water-year (1-366). Each column is a single
        water-year trace (one event/year/realization).  NaN entries are
        ignored per-row.
    n_levels : int
        Number of quantile levels, inclusive of 0 and 1.  Default 21 yields
        quantiles at 0, 0.05, 0.10, ..., 0.95, 1.0.

    Returns
    -------
    pd.DataFrame
        Rows indexed by day-of-water-year (same as input).
        Columns are float quantile levels in [0, 1].
        Values are the variable thresholds at that (DOY, quantile) pair.
    """
    quantile_levels = np.linspace(0.0, 1.0, n_levels)
    arr = aligned_traces_df.values.astype(float)
    q = np.nanquantile(arr, quantile_levels, axis=1)
    return pd.DataFrame(
        q.T, index=aligned_traces_df.index, columns=quantile_levels,
    )


def plot_doy_quantile_gradient(ax, quantile_grid, cmap='BrBG', vmin=0.0,
                                vmax=1.0, alpha=1.0, edge_linewidth=0,
                                reverse=False):
    """Fill between successive quantile bands, colored by band midpoint.

    The resulting background reads "% of traces below the y-axis value":
    bands near q=0 sit at the lowest values (few traces below); bands
    near q=1 sit at the highest values (nearly all traces below).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    quantile_grid : pd.DataFrame
        Output of `compute_doy_quantile_grid`.
    cmap : str or matplotlib colormap
        Default 'BrBG' (brown-teal diverging) to match the project convention.
    vmin, vmax : float
        Colormap normalization bounds in quantile-level space [0, 1].
    alpha : float
    edge_linewidth : float
        Width of thin boundary lines between bands (0 = none).
    reverse : bool
        If True, reverses the quantile-to-color mapping (brown at top).

    Returns
    -------
    matplotlib.cm.ScalarMappable
        Suitable for passing to fig.colorbar.
    """
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    x = quantile_grid.index.values
    levels = quantile_grid.columns.values

    for i in range(len(levels) - 1):
        q_lo = levels[i]
        q_hi = levels[i + 1]
        y_lo = quantile_grid.iloc[:, i].values
        y_hi = quantile_grid.iloc[:, i + 1].values
        mid = (q_lo + q_hi) / 2
        color_q = (1.0 - mid) if reverse else mid
        color = cmap_obj(norm(color_q))
        ax.fill_between(
            x, y_lo, y_hi,
            color=color, alpha=alpha, linewidth=edge_linewidth,
        )

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    return sm
