"""
Reusable parallel-axis (parallel-coordinates) plotting helper.

Adapted from ``../CEE6400Project/methods/plotting/plot_parallel_axis.py`` and
``../PywrDRB-ML/.../parallel_axes_with_kde_withSdyn_poster.py``, but stripped of
the multi-objective-optimization framing (no "direction of preference" arrow,
no min/max objective semantics). Here every axis is purely descriptive: the
per-axis minimum maps to the bottom of the plot, the maximum to the top.

The function draws onto a caller-supplied ``ax`` and accepts explicit, shared
per-axis bounds (``tops``/``bottoms``) plus a shared colour range
(``vmin``/``vmax``) so that several panels can be rendered on identical scales
with a single shared colourbar.

Brushing
--------
Pass ``brush_conditions`` as a list of ``(column, operator, threshold)`` tuples
(AND-combined). Rows that satisfy *all* conditions are drawn in full colour on
top; rows that fail are faded to a faint grey background. A translucent band is
drawn on each brushed axis to mark the satisfied value range. The brushing
spec is data-driven and intentionally generic so callers can change the
filter without touching this module.
"""

import numpy as np
from matplotlib import colormaps, cm
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


# Comparison operators supported in brushing specs.
_BRUSH_OPS = {
    '<':  np.less,
    '<=': np.less_equal,
    '>':  np.greater,
    '>=': np.greater_equal,
    '==': np.equal,
    '!=': np.not_equal,
}


def apply_brush(objs, brush_conditions):
    """Return a boolean mask of rows satisfying ALL brush conditions.

    Parameters
    ----------
    objs : pd.DataFrame
    brush_conditions : list of (str, str, float) or None
        Each tuple is ``(column, operator, threshold)``; operators are the
        keys of :data:`_BRUSH_OPS`. ``None`` or empty selects every row.

    Returns
    -------
    np.ndarray of bool, length ``len(objs)``.
    """
    n = len(objs)
    if not brush_conditions:
        return np.ones(n, dtype=bool)
    mask = np.ones(n, dtype=bool)
    for col, op, thresh in brush_conditions:
        if op not in _BRUSH_OPS:
            raise ValueError(f"Unsupported brush operator: {op!r}")
        mask &= _BRUSH_OPS[op](objs[col].astype(float).values, thresh)
    return mask


def _normalize_columns(objs, columns_axes, tops, bottoms):
    """Return a (n_rows x n_axes) array normalized to [0, 1] per axis.

    ``tops`` maps to 1 (top of plot), ``bottoms`` maps to 0 (bottom). Axes with
    zero range are placed at the midline (0.5).
    """
    data = objs[columns_axes].astype(float).values
    tops = np.asarray(tops, dtype=float)
    bottoms = np.asarray(bottoms, dtype=float)
    rng = tops - bottoms
    rng_safe = np.where(rng == 0, 1.0, rng)
    normed = (data - bottoms) / rng_safe
    normed = np.where(rng == 0, 0.5, normed)
    return np.clip(normed, 0.0, 1.0)


def _draw_brush_bands(ax, brush_conditions, columns_axes, tops, bottoms):
    """Shade the satisfied value range on each brushed axis that is plotted."""
    col_to_idx = {c: j for j, c in enumerate(columns_axes)}
    for col, op, thresh in brush_conditions:
        if col not in col_to_idx:
            continue  # brushed on a column that is not an axis; nothing to draw
        j = col_to_idx[col]
        rng = tops[j] - bottoms[j]
        if rng == 0:
            continue
        tn = float(np.clip((thresh - bottoms[j]) / rng, 0.0, 1.0))
        if op in ('<', '<='):
            y0, h = 0.0, tn
        elif op in ('>', '>='):
            y0, h = tn, 1.0 - tn
        else:
            continue  # == / != have no contiguous band
        ax.add_patch(Rectangle((j - 0.07, y0), 0.14, h, facecolor='0.55',
                               alpha=0.25, edgecolor='none', zorder=3))
        # dashed marker line at the threshold
        ax.plot([j - 0.07, j + 0.07], [tn, tn], c='0.3', lw=1.0,
                ls='--', zorder=3.5)


def custom_parallel_coordinates(ax, objs, columns_axes, axis_labels=None,
                                tops=None, bottoms=None,
                                color_by_continuous=None,
                                color_palette_continuous='viridis',
                                vmin=None, vmax=None,
                                alpha_base=0.7, lw_base=1.3,
                                fontsize=11, tick_decimals=1,
                                brush_conditions=None,
                                alpha_brush=0.03, lw_brush=0.5,
                                brush_grey='0.6', brush_band=True):
    """Draw a parallel-coordinates plot of *objs* onto *ax*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    objs : pd.DataFrame
        One row per line; columns include those listed in *columns_axes* and
        (optionally) *color_by_continuous* and any brushed columns.
    columns_axes : list of str
        DataFrame columns to draw as parallel axes, left to right.
    axis_labels : list of str, optional
        Display labels for each axis (defaults to *columns_axes*).
    tops, bottoms : array-like, optional
        Per-axis upper/lower bounds (length == len(columns_axes)). If omitted,
        computed from *objs* (per-axis max/min). Pass shared bounds to make
        multiple panels directly comparable.
    color_by_continuous : str, optional
        Column used to colour each line via a continuous colormap. If None, a
        single neutral colour is used.
    color_palette_continuous : str
        Matplotlib colormap name.
    vmin, vmax : float, optional
        Colour-scale limits (default: min/max of the colour column in *objs*).
    alpha_base, lw_base : float
        Line transparency and width for brushed (highlighted) lines.
    fontsize : int
        Annotation font size.
    tick_decimals : int
        Decimal places for the top/bottom value annotations.
    brush_conditions : list of (str, str, float), optional
        AND-combined ``(column, operator, threshold)`` filter. Satisfying rows
        are highlighted; the rest are faded to grey. See :func:`apply_brush`.
    alpha_brush, lw_brush : float
        Transparency / width for the faded (non-satisfying) background lines.
    brush_grey : str
        Colour for the faded background lines.
    brush_band : bool
        If True, shade the satisfied value range on each brushed axis.

    Returns
    -------
    matplotlib.cm.ScalarMappable or None
        A mappable describing the colour scale (for building a shared
        colourbar), or None when *color_by_continuous* is not used.
    """
    n_axes = len(columns_axes)
    axis_labels = axis_labels if axis_labels is not None else list(columns_axes)

    if tops is None:
        tops = objs[columns_axes].max(axis=0).values
    if bottoms is None:
        bottoms = objs[columns_axes].min(axis=0).values
    tops = np.asarray(tops, dtype=float)
    bottoms = np.asarray(bottoms, dtype=float)

    normed = _normalize_columns(objs, columns_axes, tops, bottoms)
    brushed = apply_brush(objs, brush_conditions)

    # Colour setup for the highlighted lines
    mappable = None
    if color_by_continuous is not None:
        cvals = objs[color_by_continuous].astype(float).values
        if vmin is None:
            vmin = np.nanmin(cvals)
        if vmax is None:
            vmax = np.nanmax(cvals)
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = colormaps.get_cmap(color_palette_continuous)
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        line_colors = [cmap(norm(v)) for v in cvals]
    else:
        cvals = np.zeros(normed.shape[0])
        line_colors = ['#4878a8'] * normed.shape[0]

    x = np.arange(n_axes)

    # Threshold bands first (under all lines but above the axis spines)
    if brush_band and brush_conditions:
        _draw_brush_bands(ax, brush_conditions, columns_axes, tops, bottoms)

    # Background: non-brushed lines, faint grey
    for i in np.where(~brushed)[0]:
        ax.plot(x, normed[i, :], c=brush_grey,
                alpha=alpha_brush, lw=lw_brush, zorder=2)

    # Foreground: brushed lines, worst (highest colour value) drawn last
    b_idx = np.where(brushed)[0]
    order = b_idx[np.argsort(np.nan_to_num(cvals[b_idx], nan=-np.inf))]
    for i in order:
        ax.plot(x, normed[i, :], c=line_colors[i],
                alpha=alpha_base, lw=lw_base, zorder=4)

    # Vertical axis lines + top/bottom value annotations
    for j in range(n_axes):
        ax.plot([j, j], [0, 1], c='k', lw=1.0, zorder=2)
        ax.annotate(f"{tops[j]:.{tick_decimals}f}", [j, 1.02],
                    ha='center', va='bottom', fontsize=fontsize - 1, zorder=5)
        ax.annotate(f"{bottoms[j]:.{tick_decimals}f}", [j, -0.02],
                    ha='center', va='top', fontsize=fontsize - 1, zorder=5)

    # Axis labels below each vertical axis
    for j, label in enumerate(axis_labels):
        ax.annotate(label, xy=(j, -0.13), ha='center', va='top',
                    fontsize=fontsize, zorder=5)

    # Aesthetics
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(False)
    ax.set_xlim(-0.4, n_axes - 1 + 0.4)
    ax.set_ylim(-0.35, 1.12)
    ax.patch.set_alpha(0)

    return mappable
