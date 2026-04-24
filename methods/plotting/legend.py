"""
Reusable legend elements for percentile-band + median ensemble plots.

The primary export is a legend handler that renders, in a single legend
row, the exact composite seen in the plot: an outer IQR fill, an inner
IQR fill (darker due to same-color alpha compositing), and a median line
through the vertical center.

Usage
-----
from methods.plotting.legend import IQRBandHandle, iqr_band_legend_kwargs

handles = [IQRBandHandle(color=c) for c in dataset_colors]
labels = [DATASET_LABELS[d] for d in datasets]
ax.legend(handles, labels, **iqr_band_legend_kwargs())
"""

from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from methods.plotting.styles import ALPHA_BAND_OUTER, LINEWIDTH_MEDIUM

# Legend-only: inner-band alpha is bumped above the plot's ALPHA_BAND_INNER
# so the three layers stay distinguishable at small legend-handle sizes.
# The plot itself continues to use ALPHA_BAND_INNER from styles.py.


class IQRBandHandle:
    """Proxy artist used as a legend handle for IQR-band + median glyphs.

    The legend's handler_map routes instances of this class to
    :class:`IQRBandHandler`, which draws the composite glyph.
    """

    def __init__(
        self,
        color,
        *,
        alpha_outer=ALPHA_BAND_OUTER,
        alpha_inner=0.55,
        linewidth=LINEWIDTH_MEDIUM,
        linestyle='-',
        inner_height_frac=0.5,
    ):
        self.color = color
        self.alpha_outer = alpha_outer
        self.alpha_inner = alpha_inner
        self.linewidth = linewidth
        self.linestyle = linestyle
        self.inner_height_frac = inner_height_frac


class IQRBandHandler(HandlerBase):
    """Render an :class:`IQRBandHandle` as outer-rect + inner-rect + median-line."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        x0 = -xdescent
        y0 = -ydescent

        outer = Rectangle(
            (x0, y0), width, height,
            facecolor=orig_handle.color,
            alpha=orig_handle.alpha_outer,
            edgecolor='none',
            transform=trans,
        )

        inner_h = height * orig_handle.inner_height_frac
        inner_y = y0 + (height - inner_h) / 2
        inner = Rectangle(
            (x0, inner_y), width, inner_h,
            facecolor=orig_handle.color,
            alpha=orig_handle.alpha_inner,
            edgecolor='none',
            transform=trans,
        )

        median_y = y0 + height / 2
        median = Line2D(
            [x0, x0 + width],
            [median_y, median_y],
            color=orig_handle.color,
            linewidth=orig_handle.linewidth,
            linestyle=orig_handle.linestyle,
            solid_capstyle='butt',
            transform=trans,
        )

        return [outer, inner, median]


def iqr_band_legend_kwargs(handleheight=2.6, handlelength=3.6,
                            labelspacing=0.9, **extra):
    """Default kwargs for legends containing :class:`IQRBandHandle` entries.

    Returns a dict suitable for ``**kwargs`` unpacking into ``ax.legend(...)``
    or ``fig.legend(...)``. Handle box is sized so the outer fill, inner
    fill, and median line all read clearly at normal figure sizes.
    """
    kwargs = {
        'handler_map': {IQRBandHandle: IQRBandHandler()},
        'handleheight': handleheight,
        'handlelength': handlelength,
        'labelspacing': labelspacing,
    }
    kwargs.update(extra)
    return kwargs
