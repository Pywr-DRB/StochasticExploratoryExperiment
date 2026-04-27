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

from methods.plotting.styles import ALPHA_BAND_OUTER, LINEWIDTH_THIN, LINEWIDTH_MEDIUM

# Legend-only: inner-band alpha is bumped above the plot's ALPHA_BAND_INNER
# so the three layers stay distinguishable at small legend-handle sizes.
# The plot itself continues to use ALPHA_BAND_INNER from styles.py.
ANATOMY_ALPHA_INNER = 0.55
ANATOMY_COLOR_DEFAULT = '#4d4d4d'


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
        show_inner_band=True,
        outline_only=False,
        outline_linewidth=LINEWIDTH_THIN,
    ):
        self.color = color
        self.alpha_outer = alpha_outer
        self.alpha_inner = alpha_inner
        self.linewidth = linewidth
        self.linestyle = linestyle
        self.inner_height_frac = inner_height_frac
        self.show_inner_band = show_inner_band
        self.outline_only = outline_only
        self.outline_linewidth = outline_linewidth


class IQRBandHandler(HandlerBase):
    """Render an :class:`IQRBandHandle` as outer-rect + inner-rect + median-line."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        x0 = -xdescent
        y0 = -ydescent

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

        if getattr(orig_handle, 'outline_only', False):
            top = Line2D(
                [x0, x0 + width], [y0 + height, y0 + height],
                color=orig_handle.color,
                linewidth=orig_handle.outline_linewidth,
                linestyle='-', solid_capstyle='butt', transform=trans,
            )
            bottom = Line2D(
                [x0, x0 + width], [y0, y0],
                color=orig_handle.color,
                linewidth=orig_handle.outline_linewidth,
                linestyle='-', solid_capstyle='butt', transform=trans,
            )
            return [top, bottom, median]

        outer = Rectangle(
            (x0, y0), width, height,
            facecolor=orig_handle.color,
            alpha=orig_handle.alpha_outer,
            edgecolor='none',
            transform=trans,
        )

        if getattr(orig_handle, 'show_inner_band', True):
            inner_h = height * orig_handle.inner_height_frac
            inner_y = y0 + (height - inner_h) / 2
            inner = Rectangle(
                (x0, inner_y), width, inner_h,
                facecolor=orig_handle.color,
                alpha=orig_handle.alpha_inner,
                edgecolor='none',
                transform=trans,
            )
            return [outer, inner, median]

        return [outer, median]


def draw_iqr_anatomy(
    ax,
    *,
    color=ANATOMY_COLOR_DEFAULT,
    alpha_outer=ALPHA_BAND_OUTER,
    alpha_inner=ANATOMY_ALPHA_INNER,
    inner_height_frac=0.5,
    fontsize=10,
    label_outer='1–99% range',
    label_inner='25–75% range',
    label_median='50th percentile',
    arrow_color='#555555',
    label_color='#222222',
    show_inner_band=True,
):
    """Draw the teaching/anatomy glyph for IQR-band + median legends.

    Renders a large grey glyph in the right half of ``ax`` with callout
    labels fanned vertically to the left: outer band label sits above
    and arrows up-left onto the top outer-only strip; inner band label
    is horizontal into the 25–75% band; median label sits below and
    arrows down-left onto the median line.

    The function configures ``ax`` (turns axis off, sets xlim/ylim to
    ``(0, 10)``); caller just needs to place the axes in the desired
    location on the figure.
    """
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    gx0, gx1 = 5.5, 9.4
    gy_mid = 5.0
    outer_h = 5.2
    inner_h = outer_h * inner_height_frac
    gy_outer_bot = gy_mid - outer_h / 2
    gy_outer_top = gy_mid + outer_h / 2
    gy_inner_top = gy_mid + inner_h / 2

    # Median thickness scaled to match the handle's median-to-height ratio
    median_lw = LINEWIDTH_MEDIUM * (outer_h / 2.6)

    ax.add_patch(Rectangle(
        (gx0, gy_outer_bot), gx1 - gx0, outer_h,
        facecolor=color, alpha=alpha_outer, edgecolor='none', zorder=2,
    ))
    if show_inner_band:
        ax.add_patch(Rectangle(
            (gx0, gy_mid - inner_h / 2), gx1 - gx0, inner_h,
            facecolor=color, alpha=alpha_inner, edgecolor='none', zorder=3,
        ))
    ax.plot([gx0, gx1], [gy_mid, gy_mid],
            color=color, linewidth=median_lw,
            solid_capstyle='butt', zorder=4)

    target_x = gx0 + 0.1
    outer_only_top_y = (gy_inner_top + gy_outer_top) / 2
    inner_above_median_y = (gy_mid + gy_inner_top) / 2
    label_x = 4.6

    if show_inner_band:
        callouts = [
            (label_outer,  (target_x, outer_only_top_y),     (label_x, 8.4)),
            (label_inner,  (target_x, inner_above_median_y), (label_x, inner_above_median_y)),
            (label_median, (target_x, gy_mid),               (label_x, 1.6)),
        ]
    else:
        callouts = [
            (label_outer,  (target_x, outer_only_top_y),     (label_x, 8.4)),
            (label_median, (target_x, gy_mid),               (label_x, 1.6)),
        ]

    for text, xy, xytext in callouts:
        ax.annotate(
            text, xy=xy, xytext=xytext,
            fontsize=fontsize, color=label_color,
            va='center', ha='right',
            arrowprops=dict(
                arrowstyle='-', color=arrow_color, lw=0.9,
                shrinkA=5, shrinkB=3,
            ),
            zorder=5,
        )
        ax.plot(*xy, marker='o', markersize=4.0,
                markerfacecolor=arrow_color, markeredgecolor='white',
                markeredgewidth=0.6, zorder=6)


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
