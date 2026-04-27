"""
Preview the reusable IQR-band legend handler.

Two panels, left to right:
  Left  — annotated "anatomy" of a single glyph (neutral grey).
          Callouts fan vertically: outer band label sits above and
          arrows up-left into the band, inner band is horizontal, and
          median label sits below and arrows down-left onto the line.
  Right — the compact dataset legend as it would appear in a figure.

Output: outputs/preview_iqr_legend.png
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from methods.plotting.styles import (
    DATASET_COLORS, DATASET_LABELS,
    ALPHA_BAND_OUTER, LINEWIDTH_MEDIUM,
)
from methods.plotting.legend import IQRBandHandle, iqr_band_legend_kwargs


ALL_DATASETS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']

GLYPH_ALPHA_OUTER = ALPHA_BAND_OUTER
GLYPH_ALPHA_INNER = 0.55
GLYPH_INNER_FRAC = 0.5
ANATOMY_COLOR = '#4d4d4d'
LABEL_COLOR = '#222222'
ARROW_COLOR = '#555555'


def _draw_hero_legend(ax):
    ax.set_axis_off()
    handles = [IQRBandHandle(color=DATASET_COLORS[d]) for d in ALL_DATASETS]
    labels = [DATASET_LABELS[d].replace('\n', ' ') for d in ALL_DATASETS]
    ax.legend(
        handles, labels,
        loc='center',
        frameon=False,
        fontsize=12,
        **iqr_band_legend_kwargs(),
    )


def _draw_anatomy(ax):
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    gx0, gx1 = 5.5, 9.4
    gy_mid = 5.0
    outer_h = 5.2
    inner_h = outer_h * GLYPH_INNER_FRAC
    gy_outer_bot = gy_mid - outer_h / 2     # 2.4
    gy_outer_top = gy_mid + outer_h / 2     # 7.6
    gy_inner_top = gy_mid + inner_h / 2     # 6.3

    median_lw = LINEWIDTH_MEDIUM * (outer_h / 2.6)

    ax.add_patch(Rectangle(
        (gx0, gy_outer_bot), gx1 - gx0, outer_h,
        facecolor=ANATOMY_COLOR, alpha=GLYPH_ALPHA_OUTER,
        edgecolor='none', zorder=2,
    ))
    ax.add_patch(Rectangle(
        (gx0, gy_mid - inner_h / 2), gx1 - gx0, inner_h,
        facecolor=ANATOMY_COLOR, alpha=GLYPH_ALPHA_INNER,
        edgecolor='none', zorder=3,
    ))
    ax.plot([gx0, gx1], [gy_mid, gy_mid],
            color=ANATOMY_COLOR, linewidth=median_lw,
            solid_capstyle='butt', zorder=4)

    target_x = gx0 + 0.1
    outer_only_top_y = (gy_inner_top + gy_outer_top) / 2       # 6.95
    inner_above_median_y = (gy_mid + gy_inner_top) / 2         # 5.65

    # Fanned layout: outer label sits above and aims up-left onto the
    # outer-only strip; inner label is horizontal through the inner band;
    # median label sits below and aims down-left onto the median line.
    label_x = 4.6
    callouts = [
        ('1–99% range\n(outer band)',
         (target_x, outer_only_top_y),
         (label_x, 8.4)),
        ('25–75% range\n(inner band)',
         (target_x, inner_above_median_y),
         (label_x, inner_above_median_y)),
        ('50th percentile\n(median)',
         (target_x, gy_mid),
         (label_x, 1.6)),
    ]

    for text, xy, xytext in callouts:
        ax.annotate(
            text,
            xy=xy, xytext=xytext,
            fontsize=11, color=LABEL_COLOR,
            va='center', ha='right',
            arrowprops=dict(
                arrowstyle='-', color=ARROW_COLOR,
                lw=0.9, shrinkA=6, shrinkB=4,
            ),
            zorder=5,
        )
        ax.plot(*xy, marker='o', markersize=4.5,
                markerfacecolor=ARROW_COLOR, markeredgecolor='white',
                markeredgewidth=0.8, zorder=6)


def main():
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'preview_iqr_legend.png')

    fig, (ax_anatomy, ax_legend) = plt.subplots(
        1, 2, figsize=(12.5, 3.2),
        gridspec_kw=dict(width_ratios=[1.0, 1.25], wspace=0.05),
    )
    fig.patch.set_facecolor('white')

    _draw_anatomy(ax_anatomy)
    _draw_hero_legend(ax_legend)

    fig.suptitle('Ensemble legend design',
                 fontsize=13, color='#333333', y=0.98, fontweight='semibold')

    fig.savefig(out_path, dpi=220, bbox_inches='tight',
                pad_inches=0.3, facecolor='white')
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
