"""
Fig9: Drought Satisficing Heatmaps (3 rows x 2 columns).

Row 1 shows the Stationary Baseline in absolute units (drought event
return period and percent of droughts reaching Drought Emergency). Rows
2-3 show the two climate-adjusted scenarios as absolute change versus
the Stationary Baseline. Rate ↔ return-period inversion (RP = 1/λ) is
standard practice in hydrology, climate extremes assessment, and
flood-frequency analysis (e.g. FEMA flood mapping, IPCC AR6 extremes
framing, USGS Bulletin 17C).

Usage:
    python Fig9_plot_drought_satisficing_heatmap_2col.py [ssi_window]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings("ignore")

from methods.config import FIG_DIR, N_YEARS
from methods.load import load_event_metrics
from methods.plotting.styles import (
    DATASET_LABELS, DPI_HIGH, apply_publication_style, label_panel,
)
from methods.plotting.heatmap import (
    make_shared_edges_logmag, compute_min_storage_grid, compute_emergency_grid,
    compute_exceedance_rate_grid, identify_focal_region, draw_focal_boundary,
    GRID_N_BINS, WORST_STORAGE_THRESH, FOCAL_FRAC_THRESH, FOCAL_RATE_THRESH,
)

# -- configuration -----------------------------------------------------------
FIG_OUTPUT_DIR = f"{FIG_DIR}/Fig9_drought_satisficing"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

SSI_WINDOW_DEFAULT = 3
DATASETS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']
PANEL_LETTERS = list('abcdef')

SHOW_CHANGE = True

DOR_SEV = 2.8
DOR_MAG = 48.0

AXIS_FRAME_COLOR = '#333333'
EMPTY_CELL_COLOR = '#ededed'
HATCH_EDGECOLOR = '#bbbbbb'
SEV_TICKS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]

MIN_COUNT_POPULATED = 5


def _fmt_years(v):
    if abs(v) >= 1000:
        return f'{int(round(v)):,}'
    return f'{int(round(v))}'


def _fmt_signed_years(v):
    if v == 0:
        return '0'
    if abs(v) >= 1000:
        return f'{int(round(v)):+,}'
    return f'{int(round(v)):+d}'


def _fmt_pct(v):
    if v == 0:
        return '0%'
    return f'{int(round(v))}%'


def _fmt_signed_pct(v):
    if v == 0:
        return '0'
    return f'{int(round(v)):+d}'


def _style_axis_frame(ax):
    for spine in ax.spines.values():
        spine.set_color(AXIS_FRAME_COLOR)
        spine.set_linewidth(0.8)
    ax.tick_params(color=AXIS_FRAME_COLOR, width=0.8)
    ax.grid(False)


def _draw_insufficient_hatch(ax, sev_edges, mag_edges, count_grid):
    ns, nm = count_grid.shape
    for i in range(ns):
        for j in range(nm):
            cnt = count_grid[i, j]
            if 0 < cnt < MIN_COUNT_POPULATED:
                ax.add_patch(Rectangle(
                    (sev_edges[i], mag_edges[j]),
                    sev_edges[i + 1] - sev_edges[i],
                    mag_edges[j + 1] - mag_edges[j],
                    facecolor='none', hatch='///',
                    edgecolor=HATCH_EDGECOLOR, linewidth=0, zorder=2,
                ))


def _rate_to_return_period(rate_grid):
    """RP = 1/rate, with NaN preserved and zero-rate cells masked."""
    with np.errstate(divide='ignore', invalid='ignore'):
        rp = np.where((rate_grid > 0) & np.isfinite(rate_grid),
                      1.0 / np.where(rate_grid > 0, rate_grid, np.nan),
                      np.nan)
    return rp


def _absolute_change(scenario_grid, baseline_grid):
    diff = scenario_grid - baseline_grid
    bad = np.isnan(scenario_grid) | np.isnan(baseline_grid)
    diff[bad] = np.nan
    return diff


def _discrete_norm(boundaries, cmap_name, extend='both'):
    n_intervals = len(boundaries) - 1
    if extend == 'both':
        n_colors = n_intervals + 2
    elif extend in ('min', 'max'):
        n_colors = n_intervals + 1
    else:
        n_colors = n_intervals
    cmap = plt.get_cmap(cmap_name, n_colors)
    norm = mcolors.BoundaryNorm(boundaries, n_colors, extend=extend)
    return cmap, norm


def _audit_text_overlaps(fig, extra_artists=()):
    """Flag layout defects among figure-level text artists.

    Per-subplot axis labels are intentionally duplicated across rows
    (same "Drought Magnitude" on a/c/e, same "Drought Severity" on e/f)
    and matplotlib places them; including them here produces spurious
    same-text collisions. The check is restricted to ``fig.text``
    artists, ``fig.legend`` entries, and any explicitly-passed
    *extra_artists* (e.g. colorbar labels we want collision-checked).
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    artists = [t for t in fig.texts if t.get_text().strip()]
    for leg in fig.legends:
        artists.extend([t for t in leg.get_texts() if t.get_text().strip()])
    for art in extra_artists:
        if art.get_text().strip():
            artists.append(art)

    boxes = [a.get_window_extent(renderer=renderer) for a in artists]
    collisions = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                collisions.append((artists[i].get_text(),
                                   artists[j].get_text()))
    if collisions:
        msg = '\n'.join(f'  {a!r}  ⇄  {b!r}' for a, b in collisions)
        raise RuntimeError(f'Text overlaps detected:\n{msg}')


def plot_satisficing_heatmaps(all_data, ssi_window):
    apply_publication_style()

    # 12 pt hard minimum; no bold anywhere.
    plt.rcParams.update({'font.size': 13, 'font.weight': 'normal'})
    FONTSIZE_SMALL = 12
    FONTSIZE_LABEL = 14
    FONTSIZE_TITLE = 16

    baseline_id = DATASETS[0]

    sev_edges, mag_edges, sev_centers, mag_centers = make_shared_edges_logmag(
        all_data, DATASETS, n_bins=GRID_N_BINS)

    # -- compute grids ------------------------------------------------------
    rate_grids, frac_grids, min_grids, count_grids = {}, {}, {}, {}
    for did in DATASETS:
        rg, cg = compute_exceedance_rate_grid(
            all_data[did], sev_edges, mag_edges, N_YEARS,
            min_count=MIN_COUNT_POPULATED)
        rate_grids[did] = rg
        count_grids[did] = cg
        fg, _ = compute_emergency_grid(
            all_data[did], sev_edges, mag_edges, min_count=MIN_COUNT_POPULATED)
        frac_grids[did] = fg
        mg, _ = compute_min_storage_grid(
            all_data[did], sev_edges, mag_edges, min_count=MIN_COUNT_POPULATED)
        min_grids[did] = mg

    focal_cells = identify_focal_region(rate_grids, frac_grids, min_grids, DATASETS)
    print(f"  Focal region: {len(focal_cells)} cells")

    rp_grids = {did: _rate_to_return_period(rate_grids[did]) for did in DATASETS}
    pct_de_grids = {did: (1.0 - frac_grids[did]) * 100.0 for did in DATASETS}

    # -- discrete colormaps & norms -----------------------------------------
    rp_abs_bounds = np.array([100, 500, 1000, 5000, 10000], dtype=float)
    cmap_rp_abs, norm_rp_abs = _discrete_norm(
        rp_abs_bounds, 'YlOrRd_r', extend='both')
    rp_abs_ticks = rp_abs_bounds.tolist()
    rp_abs_tick_labels = [_fmt_years(v) for v in rp_abs_ticks]

    pct_de_bounds = np.array([0, 10, 20, 30, 40, 50, 60, 70], dtype=float)
    cmap_pct_de_abs, norm_pct_de_abs = _discrete_norm(
        pct_de_bounds, 'Reds', extend='max')
    pct_de_ticks = pct_de_bounds.tolist()
    pct_de_tick_labels = [_fmt_pct(v) for v in pct_de_ticks]

    rp_diff_grids = {
        did: _absolute_change(rp_grids[did], rp_grids[baseline_id])
        for did in DATASETS[1:]
    }
    rp_diff_bounds = np.array(
        [-1000, -100, -10, 10, 100, 1000], dtype=float)
    cmap_rp_rel, norm_rp_rel = _discrete_norm(
        rp_diff_bounds, 'RdBu', extend='both')
    rp_diff_ticks = rp_diff_bounds.tolist()
    rp_diff_tick_labels = [_fmt_signed_years(v) for v in rp_diff_ticks]

    pct_de_diff_grids = {
        did: _absolute_change(pct_de_grids[did], pct_de_grids[baseline_id])
        for did in DATASETS[1:]
    }
    pct_de_diff_bounds = np.array(
        [-50, -25, -10, -5, 5, 10, 25, 50], dtype=float)
    cmap_pct_de_rel, norm_pct_de_rel = _discrete_norm(
        pct_de_diff_bounds, 'RdBu_r', extend='both')
    pct_de_diff_ticks = pct_de_diff_bounds.tolist()
    pct_de_diff_tick_labels = [_fmt_signed_pct(v) for v in pct_de_diff_ticks]

    # -- figure layout ------------------------------------------------------
    # Single 3×2 GridSpec: rows 1-3 are contiguous (no dedicated gap row).
    # The Δ colorbars now live BELOW row 3, beneath the row-3 x-axis labels,
    # so they don't consume mid-figure vertical space. The enlarged bottom
    # margin accommodates: row-3 x-label, two Δ colorbars with 2-line
    # labels, a symbol strap, and the 4-line focal-region key.
    fig = plt.figure(figsize=(10.0, 16.5))
    GS_LEFT = 0.12
    GS_RIGHT = 0.88
    GS_TOP = 0.88
    GS_BOT = 0.26

    gs = gridspec.GridSpec(
        3, 2,
        left=GS_LEFT, right=GS_RIGHT,
        top=GS_TOP, bottom=GS_BOT,
        wspace=0.08, hspace=0.12,
    )

    axes_rp, axes_pct = [], []

    def _add_panels(row_idx, did):
        is_change_row = row_idx > 0
        ax_rp = fig.add_subplot(gs[row_idx, 0])
        if is_change_row:
            grid_r = rp_diff_grids[did]
            cmap_r, norm_r = cmap_rp_rel, norm_rp_rel
        else:
            grid_r = rp_grids[did]
            cmap_r, norm_r = cmap_rp_abs, norm_rp_abs
        ax_rp.set_facecolor(EMPTY_CELL_COLOR)
        ax_rp.pcolormesh(
            sev_edges, mag_edges, np.ma.masked_invalid(grid_r.T),
            cmap=cmap_r, norm=norm_r, rasterized=True, zorder=3,
        )
        _draw_insufficient_hatch(ax_rp, sev_edges, mag_edges, count_grids[did])
        draw_focal_boundary(ax_rp, sev_edges, mag_edges, focal_cells)
        axes_rp.append(ax_rp)

        ax_pct = fig.add_subplot(gs[row_idx, 1])
        if is_change_row:
            grid_p = pct_de_diff_grids[did]
            cmap_p, norm_p = cmap_pct_de_rel, norm_pct_de_rel
        else:
            grid_p = pct_de_grids[did]
            cmap_p, norm_p = cmap_pct_de_abs, norm_pct_de_abs
        ax_pct.set_facecolor(EMPTY_CELL_COLOR)
        ax_pct.pcolormesh(
            sev_edges, mag_edges, np.ma.masked_invalid(grid_p.T),
            cmap=cmap_p, norm=norm_p, rasterized=True, zorder=3,
        )
        _draw_insufficient_hatch(ax_pct, sev_edges, mag_edges, count_grids[did])
        draw_focal_boundary(ax_pct, sev_edges, mag_edges, focal_cells)

        min_grid = min_grids[did]
        for i, sc in enumerate(sev_centers):
            for j, mc in enumerate(mag_centers):
                if not np.isnan(min_grid[i, j]) and min_grid[i, j] < WORST_STORAGE_THRESH:
                    ax_pct.scatter(
                        sc, mc, marker='x', s=34, linewidths=1.1,
                        color='#202020', alpha=0.85, zorder=7,
                    )
        axes_pct.append(ax_pct)

    for row_idx, did in enumerate(DATASETS):
        _add_panels(row_idx, did)

    # -- shared axis styling -------------------------------------------------
    all_axes = axes_rp + axes_pct
    for ax in all_axes:
        ax.set_xlim(1.0, 4.5)
        ax.set_xticks(SEV_TICKS)
        ax.set_yscale('log')
        ax.set_ylim(1.0, 100.0)
        ax.set_yticks([1, 10, 100])
        ax.set_yticks(
            [2, 3, 4, 5, 6, 7, 8, 9, 20, 30, 40, 50, 60, 70, 80, 90],
            minor=True,
        )
        _style_axis_frame(ax)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_box_aspect(1.0)

    # Keep each line short enough that, after rotation, the two-line
    # y-label height (≈ 2 × font size) does not exceed the axis height,
    # and the one-line x-label width fits within the column.
    xaxis_label_text = 'Drought Severity (peak |SSI-3|)'
    yaxis_label_text = 'Drought Magnitude\n(|SSI-3| deficit-months)'

    for row_idx in range(3):
        ax_rp = axes_rp[row_idx]
        ax_pct = axes_pct[row_idx]
        ax_rp.set_yticklabels(['1', '10', '100'])
        ax_pct.set_yticklabels([])

        # Rows share a severity axis and panels e/f carry the tick labels;
        # suppress them on the upper two rows so the numbers appear once.
        if row_idx != 2:
            ax_rp.set_xticklabels([])
            ax_pct.set_xticklabels([])

        # Per-subplot magnitude label on every left-column panel (a, c, e).
        ax_rp.set_ylabel(yaxis_label_text, fontsize=FONTSIZE_LABEL, labelpad=6)

        # Per-subplot severity label on both bottom-row panels (e, f).
        if row_idx == 2:
            ax_rp.set_xlabel(xaxis_label_text, fontsize=FONTSIZE_LABEL, labelpad=6)
            ax_pct.set_xlabel(xaxis_label_text, fontsize=FONTSIZE_LABEL, labelpad=6)

        label_panel(ax_rp, PANEL_LETTERS[row_idx * 2],
                    fontsize=FONTSIZE_LABEL, fontweight='normal')
        label_panel(ax_pct, PANEL_LETTERS[row_idx * 2 + 1],
                    fontsize=FONTSIZE_LABEL, fontweight='normal')

    # 1960s drought-of-record star + annotation on panel (a) only.
    axes_rp[0].scatter(
        DOR_SEV, DOR_MAG, marker='*', s=220,
        facecolor='white', edgecolor='#000000', linewidths=1.2,
        zorder=10,
    )
    axes_rp[0].annotate(
        '1960s drought\nof record',
        xy=(DOR_SEV, DOR_MAG), xytext=(3.3, 70),
        fontsize=FONTSIZE_SMALL, ha='left', va='center',
        arrowprops=dict(arrowstyle='-', color='#000000', lw=0.6),
        zorder=10,
    )

    # -- scenario labels on right ------------------------------------------
    # Per-subplot axis labels are now drawn by matplotlib; no figure-level
    # axis labels are needed.
    fig.canvas.draw()

    # Scenario labels on the right-hand side of each row, using each right-
    # column axis's rendered box centre so the label aligns with the row
    # and with the left-column y-tick "10" at the panel's vertical midpoint.
    scenario_label_x = 0.935
    for row_idx, did in enumerate(DATASETS):
        bb = axes_pct[row_idx].get_position()
        y_center = bb.y0 + bb.height / 2
        fig.text(scenario_label_x, y_center, DATASET_LABELS[did],
                 rotation=270, ha='center', va='center',
                 fontsize=FONTSIZE_LABEL)

    # Column headers (super-titles for each column).
    bb_top_left = axes_rp[0].get_position()
    bb_top_right = axes_pct[0].get_position()
    x_left_center = bb_top_left.x0 + bb_top_left.width / 2
    x_right_center = bb_top_right.x0 + bb_top_right.width / 2

    fig.text(x_left_center, 0.965, 'Drought Event Frequency',
             ha='center', va='bottom', fontsize=FONTSIZE_TITLE)
    fig.text(x_right_center, 0.965, 'Drought Emergency Risk',
             ha='center', va='bottom', fontsize=FONTSIZE_TITLE)

    # -- top colorbars: absolute scales for row 1 ---------------------------
    cbar_top_h = 0.010
    cbar_top_y = 0.905
    cbar_ax_rp_abs = fig.add_axes(
        [bb_top_left.x0, cbar_top_y, bb_top_left.width, cbar_top_h])
    cbar_ax_pct_abs = fig.add_axes(
        [bb_top_right.x0, cbar_top_y, bb_top_right.width, cbar_top_h])

    cb_rp_abs = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_rp_abs, norm=norm_rp_abs),
        cax=cbar_ax_rp_abs, orientation='horizontal',
        extend='both', spacing='uniform',
    )
    cb_rp_abs.set_ticks(rp_abs_ticks)
    cb_rp_abs.set_ticklabels(rp_abs_tick_labels)

    cb_pct_abs = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_pct_de_abs, norm=norm_pct_de_abs),
        cax=cbar_ax_pct_abs, orientation='horizontal',
        extend='max', spacing='uniform',
    )
    cb_pct_abs.set_ticks(pct_de_ticks)
    cb_pct_abs.set_ticklabels(pct_de_tick_labels)

    # 2-row colorbar labels. Metric name on top; quantitative detail on
    # the second line (units, direction).
    cbar_labels_top = [
        (cb_rp_abs, cbar_ax_rp_abs,
         'Drought event return period\n(years; shorter = more frequent)'),
        (cb_pct_abs, cbar_ax_pct_abs,
         'Droughts reaching Drought Emergency\n(% of events in bin)'),
    ]
    # Label above each bar, tick labels below — applied consistently to all
    # four colorbars (top absolute and bottom Δ).
    for cb, cax, label in cbar_labels_top:
        cax.xaxis.set_ticks_position('bottom')
        cax.xaxis.set_label_position('top')
        cb.set_label(label, fontsize=FONTSIZE_SMALL, linespacing=1.35)
        cb.ax.tick_params(labelsize=FONTSIZE_SMALL)
        cb.outline.set_edgecolor(AXIS_FRAME_COLOR)
        cb.outline.set_linewidth(0.8)

    # -- bottom colorbars: Δ scales, positioned below row 3's x-axis label --
    cbar_bot_h = 0.010
    cbar_bot_y = 0.185
    bb_bot_left = axes_rp[2].get_position()
    bb_bot_right = axes_pct[2].get_position()
    cbar_ax_rp_rel = fig.add_axes(
        [bb_bot_left.x0, cbar_bot_y, bb_bot_left.width, cbar_bot_h])
    cbar_ax_pct_rel = fig.add_axes(
        [bb_bot_right.x0, cbar_bot_y, bb_bot_right.width, cbar_bot_h])

    cb_rp_rel = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_rp_rel, norm=norm_rp_rel),
        cax=cbar_ax_rp_rel, orientation='horizontal',
        extend='both', spacing='uniform',
    )
    cb_rp_rel.set_ticks(rp_diff_ticks)
    cb_rp_rel.set_ticklabels(rp_diff_tick_labels)

    cb_pct_rel = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_pct_de_rel, norm=norm_pct_de_rel),
        cax=cbar_ax_pct_rel, orientation='horizontal',
        extend='both', spacing='uniform',
    )
    cb_pct_rel.set_ticks(pct_de_diff_ticks)
    cb_pct_rel.set_ticklabels(pct_de_diff_tick_labels)

    cbar_labels_bot = [
        (cb_rp_rel, cbar_ax_rp_rel,
         r'$\Delta$ Return period'
         '\n(years; scenario − baseline)'),
        (cb_pct_rel, cbar_ax_pct_rel,
         r'$\Delta$ Droughts reaching Drought Emergency'
         '\n(percentage points; scenario − baseline)'),
    ]
    for cb, cax, label in cbar_labels_bot:
        cax.xaxis.set_ticks_position('bottom')
        cax.xaxis.set_label_position('top')
        cb.set_label(label, fontsize=FONTSIZE_SMALL, linespacing=1.35)
        cb.ax.tick_params(labelsize=FONTSIZE_SMALL)
        cb.outline.set_edgecolor(AXIS_FRAME_COLOR)
        cb.outline.set_linewidth(0.8)

    # -- bottom annotations: symbols + left-aligned focal-region key --------
    # Symbol line stays centred; the focal-region criteria list reads like
    # a bulleted key and is left-aligned so the three items stack under a
    # common indent.
    symbol_line = (
        r'$\times$ = bin where $\geq$1 event drove combined NYC storage '
        '<15% of capacity    |    '
        r'$/\!/\!/$ insufficient sample (<5 events in bin)    |    '
        'gray = bin with no drought events'
    )
    fig.text(0.5, 0.115, symbol_line, ha='center', va='center',
             fontsize=FONTSIZE_SMALL, color='#333333')

    criteria_x = 0.14
    fig.text(criteria_x, 0.088,
             'Focal region (white outline): bins satisfying all three criteria simultaneously —',
             ha='left', va='center',
             fontsize=FONTSIZE_SMALL, color='#333333')
    fig.text(criteria_x + 0.02, 0.064,
             r'(i)   Return period $\leq$ 10,000 years in all three ensembles',
             ha='left', va='center',
             fontsize=FONTSIZE_SMALL, color='#333333')
    fig.text(criteria_x + 0.02, 0.040,
             r'(ii)  $\geq$5% of droughts reach Drought Emergency in all three ensembles',
             ha='left', va='center',
             fontsize=FONTSIZE_SMALL, color='#333333')
    fig.text(criteria_x + 0.02, 0.016,
             r'(iii) $\geq$1 event drove combined NYC storage <15% of capacity in any ensemble',
             ha='left', va='center',
             fontsize=FONTSIZE_SMALL, color='#333333')

    cbar_label_artists = [
        cb.ax.xaxis.label for cb in
        (cb_rp_abs, cb_pct_abs, cb_rp_rel, cb_pct_rel)
    ]
    _audit_text_overlaps(fig, extra_artists=cbar_label_artists)

    # -- save ----------------------------------------------------------------
    fname_base = (f"{FIG_OUTPUT_DIR}/Fig9_satisficing_heatmap_ssi{ssi_window}"
                  f"_rate{FOCAL_RATE_THRESH:.0e}_frac{FOCAL_FRAC_THRESH:.2f}"
                  f"_sto{WORST_STORAGE_THRESH:.0f}")
    fname_png = fname_base + '.png'
    fname_svg = fname_base + '.svg'
    fig.savefig(fname_png, dpi=DPI_HIGH, bbox_inches='tight')
    fig.savefig(fname_svg, bbox_inches='tight')
    print(f"Saved: {fname_png}")
    print(f"Saved: {fname_svg}")
    plt.close(fig)


# -- main --------------------------------------------------------------------

def main():
    ssi_window = int(sys.argv[1]) if len(sys.argv) > 1 else SSI_WINDOW_DEFAULT
    print(f"Fig9: Drought Satisficing Heatmaps (SSI-{ssi_window},"
          f" show_change={SHOW_CHANGE})")

    all_data = {}
    for did in DATASETS:
        df = load_event_metrics(did, ssi_window)
        all_data[did] = df
        print(f"  {DATASET_LABELS.get(did, did)}: {len(df)} events")

    plot_satisficing_heatmaps(all_data, ssi_window)
    print("Done.")


if __name__ == '__main__':
    main()
