"""
Fig9: Drought Satisficing Heatmaps (3 rows x 2 columns).

Row 1 shows the Stationary Baseline in absolute units.
Rows 2-3 show the two climate-adjusted scenarios as change relative to
the Stationary Baseline: relative change for the exceedance-rate column
(left) and absolute delta for the fraction-avoiding-Drought-Emergency
column (right).

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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings("ignore")

from methods.config import FIG_DIR, N_YEARS
from methods.load import load_event_metrics
from methods.plotting.styles import (
    DATASET_LABELS, FONTSIZE_SMALL, FONTSIZE_LABEL, FONTSIZE_TITLE,
    DPI_HIGH, apply_publication_style,
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

SHOW_CHANGE = True   # relative-change view is the published default

# 1960s drought-of-record anchor point (severity, magnitude in deficit-months).
DOR_SEV = 2.8
DOR_MAG = 48.0

AXIS_FRAME_COLOR = '#333333'
EMPTY_CELL_COLOR = '#ededed'
HATCH_EDGECOLOR = '#bbbbbb'
SEV_TICKS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
SSI_CLASS_BANDS = [
    (1.0, 1.5, 'Moderate'),
    (1.5, 2.0, 'Severe'),
    (2.0, 4.5, 'Extreme'),
]

MIN_COUNT_POPULATED = 5


def _pct_formatter(v, _):
    if v == 0:
        return '0'
    return f'{100 * v:+.0f}%'


def _signed_decimal_formatter(v, _):
    if v == 0:
        return '0.00'
    return f'{v:+.2f}'


def _add_panel_letter(ax, letter):
    ax.text(
        0.025, 0.955, f'({letter})', transform=ax.transAxes,
        fontsize=FONTSIZE_LABEL, fontweight='bold',
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                  edgecolor='#666666', linewidth=0.6),
        zorder=11,
    )


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


def _relative_change(scenario_grid, baseline_grid):
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = (scenario_grid - baseline_grid) / baseline_grid
    bad = (np.isnan(baseline_grid) | (baseline_grid == 0)
           | np.isnan(scenario_grid))
    rel[bad] = np.nan
    return rel


def _absolute_change(scenario_grid, baseline_grid):
    diff = scenario_grid - baseline_grid
    bad = np.isnan(scenario_grid) | np.isnan(baseline_grid)
    diff[bad] = np.nan
    return diff


def plot_satisficing_heatmaps(all_data, ssi_window):
    apply_publication_style()

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

    # -- colormaps & norms --------------------------------------------------
    cmap_rate_abs = plt.cm.YlOrRd
    norm_rate_abs = mcolors.LogNorm(vmin=1e-4, vmax=2e-2)

    cmap_frac_abs = plt.cm.viridis_r
    norm_frac_abs = mcolors.Normalize(vmin=0.3, vmax=1.0)

    # Relative change in rate vs baseline.
    rate_rel_grids = {
        did: _relative_change(rate_grids[did], rate_grids[baseline_id])
        for did in DATASETS[1:]
    }
    cmap_rate_rel = plt.cm.RdBu_r
    norm_rate_rel = mcolors.SymLogNorm(
        linthresh=0.1, linscale=0.5, vmin=-2.0, vmax=2.0, base=10)

    # Absolute delta in fraction avoiding Drought Emergency.
    frac_diff_grids = {
        did: _absolute_change(frac_grids[did], frac_grids[baseline_id])
        for did in DATASETS[1:]
    }
    all_frac_diffs = np.concatenate(
        [fd[~np.isnan(fd)].ravel() for fd in frac_diff_grids.values()])
    if all_frac_diffs.size == 0:
        frac_diff_max = 0.05
    else:
        frac_diff_max = max(
            0.05,
            np.ceil(20 * np.nanmax(np.abs(all_frac_diffs))) / 20,
        )
        frac_diff_max = min(frac_diff_max, 0.5)
    cmap_frac_rel = plt.cm.RdBu
    norm_frac_rel = mcolors.TwoSlopeNorm(
        vmin=-frac_diff_max, vcenter=0.0, vmax=frac_diff_max)

    # -- figure layout ------------------------------------------------------
    fig = plt.figure(figsize=(9.5, 12.0))
    gs = gridspec.GridSpec(
        3, 2,
        hspace=0.10, wspace=0.08,
        left=0.13, right=0.93, bottom=0.10, top=0.84,
    )

    axes_rate, axes_frac = [], []

    for row_idx, did in enumerate(DATASETS):
        is_change_row = row_idx > 0

        # ── Col 0: exceedance rate ────────────────────────────────────
        ax_rate = fig.add_subplot(gs[row_idx, 0])
        if is_change_row:
            grid_r = rate_rel_grids[did]
            cmap_r, norm_r = cmap_rate_rel, norm_rate_rel
        else:
            grid_r = rate_grids[did]
            cmap_r, norm_r = cmap_rate_abs, norm_rate_abs
        ax_rate.set_facecolor(EMPTY_CELL_COLOR)
        ax_rate.pcolormesh(
            sev_edges, mag_edges, np.ma.masked_invalid(grid_r.T),
            cmap=cmap_r, norm=norm_r, rasterized=True, zorder=3,
        )
        _draw_insufficient_hatch(ax_rate, sev_edges, mag_edges, count_grids[did])
        draw_focal_boundary(ax_rate, sev_edges, mag_edges, focal_cells)
        axes_rate.append(ax_rate)

        # ── Col 1: fraction avoiding Drought Emergency ────────────────
        ax_frac = fig.add_subplot(gs[row_idx, 1])
        if is_change_row:
            grid_f = frac_diff_grids[did]
            cmap_f, norm_f = cmap_frac_rel, norm_frac_rel
        else:
            grid_f = frac_grids[did]
            cmap_f, norm_f = cmap_frac_abs, norm_frac_abs
        ax_frac.set_facecolor(EMPTY_CELL_COLOR)
        ax_frac.pcolormesh(
            sev_edges, mag_edges, np.ma.masked_invalid(grid_f.T),
            cmap=cmap_f, norm=norm_f, rasterized=True, zorder=3,
        )
        _draw_insufficient_hatch(ax_frac, sev_edges, mag_edges, count_grids[did])
        draw_focal_boundary(ax_frac, sev_edges, mag_edges, focal_cells)

        # x markers: at least one event in bin drove combined NYC storage
        # below the worst-storage threshold (15% of capacity), evaluated in
        # this panel's own scenario even on delta rows.
        min_grid = min_grids[did]
        for i, sc in enumerate(sev_centers):
            for j, mc in enumerate(mag_centers):
                if not np.isnan(min_grid[i, j]) and min_grid[i, j] < WORST_STORAGE_THRESH:
                    ax_frac.scatter(
                        sc, mc, marker='x', s=28, linewidths=1.0,
                        color='#202020', alpha=0.85, zorder=7,
                    )
        axes_frac.append(ax_frac)

    # -- shared axis styling -------------------------------------------------
    all_axes = axes_rate + axes_frac
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
        for xv in (1.0, 1.5, 2.0):
            ax.axvline(xv, color='#888888', linewidth=0.7,
                       linestyle=(0, (4, 3)), zorder=1)

    for row_idx in range(3):
        ax_rate = axes_rate[row_idx]
        ax_frac = axes_frac[row_idx]

        ax_rate.set_yticklabels(['1', '10', '100'])
        ax_frac.set_yticklabels([])

        ax_rate.set_ylabel(
            'Drought Magnitude\n(cumulative |SSI-3| deficit, deficit-months)',
            fontsize=FONTSIZE_LABEL,
        )

        if row_idx == 2:
            for ax in (ax_rate, ax_frac):
                ax.set_xlabel(
                    'Drought Severity\n(peak |SSI-3| during event)',
                    fontsize=FONTSIZE_LABEL,
                )
        else:
            ax_rate.set_xticklabels([])
            ax_frac.set_xticklabels([])

        _add_panel_letter(ax_rate, PANEL_LETTERS[row_idx * 2])
        _add_panel_letter(ax_frac, PANEL_LETTERS[row_idx * 2 + 1])

    # SSI-class band labels on the top axis of panels (a) and (b) only.
    for ax in (axes_rate[0], axes_frac[0]):
        for lo, hi, name in SSI_CLASS_BANDS:
            x_mid = 0.5 * (lo + hi)
            ax.text(
                x_mid, 1.02, name,
                transform=ax.get_xaxis_transform(),
                ha='center', va='bottom',
                fontsize=FONTSIZE_SMALL, color='#555555',
            )

    # 1960s drought-of-record overlay — left column only.
    for ax in axes_rate:
        ax.scatter(
            DOR_SEV, DOR_MAG, marker='*', s=160,
            facecolor='white', edgecolor='#000000', linewidths=1.2,
            zorder=10,
        )
    axes_rate[0].annotate(
        '1960s drought\nof record',
        xy=(DOR_SEV, DOR_MAG), xytext=(3.3, 70),
        fontsize=FONTSIZE_SMALL, ha='left', va='center',
        arrowprops=dict(arrowstyle='-', color='#000000', lw=0.6),
        zorder=10,
    )

    # -- row labels & column headers ----------------------------------------
    fig.canvas.draw()

    row_main_x = 0.035
    row_sub_x = 0.055
    for row_idx, did in enumerate(DATASETS):
        bb = axes_rate[row_idx].get_position()
        y_center = bb.y0 + bb.height / 2
        main = DATASET_LABELS[did]
        sub = '(absolute)' if row_idx == 0 else r'($\Delta$ vs baseline)'
        fig.text(row_main_x, y_center, main,
                 rotation=90, ha='center', va='center',
                 fontsize=FONTSIZE_LABEL, fontweight='bold')
        fig.text(row_sub_x, y_center, sub,
                 rotation=90, ha='center', va='center',
                 fontsize=FONTSIZE_LABEL - 1, color='#333333')

    bb_left_col = axes_rate[0].get_position()
    bb_right_col = axes_frac[0].get_position()
    x_left_center = bb_left_col.x0 + bb_left_col.width / 2
    x_right_center = bb_right_col.x0 + bb_right_col.width / 2

    fig.text(x_left_center, 0.965, 'Drought Event Frequency',
             ha='center', va='bottom',
             fontsize=FONTSIZE_TITLE, fontweight='bold')
    fig.text(x_right_center, 0.965, 'Operational Outcome at Minimum NYC Storage',
             ha='center', va='bottom',
             fontsize=FONTSIZE_TITLE, fontweight='bold')

    # -- colorbars (top strip, 2 cols x 2 rows) -----------------------------
    cbar_h = 0.014
    cbar_gap = 0.020
    y_abs = 0.925
    y_rel = y_abs - cbar_h - cbar_gap

    cbar_ax_rate_abs = fig.add_axes([bb_left_col.x0, y_abs,
                                      bb_left_col.width, cbar_h])
    cbar_ax_frac_abs = fig.add_axes([bb_right_col.x0, y_abs,
                                      bb_right_col.width, cbar_h])
    cbar_ax_rate_rel = fig.add_axes([bb_left_col.x0, y_rel,
                                      bb_left_col.width, cbar_h])
    cbar_ax_frac_rel = fig.add_axes([bb_right_col.x0, y_rel,
                                      bb_right_col.width, cbar_h])

    cb_rate_abs = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_rate_abs, norm=norm_rate_abs),
        cax=cbar_ax_rate_abs, orientation='horizontal',
        ticks=[1e-4, 1e-3, 1e-2],
    )
    cb_frac_abs = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_frac_abs, norm=norm_frac_abs),
        cax=cbar_ax_frac_abs, orientation='horizontal',
        ticks=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    cb_rate_rel = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_rate_rel, norm=norm_rate_rel),
        cax=cbar_ax_rate_rel, orientation='horizontal',
        ticks=[-2, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2],
    )
    frac_rel_ticks = [-frac_diff_max, -frac_diff_max / 2, 0,
                      frac_diff_max / 2, frac_diff_max]
    cb_frac_rel = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_frac_rel, norm=norm_frac_rel),
        cax=cbar_ax_frac_rel, orientation='horizontal',
        ticks=frac_rel_ticks,
    )

    cb_rate_rel.ax.xaxis.set_major_formatter(FuncFormatter(_pct_formatter))
    cb_frac_rel.ax.xaxis.set_major_formatter(FuncFormatter(_signed_decimal_formatter))

    cbar_labels = [
        (cb_rate_abs, cbar_ax_rate_abs,
         r'Exceedance rate (events yr$^{-1}$, log)'),
        (cb_frac_abs, cbar_ax_frac_abs,
         'Fraction of events avoiding Drought Emergency'),
        (cb_rate_rel, cbar_ax_rate_rel,
         'Relative change in exceedance rate vs Stationary Baseline'),
        (cb_frac_rel, cbar_ax_frac_rel,
         r'$\Delta$ Fraction avoiding Drought Emergency vs Stationary Baseline'),
    ]
    for cb, cax, label in cbar_labels:
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')
        cb.set_label(label, fontsize=FONTSIZE_SMALL)
        cb.ax.tick_params(labelsize=FONTSIZE_SMALL)
        cb.outline.set_edgecolor(AXIS_FRAME_COLOR)
        cb.outline.set_linewidth(0.8)

    # Row-scope tags under each colorbar.
    tag_gap = 0.004
    y_abs_tag = y_abs - tag_gap
    y_rel_tag = y_rel - tag_gap
    for x_c, y_tag, tag in [
        (x_left_center, y_abs_tag, 'Row 1 only'),
        (x_right_center, y_abs_tag, 'Row 1 only'),
        (x_left_center, y_rel_tag, 'Rows 2-3'),
        (x_right_center, y_rel_tag, 'Rows 2-3'),
    ]:
        fig.text(x_c, y_tag, tag, ha='center', va='top',
                 fontsize=FONTSIZE_SMALL - 1, style='italic', color='#555555')

    # -- bottom annotations --------------------------------------------------
    scope_text = (
        'Spatial scope: SSI-3 computed on cumulative inflow to combined '
        'NYC reservoirs (Cannonsville + Pepacton + Neversink). '
        'Drought Emergency zone defined under current FFMP rules at the '
        "event's annual-minimum combined storage date."
    )
    fig.text(0.5, 0.055, scope_text, ha='center', va='center',
             fontsize=FONTSIZE_SMALL, color='#333333')

    h_x = Line2D(
        [0], [0], marker='x', color='#202020', linestyle='none',
        markeredgewidth=1.0, markersize=7,
        label=(r'$\times$ = at least one event in this bin drove combined '
               'NYC storage below 15% of capacity (in this scenario)'),
    )
    h_hatch = Patch(
        facecolor='none', hatch='///', edgecolor=HATCH_EDGECOLOR,
        linewidth=0,
        label=f'Insufficient sample (< {MIN_COUNT_POPULATED} events in bin)',
    )
    h_empty = Patch(
        facecolor=EMPTY_CELL_COLOR, edgecolor=AXIS_FRAME_COLOR, linewidth=0.6,
        label='No drought events in this range',
    )
    fig.legend(
        handles=[h_x, h_hatch, h_empty],
        bbox_to_anchor=(0.5, 0.035), loc='lower center', ncol=3,
        fontsize=FONTSIZE_SMALL, frameon=False,
    )

    focal_text = (
        'Focal region (white outline): bins satisfying all three criteria '
        '—\n'
        f'(i) exceedance rate ≥ 1×10⁻⁴ events yr⁻¹ '
        'in all three ensembles; '
        f'(ii) ≥ 5% of events trigger Drought Emergency in all three '
        'ensembles; '
        f'(iii) at least one event drove combined NYC storage below '
        f'{WORST_STORAGE_THRESH:.0f}% of capacity in at least one ensemble.'
    )
    fig.text(0.5, 0.012, focal_text, ha='center', va='bottom',
             fontsize=FONTSIZE_SMALL, color='#333333')

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
