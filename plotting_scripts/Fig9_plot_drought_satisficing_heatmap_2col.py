"""
Fig9: Drought Satisficing Heatmaps (3 rows x 2 columns).

Row 1 shows the Stationary Baseline in absolute units (drought-free
interval between events ``T_W`` and percent of droughts reaching
Drought Emergency). Rows 2-3 show the two climate-adjusted scenarios as
absolute change versus the Stationary Baseline.

The drought-free interval ``T_W = T_R − E[D|exc]`` is the
Bonaccorso-Shiau interarrival time minus the mean event duration over
the joint exceedance region. Each cell's value is evaluated at its
lower-left corner, giving the recurrence interval for events
*at least as severe and as long* as that point — the bivariate "AND"
return period (Shiau & Shen 2001; Bonaccorso, Cancelliere & Rossi 2003;
Shiau 2006; Salvadori & De Michele 2004, 2010), with the empirical
copula (Deheuvels 1979; Nelsen 2006) used to estimate the joint
distribution. For multi-year droughts the duration adjustment avoids
inflating the recurrence interval with the drought event itself
(Loaiciga & Mariño 1991; Fernández & Salas 1999; Salas & Obeysekera 2014).

Usage:
    python Fig9_plot_drought_satisficing_heatmap_2col.py [ssi_window]
"""

import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings("ignore")

from methods.config import (
    FIG_DIR, N_YEARS,
    GRID_N_BINS, FOCAL_WORST_STORAGE_THRESH,
    FOCAL_FRAC_THRESH, FOCAL_RP_THRESH_YEARS,
)
from methods.load import load_event_metrics
from methods.return_period import compute_return_period_grid_exceedance as compute_return_period_grid
from methods.plotting.styles import (
    DATASET_LABELS, DATASET_LABELS_SHORT, DATASET_COLORS,
    DPI_HIGH, apply_publication_style, label_panel,
    MANUSCRIPT_CMAPS,
)
from methods.plotting.heatmap import (
    make_shared_edges_logmag, compute_min_storage_grid, compute_emergency_grid,
    identify_focal_region, draw_focal_boundary,
)

# -- configuration -----------------------------------------------------------
FIG_OUTPUT_DIR = f"{FIG_DIR}/Fig9_drought_satisficing"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

SSI_WINDOW_DEFAULT = 3
DATASETS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']
PANEL_LETTERS = list('abcdef')

SHOW_CHANGE = False

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


def _absolute_change(scenario_grid, baseline_grid):
    diff = scenario_grid - baseline_grid
    bad = np.isnan(scenario_grid) | np.isnan(baseline_grid)
    diff[bad] = np.nan
    return diff


def _ceil_to_5pct(v):
    """Round *v* (in percentage points) up to the next multiple of 5."""
    if not np.isfinite(v):
        return 5.0
    return float(math.ceil(v / 5.0) * 5.0)


def _focal_region_event_mask(df, sev_edges, mag_edges, focal_cells):
    if not focal_cells:
        return np.zeros(len(df), dtype=bool)
    ns = len(sev_edges) - 1
    nm = len(mag_edges) - 1
    sev_idx = np.clip(np.digitize(df['severity'].values, sev_edges) - 1, 0, ns - 1)
    mag_idx = np.clip(np.digitize(df['magnitude'].values, mag_edges) - 1, 0, nm - 1)
    mask = np.zeros(len(df), dtype=bool)
    for i, j in focal_cells:
        mask |= (sev_idx == i) & (mag_idx == j)
    return mask


def _count_events_in_focal_region(df, sev_edges, mag_edges, focal_cells):
    """Count events (rows of ``df``) whose (sev_bin, mag_bin) falls in
    ``focal_cells``. Uses the same digitize logic as the grid builders in
    ``methods.plotting.heatmap`` so the count is consistent with the
    rendered heatmaps.
    """
    return int(_focal_region_event_mask(df, sev_edges, mag_edges, focal_cells).sum())


def _sum_focal_region_event_years(df, sev_edges, mag_edges, focal_cells):
    """Total drought-years inside the focal region: the sum of event
    durations (converted from days to years) for events whose (sev, mag)
    bin lies in ``focal_cells``. Counts every realization, so a 2-yr
    event in each of 100 realizations contributes 200 yr.
    """
    mask = _focal_region_event_mask(df, sev_edges, mag_edges, focal_cells)
    if not mask.any():
        return 0.0
    return float(df.loc[mask, 'duration_days'].sum() / 365.25)


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
    # Bonaccorso-Shiau return-period grids: T_R is the mean interarrival
    # time per bin; T_W = T_R − E[D|bin] is the duration-adjusted
    # drought-free interval (the displayed quantity and the focal-region
    # threshold metric).
    T_R_grids, dur_grids, T_W_grids = {}, {}, {}
    frac_grids, min_grids, count_grids = {}, {}, {}
    for did in DATASETS:
        T_R, dur, T_W, cg = compute_return_period_grid(
            all_data[did], sev_edges, mag_edges, N_YEARS,
            min_count=MIN_COUNT_POPULATED)
        T_R_grids[did] = T_R
        dur_grids[did] = dur
        T_W_grids[did] = T_W
        count_grids[did] = cg
        fg, _ = compute_emergency_grid(
            all_data[did], sev_edges, mag_edges, min_count=MIN_COUNT_POPULATED)
        frac_grids[did] = fg
        mg, _ = compute_min_storage_grid(
            all_data[did], sev_edges, mag_edges, min_count=MIN_COUNT_POPULATED)
        min_grids[did] = mg

    focal_cells = identify_focal_region(T_W_grids, frac_grids, min_grids, DATASETS)
    focal_event_counts = {
        did: _count_events_in_focal_region(
            all_data[did], sev_edges, mag_edges, focal_cells)
        for did in DATASETS
    }
    focal_overlap_years = {
        did: _sum_focal_region_event_years(
            all_data[did], sev_edges, mag_edges, focal_cells)
        for did in DATASETS
    }
    print(f"  Focal region (T_W ≤ {FOCAL_RP_THRESH_YEARS} yr): "
          f"{len(focal_cells)} cells")
    for did in DATASETS:
        cell_durs = [dur_grids[did][i, j]
                     for (i, j) in focal_cells
                     if not np.isnan(dur_grids[did][i, j])]
        mean_dur = float(np.mean(cell_durs)) if cell_durs else float('nan')
        print(f"    {DATASET_LABELS[did]}: "
              f"{focal_event_counts[did]:,} events in focal region; "
              f"mean E[D|focal] = {mean_dur:.2f} yr; "
              f"total drought-years in focal region = "
              f"{focal_overlap_years[did]:,.0f}")

    # Apply a unified per-bin sample-size mask to BOTH columns so the
    # displayed support is identical. The right-column emergency grid is
    # already NaN where bin count < MIN_COUNT_POPULATED via
    # compute_emergency_grid; mirror that on the left column. The joint-
    # exceedance return period is mathematically defined for every cell
    # (it pools events from the upper-right quadrant), but displaying it
    # over bins the ensemble itself never produced is visually misleading.
    rp_grids = {}
    pct_de_grids = {}
    for did in DATASETS:
        insufficient = count_grids[did] < MIN_COUNT_POPULATED
        rp_masked = T_W_grids[did].copy()
        rp_masked[insufficient] = np.nan
        rp_grids[did] = rp_masked
        pct_masked = ((1.0 - frac_grids[did]) * 100.0)
        pct_masked[insufficient] = np.nan
        pct_de_grids[did] = pct_masked

    # -- colormaps & continuous norms ---------------------------------------
    # Manuscript-wide palette: sequential = viridis_r, diverging = BrBG_r.
    # For the two Δ panels we want "brown = adverse in each metric", so
    # the diverging palette is used as-is where *positive* Δ is adverse
    # (%DE increases → more emergencies), and reversed where *negative* Δ
    # is adverse (RP shortens → more frequent droughts). Reversing BrBG_r
    # yields BrBG (brown on the low / negative end).
    cmap_sequential = MANUSCRIPT_CMAPS['sequential']         # viridis_r
    cmap_diverging_pos_brown = MANUSCRIPT_CMAPS['diverging']  # BrBG_r
    cmap_diverging_neg_brown = cmap_diverging_pos_brown.reversed()  # BrBG

    # All four colormaps are discretized via BoundaryNorm so the heatmaps
    # show distinct colour bands rather than continuous gradients. Each
    # palette is the same as before; only the binning is new.

    # Panel (a): joint-exceedance return period (years), discrete log bands
    # from 10 → 10,000 yr using a 1-2-5 pattern (3 bands per decade).
    rp_abs_boundaries = np.array(
        [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000], dtype=float)
    n_rp_bins = len(rp_abs_boundaries) - 1
    cmap_rp_abs = cmap_sequential.resampled(n_rp_bins)
    norm_rp_abs = mcolors.BoundaryNorm(rp_abs_boundaries, ncolors=n_rp_bins)
    # Label only the decade boundaries to prevent tick-label overlap on
    # the discrete 1-2-5 colorbar; the underlying binning is unchanged.
    rp_abs_ticks = [10, 100, 1000, 10000]
    rp_abs_tick_labels = [_fmt_years(v) for v in rp_abs_ticks]

    # Panel (b): percent reaching DE, discrete 10-pp bands from 0 → ceil-to-10pp
    # of the observed max.
    pct_de_data_max = float(np.nanmax(
        [np.nanmax(pct_de_grids[did]) for did in DATASETS]
    ))
    pct_de_vmax = max(10.0, math.ceil(pct_de_data_max / 10.0) * 10.0)
    pct_de_boundaries = np.arange(0.0, pct_de_vmax + 0.01, 10.0)
    n_pct_bins = len(pct_de_boundaries) - 1
    cmap_pct_de_abs = cmap_sequential.resampled(n_pct_bins)
    norm_pct_de_abs = mcolors.BoundaryNorm(pct_de_boundaries, ncolors=n_pct_bins)
    pct_de_ticks = list(pct_de_boundaries)
    pct_de_tick_labels = [f'{int(round(v))}' for v in pct_de_ticks]

    # Panels (c, e): Δ return period (years), discrete sym-log bands.
    # Brown should mark the adverse side (negative Δ = T_W shortened).
    # Central bin [-10, 10] is the "no significant change" band (≈ white).
    rp_diff_grids = {
        did: _absolute_change(rp_grids[did], rp_grids[baseline_id])
        for did in DATASETS[1:]
    }
    rp_diff_boundaries = np.array(
        [-1000, -300, -100, -30, -10, 10, 30, 100, 300, 1000], dtype=float)
    n_rp_diff_bins = len(rp_diff_boundaries) - 1
    cmap_rp_rel = cmap_diverging_neg_brown.resampled(n_rp_diff_bins)
    norm_rp_rel = mcolors.BoundaryNorm(rp_diff_boundaries, ncolors=n_rp_diff_bins)
    rp_diff_ticks = [-1000, -100, -10, 10, 100, 1000]
    rp_diff_tick_labels = [_fmt_signed_years(v) for v in rp_diff_ticks]

    # Panels (d, f): Δ percent reaching DE (percentage points), discrete 10-pp
    # bands from -50 to +50. Brown marks the adverse side (positive Δ).
    pct_de_diff_grids = {
        did: _absolute_change(pct_de_grids[did], pct_de_grids[baseline_id])
        for did in DATASETS[1:]
    }
    pct_de_diff_boundaries = np.arange(-50.0, 50.0 + 0.01, 10.0)
    n_pct_diff_bins = len(pct_de_diff_boundaries) - 1
    cmap_pct_de_rel = cmap_diverging_pos_brown.resampled(n_pct_diff_bins)
    norm_pct_de_rel = mcolors.BoundaryNorm(
        pct_de_diff_boundaries, ncolors=n_pct_diff_bins)
    pct_de_diff_ticks = [-50, -25, 0, 25, 50]
    pct_de_diff_tick_labels = [_fmt_signed_pct(v) for v in pct_de_diff_ticks]

    # -- figure layout ------------------------------------------------------
    # 3×3 GridSpec. Columns 0-1 hold the heatmap matrix; column 2 holds the
    # focal-region summary block: the bar chart of total drought-years inside
    # the focal region (top cell) and the focal-region criteria text (bottom
    # two cells, merged). Top/bottom colorbars span only cols 0-1 (their
    # geometry pulls bb of axes_rp[0]/axes_pct[0] / [2]).
    fig = plt.figure(figsize=(13.0, 14.5))
    GS_LEFT = 0.085
    GS_RIGHT = 0.97
    GS_TOP = 0.86
    GS_BOT = 0.18

    gs = gridspec.GridSpec(
        3, 3,
        left=GS_LEFT, right=GS_RIGHT,
        top=GS_TOP, bottom=GS_BOT,
        wspace=0.10, hspace=0.12,
        width_ratios=[1.0, 1.0, 0.95],
    )

    axes_rp, axes_pct = [], []
    ax_bar = fig.add_subplot(gs[0, 2])
    ax_criteria = fig.add_subplot(gs[1:3, 2])

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
                if not np.isnan(min_grid[i, j]) and min_grid[i, j] < FOCAL_WORST_STORAGE_THRESH:
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
        # Major tick labels on every subplot (both columns, all rows).
        ax_rp.set_yticklabels(['1', '10', '100'])
        ax_pct.set_yticklabels(['1', '10', '100'])

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

    # 1960s drought-of-record marker + annotation on panel (a) only.
    # Red triangle to match Fig5's drought-of-record marker style.
    axes_rp[0].scatter(
        DOR_SEV, DOR_MAG, marker='^', s=100,
        color='red', edgecolors='white', linewidths=1.0,
        zorder=10,
    )
    axes_rp[0].annotate(
        '1960s drought\nof record',
        xy=(DOR_SEV, DOR_MAG), xytext=(3.3, 70),
        fontsize=FONTSIZE_SMALL, ha='left', va='center',
        arrowprops=dict(arrowstyle='-', color='#000000', lw=0.6),
        zorder=10,
    )

    # -- focal-region bar chart (top right cell) ---------------------------
    # Total drought-years within the focal region per scenario. "Drought-
    # years" = sum of event durations across all realizations for events
    # whose (severity, magnitude) bin falls inside the focal region.
    bar_xs = np.arange(len(DATASETS))
    bar_values = [focal_overlap_years[did] for did in DATASETS]
    bar_colors = [DATASET_COLORS[did] for did in DATASETS]
    ax_bar.bar(bar_xs, bar_values, color=bar_colors,
               edgecolor=AXIS_FRAME_COLOR, linewidth=0.8, zorder=3)
    ax_bar.set_xticks(bar_xs)
    ax_bar.set_xticklabels(
        [DATASET_LABELS_SHORT[did] for did in DATASETS],
        fontsize=FONTSIZE_SMALL, rotation=0,
    )
    bar_max = max(bar_values) if bar_values else 0.0
    ax_bar.set_ylim(0.0, bar_max * 1.18 if bar_max > 0 else 1.0)
    for x, v in zip(bar_xs, bar_values):
        ax_bar.text(x, v + bar_max * 0.02, f'{v:,.0f}',
                    ha='center', va='bottom',
                    fontsize=FONTSIZE_SMALL, color='#222222')
    # Clean bar-chart left side so the scenario row labels (in the gap
    # between heatmaps and this column) have unobstructed space.
    ax_bar.set_yticks([])
    ax_bar.set_ylabel('')
    _style_axis_frame(ax_bar)
    for side in ('top', 'right', 'left'):
        ax_bar.spines[side].set_visible(False)
    ax_bar.tick_params(axis='y', length=0)
    ax_bar.set_title('Drought-years in\nfocal region',
                     fontsize=FONTSIZE_LABEL, pad=8)
    label_panel(ax_bar, 'g', fontsize=FONTSIZE_LABEL, fontweight='normal')

    # -- focal-region criteria text (bottom-right merged cell) -------------
    ax_criteria.set_axis_off()
    crit_lines = [
        ('header', 'Focal Region (white outline)'),
        ('sub',    'All three criteria must hold:'),
        ('item',   rf'(i)   Return Period (joint exc.) $\leq$ '
                   rf'{FOCAL_RP_THRESH_YEARS:,} yr'
                   '\n        in all 3 scenarios'),
        ('item',   r'(ii)  $\geq$5% of droughts reach DE'
                   '\n        in all 3 scenarios'),
        ('item',   r'(iii) NYC storage <15% in $\geq$1 event'
                   '\n        in at least 1 of the 3 scenarios'),
        ('symbol', r'$\times$  bin where $\geq$1 event drove combined NYC'
                   '\n     storage <15% of capacity'),
        ('symbol', r'$/\!/\!/$  insufficient sample (<5 events in bin)'),
        ('symbol', r'gray  bin with no drought events'),
    ]
    y_cursor = 0.96
    line_pitch = {'header': 0.085, 'sub': 0.080, 'item': 0.115,
                  'symbol': 0.090}
    for kind, text in crit_lines:
        if kind == 'header':
            ax_criteria.text(0.02, y_cursor, text,
                             ha='left', va='top',
                             fontsize=FONTSIZE_LABEL, color='#222222',
                             transform=ax_criteria.transAxes)
        elif kind == 'sub':
            ax_criteria.text(0.02, y_cursor, text,
                             ha='left', va='top',
                             fontsize=FONTSIZE_SMALL, color='#333333',
                             transform=ax_criteria.transAxes)
        elif kind == 'item':
            ax_criteria.text(0.02, y_cursor, text,
                             ha='left', va='top',
                             fontsize=FONTSIZE_SMALL, color='#222222',
                             transform=ax_criteria.transAxes)
        elif kind == 'symbol':
            ax_criteria.text(0.02, y_cursor, text,
                             ha='left', va='top',
                             fontsize=FONTSIZE_SMALL, color='#333333',
                             transform=ax_criteria.transAxes,
                             linespacing=1.25)
        y_cursor -= line_pitch[kind]

    # -- scenario labels between col 1 and col 2 ---------------------------
    # Place each row's scenario label vertically in the gap between the
    # right heatmap (axes_pct) and the right-column summary panel.
    fig.canvas.draw()
    bb_pct_top = axes_pct[0].get_position()
    bb_bar = ax_bar.get_position()
    scenario_label_x = (bb_pct_top.x1 + bb_bar.x0) / 2
    for row_idx, did in enumerate(DATASETS):
        bb = axes_pct[row_idx].get_position()
        y_center = bb.y0 + bb.height / 2
        fig.text(scenario_label_x, y_center, DATASET_LABELS[did],
                 rotation=270, ha='center', va='center',
                 fontsize=FONTSIZE_LABEL)

    # Column geometry, reused below for colorbar placement.
    bb_top_left = axes_rp[0].get_position()
    bb_top_right = axes_pct[0].get_position()

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
         'Return Period, joint exceedance (years)'),
        (cb_pct_abs, cbar_ax_pct_abs,
         'Droughts reaching DE\n(% of events at this severity & magnitude)'),
    ]
    # Label above each bar, tick labels below — applied consistently to all
    # four colorbars (top absolute and bottom Δ).
    for cb, cax, label in cbar_labels_top:
        cax.xaxis.set_ticks_position('bottom')
        cax.xaxis.set_label_position('top')
        cb.set_label(label, fontsize=FONTSIZE_SMALL, linespacing=1.35)
        cb.ax.tick_params(labelsize=10)
        cb.outline.set_edgecolor(AXIS_FRAME_COLOR)
        cb.outline.set_linewidth(0.8)

    # -- bottom colorbars: Δ scales, positioned below row 3's x-axis label --
    # Sits below GS_BOT with room above for panel x-tick labels, panel
    # x-axis label, and the cbar's own 2-line top label.
    cbar_bot_h = 0.010
    cbar_bot_y = 0.10
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
         'Δ Return Period, joint exceedance (years)'
         '\nbrown = more frequent than baseline'),
        (cb_pct_rel, cbar_ax_pct_rel,
         r'$\Delta$ Droughts reaching DE (pp)'
         '\nbrown = more emergencies than baseline'),
    ]
    for cb, cax, label in cbar_labels_bot:
        cax.xaxis.set_ticks_position('bottom')
        cax.xaxis.set_label_position('top')
        cb.set_label(label, fontsize=FONTSIZE_SMALL, linespacing=1.35)
        cb.ax.tick_params(labelsize=FONTSIZE_SMALL)
        cb.outline.set_edgecolor(AXIS_FRAME_COLOR)
        cb.outline.set_linewidth(0.8)

    # Focal-region criteria, drought-years bar chart, and the symbol legend
    # all live in column 2 of the GridSpec — see the bar/criteria block above.

    cbar_label_artists = [
        cb.ax.xaxis.label for cb in
        (cb_rp_abs, cb_pct_abs, cb_rp_rel, cb_pct_rel)
    ]
    _audit_text_overlaps(fig, extra_artists=cbar_label_artists)

    # -- save ----------------------------------------------------------------
    fname_base = (f"{FIG_OUTPUT_DIR}/Fig9_satisficing_heatmap_ssi{ssi_window}"
                  f"_rp{FOCAL_RP_THRESH_YEARS}_frac{FOCAL_FRAC_THRESH:.2f}"
                  f"_sto{FOCAL_WORST_STORAGE_THRESH:.0f}")
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
