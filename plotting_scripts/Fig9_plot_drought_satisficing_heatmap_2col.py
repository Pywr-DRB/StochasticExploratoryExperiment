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
import pandas as pd
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
from methods.load import load_event_metrics, load_drought_events
from methods.return_period import compute_return_period_grid_exceedance as compute_return_period_grid
from methods.plotting.styles import (
    DATASET_LABELS, DATASET_LABELS_SHORT, DATASET_COLORS,
    DPI_HIGH, apply_publication_style,
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
# Reading order: a-c top heatmap row, d-f middle heatmap row,
# g-i bottom summary row (bar / criteria / legend).
PANEL_LETTERS = list('abcdefghi')

SHOW_CHANGE = False

# Override FOCAL_WORST_STORAGE_THRESH for ad-hoc sensitivity figures (e.g.
# regenerate the same plot with the × marker / criterion (iii) threshold
# at 25 % instead of the manuscript default 15 %). Set to None to use the
# value from methods.config; the output filename's `_stoNN` suffix tracks
# whichever threshold was used, so 15 % and 25 % versions don't overwrite.
STORAGE_THRESH_OVERRIDE = None

# Grey dividing lines that separate the absolute (col 0) from Δ (cols 1-2)
# heatmaps and the metric rows from each other. Toggle off if these will be
# added in postprocessing of the SVG.
SHOW_DIVIDERS = True
DIVIDER_COLOR = '#bfbfbf'
DIVIDER_LW = 1.0

# Date used to pick the 1960s drought-of-record from the observed-event
# table. The historic event active during this date is highlighted in red;
# its (severity, magnitude) is read from the data rather than hardcoded.
DOR_TARGET_DATE = pd.Timestamp('1964-12-01')

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

    # Resolve the worst-case-storage threshold once. Propagated to:
    # (i) identify_focal_region's `storage_thresh` argument so criterion
    #     (iii) of the focal-region selection uses the override, (ii) the
    # × marker draw loop, (iii) the criteria + legend text, and
    # (iv) the output filename suffix.
    storage_thresh = (FOCAL_WORST_STORAGE_THRESH
                      if STORAGE_THRESH_OVERRIDE is None
                      else float(STORAGE_THRESH_OVERRIDE))

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

    focal_cells = identify_focal_region(
        T_W_grids, frac_grids, min_grids, DATASETS,
        storage_thresh=storage_thresh)
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
    # Three palettes total. The two Δ-panel pairs share a single
    # brown-green diverging family — brown always marks the adverse side,
    # which lives on the *negative* axis for ΔRP (shorter return period =
    # more frequent droughts) and on the *positive* axis for Δ%DE (more
    # emergencies). The shared palette signals "these four panels show
    # the same kind of thing — change vs. baseline".
    #
    # Row 0 — Drought Return Period
    #   abs (a)        viridis_r — manuscript sequential
    #   Δ   (b, c)     BrBG — brown on negative side = more frequent
    #
    # Row 1 — Droughts reaching DE
    #   abs (d)        Reds — red intensity = more droughts hit DE
    #   Δ   (e, f)     BrBG_r — brown on positive side = more emergencies
    cmap_sequential = MANUSCRIPT_CMAPS['sequential']         # viridis_r
    cmap_diverging_pos_brown = MANUSCRIPT_CMAPS['diverging']  # BrBG_r
    cmap_diverging_neg_brown = cmap_diverging_pos_brown.reversed()  # BrBG
    cmap_pct_abs_base = plt.get_cmap('Reds')

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
    cmap_pct_de_abs = cmap_pct_abs_base.resampled(n_pct_bins)
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
    # Two GridSpecs share the figure so the gap between the heatmap region
    # and the summary footer is controlled explicitly (rather than emerging
    # from a uniform hspace). The bottom colorbars sit in that gap.
    #
    #   gs_heat  — 2 rows × 3 cols of heatmaps (top ~70% of figure)
    #     row 0   joint-exceedance return period (RP)
    #     row 1   % droughts reaching Drought Emergency (%DE)
    #   gs_summary — 1 row × 3 cols of summary panels (bottom ~15%)
    #     col 0   bar chart (drought-years per scenario)
    #     col 1   focal-region criteria text
    #     col 2   symbol legend
    #
    # Column meaning (heatmap rows):
    #   col 0   Stationary Baseline (absolute units)
    #   col 1   Δ vs. baseline, Wetter Winter / Drier Summer
    #   col 2   Δ vs. baseline, Wetter Winter / Similar Summer
    fig = plt.figure(figsize=(14.0, 13.5))
    GS_LEFT = 0.085
    GS_RIGHT = 0.97

    HEAT_TOP = 0.86
    HEAT_BOT = 0.32
    SUM_TOP = 0.18
    SUM_BOT = 0.04

    gs_heat = gridspec.GridSpec(
        2, 3,
        left=GS_LEFT, right=GS_RIGHT,
        top=HEAT_TOP, bottom=HEAT_BOT,
        wspace=0.10, hspace=0.35,
        width_ratios=[1.0, 1.0, 1.0],
    )
    # Match gs_heat's column geometry (left/right margins, wspace, and
    # width_ratios) so that g/h/i align vertically with a/d, b/e, c/f.
    gs_summary = gridspec.GridSpec(
        1, 3,
        left=GS_LEFT, right=GS_RIGHT,
        top=SUM_TOP, bottom=SUM_BOT,
        wspace=0.10,
        width_ratios=[1.0, 1.0, 1.0],
    )
    gs = gs_heat  # alias used by _add_heatmap below

    # axes_heat[metric_row][scenario_col]
    axes_heat = [[None, None, None], [None, None, None]]

    # Per-(metric, col) selection of grid + cmap + norm. Col 0 always uses the
    # absolute scale; cols 1-2 use the Δ scale. Hatching uses the *displayed*
    # dataset's count grid: stationary for col 0 (so 1-4-event bins of the
    # baseline are flagged on panel a/d) and the climate-adjusted dataset
    # for cols 1-2 (where insufficiency reflects the scenario, not baseline).
    rp_panel_specs = [
        ('stationary_ensemble',    rp_grids,      cmap_rp_abs,     norm_rp_abs),
        ('climate_adjusted_low',   rp_diff_grids, cmap_rp_rel,     norm_rp_rel),
        ('climate_adjusted_high',  rp_diff_grids, cmap_rp_rel,     norm_rp_rel),
    ]
    pct_panel_specs = [
        ('stationary_ensemble',    pct_de_grids,      cmap_pct_de_abs, norm_pct_de_abs),
        ('climate_adjusted_low',   pct_de_diff_grids, cmap_pct_de_rel, norm_pct_de_rel),
        ('climate_adjusted_high',  pct_de_diff_grids, cmap_pct_de_rel, norm_pct_de_rel),
    ]

    def _add_heatmap(metric_row, scenario_col, did, grid_dict, cmap, norm,
                     overlay_min_storage):
        ax = fig.add_subplot(gs[metric_row, scenario_col])
        ax.set_facecolor(EMPTY_CELL_COLOR)
        ax.pcolormesh(
            sev_edges, mag_edges, np.ma.masked_invalid(grid_dict[did].T),
            cmap=cmap, norm=norm, rasterized=True, zorder=3,
        )
        _draw_insufficient_hatch(ax, sev_edges, mag_edges, count_grids[did])
        draw_focal_boundary(ax, sev_edges, mag_edges, focal_cells)
        if overlay_min_storage:
            mg = min_grids[did]
            for i, sc in enumerate(sev_centers):
                for j, mc in enumerate(mag_centers):
                    if not np.isnan(mg[i, j]) and mg[i, j] < storage_thresh:
                        ax.scatter(
                            sc, mc, marker='x', s=34, linewidths=1.1,
                            color='#202020', alpha=0.85, zorder=7,
                        )
        axes_heat[metric_row][scenario_col] = ax
        return ax

    for col_idx, (did, gd, cm, nm) in enumerate(rp_panel_specs):
        _add_heatmap(0, col_idx, did, gd, cm, nm, overlay_min_storage=False)
    for col_idx, (did, gd, cm, nm) in enumerate(pct_panel_specs):
        _add_heatmap(1, col_idx, did, gd, cm, nm, overlay_min_storage=True)

    ax_bar = fig.add_subplot(gs_summary[0, 0])
    ax_criteria = fig.add_subplot(gs_summary[0, 1])
    ax_legend = fig.add_subplot(gs_summary[0, 2])

    # -- shared axis styling -------------------------------------------------
    all_heatmap_axes = [ax for row in axes_heat for ax in row]
    for ax in all_heatmap_axes:
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

    xaxis_label_text = 'Drought Severity (peak |SSI-3|)'
    yaxis_label_text = 'Drought Magnitude\n(|SSI-3| deficit-months)'

    # Subpanel descriptive titles (panel letter + metric/scenario phrase).
    # The column titles supply the scenario; the title here disambiguates
    # absolute vs. Δ within each row.
    panel_titles = [
        [  # row 0 — joint-exceedance return period
            '(a) Joint-exc. return period',
            '(b) Δ Return period vs. Stationary',
            '(c) Δ Return period vs. Stationary',
        ],
        [  # row 1 — % droughts reaching DE
            '(d) % Droughts reaching DE',
            '(e) Δ % reaching DE vs. Stationary',
            '(f) Δ % reaching DE vs. Stationary',
        ],
    ]

    for metric_row in range(2):
        for col_idx in range(3):
            ax = axes_heat[metric_row][col_idx]
            ax.set_yticklabels(['1', '10', '100'])
            # y-axis label only on col 0 (panels a, d).
            if col_idx == 0:
                ax.set_ylabel(yaxis_label_text,
                              fontsize=FONTSIZE_LABEL, labelpad=6)
            # x-axis label only on the bottom heatmap row (panels d, e, f).
            if metric_row == 1:
                ax.set_xlabel(xaxis_label_text,
                              fontsize=FONTSIZE_LABEL, labelpad=6)
            ax.set_title(panel_titles[metric_row][col_idx],
                         fontsize=FONTSIZE_SMALL, loc='left', pad=4)

    # Historic-drought markers on panel (a) only. All observed droughts
    # whose (severity, magnitude) bin falls inside the focal region get a
    # black triangle; the 1960s drought-of-record (the historic event
    # active at DOR_TARGET_DATE) is highlighted in red on top.
    obs_droughts = load_drought_events(
        baseline_id, ssi_window, observed=True)
    obs_in_focal_mask = _focal_region_event_mask(
        obs_droughts, sev_edges, mag_edges, focal_cells)

    obs_start = pd.to_datetime(obs_droughts['start'])
    obs_end = pd.to_datetime(obs_droughts['end'])
    is_1960s = (obs_start <= DOR_TARGET_DATE) & (obs_end >= DOR_TARGET_DATE)

    # Black triangles: any historic event inside focal region except the
    # 1960s drought-of-record.
    other_focal_obs = obs_droughts[obs_in_focal_mask & ~is_1960s.values]
    if len(other_focal_obs):
        axes_heat[0][0].scatter(
            other_focal_obs['severity'].values,
            other_focal_obs['magnitude'].values,
            marker='^', s=70, color='#000000', edgecolors='white',
            linewidths=0.8, zorder=10,
        )
        for _, row in other_focal_obs.iterrows():
            yr = pd.to_datetime(row['start']).year
            axes_heat[0][0].annotate(
                str(yr),
                xy=(row['severity'], row['magnitude']),
                xytext=(4, 4), textcoords='offset points',
                fontsize=FONTSIZE_SMALL - 2, color='#000000',
                ha='left', va='bottom', zorder=11,
            )

    # Red triangle for the 1960s drought-of-record (drawn last, on top).
    drought_1960s = obs_droughts[is_1960s.values]
    if len(drought_1960s):
        row_dor = drought_1960s.iloc[0]
        dor_sev = float(row_dor['severity'])
        dor_mag = float(row_dor['magnitude'])
        axes_heat[0][0].scatter(
            dor_sev, dor_mag, marker='^', s=100,
            color='red', edgecolors='white', linewidths=1.0,
            zorder=12,
        )
        axes_heat[0][0].annotate(
            '1960s drought\nof record',
            xy=(dor_sev, dor_mag), xytext=(3.3, 70),
            fontsize=FONTSIZE_SMALL, ha='left', va='center',
            arrowprops=dict(arrowstyle='-', color='#000000', lw=0.6),
            zorder=12,
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
    ax_bar.set_ylabel('Drought-years', fontsize=FONTSIZE_SMALL, labelpad=4)
    _style_axis_frame(ax_bar)
    for side in ('top', 'right'):
        ax_bar.spines[side].set_visible(False)
    ax_bar.tick_params(axis='y', labelsize=FONTSIZE_SMALL - 1)
    # Panel letter + description as a single in-panel label so the title
    # does not collide with the bottom colorbar above the summary row.
    ax_bar.text(0.02, 1.04, '(g) Drought-years in focal region',
                transform=ax_bar.transAxes,
                fontsize=FONTSIZE_LABEL, ha='left', va='bottom')

    # -- focal-region criteria text (bottom row, middle cell) -------------
    # Header + sub + 3 numbered items live here; the symbol legend goes in
    # the adjacent ax_legend cell so each summary panel is one self-contained
    # text block. Spines stay visible so the criteria block reads as a
    # boxed callout.
    ax_criteria.set_xticks([])
    ax_criteria.set_yticks([])
    for spine in ax_criteria.spines.values():
        spine.set_visible(True)
        spine.set_color('#000000')
        spine.set_linewidth(0.8)
    crit_lines = [
        ('header', '(h) Focal Region (white outline)'),
        ('sub',    'All three criteria must hold:'),
        ('item',   rf'(i)   Return Period (joint exc.) $\leq$ '
                   rf'{FOCAL_RP_THRESH_YEARS:,} yr'
                   '\n        in all 3 scenarios'),
        ('item',   r'(ii)  $\geq$5% of droughts reach DE'
                   '\n        in all 3 scenarios'),
        ('item',   rf'(iii) NYC storage <{int(round(storage_thresh))}% in $\geq$1 event'
                   '\n        in at least 1 of the 3 scenarios'),
    ]
    y_cursor = 0.96
    line_pitch = {'header': 0.10, 'sub': 0.10, 'item': 0.20,
                  'symbol': 0.20}
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
        y_cursor -= line_pitch[kind]

    # -- symbol legend (bottom row, right cell) ----------------------------
    ax_legend.set_axis_off()
    legend_lines = [
        ('header', '(i) Symbol legend'),
        ('symbol', r'$\times$  bin where $\geq$1 event drove combined NYC'
                   '\n     storage <' + f'{int(round(storage_thresh))}'
                   '% of capacity'),
        ('symbol', r'$/\!/\!/$  insufficient sample (<5 events in bin)'),
        ('symbol', r'gray  bin with no drought events'),
    ]
    y_cursor = 0.96
    for kind, text in legend_lines:
        if kind == 'header':
            ax_legend.text(0.02, y_cursor, text,
                           ha='left', va='top',
                           fontsize=FONTSIZE_LABEL, color='#222222',
                           transform=ax_legend.transAxes)
        elif kind == 'symbol':
            ax_legend.text(0.02, y_cursor, text,
                           ha='left', va='top',
                           fontsize=FONTSIZE_SMALL, color='#333333',
                           transform=ax_legend.transAxes,
                           linespacing=1.25)
        y_cursor -= line_pitch[kind]

    # -- layout sync: real (post-box_aspect) heatmap positions -------------
    # set_box_aspect(1.0) shrinks each heatmap inside its GridSpec column
    # so the panels are square. Axes.get_position() returns the original
    # GridSpec bbox (unaware of the shrink), so we query the renderer to
    # get the actual on-screen extent, then propagate that geometry to
    # column titles, colorbars, summary panels, and dividers — keeping
    # all rows the same visual width.
    fig.canvas.draw()
    _renderer = fig.canvas.get_renderer()

    def _rendered_bbox(ax):
        return ax.get_window_extent(_renderer).transformed(
            fig.transFigure.inverted())

    bb_row0_col0 = _rendered_bbox(axes_heat[0][0])
    bb_row0_col1 = _rendered_bbox(axes_heat[0][1])
    bb_row0_col2 = _rendered_bbox(axes_heat[0][2])
    bb_row1_col0 = _rendered_bbox(axes_heat[1][0])
    bb_row1_col1 = _rendered_bbox(axes_heat[1][1])
    bb_row1_col2 = _rendered_bbox(axes_heat[1][2])

    # Move summary panels (g/h/i) so they share x-extent with the heatmap
    # columns above. Their y-extent stays where gs_summary placed it.
    for col_idx, ax_sum in enumerate((ax_bar, ax_criteria, ax_legend)):
        target_x = bb_row0_col0.x0 if col_idx == 0 else (
            bb_row0_col1.x0 if col_idx == 1 else bb_row0_col2.x0)
        target_w = bb_row0_col0.width  # all heatmap cols are equal width
        bb_sum = ax_sum.get_position()
        ax_sum.set_position([target_x, bb_sum.y0, target_w, bb_sum.height])

    # -- column titles & row metric labels ---------------------------------
    # Column titles: scenario name above the top-row colorbars. Col 0 is
    # the absolute baseline; cols 1-2 are Δ vs. baseline (the "Δ" prefix
    # is added explicitly so a reader skimming column titles knows the
    # column displays a difference, not absolute values).
    column_titles = [
        DATASET_LABELS['stationary_ensemble'],
        f"Δ {DATASET_LABELS['climate_adjusted_low']}",
        f"Δ {DATASET_LABELS['climate_adjusted_high']}",
    ]
    col_title_y = 0.965
    col_bbs_top = [bb_row0_col0, bb_row0_col1, bb_row0_col2]
    for col_idx, title in enumerate(column_titles):
        bb = col_bbs_top[col_idx]
        x_center = bb.x0 + bb.width / 2
        fig.text(x_center, col_title_y, title,
                 ha='center', va='top',
                 fontsize=FONTSIZE_LABEL)

    # Row metric labels on the left, rotated 270, anchored to each metric
    # row's vertical centre. The figure-x position sits to the left of the
    # heatmap y-axis labels (i.e. left of col 0).
    row_metric_labels = [
        'Drought Return Period\n(joint exceedance, years)',
        'NYC Storage during droughts\n(% reaching Drought Emergency)',
    ]
    row_label_x = 0.012
    row_bbs_col0 = [bb_row0_col0, bb_row1_col0]
    for metric_row in range(2):
        bb = row_bbs_col0[metric_row]
        y_center = bb.y0 + bb.height / 2
        fig.text(row_label_x, y_center, row_metric_labels[metric_row],
                 rotation=90, ha='left', va='center',
                 fontsize=FONTSIZE_LABEL)

    # -- colorbar geometry: col 0 (abs) vs. cols 1-2 (Δ) -------------------

    rel_top_x0 = bb_row0_col1.x0
    rel_top_x1 = bb_row0_col2.x1
    rel_bot_x0 = bb_row1_col1.x0
    rel_bot_x1 = bb_row1_col2.x1

    cbar_h = 0.010
    cbar_top_y = bb_row0_col0.y1 + 0.045
    # Position the bottom colorbar in the gap between the heatmap GridSpec
    # bottom (HEAT_BOT) and the summary GridSpec top (SUM_TOP). Offset
    # downward from HEAT_BOT to leave room for row 1's x-axis label and
    # the colorbar's own top label.
    cbar_bot_y = HEAT_BOT - 0.085

    # -- top colorbars: above RP heatmap row -------------------------------
    cbar_ax_rp_abs = fig.add_axes(
        [bb_row0_col0.x0, cbar_top_y, bb_row0_col0.width, cbar_h])
    cbar_ax_rp_rel = fig.add_axes(
        [rel_top_x0, cbar_top_y, rel_top_x1 - rel_top_x0, cbar_h])

    cb_rp_abs = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_rp_abs, norm=norm_rp_abs),
        cax=cbar_ax_rp_abs, orientation='horizontal',
        extend='both', spacing='uniform',
    )
    cb_rp_abs.set_ticks(rp_abs_ticks)
    cb_rp_abs.set_ticklabels(rp_abs_tick_labels)

    cb_rp_rel = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_rp_rel, norm=norm_rp_rel),
        cax=cbar_ax_rp_rel, orientation='horizontal',
        extend='both', spacing='uniform',
    )
    cb_rp_rel.set_ticks(rp_diff_ticks)
    cb_rp_rel.set_ticklabels(rp_diff_tick_labels)

    # -- bottom colorbars: below %DE heatmap row ---------------------------
    cbar_ax_pct_abs = fig.add_axes(
        [bb_row1_col0.x0, cbar_bot_y, bb_row1_col0.width, cbar_h])
    cbar_ax_pct_rel = fig.add_axes(
        [rel_bot_x0, cbar_bot_y, rel_bot_x1 - rel_bot_x0, cbar_h])

    cb_pct_abs = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_pct_de_abs, norm=norm_pct_de_abs),
        cax=cbar_ax_pct_abs, orientation='horizontal',
        extend='max', spacing='uniform',
    )
    cb_pct_abs.set_ticks(pct_de_ticks)
    cb_pct_abs.set_ticklabels(pct_de_tick_labels)

    cb_pct_rel = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_pct_de_rel, norm=norm_pct_de_rel),
        cax=cbar_ax_pct_rel, orientation='horizontal',
        extend='both', spacing='uniform',
    )
    cb_pct_rel.set_ticks(pct_de_diff_ticks)
    cb_pct_rel.set_ticklabels(pct_de_diff_tick_labels)

    # 2-row colorbar labels. Metric name on top; quantitative detail on
    # the second line (units, direction). Tick labels below.
    cbar_labels = [
        (cb_rp_abs, cbar_ax_rp_abs,
         'Return Period, joint exceedance (years)'),
        (cb_rp_rel, cbar_ax_rp_rel,
         'Δ Return Period, joint exceedance (years)'
         '\nbrown = more frequent than baseline'),
        (cb_pct_abs, cbar_ax_pct_abs,
         'Droughts reaching DE\n(% of events at this severity & magnitude)'),
        (cb_pct_rel, cbar_ax_pct_rel,
         r'$\Delta$ Droughts reaching DE (pp)'
         '\nbrown = more emergencies than baseline'),
    ]
    for cb, cax, label in cbar_labels:
        cax.xaxis.set_ticks_position('bottom')
        cax.xaxis.set_label_position('top')
        cb.set_label(label, fontsize=FONTSIZE_SMALL, linespacing=1.35)
        cb.ax.tick_params(labelsize=FONTSIZE_SMALL)
        cb.outline.set_edgecolor(AXIS_FRAME_COLOR)
        cb.outline.set_linewidth(0.8)

    # -- optional grey dividing lines --------------------------------------
    # Three dividers: vertical between col 0 (absolute) and col 1 (Δ), and
    # horizontal between metric rows + between heatmap rows and the summary
    # row. Drawn in figure coords and gated by SHOW_DIVIDERS so the user
    # can switch them off and add cleaner versions to the SVG by hand.
    if SHOW_DIVIDERS:
        from matplotlib.lines import Line2D

        # Refresh bboxes after colorbars/text were placed; re-use the
        # renderer-based positions so dividers align with what's drawn.
        fig.canvas.draw()
        bb_row0_col0 = _rendered_bbox(axes_heat[0][0])
        bb_row0_col1 = _rendered_bbox(axes_heat[0][1])
        bb_row1_col0 = _rendered_bbox(axes_heat[1][0])
        bb_row1_col2 = _rendered_bbox(axes_heat[1][2])
        bb_summary = ax_bar.get_position()

        # Vertical: midway between col 0 right edge and col 1 left edge,
        # spanning from just above the top colorbar down to just below the
        # bottom colorbar so the divider visually brackets the entire
        # absolute-vs-Δ split for both metric rows.
        vx = (bb_row0_col0.x1 + bb_row0_col1.x0) / 2
        vy_top = cbar_top_y + cbar_h + 0.020
        vy_bot = cbar_bot_y - 0.020
        fig.add_artist(Line2D(
            [vx, vx], [vy_bot, vy_top],
            transform=fig.transFigure,
            color=DIVIDER_COLOR, linewidth=DIVIDER_LW, zorder=0.5,
        ))

        # Horizontal between metric rows: midway between row 0 bottom and
        # row 1 top, spanning the heatmap region.
        hx_left = bb_row0_col0.x0
        hx_right = bb_row1_col2.x1
        hy_between = (bb_row0_col0.y0 + bb_row1_col0.y1) / 2
        fig.add_artist(Line2D(
            [hx_left, hx_right], [hy_between, hy_between],
            transform=fig.transFigure,
            color=DIVIDER_COLOR, linewidth=DIVIDER_LW, zorder=0.5,
        ))

        # Horizontal between heatmaps + colorbars and summary row: just
        # above the bar/criteria/legend strip, spanning only the heatmap
        # region (matches the divider between metric rows).
        hy_summary = (cbar_bot_y - 0.020 + bb_summary.y1) / 2
        fig.add_artist(Line2D(
            [hx_left, hx_right], [hy_summary, hy_summary],
            transform=fig.transFigure,
            color=DIVIDER_COLOR, linewidth=DIVIDER_LW, zorder=0.5,
        ))

    cbar_label_artists = [
        cb.ax.xaxis.label for cb in
        (cb_rp_abs, cb_rp_rel, cb_pct_abs, cb_pct_rel)
    ]
    _audit_text_overlaps(fig, extra_artists=cbar_label_artists)

    # -- save ----------------------------------------------------------------
    fname_base = (f"{FIG_OUTPUT_DIR}/Fig9_satisficing_heatmap_ssi{ssi_window}"
                  f"_rp{FOCAL_RP_THRESH_YEARS}_frac{FOCAL_FRAC_THRESH:.2f}"
                  f"_sto{storage_thresh:.0f}")
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
