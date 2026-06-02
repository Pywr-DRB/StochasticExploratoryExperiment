"""
Fig10: Focal-region drought event landscape + two anchored droughts
(5 rows x 2 cols, panels a-j).

Row 1   (a) Grey 2-D drought-event support grid (stationary, focal-region
            outline, A/B markers)               | (b) Text key / event metadata
Row 2   (c) SSI-3 monthly for event A            | (d) SSI-3 monthly for event B
Row 3   (e) NYC storage % with FFMP dotted       | (f) same for event B
            zone thresholds for event A
Row 4   (g) NYC outflow — directed Montague      | (h) same for event B
            release (solid) vs NYC diversion
            (dashed), both ensemble-colored
Row 5   (i) NYC diversion shortage (% of demand) | (j) same for event B

Both columns share a fixed 3-year June-anchored x-axis. Multi-year
droughts (duration > 365 d) are flush-left with the window starting at
the June 1 on or before the drought start. Shorter droughts are
centered so the drought midpoint sits near the window's geometric
middle. Dynamics in each column are colored by the event's ensemble
(Stationary blue / WWDS orange); row 4 uses line style for release vs
diversion within a column.

Event slots are picked from the focal-region pool by the per-slot
`selection_method` ('rank_max' / 'rank_min' for extremum on a column,
'target_nearest' for z-scored (severity, min_storage) nearest to a
target tuple). Edit FOCAL_EVENTS at the top of the script to iterate
without code changes.

Usage:
    python Fig10_focal_drought_dynamics.py
"""

import sys
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.legend import Legend

from methods.config import (
    FIG_DIR, OUTPUT_DIR, N_YEARS,
    NYC_RESERVOIRS, NYC_TOTAL_CAPACITY,
    DATASET_CONFIGS,
    GRID_N_BINS,
    FOCAL_FRAC_THRESH, FOCAL_RP_THRESH_YEARS, FOCAL_WORST_STORAGE_THRESH,
)
from methods.load import (
    load_event_metrics, load_rank_subset_from_export, load_ffmp_boundaries,
)
from methods.return_period import (
    compute_return_period_grid_exceedance as compute_return_period_grid,
)
from methods.drought_analysis import fit_ssi_calculator
from methods.plotting.heatmap import (
    make_shared_edges_logmag, assign_grid_bins,
    compute_emergency_grid, compute_min_storage_grid,
    identify_focal_region, draw_focal_boundary,
    select_events_from_focal_region,
)
from methods.plotting.styles import (
    DATASET_COLORS, DATASET_LABELS_SHORT, DATASET_LABELS,
    FFMP_ZONE_COLORS,
    apply_publication_style, label_panel, save_fig,
    DPI_HIGH,
)
from methods.plotting.ensemble_summary import MGD_TO_MCM


# -- configuration -----------------------------------------------------------

SSI_WINDOW = 3
DATASETS = list(DATASET_CONFIGS.keys())
MIN_COUNT = 1  # presence threshold for the support grid (1 = any event)
SEV_TICKS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
AXIS_FRAME_COLOR = '#333333'
EMPTY_CELL_COLOR = '#ededed'
SUPPORT_FILL_COLOR = '#cccccc'

# The two anchored events. Each slot picks one in-focal event from the
# named ensemble via the chosen `selection_method`:
#   - 'target_nearest' (Euclidean nearest in z-scored (severity, min_storage)
#     to (target_severity, target_min_storage_pct))
#   - 'rank_max' / 'rank_min' (extremum of `rank_column` over the focal pool)
# Optional `window_days_before` / `window_days_after` override the module-
# level defaults for multi-year droughts that need a wider extraction span.
FOCAL_EVENTS = {
    'A': dict(
        label='A',
        title='Worst NYC-shortage focal drought (stationary)',
        selection_method='rank_max',
        rank_column='nyc_shortage_pct',
        ensemble='stationary_ensemble',
        marker='o',
    ),
    'B': dict(
        label='B',
        title='High-severity WWDS focal event',
        selection_method='target_nearest',
        target_severity=3.5,
        target_min_storage_pct=10.0,
        ensemble='climate_adjusted_low',
        marker='s',
    ),
}
EVENT_SLOT_ORDER = ['A', 'B']

# Length (in years) of the rows-2/3/4/5 x-axis window. All event slots
# share this length so the two columns are directly comparable. The
# anchor logic (`_compute_3yr_window`) decides which June 1 the window
# starts on so multi-year droughts are flush-left in the window and
# sub-yearly droughts are centered in the middle year.
WINDOW_YEARS = 3

# Rolling-mean window applied to NYC diversion + NYC release-to-Montague
# (row 4) and the daily shortage (row 5) before plotting. 30 days is
# coarse enough to suppress weekday/sub-monthly noise so the priority
# crossover and the multi-month shortage envelope read clearly.
DIVERSION_SMOOTHING_DAYS = 30

# Line-style scheme for row 4 (and any future panel) where two release
# types share an axis — color = ensemble (per panel column), style =
# release type. Convention matches Fig7_2x3_kdes_3mo_9mo_windows.py:
# Montague release = solid, NYC diversion = dashed.
NYC_RELEASE_LINE_STYLE = '-'
NYC_DIVERSION_LINE_STYLE = '--'

# FFMP rule-curve overlay (rows e, f) — drawn as faint dotted lines per
# zone threshold (Watch / Warning / Emergency), colored by zone via
# FFMP_ZONE_COLORS. The Normal zone has no upper line of its own (it is
# everything above the Watch threshold).
FFMP_LINE_STYLE = ':'
FFMP_LINE_ALPHA = 0.85
FFMP_LINE_WIDTH = 1.2

# Output destination.
FIG_OUTPUT_DIR = os.path.join(FIG_DIR, 'Fig10')
FIG_NAME_STEM = f'Fig10_focal_drought_dynamics_ssi{SSI_WINDOW}'

# Cache for the focal-region setup (grid edges, focal_cells, stationary
# count grid). Invariant to FOCAL_EVENTS — only invalidated when SSI_WINDOW
# or the focal thresholds change, or when CACHE_VERSION is bumped.
CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')
CACHE_VERSION = 'v1'
REBUILD_CACHE = False

# Font sizes (match Fig9's 12 / 14 / 16 trio).
FONTSIZE_SMALL = 12
FONTSIZE_LABEL = 14
FONTSIZE_TITLE = 16


# -- helpers -----------------------------------------------------------------

def _style_axis_frame(ax):
    for spine in ax.spines.values():
        spine.set_color(AXIS_FRAME_COLOR)
        spine.set_linewidth(0.8)
    ax.tick_params(color=AXIS_FRAME_COLOR, width=0.8)
    ax.grid(False)


def _mark_drought_period(ax, event):
    """Dashed verticals at the SSI-classified drought start and end."""
    for date_key in ('start', 'end'):
        ax.axvline(event[date_key], color='#666666',
                   linestyle='--', linewidth=0.9, alpha=0.7, zorder=2)


def _compute_3yr_window(event, total_years=WINDOW_YEARS):
    """A fixed `total_years`-year window anchored on June 1.

    - If the drought is longer than one year (`duration_days > 365`), the
      window starts at the June 1 on or before the drought start. The
      drought lead-up sits at the beginning of the window and the
      recovery occupies the back of the window.
    - If the drought is one year or shorter, the window starts at the
      June 1 that places the drought midpoint as close as possible to
      the window's geometric center — the drought lives in the middle
      of the 3-year span.

    Both events end up with the same window length so panels c-j across
    the two columns are directly comparable left-to-right.
    """
    start = pd.Timestamp(event['start'])
    end = pd.Timestamp(event['end'])
    duration_days = int(event['duration_days'])

    def june_floor(d):
        if d.month >= 6:
            return pd.Timestamp(year=d.year, month=6, day=1)
        return pd.Timestamp(year=d.year - 1, month=6, day=1)

    if duration_days > 365:
        w_start = june_floor(start)
    else:
        drought_center = start + (end - start) / 2
        # Target window-start so drought midpoint sits at window center.
        target = drought_center - pd.DateOffset(months=6 * total_years)
        floor = june_floor(target)
        ceil = pd.Timestamp(year=floor.year + 1, month=6, day=1)
        # Pick whichever June 1 puts the drought's midpoint nearest the
        # center of the resulting 3-year window.
        w_start = floor if (target - floor) <= (ceil - target) else ceil

    w_end = (pd.Timestamp(year=w_start.year + total_years, month=6, day=1)
             - pd.Timedelta(days=1))
    return w_start, w_end


def _select_anchored_event(all_data, sev_edges, mag_edges, focal_cells, cfg):
    """Pick one in-focal event from ``cfg['ensemble']`` per the slot's
    ``selection_method``. Supported methods:

    - ``'target_nearest'``: Euclidean nearest in z-scored
      (severity, event_min_storage_pct) to (cfg['target_severity'],
      cfg['target_min_storage_pct']).
    - ``'rank_max'``: event with the largest value of
      ``cfg['rank_column']`` over the focal pool.
    - ``'rank_min'``: event with the smallest value.
    """
    ds = cfg['ensemble']
    df_binned = assign_grid_bins(all_data[ds], sev_edges, mag_edges)
    pool = select_events_from_focal_region(
        df_binned, focal_cells,
        rank_col='event_min_storage_pct', ascending=True, n=None,
    )
    if pool.empty:
        raise RuntimeError(
            f"No focal-region events available in ensemble '{ds}' for slot "
            f"'{cfg['label']}'. Loosen focal thresholds or pick a different "
            f"ensemble in FOCAL_EVENTS."
        )

    method = cfg.get('selection_method', 'target_nearest')
    if method == 'target_nearest':
        sev = pool['severity'].values
        sto = pool['event_min_storage_pct'].values
        sev_mu, sev_sd = float(np.mean(sev)), float(np.std(sev) or 1.0)
        sto_mu, sto_sd = float(np.mean(sto)), float(np.std(sto) or 1.0)
        z_sev = (sev - sev_mu) / sev_sd
        z_sto = (sto - sto_mu) / sto_sd
        tgt_z_sev = (cfg['target_severity'] - sev_mu) / sev_sd
        tgt_z_sto = (cfg['target_min_storage_pct'] - sto_mu) / sto_sd
        dist = np.hypot(z_sev - tgt_z_sev, z_sto - tgt_z_sto)
        row = pool.iloc[int(np.argmin(dist))]
    elif method in ('rank_max', 'rank_min'):
        col = cfg['rank_column']
        if col not in pool.columns:
            raise KeyError(
                f"selection_method='{method}' for slot '{cfg['label']}' "
                f"requires rank_column='{col}' to be a column of the event "
                f"metrics CSV. Available: {sorted(pool.columns)}")
        ascending = (method == 'rank_min')
        row = pool.sort_values(col, ascending=ascending).iloc[0]
    else:
        raise ValueError(
            f"Unknown selection_method='{method}' for slot '{cfg['label']}'. "
            f"Use 'target_nearest', 'rank_max', or 'rank_min'.")

    return dict(
        ensemble=ds,
        realization_id=int(row['realization_id']),
        severity=float(row['severity']),
        magnitude=float(row['magnitude']),
        duration_days=int(row['duration_days']),
        event_min_storage_pct=float(row['event_min_storage_pct']),
        nyc_shortage_pct=float(row.get('nyc_shortage_pct', float('nan'))),
        min_storage_date=pd.Timestamp(row['min_storage_date']),
        start=pd.Timestamp(row['start']),
        end=pd.Timestamp(row['end']),
        config=cfg,
    )


def _load_event_window(event, results_sets, ssi_calc):
    """Load the daily NYC storage / diversion / Montague-release window for
    *event*, and compute the monthly SSI3 series over the same window.

    Returns
    -------
    dict
        ``{'daily': pd.DataFrame, 'ssi3': pd.Series}``
        - ``daily`` is the 2-yr daily window of:
            nyc_storage_pct, nyc_diversion (MCM/day),
            nyc_release_montague (MCM/day).
        - ``ssi3`` is the monthly SSI-3 series clipped to the same window,
          computed from this realization's NYC-aggregate monthly inflow
          standardized by the baseline-fit SSI calculator.
    """
    ds = event['ensemble']
    r = event['realization_id']
    fname = os.path.join(OUTPUT_DIR, f'{ds}_with_postprocessing.hdf5')
    data = load_rank_subset_from_export(
        fname, [r], results_sets, rank=0, size=1,
    )

    storage_raw = data.res_storage[ds][r][NYC_RESERVOIRS].sum(axis=1)
    nyc_storage_pct = 100.0 * storage_raw / NYC_TOTAL_CAPACITY

    nyc_diversion = (
        data.ibt_diversions[ds][r]['delivery_nyc'] * MGD_TO_MCM
    )
    nyc_demand = (
        data.ibt_demands[ds][r]['demand_nyc'] * MGD_TO_MCM
    )

    # NYC directed release to the Montague flow target. The contribution
    # results_set's column 'mrf_montagueTrenton_nyc' is the daily NYC
    # contribution to the Montague-Trenton MRF target (MGD).
    contribution = data.contribution[ds][r]
    if isinstance(contribution, pd.DataFrame):
        nyc_release_mont = contribution['mrf_montagueTrenton_nyc']
    else:
        nyc_release_mont = contribution
    nyc_release_mont = nyc_release_mont * MGD_TO_MCM

    # Daily NYC diversion shortage as a % of that day's demand. Where
    # demand is zero (shouldn't happen in practice but defend the divide),
    # the percent is reported as 0.
    nyc_shortage = (nyc_demand - nyc_diversion).clip(lower=0)
    nyc_shortage_pct_daily = (
        100.0 * nyc_shortage / nyc_demand.where(nyc_demand > 0, 1.0)
    ).where(nyc_demand > 0, 0.0)

    df_daily = pd.DataFrame({
        'nyc_storage_pct': nyc_storage_pct,
        'nyc_diversion': nyc_diversion,
        'nyc_demand': nyc_demand,
        'nyc_release_montague': nyc_release_mont,
        'nyc_shortage_pct': nyc_shortage_pct_daily,
    })

    # SSI-3 monthly series. NYC-aggregate daily inflow → monthly sum →
    # baseline-fit SSI calculator. Slice to the plotting window after
    # transform so the rolling sum has the full realization context.
    inflow_daily = data.inflow[ds][r][NYC_RESERVOIRS].sum(axis=1)
    inflow_monthly = (
        inflow_daily.resample('MS').sum().replace(0, np.nan).dropna()
    )
    ssi3_full = ssi_calc.transform(inflow_monthly)

    w_start, w_end = _compute_3yr_window(event)
    return dict(
        daily=df_daily.loc[w_start:w_end],
        ssi3=ssi3_full.loc[w_start:w_end],
    )


# -- panel builders ----------------------------------------------------------

def _panel_support_grid(ax, support_grid, sev_edges, mag_edges, focal_cells,
                        selected_events):
    """Panel (a): grey presence map + focal boundary + A/B markers."""
    ax.set_facecolor('white')
    display = np.where(support_grid > 0, 1.0, np.nan).T
    ax.pcolormesh(
        sev_edges, mag_edges,
        np.ma.masked_invalid(display),
        cmap=ListedColormap([SUPPORT_FILL_COLOR]),
        vmin=0.5, vmax=1.5,
        rasterized=True, zorder=3,
    )
    draw_focal_boundary(ax, sev_edges, mag_edges, focal_cells)

    for slot in EVENT_SLOT_ORDER:
        ev = selected_events[slot]
        cfg = ev['config']
        ens_color = DATASET_COLORS[ev['ensemble']]
        ax.scatter(
            ev['severity'], ev['magnitude'],
            marker=cfg['marker'], s=140,
            color=ens_color, edgecolor='white', linewidth=1.4,
            zorder=11,
        )
        ax.annotate(
            cfg['label'],
            xy=(ev['severity'], ev['magnitude']),
            xytext=(7, 5), textcoords='offset points',
            fontsize=FONTSIZE_LABEL, color=ens_color,
            ha='left', va='bottom', zorder=12,
        )

    ax.set_xlim(1.0, 4.5)
    ax.set_xticks(SEV_TICKS)
    ax.set_yscale('log')
    ax.set_ylim(1.0, 100.0)
    ax.set_yticks([1, 10, 100])
    ax.set_yticklabels(['1', '10', '100'])
    ax.set_yticks(
        [2, 3, 4, 5, 6, 7, 8, 9, 20, 30, 40, 50, 60, 70, 80, 90],
        minor=True,
    )
    ax.set_box_aspect(1.0)
    ax.set_xlabel('Drought Severity (peak |SSI-3|)', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Drought Magnitude\n(|SSI-3| deficit-months)',
                  fontsize=FONTSIZE_LABEL)
    _style_axis_frame(ax)


def _panel_key(ax, selected_events):
    """Panel (b): text legend / marker key for panel (a)."""
    ax.set_axis_off()
    label_panel(ax, 'b', fontsize=FONTSIZE_LABEL)
    lines = [
        ('item',   r'Grey cells: stationary-ensemble drought support'
                   '\n(at least one event in that severity-magnitude bin).'),
        ('item',   r'White outline: focal region (Fig 9 criteria;'
                   '\nall 3 ensembles, NYC storage stressed).'),
    ]
    y = 0.86
    pitch = {'header': 0.10, 'item': 0.16, 'symbol': 0.16}
    for kind, text in lines:
        ax.text(0.02, y, text, transform=ax.transAxes,
                ha='left', va='top',
                fontsize=(FONTSIZE_LABEL if kind == 'header' else FONTSIZE_SMALL),
                color='#222222', linespacing=1.3)
        y -= pitch[kind]

    for slot in EVENT_SLOT_ORDER:
        ev = selected_events[slot]
        cfg = ev['config']
        ens_color = DATASET_COLORS[ev['ensemble']]
        ax.plot([0.04], [y - 0.025], marker=cfg['marker'],
                color=ens_color, markeredgecolor='white',
                markeredgewidth=1.2, markersize=11,
                transform=ax.transAxes, clip_on=False)
        shortage_txt = (
            f"{ev['nyc_shortage_pct']:.1f}% of demand"
            if np.isfinite(ev.get('nyc_shortage_pct', float('nan')))
            else 'n/a'
        )
        ax.text(
            0.10, y,
            f"Event {cfg['label']} — {cfg['title']}\n"
            f"  ensemble: {DATASET_LABELS_SHORT.get(ev['ensemble'], ev['ensemble'])}\n"
            f"  severity = {ev['severity']:.2f}, "
            f"magnitude = {ev['magnitude']:.1f}, "
            f"min storage = {ev['event_min_storage_pct']:.1f}%\n"
            f"  duration = {ev['duration_days']} d, "
            f"NYC shortage = {shortage_txt}",
            transform=ax.transAxes, ha='left', va='top',
            fontsize=FONTSIZE_SMALL, color='#222222', linespacing=1.3,
        )
        y -= 0.26


def _panel_ssi3(ax, ssi3, daily_index, event):
    """Row 2 panel: SSI-3 monthly series across the event window, drawn in
    the event's ensemble color. The light fill is restricted to the
    SSI-classified drought period (event['start'] - event['end']) — not
    to every negative SSI excursion in the window."""
    ens_color = DATASET_COLORS[event['ensemble']]
    if ssi3 is not None and len(ssi3) > 0:
        ax.plot(ssi3.index, ssi3.values, color=ens_color,
                linewidth=1.8, zorder=4)
        in_drought = ((ssi3.index >= event['start']) &
                      (ssi3.index <= event['end']))
        ax.fill_between(
            ssi3.index, 0.0, ssi3.values,
            where=in_drought, interpolate=True,
            color=ens_color, alpha=0.18, linewidth=0, zorder=3,
        )
    ax.axhline(0.0, color='#666666', linewidth=0.8, zorder=2)
    _mark_drought_period(ax, event)
    # x-axis matches rows 3/4 (daily window extents) so columns align.
    ax.set_xlim(daily_index.min(), daily_index.max())
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.tick_params(axis='x', labelsize=FONTSIZE_SMALL)
    _style_axis_frame(ax)


def _build_ffmp_seasonal_lookup():
    """Seasonal (cal_doy → Watch / Warning / Emergency) FFMP threshold table.

    Returns a DataFrame indexed by calendar day-of-year (1..366) with
    columns Watch / Warning / Emergency (% of capacity), aggregated as
    the median across years from the operational rule-curve file. Used
    by ``_panel_storage`` to draw time-varying FFMP zone bands behind
    each event's storage trajectory. Returns ``None`` if the rule-curve
    file is missing or has no matchable columns.
    """
    try:
        fb = load_ffmp_boundaries().copy()
    except Exception:
        return None
    fb['cal_doy'] = fb.index.dayofyear
    col_map = {}
    for candidate, zone in [('L3', 'Watch'), ('level3', 'Watch'),
                             ('L4', 'Warning'), ('level4', 'Warning'),
                             ('L5', 'Emergency'), ('level5', 'Emergency')]:
        if candidate in fb.columns:
            col_map[candidate] = zone
    if not col_map:
        return None
    fb = fb.rename(columns=col_map)
    zone_cols = [z for z in ('Watch', 'Warning', 'Emergency')
                 if z in fb.columns]
    seasonal = fb.groupby('cal_doy')[zone_cols].median()
    # Reindex to full 1..366 with forward+back fill so leap-day gaps are
    # filled in without spurious NaNs at the band edges.
    seasonal = seasonal.reindex(np.arange(1, 367)).ffill().bfill()
    return seasonal


def _draw_ffmp_lines(ax, daily_index, ffmp_seasonal):
    """Draw faint dotted FFMP rule-curve lines behind a date-indexed storage
    panel — one line per zone threshold (Watch / Warning / Emergency),
    colored by zone. The threshold values vary by calendar day-of-year so
    the lines track the seasonal rule curve across the full window.
    """
    if ffmp_seasonal is None or len(daily_index) == 0:
        return
    needed = ('Watch', 'Warning', 'Emergency')
    if not all(c in ffmp_seasonal.columns for c in needed):
        return
    cal_doys = daily_index.dayofyear
    daily_thresh = ffmp_seasonal.reindex(cal_doys)
    daily_thresh.index = daily_index
    for zone in needed:
        ax.plot(
            daily_index, daily_thresh[zone].astype(float).values,
            color=FFMP_ZONE_COLORS[zone],
            linestyle=FFMP_LINE_STYLE,
            linewidth=FFMP_LINE_WIDTH,
            alpha=FFMP_LINE_ALPHA,
            zorder=2,
        )


def _panel_storage(ax, window, event, ffmp_seasonal=None):
    """Row 3 panel: NYC storage % with dotted FFMP zone thresholds.
    Line drawn in the event's ensemble color."""
    _draw_ffmp_lines(ax, window.index, ffmp_seasonal)
    ax.plot(window.index, window['nyc_storage_pct'],
            color=DATASET_COLORS[event['ensemble']], linewidth=1.8, zorder=4)
    _mark_drought_period(ax, event)
    ax.set_ylim(0, 100)
    ax.set_xlim(window.index.min(), window.index.max())
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.tick_params(axis='x', labelsize=FONTSIZE_SMALL)
    _style_axis_frame(ax)


def _panel_div_release(ax, window, event):
    """Row 4 panel: NYC diversion vs NYC directed release to Montague.

    Both lines drawn in the event's ensemble color; differentiated by
    line style (Fig7 convention) — Montague release is solid, NYC
    diversion is dashed. Where the lines cross, the relative priority
    between NYC's own diversion and the Montague-MRF contribution flips.
    """
    t = window.index
    ens_color = DATASET_COLORS[event['ensemble']]
    diversion = window['nyc_diversion'].rolling(
        DIVERSION_SMOOTHING_DAYS, center=True, min_periods=1).mean()
    release = window['nyc_release_montague'].rolling(
        DIVERSION_SMOOTHING_DAYS, center=True, min_periods=1).mean()
    ax.plot(t, release, color=ens_color,
            linestyle=NYC_RELEASE_LINE_STYLE,
            linewidth=1.6, zorder=4)
    ax.plot(t, diversion, color=ens_color,
            linestyle=NYC_DIVERSION_LINE_STYLE,
            linewidth=1.6, zorder=4)
    _mark_drought_period(ax, event)
    ax.set_ylim(bottom=0)
    ax.set_xlim(t.min(), t.max())
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.tick_params(axis='x', labelsize=FONTSIZE_SMALL)
    _style_axis_frame(ax)


def _panel_shortage(ax, window, event):
    """Row 5 panel: NYC diversion shortage as % of demand.

    Smoothing parity with _panel_div_release: numerator (shortage MCM) and
    denominator (demand MCM) are each 7-day-centered-mean smoothed
    independently, then divided. Smoothing the ratio directly amplifies
    noise when daily demand and diversion are both small; smoothing the
    flows first matches the visual smoothness of rows g/h.
    """
    t = window.index
    shortage_mcm = (
        window['nyc_demand'] - window['nyc_diversion']
    ).clip(lower=0)
    shortage_smooth = shortage_mcm.rolling(
        DIVERSION_SMOOTHING_DAYS, center=True, min_periods=1).mean()
    demand_smooth = window['nyc_demand'].rolling(
        DIVERSION_SMOOTHING_DAYS, center=True, min_periods=1).mean()
    shortage_pct = (
        100.0 * shortage_smooth / demand_smooth.where(demand_smooth > 0, 1.0)
    )
    ax.plot(t, shortage_pct,
            color=DATASET_COLORS[event['ensemble']],
            linewidth=1.6, zorder=4)
    _mark_drought_period(ax, event)
    ax.set_ylim(bottom=0)
    ax.set_xlim(t.min(), t.max())
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.tick_params(axis='x', labelsize=FONTSIZE_SMALL)
    _style_axis_frame(ax)


# -- focal-region setup cache ------------------------------------------------

def _focal_setup_cache_path():
    tag = (f"ssi{SSI_WINDOW}"
           f"_rp{FOCAL_RP_THRESH_YEARS}"
           f"_fr{FOCAL_FRAC_THRESH}"
           f"_st{FOCAL_WORST_STORAGE_THRESH}")
    return os.path.join(
        CACHE_DIR, f"fig10_focal_setup_{CACHE_VERSION}_{tag}.pkl")


def _compute_focal_setup(all_data):
    """Heavy step: shared grid edges, per-dataset return-period / emergency /
    min-storage grids, focal-cell identification, stationary count grid.

    All outputs depend only on the event_metrics CSVs and the focal
    thresholds — they are invariant to FOCAL_EVENTS, so they cache cleanly.
    """
    sev_edges, mag_edges, _, _ = make_shared_edges_logmag(
        all_data, DATASETS, n_bins=GRID_N_BINS,
    )

    T_W_grids, frac_grids, min_grids = {}, {}, {}
    for ds in DATASETS:
        _, _, T_W_grids[ds], _ = compute_return_period_grid(
            all_data[ds], sev_edges, mag_edges, N_YEARS, min_count=MIN_COUNT)
        frac_grids[ds], _ = compute_emergency_grid(
            all_data[ds], sev_edges, mag_edges, min_count=MIN_COUNT)
        min_grids[ds], _ = compute_min_storage_grid(
            all_data[ds], sev_edges, mag_edges, min_count=MIN_COUNT)

    focal_cells = identify_focal_region(
        T_W_grids, frac_grids, min_grids, DATASETS,
        rp_thresh_years=FOCAL_RP_THRESH_YEARS,
        frac_thresh=FOCAL_FRAC_THRESH,
        storage_thresh=FOCAL_WORST_STORAGE_THRESH,
    )

    _, count_grid_stationary = compute_min_storage_grid(
        all_data['stationary_ensemble'], sev_edges, mag_edges,
        min_count=MIN_COUNT,
    )

    return dict(
        sev_edges=sev_edges,
        mag_edges=mag_edges,
        focal_cells=focal_cells,
        count_grid_stationary=count_grid_stationary,
    )


def _load_or_compute_focal_setup(all_data):
    cache_path = _focal_setup_cache_path()

    if (not REBUILD_CACHE) and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            print(f"  Loaded focal-region setup from cache: {cache_path}")
            return cached
        except Exception as e:
            print(f"  Warning: cache load failed ({e}); recomputing.")

    print("  Computing focal-region setup (grids + focal cells)...")
    result = _compute_focal_setup(all_data)

    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Cached focal-region setup to {cache_path}")
    except Exception as e:
        print(f"  Warning: cache save failed ({e}).")

    return result


# -- main --------------------------------------------------------------------

def main():
    apply_publication_style()
    plt.rcParams.update({'font.size': 13, 'font.weight': 'normal'})
    os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

    print(f"Fig10: focal-region drought event landscape "
          f"(SSI-{SSI_WINDOW})")

    # 1. Event metrics per ensemble.
    all_data = {}
    for ds in DATASETS:
        all_data[ds] = load_event_metrics(ds, SSI_WINDOW)
        print(f"  {ds}: {len(all_data[ds])} events")

    # 2. Focal-region setup (grid edges, focal cells, stationary count grid)
    #    — cached because it is the dominant cost and is invariant to
    #    FOCAL_EVENTS configuration.
    setup = _load_or_compute_focal_setup(all_data)
    sev_edges = setup['sev_edges']
    mag_edges = setup['mag_edges']
    focal_cells = setup['focal_cells']
    count_grid_stationary = setup['count_grid_stationary']
    print(f"  Focal region: {len(focal_cells)} cells")
    if len(focal_cells) == 0:
        raise RuntimeError("Empty focal region; cannot anchor events.")

    selected_events = {}
    for slot in EVENT_SLOT_ORDER:
        cfg = FOCAL_EVENTS[slot]
        ev = _select_anchored_event(
            all_data, sev_edges, mag_edges, focal_cells, cfg,
        )
        selected_events[slot] = ev
        method = cfg.get('selection_method', 'target_nearest')
        if method == 'target_nearest':
            sel_detail = (f"target sev={cfg['target_severity']}, "
                          f"target min_storage={cfg['target_min_storage_pct']}%")
        else:
            sel_detail = f"{method} on {cfg['rank_column']}"
        print(
            f"  Event {cfg['label']} ({cfg['title']}): "
            f"{DATASET_LABELS_SHORT.get(ev['ensemble'], ev['ensemble'])} "
            f"R{ev['realization_id']:04d}, "
            f"sev={ev['severity']:.2f}, mag={ev['magnitude']:.1f}, "
            f"dur={ev['duration_days']}d, "
            f"min_storage={ev['event_min_storage_pct']:.1f}% on "
            f"{ev['min_storage_date'].date()}, "
            f"shortage={ev['nyc_shortage_pct']:.1f}%  "
            f"[{sel_detail}]"
        )

    # 4. Fit SSI-3 calculator on baseline historical inflow (used to
    #    standardize each event's NYC-aggregate monthly inflow into SSI-3).
    print("  Fitting SSI-3 calculator on baseline historical inflow...")
    ssi_calc = fit_ssi_calculator(SSI_WINDOW)

    # 4b. FFMP seasonal rule curves (Watch / Warning / Emergency by
    #     calendar day-of-year), drawn as faint bands behind panels e/f.
    ffmp_seasonal = _build_ffmp_seasonal_lookup()
    if ffmp_seasonal is None:
        print("  Warning: FFMP boundaries unavailable; storage panels "
              "will be drawn without rule-curve bands.")

    # 5. Load 2-yr HDF5 windows for the two anchored events. ibt_demands
    #    is kept in the load even though it is not currently plotted, so a
    #    later shortage-metric overlay (demand - diversion) can be added
    #    without changing the HDF5 read.
    results_sets = ['res_storage', 'ibt_diversions', 'ibt_demands',
                    'inflow', 'contribution']
    event_windows = {}
    for slot in EVENT_SLOT_ORDER:
        ev = selected_events[slot]
        print(f"  Loading HDF5 window for event {slot} "
              f"({ev['ensemble']} R{ev['realization_id']:04d})...")
        event_windows[slot] = _load_event_window(ev, results_sets, ssi_calc)

    # 6. Figure layout — two stacked GridSpecs so the gap between the
    #    landscape row (a/b) and the timeseries stack (c-j) stays generous,
    #    while the four timeseries rows themselves sit tight against each
    #    other. Rows 2-5 share x within each column so SSI / storage /
    #    diversion / shortage align vertically per event.
    fig = plt.figure(figsize=(13.8, 14.6))
    outer = gridspec.GridSpec(
        2, 1,
        # A touch more left margin so the rotated y-axis labels in the
        # timeseries stack don't get pinched by the figure edge.
        left=0.095, right=0.975, top=0.965, bottom=0.085,
        height_ratios=[1.0, 2.95],
        # Tighten the gap below row 1 (a/b -> c/d).
        hspace=0.10,
    )
    top_gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0], wspace=0.24,
    )
    bot_gs = gridspec.GridSpecFromSubplotSpec(
        4, 2, subplot_spec=outer[1],
        # c/d and g-j shrunk a further 20% off their prior 70% height;
        # e/f — the storage row, carrying FFMP curves — kept at full
        # narrative-anchor height.
        height_ratios=[0.476, 0.78, 0.437, 0.437],
        hspace=0.10, wspace=0.22,
    )

    ax_a = fig.add_subplot(top_gs[0, 0])
    ax_b = fig.add_subplot(top_gs[0, 1])
    ax_c = fig.add_subplot(bot_gs[0, 0])
    ax_d = fig.add_subplot(bot_gs[0, 1])
    ax_e = fig.add_subplot(bot_gs[1, 0], sharex=ax_c)
    ax_f = fig.add_subplot(bot_gs[1, 1], sharex=ax_d)
    ax_g = fig.add_subplot(bot_gs[2, 0], sharex=ax_c)
    ax_h = fig.add_subplot(bot_gs[2, 1], sharex=ax_d)
    ax_i = fig.add_subplot(bot_gs[3, 0], sharex=ax_c)
    ax_j = fig.add_subplot(bot_gs[3, 1], sharex=ax_d)

    # Panel (a)
    _panel_support_grid(ax_a, count_grid_stationary, sev_edges, mag_edges,
                        focal_cells, selected_events)
    label_panel(ax_a, 'a', fontsize=FONTSIZE_LABEL)

    # Panel (b)
    _panel_key(ax_b, selected_events)

    # Panels (c), (d): SSI-3 monthly timeseries per anchored event.
    _panel_ssi3(ax_c, event_windows['A']['ssi3'],
                event_windows['A']['daily'].index, selected_events['A'])
    _panel_ssi3(ax_d, event_windows['B']['ssi3'],
                event_windows['B']['daily'].index, selected_events['B'])
    # Timeseries y-labels are kept short + one line so the rotated text
    # fits within the shortened c/d, g/h, i/j panel heights and the
    # bottom of one label doesn't bleed into the panel below. FONTSIZE 11
    # is a touch smaller than the panel-letter font, which keeps the
    # column's letter labels visually dominant.
    YAX_LABEL_FONTSIZE = 11
    ax_c.set_ylabel('NYC inflow SSI-3',
                    fontsize=YAX_LABEL_FONTSIZE, labelpad=6)
    # Shared y on (c, d) so the depth/breadth of the SSI-3 trough is
    # directly comparable between the two events.
    ssi_concat = pd.concat([
        event_windows['A']['ssi3'].dropna(),
        event_windows['B']['ssi3'].dropna(),
    ])
    if len(ssi_concat):
        ssi_lo = float(min(ssi_concat.min(), -2.5)) - 0.2
        ssi_hi = float(max(ssi_concat.max(), 1.0)) + 0.2
    else:
        ssi_lo, ssi_hi = -3.0, 1.5
    ax_c.set_ylim(ssi_lo, ssi_hi)
    ax_d.set_ylim(ssi_lo, ssi_hi)
    label_panel(ax_c, 'c', fontsize=FONTSIZE_LABEL)
    label_panel(ax_d, 'd', fontsize=FONTSIZE_LABEL)

    # Panels (e), (f): NYC storage timeseries with FFMP rule-curve bands.
    _panel_storage(ax_e, event_windows['A']['daily'], selected_events['A'],
                   ffmp_seasonal=ffmp_seasonal)
    _panel_storage(ax_f, event_windows['B']['daily'], selected_events['B'],
                   ffmp_seasonal=ffmp_seasonal)
    ax_e.set_ylabel('NYC reservoir storage\n(% of total capacity)',
                    fontsize=YAX_LABEL_FONTSIZE, labelpad=6)
    label_panel(ax_e, 'e', fontsize=FONTSIZE_LABEL)
    label_panel(ax_f, 'f', fontsize=FONTSIZE_LABEL)

    # Panels (g), (h): NYC diversion vs NYC release to Montague — the
    # crossing point marks the priority shift.
    _panel_div_release(ax_g, event_windows['A']['daily'], selected_events['A'])
    _panel_div_release(ax_h, event_windows['B']['daily'], selected_events['B'])
    ax_g.set_ylabel('NYC outflow (MCM/day)',
                    fontsize=YAX_LABEL_FONTSIZE, labelpad=6)
    # Shared y across (g, h) using the max of either line over both events.
    y_max = max(
        float(event_windows['A']['daily']
              [['nyc_diversion', 'nyc_release_montague']].max().max()),
        float(event_windows['B']['daily']
              [['nyc_diversion', 'nyc_release_montague']].max().max()),
    ) * 1.08
    ax_g.set_ylim(0, y_max)
    ax_h.set_ylim(0, y_max)
    label_panel(ax_g, 'g', fontsize=FONTSIZE_LABEL)
    label_panel(ax_h, 'h', fontsize=FONTSIZE_LABEL)

    # Panels (i), (j): NYC diversion shortage (% of daily demand).
    _panel_shortage(ax_i, event_windows['A']['daily'], selected_events['A'])
    _panel_shortage(ax_j, event_windows['B']['daily'], selected_events['B'])
    ax_i.set_ylabel('Diversion shortage (%)',
                    fontsize=YAX_LABEL_FONTSIZE, labelpad=6)
    shortage_max = max(
        float(event_windows['A']['daily']['nyc_shortage_pct'].max()),
        float(event_windows['B']['daily']['nyc_shortage_pct'].max()),
    ) * 1.10 or 1.0
    ax_i.set_ylim(0, shortage_max)
    ax_j.set_ylim(0, shortage_max)
    label_panel(ax_i, 'i', fontsize=FONTSIZE_LABEL)
    label_panel(ax_j, 'j', fontsize=FONTSIZE_LABEL)

    # Hide x-tick labels on rows 2-4 of the timeseries stack; only row 5
    # (panels i, j) carries the month labels.
    for ax in (ax_c, ax_d, ax_e, ax_f, ax_g, ax_h):
        plt.setp(ax.get_xticklabels(), visible=False)

    # Align the four left-column y-axis labels at the same x so the
    # vertical stack reads as one cohesive timeseries column.
    fig.align_ylabels([ax_c, ax_e, ax_g, ax_i])

    # -- figure-level legend -------------------------------------------------
    # Three groups along the bottom:
    #   1) Ensemble identity + marker per anchored event (color = column).
    #   2) FFMP zone threshold lines (panels e, f).
    #   3) Row-4 line styles — solid = Montague release, dashed = diversion.
    ensemble_handles = [
        Line2D(
            [0], [0],
            marker=FOCAL_EVENTS[slot]['marker'], linestyle='-',
            color=DATASET_COLORS[selected_events[slot]['ensemble']],
            markeredgecolor='white', markeredgewidth=1.0,
            markersize=10, linewidth=2.0,
            label=(f"Event {FOCAL_EVENTS[slot]['label']} — "
                   f"{DATASET_LABELS[selected_events[slot]['ensemble']]}"),
        )
        for slot in EVENT_SLOT_ORDER
    ]
    ffmp_handles = [
        Line2D([0], [0], color=FFMP_ZONE_COLORS[z],
               linestyle=FFMP_LINE_STYLE, linewidth=FFMP_LINE_WIDTH,
               alpha=FFMP_LINE_ALPHA,
               label=f'FFMP {z} threshold')
        for z in ('Watch', 'Warning', 'Emergency')
    ] if ffmp_seasonal is not None else []
    row4_handles = [
        Line2D([0], [0], color='#444444',
               linestyle=NYC_RELEASE_LINE_STYLE, linewidth=1.6,
               label='NYC directed release to Montague'),
        Line2D([0], [0], color='#444444',
               linestyle=NYC_DIVERSION_LINE_STYLE, linewidth=1.6,
               label='NYC diversion (delivered)'),
    ]

    leg_ensembles = Legend(
        fig, ensemble_handles, [h.get_label() for h in ensemble_handles],
        loc='upper left', bbox_to_anchor=(0.085, 0.065),
        bbox_transform=fig.transFigure,
        ncol=1, fontsize=FONTSIZE_SMALL, frameon=False,
        title='Anchored events / dynamics color (panels c-j)',
        title_fontsize=FONTSIZE_SMALL,
        handlelength=2.0, handletextpad=0.6,
        borderpad=0.2, labelspacing=0.3,
    )
    leg_ensembles._legend_box.align = 'left'
    fig.add_artist(leg_ensembles)

    if ffmp_handles:
        leg_ffmp = Legend(
            fig, ffmp_handles, [h.get_label() for h in ffmp_handles],
            loc='upper left', bbox_to_anchor=(0.40, 0.065),
            bbox_transform=fig.transFigure,
            ncol=1, fontsize=FONTSIZE_SMALL, frameon=False,
            title='FFMP storage zones (panels e, f)',
            title_fontsize=FONTSIZE_SMALL,
            handlelength=2.0, handletextpad=0.6,
            borderpad=0.2, labelspacing=0.3,
        )
        leg_ffmp._legend_box.align = 'left'
        fig.add_artist(leg_ffmp)

    leg_row4 = Legend(
        fig, row4_handles, [h.get_label() for h in row4_handles],
        loc='upper left', bbox_to_anchor=(0.72, 0.065),
        bbox_transform=fig.transFigure,
        ncol=1, fontsize=FONTSIZE_SMALL, frameon=False,
        title='Row 4 line styles (color = column ensemble)',
        title_fontsize=FONTSIZE_SMALL,
        handlelength=2.4, handletextpad=0.6,
        borderpad=0.2, labelspacing=0.3,
    )
    leg_row4._legend_box.align = 'left'
    fig.add_artist(leg_row4)

    # -- save ----------------------------------------------------------------
    out_stem = os.path.join(FIG_OUTPUT_DIR, FIG_NAME_STEM)
    save_fig(fig, out_stem, dpi=DPI_HIGH)
    plt.close(fig)
    print(f"Done. Saved: {out_stem}.{{png,svg,pdf}}")


if __name__ == '__main__':
    main()
