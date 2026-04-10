"""
Drought Dynamics Overlay Plot

Multi-panel figure overlaying smoothed timeseries from multiple drought events
on a year-agnostic axis. Panels: drought duration bars, NYC inflow, NYC storage
(with FFMP zones), NYC releases, Montague flow (log-scale).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

from methods.config import NYC_RESERVOIRS, NYC_TOTAL_CAPACITY
from methods.plotting.styles import (
    DATASET_COLORS, DATASET_LABELS, FFMP_ZONE_COLORS,
    DPI_HIGH, apply_publication_style,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def get_plot_window(start, end):
    """
    Compute plotting window: June 1 before drought start through
    May 31 after drought end, ensuring full annual cycles.
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    if start.month >= 6:
        plot_start = pd.Timestamp(year=start.year, month=6, day=1)
    else:
        plot_start = pd.Timestamp(year=start.year - 1, month=6, day=1)

    if end.month <= 5:
        plot_end = pd.Timestamp(year=end.year, month=5, day=31)
    else:
        plot_end = pd.Timestamp(year=end.year + 1, month=5, day=31)

    return plot_start, plot_end


def extract_drought_timeseries(data, dataset_id, realization_id,
                                plot_start, plot_end):
    """
    Extract the 4 timeseries needed for the drought dynamics overlay.

    Returns
    -------
    dict with keys: nyc_inflow, nyc_storage_pct, nyc_release, montague_flow
        Each value is a pd.Series sliced to [plot_start, plot_end].
    """
    # NYC aggregate inflow (MGD)
    nyc_inflow = data.inflow[dataset_id][realization_id][NYC_RESERVOIRS].sum(axis=1)

    # NYC aggregate storage as % of capacity
    storage_raw = data.res_storage[dataset_id][realization_id][NYC_RESERVOIRS].sum(axis=1)
    nyc_storage_pct = 100.0 * storage_raw / NYC_TOTAL_CAPACITY

    # NYC releases to Montague
    contribution = data.contribution[dataset_id][realization_id]
    if isinstance(contribution, pd.DataFrame):
        nyc_release = contribution['mrf_montagueTrenton_nyc']
    else:
        nyc_release = contribution

    # Montague flow
    montague_flow = data.major_flow[dataset_id][realization_id]['delMontague']

    # Slice to plot window
    ts = {}
    for name, series in [
        ('nyc_inflow', nyc_inflow),
        ('nyc_storage_pct', nyc_storage_pct),
        ('nyc_release', nyc_release),
        ('montague_flow', montague_flow),
    ]:
        mask = (series.index >= plot_start) & (series.index <= plot_end)
        ts[name] = series[mask]

    return ts


def align_to_reference(timeseries_dict, event_plot_start, reference_start):
    """
    Shift all timeseries so that event_plot_start maps to reference_start.

    Parameters
    ----------
    timeseries_dict : dict of pd.Series
    event_plot_start : pd.Timestamp
    reference_start : pd.Timestamp

    Returns
    -------
    dict of pd.Series with shifted DatetimeIndex
    """
    delta = reference_start - event_plot_start
    shifted = {}
    for key, series in timeseries_dict.items():
        s = series.copy()
        s.index = s.index + delta
        shifted[key] = s
    return shifted


def align_to_water_year(timeseries_dict, event_start, event_end,
                        min_storage_date, reference_wy_start=None):
    """
    Shift timeseries so that the water year containing the minimum-storage
    date maps to a reference water year.  Months are preserved exactly.

    The shift is always a whole number of years so that calendar months
    remain unchanged — only the year changes.

    Parameters
    ----------
    timeseries_dict : dict of pd.Series
    event_start, event_end : pd.Timestamp
        Actual drought start/end dates.
    min_storage_date : pd.Timestamp
        Date of minimum storage during the event (determines which water
        year the drought is anchored to).
    reference_wy_start : pd.Timestamp, optional
        June 1 of the reference water year.  Defaults to 2000-06-01.

    Returns
    -------
    shifted : dict of pd.Series with shifted DatetimeIndex
    shifted_start : pd.Timestamp
    shifted_end : pd.Timestamp
    """
    if reference_wy_start is None:
        reference_wy_start = pd.Timestamp('2000-06-01')

    # Determine the water year of min_storage_date (Jun–May)
    min_d = pd.Timestamp(min_storage_date)
    if min_d.month >= 6:
        event_wy_start = pd.Timestamp(year=min_d.year, month=6, day=1)
    else:
        event_wy_start = pd.Timestamp(year=min_d.year - 1, month=6, day=1)

    # Shift by whole years only so months are preserved exactly
    delta = reference_wy_start - event_wy_start

    shifted = {}
    for key, series in timeseries_dict.items():
        s = series.copy()
        s.index = s.index + delta
        shifted[key] = s
    return shifted, pd.Timestamp(event_start) + delta, pd.Timestamp(event_end) + delta


def compute_reference_window(events, reference_start=None):
    """
    Compute the reference start/end from all events' plot windows.

    Finds the longest event span and creates a reference window of that length.

    Parameters
    ----------
    events : list of dict
        Each must have 'start' and 'end' keys (drought dates).
    reference_start : pd.Timestamp, optional
        Defaults to 2000-06-01.

    Returns
    -------
    reference_start, reference_end : pd.Timestamp
    """
    if reference_start is None:
        reference_start = pd.Timestamp('2000-06-01')

    max_days = 0
    for ev in events:
        ps, pe = get_plot_window(ev['start'], ev['end'])
        span = (pe - ps).days
        if span > max_days:
            max_days = span

    reference_end = reference_start + pd.Timedelta(days=max_days)
    return reference_start, reference_end


def compute_reference_window_from_shifted(events, pad_months=1):
    """
    Compute reference x-axis range from events that already have
    'shifted_start' and 'shifted_end' keys, padded to month boundaries.

    Parameters
    ----------
    events : list of dict
        Each must have 'shifted_start' and 'shifted_end' keys.
    pad_months : int
        Months of padding before/after the earliest/latest shifted dates.

    Returns
    -------
    reference_start, reference_end : pd.Timestamp
    """
    earliest = min(ev['shifted_start'] for ev in events)
    latest = max(ev['shifted_end'] for ev in events)

    # Pad to 1st of month, minus pad_months
    ref_start = pd.Timestamp(year=earliest.year, month=earliest.month, day=1)
    ref_start -= pd.DateOffset(months=pad_months)

    # Pad to end of month, plus pad_months
    ref_end = pd.Timestamp(year=latest.year, month=latest.month, day=28)
    ref_end += pd.DateOffset(months=pad_months + 1)
    ref_end = pd.Timestamp(year=ref_end.year, month=ref_end.month, day=1) - pd.Timedelta(days=1)

    return ref_start, ref_end


def compute_fixed_extraction_window(min_storage_date, pad_before_wy=1, pad_after_wy=1):
    """Compute a fixed timeseries extraction window centered on the water year
    of minimum storage.

    The window spans from *pad_before_wy* water years before the min-storage
    water year through *pad_after_wy* water years after it.

    Parameters
    ----------
    min_storage_date : pd.Timestamp
    pad_before_wy, pad_after_wy : int

    Returns
    -------
    window_start, window_end : pd.Timestamp
    """
    min_d = pd.Timestamp(min_storage_date)
    # Water year containing min_storage_date (Jun–May)
    if min_d.month >= 6:
        wy_start_year = min_d.year
    else:
        wy_start_year = min_d.year - 1

    window_start = pd.Timestamp(year=wy_start_year - pad_before_wy, month=6, day=1)
    window_end = pd.Timestamp(year=wy_start_year + 1 + pad_after_wy, month=5, day=31)
    return window_start, window_end


def compute_fixed_reference_window(reference_wy_start=None,
                                   pad_before_wy=1, pad_after_wy=1):
    """Compute the fixed reference window for aligned timeseries.

    Since all events are aligned so the min-storage water year maps to the
    reference water year, and all use the same padding, the window is
    deterministic.

    Parameters
    ----------
    reference_wy_start : pd.Timestamp, optional
        June 1 of the reference water year. Defaults to 2000-06-01.
    pad_before_wy, pad_after_wy : int

    Returns
    -------
    reference_start, reference_end : pd.Timestamp
    """
    if reference_wy_start is None:
        reference_wy_start = pd.Timestamp('2000-06-01')

    ref_start = reference_wy_start - pd.DateOffset(years=pad_before_wy)
    ref_end = pd.Timestamp(
        year=reference_wy_start.year + 1 + pad_after_wy,
        month=5, day=31,
    )
    return ref_start, ref_end


def _build_ffmp_doy_lookup(ffmp_boundaries):
    """
    Build a day-of-year lookup for FFMP zone boundaries (%).

    Parameters
    ----------
    ffmp_boundaries : pd.DataFrame
        From load_ffmp_boundaries(). Has DatetimeIndex and columns
        like L3 (Watch), L4 (Warning), L5 (Emergency).

    Returns
    -------
    pd.DataFrame
        Indexed by day-of-year (1-366), columns: Watch, Warning, Emergency.
    """
    fb = ffmp_boundaries.copy()
    fb['doy'] = fb.index.dayofyear

    # Map column names — boundaries may use L3/L4/L5 or level3/level4/level5
    col_map = {}
    for candidate, label in [('L3', 'Watch'), ('level3', 'Watch'),
                              ('L4', 'Warning'), ('level4', 'Warning'),
                              ('L5', 'Emergency'), ('level5', 'Emergency')]:
        if candidate in fb.columns:
            col_map[candidate] = label

    if not col_map:
        return None

    fb = fb.rename(columns=col_map)
    zone_cols = [v for v in ['Watch', 'Warning', 'Emergency'] if v in fb.columns]
    return fb.groupby('doy')[zone_cols].median()


def _map_ffmp_to_reference_dates(ffmp_doy_lookup, reference_start, reference_end):
    """
    Map FFMP DOY boundaries onto the reference date range.

    Returns a DataFrame indexed by reference dates with zone boundary columns.
    """
    dates = pd.date_range(reference_start, reference_end, freq='D')
    doys = dates.dayofyear

    result = pd.DataFrame(index=dates)
    for col in ffmp_doy_lookup.columns:
        vals = []
        for doy in doys:
            if doy in ffmp_doy_lookup.index:
                vals.append(ffmp_doy_lookup.loc[doy, col])
            else:
                vals.append(np.nan)
        result[col] = vals

    return result


# ── Envelope helpers ─────────────────────────────────────────────────

def _smooth(series, window):
    """Apply centered rolling mean if window > 1."""
    if window > 1:
        return series.rolling(window, center=True, min_periods=1).mean()
    return series


def _compute_dataset_envelopes(events, aligned_timeseries, smoothing_window,
                                reference_start, reference_end):
    """Compute per-dataset min/max/median envelopes across all events.

    Returns
    -------
    dict : {dataset_id: {var_key: DataFrame with columns 'min','max','median'}}
    """
    from collections import defaultdict

    daily_index = pd.date_range(reference_start, reference_end, freq='D')
    var_keys = ['nyc_inflow', 'nyc_storage_pct', 'nyc_release', 'montague_flow']

    # Collect smoothed series per dataset per variable
    dataset_series = defaultdict(lambda: defaultdict(list))
    for ev, ts_dict in zip(events, aligned_timeseries):
        did = ev['dataset_id']
        for key in var_keys:
            s = _smooth(ts_dict[key], smoothing_window)
            # Reindex to common daily grid (NaN where event has no data)
            s = s.reindex(daily_index)
            dataset_series[did][key].append(s)

    # Compute envelopes
    envelopes = {}
    for did, var_dict in dataset_series.items():
        envelopes[did] = {}
        for key, series_list in var_dict.items():
            combined = pd.concat(series_list, axis=1)
            envelopes[did][key] = pd.DataFrame({
                'min': combined.min(axis=1),
                'max': combined.max(axis=1),
                'median': combined.median(axis=1),
            }, index=daily_index)

    return envelopes


# ── Main plotting function ──────────────────────────────────────────────

def plot_drought_dynamics_overlay(
    events,
    aligned_timeseries,
    reference_start,
    reference_end,
    smoothing_window=7,
    figsize=(14, 12),
    fname=None,
    alpha=0.7,
    ffmp_boundaries=None,
    envelope_mode=False,
    highlight_indices=None,
    show_median=True,
):
    """
    Create a 5-panel figure overlaying smoothed drought dynamics.

    Panel 0: Drought duration bars (horizontal lines showing each event's period).
    Panels 1-4: NYC inflow, NYC storage (with FFMP zones), NYC releases,
    Montague flow (log-scale).

    Parameters
    ----------
    events : list of dict
        Each dict has: dataset_id, realization_id, start, end.
    aligned_timeseries : list of dict
        Each dict has keys: nyc_inflow, nyc_storage_pct, nyc_release, montague_flow.
        All with DatetimeIndex shifted to the reference year.
    reference_start, reference_end : pd.Timestamp
        X-axis limits in reference-year space.
    smoothing_window : int
        Rolling mean window in days.
    figsize : tuple
    fname : str or None
        If provided, save figure to this path.
    alpha : float
        Line transparency (individual mode) or envelope fill alpha (envelope mode).
    ffmp_boundaries : pd.DataFrame or None
        From load_ffmp_boundaries(). If provided, FFMP zone boundaries are
        drawn on the NYC storage panel.
    envelope_mode : bool
        If True, show per-dataset min/max envelopes and overlay only the
        highlighted event(s) as solid lines.  If False (default), plot all
        events as individual lines.
    highlight_indices : list of int or None
        Event indices to draw as solid lines in envelope mode.  Ignored when
        envelope_mode is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    apply_publication_style()

    n_events = len(events)
    if highlight_indices is None:
        highlight_indices = []

    # Get shifted drought periods for the duration bar panel.
    # Use pre-computed shifted_start/shifted_end if available (midpoint alignment),
    # otherwise fall back to plot-window-based alignment.
    shifted_drought_periods = []
    for ev in events:
        if 'shifted_start' in ev and 'shifted_end' in ev:
            shifted_drought_periods.append(
                (pd.Timestamp(ev['shifted_start']),
                 pd.Timestamp(ev['shifted_end']))
            )
        else:
            ps, _pe = get_plot_window(ev['start'], ev['end'])
            delta = reference_start - ps
            shifted_drought_periods.append(
                (pd.Timestamp(ev['start']) + delta,
                 pd.Timestamp(ev['end']) + delta)
            )

    # ── Figure layout ─────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        5, 1,
        height_ratios=[0.4, 1, 1, 1, 1],
        hspace=0.15,
        left=0.09, right=0.95, top=0.96, bottom=0.05,
    )
    axes = [fig.add_subplot(gs[i]) for i in range(5)]

    # ── Panel 0: Drought duration bars ────────────────────────────────
    ax_bars = axes[0]
    if envelope_mode:
        # In envelope mode, show individual event bars stacked by dataset,
        # sorted by start month (water-year order: Jun=0 … May=11).
        # Highlight the worst-case event (solid) and the median-duration
        # event (dashed).
        from collections import defaultdict

        # Group events by dataset, sorted by start month (June-first order)
        dataset_events = defaultdict(list)
        for idx, (ev, (ds, de)) in enumerate(
            zip(events, shifted_drought_periods)
        ):
            dur = (de - ds).days
            dataset_events[ev['dataset_id']].append((idx, ds, de, dur))

        def _water_year_month_then_duration(item):
            """Sort key: month offset from June, then duration (shortest first).

            This produces a cascading bar chart: events grouped by start month
            (Jun=0, Jul=1, … May=11), with shorter events before longer ones
            within each month.
            """
            m = item[1].month  # shifted start month
            dur = item[3]       # duration in days
            return ((m - 6) % 12, dur)

        for did in dataset_events:
            dataset_events[did].sort(key=_water_year_month_then_duration)

        # Stack datasets vertically; within each dataset stack events
        unique_datasets = list(dataset_events.keys())
        y = 0.0
        y_spacing = 0.35  # spacing between bars within a dataset
        dataset_gap = 0.6  # gap between datasets

        for ds_idx, did in enumerate(unique_datasets):
            ev_list = dataset_events[did]
            color = DATASET_COLORS.get(did, '#808080')
            n_in_cell = len(ev_list)

            # Find median-duration event index (middle of sorted list)
            median_pos = n_in_cell // 2
            median_event_idx = ev_list[median_pos][0]

            if ds_idx > 0:
                y += dataset_gap

            for rank, (idx, ds, de, dur) in enumerate(ev_list):
                y += y_spacing
                is_highlight = idx in highlight_indices
                is_median = (idx == median_event_idx)

                if is_highlight:
                    ax_bars.plot(
                        [ds, de], [y, y],
                        color=color, linewidth=4, solid_capstyle='butt',
                        alpha=1.0, zorder=4,
                    )
                elif is_median:
                    ax_bars.plot(
                        [ds, de], [y, y],
                        color=color, linewidth=2.5, solid_capstyle='butt',
                        alpha=0.8, linestyle='--', zorder=3,
                    )
                else:
                    ax_bars.plot(
                        [ds, de], [y, y],
                        color=color, linewidth=2, solid_capstyle='butt',
                        alpha=0.25, zorder=2,
                    )

            # Dataset label
            y_mid = y - (n_in_cell - 1) * y_spacing / 2
            label = DATASET_LABELS.get(did, did)
            ax_bars.text(
                reference_start, y_mid,
                f' {label} (n={n_in_cell}) ',
                va='center', ha='right', fontsize=7, color=color,
                clip_on=False,
            )

        ax_bars.set_ylim(0, y + y_spacing)
    else:
        for i, (ev, (d_start, d_end)) in enumerate(zip(events, shifted_drought_periods)):
            dataset_id = ev['dataset_id']
            color = DATASET_COLORS.get(dataset_id, '#808080')
            event_label = f'({i + 1})'
            y_pos = n_events - i

            ax_bars.plot(
                [d_start, d_end], [y_pos, y_pos],
                color=color, linewidth=4, solid_capstyle='butt', zorder=3,
            )
            ax_bars.text(
                d_start, y_pos, f' {event_label}',
                va='center', ha='right', fontsize=8, color=color, fontweight='bold',
            )

        ax_bars.set_ylim(0.3, n_events + 0.7)

    ax_bars.set_xlim(reference_start, reference_end)
    ax_bars.set_yticks([])
    ax_bars.set_ylabel('Drought\nPeriods', fontsize=10)
    ax_bars.grid(axis='x', alpha=0.2, linestyle='--')
    ax_bars.set_axisbelow(True)
    ax_bars.set_xticklabels([])

    # ── FFMP DOY lookup ───────────────────────────────────────────────
    ffmp_ref = None
    if ffmp_boundaries is not None:
        ffmp_doy = _build_ffmp_doy_lookup(ffmp_boundaries)
        if ffmp_doy is not None:
            ffmp_ref = _map_ffmp_to_reference_dates(
                ffmp_doy, reference_start, reference_end
            )

    # ── Panels 1-4: Timeseries ────────────────────────────────────────
    panel_config = [
        ('nyc_inflow', 'NYC Inflow (MGD)'),
        ('nyc_storage_pct', 'NYC Storage (%)'),
        ('nyc_release', 'NYC Release (MGD)'),
        ('montague_flow', 'Montague Flow (MGD)'),
    ]
    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    ts_axes = axes[1:]

    if envelope_mode:
        # Compute envelopes
        envelopes = _compute_dataset_envelopes(
            events, aligned_timeseries, smoothing_window,
            reference_start, reference_end,
        )

        # Draw envelopes per dataset
        legend_datasets = set()
        for did, var_envelopes in envelopes.items():
            color = DATASET_COLORS.get(did, '#808080')
            label_base = DATASET_LABELS.get(did, did)
            show_label = did not in legend_datasets
            legend_datasets.add(did)

            for i, (key, _ylabel) in enumerate(panel_config):
                ax = ts_axes[i]
                env = var_envelopes[key]
                ax.fill_between(
                    env.index, env['min'], env['max'],
                    color=color, alpha=0.2, zorder=1,
                    label=(f'{label_base} range' if (show_label and i == 0)
                           else None),
                )
                if show_median:
                    ax.plot(
                        env.index, env['median'],
                        color=color, alpha=0.6, linewidth=1.2,
                        linestyle='--', zorder=2,
                        label=(f'{label_base} median'
                               if (show_label and i == 0) else None),
                    )

        # Draw highlighted event lines on top
        for hi in highlight_indices:
            ev = events[hi]
            ts_dict = aligned_timeseries[hi]
            color = DATASET_COLORS.get(ev['dataset_id'], '#808080')
            label_base = DATASET_LABELS.get(ev['dataset_id'], ev['dataset_id'])

            for i, (key, _ylabel) in enumerate(panel_config):
                ax = ts_axes[i]
                series = _smooth(ts_dict[key], smoothing_window)
                ax.plot(
                    series.index, series.values,
                    color=color, alpha=1.0, linewidth=2.0, zorder=4,
                    label=(f'{label_base} (worst)' if i == 0 else None),
                )
    else:
        # Individual line mode (original behavior)
        legend_datasets = set()

        for ev_idx, (event, ts_dict) in enumerate(zip(events, aligned_timeseries)):
            dataset_id = event['dataset_id']
            color = DATASET_COLORS.get(dataset_id, '#808080')
            label_base = DATASET_LABELS.get(dataset_id, dataset_id)
            event_num = f'({ev_idx + 1})'

            show_label = dataset_id not in legend_datasets
            legend_datasets.add(dataset_id)

            for i, (key, _ylabel) in enumerate(panel_config):
                ax = ts_axes[i]
                series = _smooth(ts_dict[key], smoothing_window)

                ax.plot(
                    series.index, series.values,
                    color=color, alpha=alpha, linewidth=1.5,
                    label=label_base if (show_label and i == 0) else None,
                    zorder=3,
                )

                # Add event number label at the end of each line (first panel only)
                if i == 0 and len(series) > 0:
                    last_valid = series.last_valid_index()
                    if last_valid is not None:
                        ax.annotate(
                            event_num,
                            xy=(last_valid, series[last_valid]),
                            fontsize=7, color=color, fontweight='bold',
                            va='center', ha='left',
                            xytext=(3, 0), textcoords='offset points',
                        )

    # ── Format timeseries panels ──────────────────────────────────────
    for i, (key, ylabel) in enumerate(panel_config):
        ax = ts_axes[i]
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlim(reference_start, reference_end)
        ax.grid(axis='both', alpha=0.2, linestyle='--')
        ax.set_axisbelow(True)
        ax.text(0.01, 0.92, panel_labels[i], transform=ax.transAxes,
                fontsize=12, va='top')

        if key == 'nyc_storage_pct':
            ax.set_ylim(0, 105)
            # Add FFMP zone boundaries
            if ffmp_ref is not None:
                zone_style = {
                    'Watch': {'color': FFMP_ZONE_COLORS['Watch'], 'label': 'Watch'},
                    'Warning': {'color': FFMP_ZONE_COLORS['Warning'], 'label': 'Warning'},
                    'Emergency': {'color': FFMP_ZONE_COLORS['Emergency'], 'label': 'Emergency'},
                }
                for zone_name, style in zone_style.items():
                    if zone_name in ffmp_ref.columns:
                        ax.plot(
                            ffmp_ref.index, ffmp_ref[zone_name],
                            color=style['color'], linewidth=1.0,
                            linestyle='--', alpha=0.8, label=style['label'],
                            zorder=2,
                        )
                ax.legend(loc='lower left', fontsize=7, ncol=3, framealpha=0.9)

        if key == 'nyc_inflow':
            ax.set_yscale('log')
            ax.set_ylim(bottom=1.0)

        if key == 'montague_flow':
            ax.set_yscale('log')
            ax.set_ylim(bottom=900)

        if i < len(ts_axes) - 1:
            ax.set_xticklabels([])
        else:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax.tick_params(axis='x', labelsize=9)

    # Legend on first timeseries panel (dataset colors)
    ts_axes[0].legend(loc='upper right', fontsize=9, framealpha=0.9)

    if fname:
        fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
        print(f"Saved: {fname}")

    return fig
