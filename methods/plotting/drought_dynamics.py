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


# ── Main plotting function ──────────────────────────────────────────────

def plot_drought_dynamics_overlay(
    events,
    aligned_timeseries,
    reference_start,
    reference_end,
    smoothing_window=7,
    figsize=(14, 14),
    fname=None,
    alpha=0.7,
    ffmp_boundaries=None,
    all_event_data=None,
):
    """
    Create a multi-panel figure overlaying smoothed drought dynamics.

    Panel 0: Drought duration bars (horizontal lines showing each event's period).
    Panels 1-4: NYC inflow, NYC storage (with FFMP zones), NYC releases,
    Montague flow (log-scale).
    Panel 5 (optional): Severity vs magnitude scatter of all events, with
    selected events labeled numerically.

    Parameters
    ----------
    events : list of dict
        Each dict has: dataset_id, realization_id, start, end, and optionally
        severity, magnitude.
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
        Line transparency.
    ffmp_boundaries : pd.DataFrame or None
        From load_ffmp_boundaries(). If provided, FFMP zone boundaries are
        drawn on the NYC storage panel.
    all_event_data : dict or None
        ``{dataset_id: DataFrame}`` of all event metrics (from load_event_metrics).
        If provided, a severity vs magnitude scatter panel is appended at the bottom.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    apply_publication_style()

    n_events = len(events)
    show_scatter = (all_event_data is not None and
                    all(('severity' in ev and 'magnitude' in ev) for ev in events))

    # Compute shifted drought periods for the duration bar panel
    shifted_drought_periods = []
    for ev in events:
        ps, _pe = get_plot_window(ev['start'], ev['end'])
        delta = reference_start - ps
        shifted_start = pd.Timestamp(ev['start']) + delta
        shifted_end = pd.Timestamp(ev['end']) + delta
        shifted_drought_periods.append((shifted_start, shifted_end))

    # ── Figure layout ─────────────────────────────────────────────────
    n_rows = 6 if show_scatter else 5
    height_ratios = [0.4, 1, 1, 1, 1]
    if show_scatter:
        height_ratios.append(0.8)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        n_rows, 1,
        height_ratios=height_ratios,
        hspace=0.15,
        left=0.09, right=0.95, top=0.96, bottom=0.05,
    )
    axes = [fig.add_subplot(gs[i]) for i in range(n_rows)]

    # ── Panel 0: Drought duration bars ────────────────────────────────
    ax_bars = axes[0]
    for i, (ev, (d_start, d_end)) in enumerate(zip(events, shifted_drought_periods)):
        dataset_id = ev['dataset_id']
        color = DATASET_COLORS.get(dataset_id, '#808080')
        event_label = f'({i + 1})'
        y_pos = n_events - i  # stack top-to-bottom

        ax_bars.plot(
            [d_start, d_end], [y_pos, y_pos],
            color=color, linewidth=4, solid_capstyle='butt', zorder=3,
        )
        # Label at the left end of the bar
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

    # Track which datasets have been added to legend
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
            series = ts_dict[key]

            if smoothing_window > 1:
                series = series.rolling(
                    smoothing_window, center=True, min_periods=1
                ).mean()

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

        if key == 'montague_flow':
            ax.set_yscale('log')
            ax.set_ylim(bottom=100)

        if i < len(ts_axes) - 1:
            ax.set_xticklabels([])
        else:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax.tick_params(axis='x', labelsize=9)

    # Legend on first timeseries panel (dataset colors)
    ts_axes[0].legend(loc='upper right', fontsize=9, framealpha=0.9)

    # ── Optional scatter panel: severity vs magnitude ─────────────────
    if show_scatter:
        ax_scat = axes[-1]

        # Background: all events per dataset
        for dataset_id, df in all_event_data.items():
            color = DATASET_COLORS.get(dataset_id, '#808080')
            ax_scat.scatter(
                df['severity'], df['magnitude'],
                c=color, s=12, alpha=0.25, edgecolors='none', zorder=2,
            )

        # Highlight selected events with numbered labels
        for ev_idx, ev in enumerate(events):
            color = DATASET_COLORS.get(ev['dataset_id'], '#808080')
            sev = ev['severity']
            mag = ev['magnitude']
            ax_scat.scatter(
                sev, mag, c=color, s=80, edgecolors='black',
                linewidths=1.0, zorder=4,
            )
            ax_scat.annotate(
                f'({ev_idx + 1})',
                xy=(sev, mag), fontsize=8, fontweight='bold',
                color='black', va='bottom', ha='center',
                xytext=(0, 5), textcoords='offset points', zorder=5,
            )

        ax_scat.set_xlabel('Drought Severity (peak SSI deviation)', fontsize=11)
        ax_scat.set_ylabel('Drought Magnitude\n(cumul. SSI deficit)', fontsize=11)
        ax_scat.grid(True, alpha=0.2, linestyle='--')
        ax_scat.set_axisbelow(True)
        ax_scat.text(0.01, 0.92, '(e)', transform=ax_scat.transAxes,
                     fontsize=12, va='top')

    if fname:
        fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
        print(f"Saved: {fname}")

    return fig
