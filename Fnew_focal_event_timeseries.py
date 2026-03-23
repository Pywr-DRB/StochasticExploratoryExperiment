"""
Fnew: Focal Drought Event Timeseries

Plots detailed timeseries dynamics for individual drought events selected
by 07_select_focal_events.py. Each event gets a multi-panel figure showing:

  Panel 1: NYC aggregate storage (% of capacity)
  Panel 2: NYC releases to Montague (MGD)
  Panel 3: Montague flow (MGD) and % of flow from NYC releases
  Panel 4: Shortage periods (binary indicator for Montague)

The x-axis spans full annual cycles: June 1 before the drought start
through May 31 after the drought ends.

Usage:
    python Fnew_focal_event_timeseries.py [ssi_window]
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from methods.config import (
    ROOT_DIR, FIG_DIR,
    NYC_RESERVOIRS, NYC_TOTAL_CAPACITY,
    DEFAULT_SHORTAGE_TOLERANCE_MGD,
)
from methods.load import load_rank_subset_from_export, load_ffmp_boundaries
from methods.plotting.styles import (
    DATASET_COLORS, DATASET_LABELS,
    DPI_HIGH, apply_publication_style,
)

FIG_OUTPUT_DIR = os.path.join(FIG_DIR, 'Fnew_focal_event_timeseries')
FOCAL_EVENTS_DIR = os.path.join(ROOT_DIR, 'pywrdrb', 'focal_events')


# ── Data loading ──────────────────────────────────────────────────────

def load_focal_events(ssi_window):
    """Load the focal events CSV produced by 07_select_focal_events.py."""
    fname = os.path.join(FOCAL_EVENTS_DIR, f'focal_events_ssi{ssi_window}.csv')
    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Focal events file not found: {fname}\n"
            f"Run 07_select_focal_events.py first!"
        )
    df = pd.read_csv(fname)
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    return df


def get_plot_window(start, end):
    """
    Compute the plotting window: June 1 before drought start through
    May 31 after drought end, ensuring full annual cycles.
    """
    # June 1 on or before drought start
    if start.month >= 6:
        plot_start = pd.Timestamp(year=start.year, month=6, day=1)
    else:
        plot_start = pd.Timestamp(year=start.year - 1, month=6, day=1)

    # May 31 on or after drought end
    if end.month <= 5:
        plot_end = pd.Timestamp(year=end.year, month=5, day=31)
    else:
        plot_end = pd.Timestamp(year=end.year + 1, month=5, day=31)

    return plot_start, plot_end


def load_realization_data(dataset_id, realization_id):
    """
    Load timeseries data for a single realization from the postprocessed
    HDF5 file using the existing load_rank_subset_from_export function.
    """
    fname = os.path.join(
        ROOT_DIR, 'pywrdrb', 'outputs',
        f'{dataset_id}_with_postprocessing.hdf5'
    )
    results_sets = [
        'res_storage', 'major_flow', 'mrf_target',
        'contribution', 'shortage',
    ]
    data = load_rank_subset_from_export(
        fname, [realization_id], results_sets, rank=0, size=1
    )
    return data


def extract_timeseries(data, dataset_id, realization_id, plot_start, plot_end):
    """
    Extract and slice all timeseries needed for the focal event plot.

    Returns dict with keys:
        storage_pct, nyc_release, montague_flow, montague_target,
        nyc_pct_of_montague, montague_shortage_binary
    """
    # NYC aggregate storage as %
    storage_raw = data.res_storage[dataset_id][realization_id][NYC_RESERVOIRS].sum(axis=1)
    storage_pct = 100.0 * storage_raw / NYC_TOTAL_CAPACITY

    # NYC releases to Montague (contribution)
    contribution = data.contribution[dataset_id][realization_id]
    if isinstance(contribution, pd.DataFrame):
        nyc_release = contribution['mrf_montagueTrenton_nyc']
    else:
        nyc_release = contribution

    # Montague flow and target
    montague_flow = data.major_flow[dataset_id][realization_id]['delMontague']
    montague_target = data.mrf_target[dataset_id][realization_id]['delMontague']

    # NYC contribution as % of Montague flow
    nyc_pct = np.where(
        montague_flow > 0,
        100.0 * nyc_release / montague_flow,
        0.0
    )
    nyc_pct_series = pd.Series(nyc_pct, index=nyc_release.index)

    # Montague shortage magnitude (MGD)
    montague_shortage = data.shortage[dataset_id][realization_id]['delMontague']

    # Slice to plot window
    ts = {}
    for name, series in [
        ('storage_pct', storage_pct),
        ('nyc_release', nyc_release),
        ('montague_flow', montague_flow),
        ('montague_target', montague_target),
        ('nyc_pct_of_montague', nyc_pct_series),
        ('montague_shortage', montague_shortage),
    ]:
        mask = (series.index >= plot_start) & (series.index <= plot_end)
        ts[name] = series[mask]

    return ts


# ── Plotting ──────────────────────────────────────────────────────────

def plot_focal_event(event_row, ts, plot_start, plot_end, ffmp_boundaries,
                     fname):
    """
    Plot a 4-panel focal event timeseries figure.

    Parameters
    ----------
    event_row : pd.Series
        Row from the focal events CSV.
    ts : dict
        Timeseries dict from extract_timeseries.
    plot_start, plot_end : pd.Timestamp
        Plotting window boundaries.
    ffmp_boundaries : pd.DataFrame
        FFMP storage zone boundaries (percentage scale).
    fname : str
        Output filename.
    """
    apply_publication_style()

    drought_start = pd.Timestamp(event_row['start'])
    drought_end = pd.Timestamp(event_row['end'])
    dataset_id = event_row['dataset_id']
    realization_id = int(event_row['realization_id'])
    ds_color = DATASET_COLORS.get(dataset_id, '#333333')
    ds_label = DATASET_LABELS.get(dataset_id, dataset_id)

    fig = plt.figure(figsize=(14, 11))
    gs = gridspec.GridSpec(
        4, 1,
        height_ratios=[1.2, 1, 1.2, 0.5],
        hspace=0.15,
        left=0.09, right=0.95, top=0.93, bottom=0.07,
    )

    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    panel_labels = ['(a)', '(b)', '(c)', '(d)']

    # ── Drought shading helper ────────────────────────────────────────
    def shade_drought(ax):
        ax.axvspan(drought_start, drought_end,
                   color='#FFE0B2', alpha=0.5, zorder=0,
                   label='SSI drought period')

    # ── Panel 1: NYC aggregate storage (%) ────────────────────────────
    ax = axes[0]
    shade_drought(ax)
    ax.plot(ts['storage_pct'].index, ts['storage_pct'].values,
            color=ds_color, linewidth=1.5, zorder=3)

    # FFMP zone boundaries (dynamic by day of year)
    if ffmp_boundaries is not None:
        plot_dates = ts['storage_pct'].index
        zone_colors = {'level3': '#f9a825', 'level4': '#ef6c00', 'level5': '#d32f2f'}
        zone_labels = {'level3': 'Watch', 'level4': 'Warning', 'level5': 'Emergency'}
        for level in ['level3', 'level4', 'level5']:
            if level in ffmp_boundaries.columns:
                boundary_vals = []
                for d in plot_dates:
                    doy = d.dayofyear
                    if doy > 365:
                        doy = 365
                    if doy in ffmp_boundaries.index.dayofyear.values:
                        mask = ffmp_boundaries.index.dayofyear == doy
                        boundary_vals.append(ffmp_boundaries.loc[mask, level].median())
                    else:
                        boundary_vals.append(np.nan)
                ax.plot(plot_dates, boundary_vals,
                        color=zone_colors[level], linewidth=0.8,
                        linestyle='--', alpha=0.7, label=zone_labels[level])

    ax.set_ylabel('NYC Storage (%)', fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(loc='lower left', fontsize=8, ncol=4, framealpha=0.9)

    # ── Panel 2: NYC releases to Montague ─────────────────────────────
    ax = axes[1]
    shade_drought(ax)
    ax.plot(ts['nyc_release'].index, ts['nyc_release'].values,
            color='#2E7D32', linewidth=1.2, zorder=3)
    ax.set_ylabel('NYC Release to\nMontague (MGD)', fontsize=11)
    ax.set_ylim(bottom=0)

    # ── Panel 3: Montague flow and NYC fraction ───────────────────────
    ax = axes[2]
    shade_drought(ax)

    # Montague flow
    ax.plot(ts['montague_flow'].index, ts['montague_flow'].values,
            color='#1565C0', linewidth=1.2, label='Montague flow', zorder=3)
    # Montague target
    ax.plot(ts['montague_target'].index, ts['montague_target'].values,
            color='#D32F2F', linewidth=1.0, linestyle='--', alpha=0.8,
            label='MRF target', zorder=3)
    ax.set_ylabel('Montague Flow (MGD)', fontsize=11)
    ax.set_yscale('log')
    ax.set_ylim(bottom=100)

    # Secondary axis for NYC contribution %
    ax2 = ax.twinx()
    ax2.fill_between(ts['nyc_pct_of_montague'].index,
                     0, ts['nyc_pct_of_montague'].values,
                     color='#2E7D32', alpha=0.15, zorder=1,
                     label='NYC % of Montague')
    ax2.set_ylabel('NYC % of\nMontague Flow', fontsize=11, color='#2E7D32')
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='y', colors='#2E7D32')

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              loc='upper right', fontsize=8, ncol=3, framealpha=0.9)

    # ── Panel 4: Shortage magnitude (MGD) ───────────────────────────
    ax = axes[3]
    shade_drought(ax)
    ax.fill_between(ts['montague_shortage'].index,
                    0, ts['montague_shortage'].values,
                    color='#D32F2F', alpha=0.7, step='mid', zorder=3)
    ax.set_ylabel('Montague\nShortage (MGD)', fontsize=11)
    ax.set_ylim(bottom=0)

    # ── Shared x-axis formatting ──────────────────────────────────────
    for i, ax in enumerate(axes):
        ax.set_xlim(plot_start, plot_end)
        ax.grid(axis='both', alpha=0.2, linestyle='--')
        ax.set_axisbelow(True)
        ax.text(0.01, 0.92, panel_labels[i], transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top')

        if i < len(axes) - 1:
            ax.set_xticklabels([])
        else:
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[6, 9, 12, 3]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
            ax.tick_params(axis='x', labelsize=9)

    # ── Title ─────────────────────────────────────────────────────────
    mag = event_row.get('magnitude', np.nan)
    sev = event_row.get('severity', np.nan)
    sel_label = event_row.get('selection_label', '')
    outcome_metric = event_row.get('outcome_metric', 'max_consec_montague_days')

    # Format the outcome metric value with appropriate units
    outcome_val = event_row.get(outcome_metric, np.nan)
    metric_display = {
        'max_consec_montague_days': f'Max Consec. Shortage={outcome_val:.0f}d',
        'total_montague_shortage_mg': f'Total Shortage={outcome_val:.0f} MG',
    }.get(outcome_metric, f'{outcome_metric}={outcome_val:.1f}')

    fig.suptitle(
        f'{ds_label}  |  Realization {realization_id}  |  '
        f'Drought: {drought_start.strftime("%Y-%m-%d")} to '
        f'{drought_end.strftime("%Y-%m-%d")}\n'
        f'Severity={sev:.2f}  Magnitude={mag:.1f}  '
        f'{metric_display}  [{sel_label}]',
        fontsize=12, fontweight='bold', y=0.98,
    )

    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    ssi_window = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"Fnew: Focal Drought Event Timeseries (SSI-{ssi_window})")

    os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)
    focal_events = load_focal_events(ssi_window)
    print(f"  Loaded {len(focal_events)} focal events")

    # Load FFMP boundaries once
    ffmp = load_ffmp_boundaries()

    # Group by (dataset_id, realization_id) to minimize HDF5 reloads
    focal_events['_group'] = (
        focal_events['dataset_id'] + '_' +
        focal_events['realization_id'].astype(str)
    )
    grouped = focal_events.groupby('_group')

    for group_key, group_df in grouped:
        dataset_id = group_df.iloc[0]['dataset_id']
        realization_id = int(group_df.iloc[0]['realization_id'])

        print(f"\n  Loading data: {dataset_id} R{realization_id:04d}...")
        data = load_realization_data(dataset_id, realization_id)

        for _, event_row in group_df.iterrows():
            drought_start = pd.Timestamp(event_row['start'])
            drought_end = pd.Timestamp(event_row['end'])
            plot_start, plot_end = get_plot_window(drought_start, drought_end)

            print(f"    Plotting: {drought_start.date()} to {drought_end.date()} "
                  f"(window: {plot_start.date()} to {plot_end.date()}) "
                  f"[{event_row.get('selection_label', '')}]")

            ts = extract_timeseries(
                data, dataset_id, realization_id, plot_start, plot_end
            )

            sel_label = str(event_row.get('selection_label', 'event')).replace('=', '')
            outcome_metric = event_row.get('outcome_metric', 'unknown')
            # Short tag for filename
            metric_tag = {
                'max_consec_montague_days': 'consec',
                'total_montague_shortage_mg': 'totalMG',
            }.get(outcome_metric, outcome_metric[:10])
            fname = os.path.join(
                FIG_OUTPUT_DIR,
                f'focal_{dataset_id}_R{realization_id:04d}_'
                f'{drought_start.strftime("%Y%m%d")}_{sel_label}'
                f'_{metric_tag}_ssi{ssi_window}.png'
            )
            plot_focal_event(
                event_row, ts, plot_start, plot_end, ffmp, fname
            )

        del data

    print("\nDone.")


if __name__ == '__main__':
    main()
