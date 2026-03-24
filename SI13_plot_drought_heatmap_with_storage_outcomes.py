"""
SI13: Drought Outcome Heatmaps with Storage Outcomes

Heatmap figures showing how drought outcomes (min storage, Montague shortage)
vary across the severity × magnitude space, with baseline absolute values
and climate scenario deltas.

  Row 1:   Baseline median and 5th-pctile (severity x magnitude grid)
  Row 2-3: Relative change for each climate scenario
  Grey hatch marks cells where baseline/scenario data don't align.

Usage:
    python SI13_plot_drought_heatmap_with_storage_outcomes.py [ssi_window]
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from scipy.stats import binned_statistic_2d
import warnings
warnings.filterwarnings("ignore")

from methods.config import ROOT_DIR, FIG_DIR
from methods.plotting.styles import (
    DATASET_COLORS, DATASET_LABELS,
    FONTSIZE_SMALL, FONTSIZE_MEDIUM,
    DPI_HIGH,
)

FIG_OUTPUT_DIR = f"{FIG_DIR}/SI10_drought_heatmap"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

MIN_DURATION = 30
MIN_SEVERITY = 1.0
MAX_SEVERITY = 6.2
N_GRID = 25
MIN_COUNT = 3
DATASETS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']


# ── helpers ───────────────────────────────────────────────────────────

def load_events(dataset_id, ssi_window):
    df = pd.read_csv(
        f'{ROOT_DIR}/pywrdrb/event_metrics/'
        f'{dataset_id}_ssi{ssi_window}_event_metrics.csv'
    )
    df = df[df['duration_days'] >= MIN_DURATION].copy()
    df['severity'] = df['severity'].abs()
    df['magnitude'] = df['magnitude'].abs()
    df = df[(df['severity'] >= MIN_SEVERITY) & (df['severity'] <= MAX_SEVERITY)]
    return df


def _pct5(x):
    return np.percentile(x, 5)


def _pct95(x):
    return np.percentile(x, 95)


def grid_stat(sev, mag, vals, sev_edges, mag_edges, stat_func):
    counts = binned_statistic_2d(
        sev, mag, vals, statistic='count', bins=[sev_edges, mag_edges],
    ).statistic
    stat = binned_statistic_2d(
        sev, mag, vals, statistic=stat_func, bins=[sev_edges, mag_edges],
    ).statistic
    stat[counts < MIN_COUNT] = np.nan
    return stat, counts


def _overlay_hatch(ax, sev_edges, mag_edges, misaligned):
    dx = sev_edges[1] - sev_edges[0]
    dy = mag_edges[1] - mag_edges[0]
    for i in range(len(sev_edges) - 1):
        for j in range(len(mag_edges) - 1):
            if misaligned[i, j]:
                ax.add_patch(Rectangle(
                    (sev_edges[i], mag_edges[j]), dx, dy,
                    facecolor='#d0d0d0', edgecolor='grey',
                    hatch='///', linewidth=0.3, alpha=0.5, zorder=5,
                ))


def _plot_heatmap(ax, sev_edges, mag_edges, stat, cmap, vmin, vmax, title):
    im = ax.pcolormesh(
        sev_edges, mag_edges, np.ma.masked_invalid(stat.T),
        cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True,
    )
    ax.set_facecolor('#f0f0f0')
    ax.set_xlim(sev_edges[0], sev_edges[-1])
    ax.set_ylim(mag_edges[0], mag_edges[-1])
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=FONTSIZE_SMALL)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return im


# ── heatmap figure ────────────────────────────────────────────────────

def plot_heatmap_figure(ssi_window, all_data, sev_edges, mag_edges,
                        metric_col, metric_label, stat_funcs,
                        base_cmap, base_vmin, base_vmax,
                        delta_cmap, cb_label, cb_delta_label,
                        fname):
    """Generic heatmap figure: baseline median/p5 + scenario deltas.

    Parameters
    ----------
    stat_funcs : dict with keys 'median' and 'p5' — callables or str
    base_cmap : colormap for absolute baseline panels
    base_vmin, base_vmax : color scale for baseline
    delta_cmap : colormap for delta panels
    """

    baseline = all_data['stationary_ensemble']

    # ── compute grid stats ────────────────────────────────────────────
    stats_all = {}
    for did in DATASETS:
        df = all_data[did]
        s, m, v = df['severity'].values, df['magnitude'].values, df[metric_col].values
        med, cnt = grid_stat(s, m, v, sev_edges, mag_edges, stat_funcs['median'])
        p5, _   = grid_stat(s, m, v, sev_edges, mag_edges, stat_funcs['p5'])
        stats_all[did] = {'median': med, 'p5': p5, 'count': cnt}

    base = stats_all['stationary_ensemble']
    base_has = base['count'] >= MIN_COUNT

    deltas, misalign = {}, {}
    for did in DATASETS[1:]:
        scen_has = stats_all[did]['count'] >= MIN_COUNT
        valid = base_has & scen_has
        mis = base_has ^ scen_has
        d_med = np.where(valid, stats_all[did]['median'] - base['median'], np.nan)
        d_p5  = np.where(valid, stats_all[did]['p5'] - base['p5'], np.nan)
        deltas[did] = (d_med, d_p5)
        misalign[did] = mis

    all_d = np.concatenate([d.ravel() for pair in deltas.values() for d in pair])
    delta_lim = min(np.nanquantile(np.abs(all_d), 0.98), base_vmax * 0.5)
    delta_lim = max(5, int(np.ceil(delta_lim / 5) * 5))

    # ── figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 18))
    gs = gridspec.GridSpec(
        5, 2,
        height_ratios=[1, 0.04, 1, 1, 0.04],
        hspace=0.22, wspace=0.28,
        top=0.96, bottom=0.06, left=0.08, right=0.92,
    )

    ax_bm = fig.add_subplot(gs[0, 0])
    ax_bp = fig.add_subplot(gs[0, 1])
    ax_cb1 = fig.add_subplot(gs[1, :])
    ax_lm = fig.add_subplot(gs[2, 0])
    ax_lp = fig.add_subplot(gs[2, 1])
    ax_hm = fig.add_subplot(gs[3, 0])
    ax_hp = fig.add_subplot(gs[3, 1])
    ax_cb2 = fig.add_subplot(gs[4, :])

    xlabel = 'Severity (max deviation)'
    ylabel = 'Magnitude (cumulative deficit)'

    # ── row 0: baseline ───────────────────────────────────────────────
    im1 = _plot_heatmap(ax_bm, sev_edges, mag_edges, base['median'],
                        base_cmap, base_vmin, base_vmax,
                        f'(a)  Baseline: Median {metric_label}')
    ax_bm.set_ylabel(ylabel, fontsize=FONTSIZE_MEDIUM)

    _plot_heatmap(ax_bp, sev_edges, mag_edges, base['p5'],
                  base_cmap, base_vmin, base_vmax,
                  f'(b)  Baseline: 5th Percentile {metric_label}')

    ax_bp.text(1.06, 0.5, 'Baseline Climate',
               transform=ax_bp.transAxes, fontsize=11, va='center', ha='left',
               rotation=-90, fontweight='bold',
               color=DATASET_COLORS.get('stationary_ensemble', 'k'))

    cb1 = fig.colorbar(im1, cax=ax_cb1, orientation='horizontal')
    cb1.set_label(cb_label, fontsize=FONTSIZE_SMALL)
    cb1.ax.tick_params(labelsize=FONTSIZE_SMALL - 1)

    # ── rows 2-3: deltas ──────────────────────────────────────────────
    panels = [
        (ax_lm, ax_lp, 'climate_adjusted_low',  'c', 'd'),
        (ax_hm, ax_hp, 'climate_adjusted_high', 'e', 'f'),
    ]
    im2 = None
    for ax_med, ax_p5, did, lm, lp in panels:
        d_med, d_p5 = deltas[did]
        mis = misalign[did]
        label = DATASET_LABELS.get(did, did)

        im2 = _plot_heatmap(ax_med, sev_edges, mag_edges, d_med,
                            delta_cmap, -delta_lim, delta_lim,
                            f'({lm})  {label}: \u0394 Median')
        _overlay_hatch(ax_med, sev_edges, mag_edges, mis)
        ax_med.set_ylabel(ylabel, fontsize=FONTSIZE_MEDIUM)

        _plot_heatmap(ax_p5, sev_edges, mag_edges, d_p5,
                      delta_cmap, -delta_lim, delta_lim,
                      f'({lp})  {label}: \u0394 5th Percentile')
        _overlay_hatch(ax_p5, sev_edges, mag_edges, mis)

        ax_p5.text(1.06, 0.5, label,
                   transform=ax_p5.transAxes, fontsize=11, va='center', ha='left',
                   rotation=-90, fontweight='bold',
                   color=DATASET_COLORS.get(did, 'k'))

    ax_hm.set_xlabel(xlabel, fontsize=FONTSIZE_MEDIUM)
    ax_hp.set_xlabel(xlabel, fontsize=FONTSIZE_MEDIUM)

    cb2 = fig.colorbar(im2, cax=ax_cb2, orientation='horizontal')
    cb2.set_label(cb_delta_label, fontsize=FONTSIZE_SMALL)
    cb2.ax.tick_params(labelsize=FONTSIZE_SMALL - 1)

    hatch_handle = Rectangle(
        (0, 0), 1, 1, facecolor='#d0d0d0', edgecolor='grey',
        hatch='///', linewidth=0.3, alpha=0.5,
        label='Data in one scenario only (delta not computed)',
    )
    fig.legend(
        handles=[hatch_handle], loc='lower center',
        fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, 0.003),
        handlelength=2.5, handleheight=1.5,
    )

    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


# ── main ──────────────────────────────────────────────────────────────

def main():
    ssi_window = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"SI13: Drought Outcome Heatmaps (SSI-{ssi_window})")
    print(f"  Severity range: [{MIN_SEVERITY}, {MAX_SEVERITY}]")

    # ── load ──────────────────────────────────────────────────────────
    all_data = {}
    for did in DATASETS:
        df = load_events(did, ssi_window)
        all_data[did] = df
        print(f"  {DATASET_LABELS.get(did, did)}: {len(df)} events")

    # Tight axis limits
    all_sev = np.concatenate([all_data[d]['severity'].values for d in DATASETS])
    all_mag = np.concatenate([all_data[d]['magnitude'].values for d in DATASETS])
    sev_limit = min(np.percentile(all_sev, 99.9) * 1.15, MAX_SEVERITY)
    mag_limit = np.percentile(all_mag, 99.9) * 1.15
    print(f"  Axis limits: severity {MIN_SEVERITY}\u2013{sev_limit:.1f}, "
          f"magnitude 0\u2013{mag_limit:.0f}")

    # ── heatmap grid ─────────────────────────────────────────────────
    sev_edges = np.linspace(MIN_SEVERITY, sev_limit, N_GRID + 1)
    mag_edges = np.linspace(0, mag_limit, N_GRID + 1)

    heatmap_specs = [
        ('event_min_storage_pct', 'Min. Storage',
         {'median': 'median', 'p5': _pct5},
         'RdYlGn', 0, 100, 'RdBu',
         'Min. Storage During Event (%)',
         '\u0394 Min. Storage vs. Baseline (%-points)',
         'storage'),
        ('max_consec_montague_days', 'Consec. Montague Days',
         {'median': 'median', 'p5': _pct95},
         'YlOrRd_r', 0, 30, 'RdBu_r',
         'Max Consec. Montague Shortage Days',
         '\u0394 Max Consec. Montague Days vs. Baseline',
         'montague_days'),
        ('total_montague_shortage_mg', 'Montague Shortage (MG)',
         {'median': 'median', 'p5': _pct95},
         'YlOrRd_r', 0, 500, 'RdBu_r',
         'Total Montague Shortage Volume (MG)',
         '\u0394 Total Montague Shortage vs. Baseline (MG)',
         'montague_volume'),
    ]

    for (col, label, sfuncs, cmap, vmin, vmax,
         dcmap, cbl, dcbl, tag) in heatmap_specs:
        print(f"\n--- Heatmap: {label} ---")
        plot_heatmap_figure(
            ssi_window, all_data, sev_edges, mag_edges,
            metric_col=col, metric_label=label, stat_funcs=sfuncs,
            base_cmap=cmap, base_vmin=vmin, base_vmax=vmax,
            delta_cmap=dcmap, cb_label=cbl, cb_delta_label=dcbl,
            fname=f"{FIG_OUTPUT_DIR}/heatmap_{tag}_ssi{ssi_window}.png",
        )

    print("\nDone.")


if __name__ == '__main__':
    main()
