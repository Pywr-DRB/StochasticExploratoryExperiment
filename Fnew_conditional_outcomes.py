"""
Fnew: Conditional Outcome Distributions

Two complementary figure sets bridging F2 (drought hazard) to F5 (system performance):

  Figure 1 — Outcome Heatmaps:
    Row 1:   Baseline median and 5th-pctile min storage (severity x magnitude)
    Row 2-3: Relative change for each climate scenario
    Grey hatch marks cells where baseline/scenario data don't align.

  Figures 2a/2b — NxN Outcome Grids:
    Joint (severity x magnitude) bins with exceedance CDFs in each cell.
    Version (a): max consecutive Montague shortage days
    Version (b): total Montague shortage volume (MG)

Usage:
    python Fnew_conditional_outcomes.py [ssi_window]
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import binned_statistic_2d
import warnings
warnings.filterwarnings("ignore")

from methods.config import ROOT_DIR, FIG_DIR
from methods.plotting.styles import (
    DATASET_COLORS, DATASET_LINESTYLES, DATASET_LABELS,
    FONTSIZE_SMALL, FONTSIZE_MEDIUM,
    CMAP_SEQUENTIAL,
    DPI_HIGH, apply_publication_style,
)

FIG_OUTPUT_DIR = f"{FIG_DIR}/Fnew_conditional_outcomes"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

MIN_DURATION = 30
MIN_SEVERITY = 1.0
MAX_SEVERITY = 6.2
N_GRID = 25
MIN_COUNT = 3
DATASETS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']

# Magnitude bins (approximately tercile boundaries of baseline, rounded)
MAG_BINS = [
    (0, 4, '< 4'),
    (4, 8, '4\u20138'),
    (8, np.inf, '\u2265 8'),
]
MAG_BIN_COLORS = ['#4393C3', '#FDB863', '#D73027']


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


def compute_exceedance(vals, x_grid):
    vals_sorted = np.sort(vals)
    counts = len(vals_sorted) - np.searchsorted(vals_sorted, x_grid, side='left')
    return counts / len(vals_sorted)


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


# ── figure 1: outcome heatmaps ───────────────────────────────────────

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


# ── figures 2a/2b: NxN outcome grids ─────────────────────────────────

def plot_outcome_cdfs(ssi_window, all_data, outcome_specs, fname,
                      show_marginal=True, log_kde_xaxis=True,
                      show_1960s_ref=False):
    """CDF figure with KDE reference spanning the top row and
    metric × magnitude-bin panels below (n_metrics rows × n_mag columns).

    Parameters
    ----------
    outcome_specs : list of (col_name, x_label, use_int_grid)
        Each tuple defines one row of CDF panels.
    show_marginal : bool
        If True, overlay the unconditional P(outcome) as thin dashed lines
        alongside the conditional P(outcome | magnitude bin) solid lines.
    log_kde_xaxis : bool
        If True, use log scale for the magnitude KDE x-axis.
    """

    apply_publication_style()
    plt.rcParams.update({'font.size': 13, 'axes.labelsize': 14,
                         'axes.titlesize': 14, 'xtick.labelsize': 12,
                         'ytick.labelsize': 12})
    from scipy.stats import gaussian_kde

    n_mag = len(MAG_BINS)
    n_metrics = len(outcome_specs)
    baseline = all_data['stationary_ensemble']

    # ── layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(4.5 * n_mag, 3 + 4 * n_metrics))
    gs = gridspec.GridSpec(
        1 + n_metrics, n_mag,
        height_ratios=[0.7] + [1] * n_metrics,
        hspace=0.38, wspace=0.28,
        left=0.08, right=0.97, top=0.95, bottom=0.07,
    )

    # ── load 1964 reconstruction reference values ───────────────────
    recon_1964 = {}
    recon_mag_bin = None
    if show_1960s_ref:
        try:
            recon_df = pd.read_csv(
                f'{ROOT_DIR}/pywrdrb/event_metrics/reconstruction_ssi{ssi_window}_event_metrics.csv')
            recon_df['start'] = pd.to_datetime(recon_df['start'])
            recon_df['end'] = pd.to_datetime(recon_df['end'])
            target = pd.Timestamp('1964-12-01')
            d64 = recon_df[(recon_df['start'] <= target) & (recon_df['end'] >= target)]
            if len(d64) > 0:
                r = d64.iloc[0]
                recon_1964['magnitude'] = abs(r['magnitude'])
                for spec_col, _, _ in outcome_specs:
                    if spec_col in r.index:
                        recon_1964[spec_col] = float(r[spec_col])
                print(f"  1960s drought reference: mag={recon_1964['magnitude']:.0f}, "
                      + ', '.join(f'{k}={v:.1f}' for k, v in recon_1964.items()
                                  if k != 'magnitude'))
        except Exception as e:
            print(f"  1960s drought reference not available: {e}")

        if 'magnitude' in recon_1964:
            for col, (mlo, mhi, _) in enumerate(MAG_BINS):
                if recon_1964['magnitude'] >= mlo and recon_1964['magnitude'] < mhi:
                    recon_mag_bin = col
                    break
            if recon_mag_bin is None and recon_1964['magnitude'] >= MAG_BINS[-1][0]:
                recon_mag_bin = len(MAG_BINS) - 1

    # ── top row: KDE reference spanning all columns ───────────────────
    ax_kde = fig.add_subplot(gs[0, :])
    kde_x = np.linspace(0, baseline['magnitude'].quantile(0.995), 300)
    for did in DATASETS:
        df = all_data[did]
        kde = gaussian_kde(df['magnitude'].values, bw_method=0.15)
        ax_kde.plot(kde_x, kde(kde_x),
                    color=DATASET_COLORS.get(did),
                    linestyle='-',
                    linewidth=2.5, alpha=0.85)

    # Bin boundaries as vertical lines with labels
    for i, (lo, hi, label) in enumerate(MAG_BINS):
        if i > 0:
            ax_kde.axvline(lo, color='black', ls='--', lw=1.5, alpha=0.7, zorder=5)
            ax_kde.text(lo, ax_kde.get_ylim()[1] * 0.02, f'  {lo}',
                        fontsize=8, ha='left', va='bottom', alpha=0.6)

    # 1964 drought reference on KDE
    if show_1960s_ref:
        from methods.load import load_drought_events
        try:
            obs = load_drought_events(DATASETS[0], ssi_window, observed=True)
            target = pd.Timestamp('1964-12-01')
            d64 = obs[(pd.to_datetime(obs['start']) <= target) &
                      (pd.to_datetime(obs['end']) >= target)]
            if len(d64) > 0:
                mag_64 = abs(d64.iloc[0]['magnitude'])
                ax_kde.axvline(mag_64, color='red', ls='-', lw=2, alpha=0.8, zorder=6)
                ax_kde.text(mag_64 * 1.05, ax_kde.get_ylim()[1] * 0.5,
                            f'1960s Drought\n(mag={mag_64:.0f})',
                            fontsize=8, color='red', va='center',
                            bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                      alpha=0.85, ec='red', lw=0.8))
        except Exception:
            pass

    ax_kde.set_xlabel('Magnitude (cumulative deficit)', fontsize=13)
    ax_kde.set_ylabel('Density', fontsize=13)
    ax_kde.text(0.02, 0.95, '(a)', transform=ax_kde.transAxes,
                fontsize=13, va='top', ha='left')
    if log_kde_xaxis:
        ax_kde.set_xscale('log')
        ax_kde.set_xlim(0.8, kde_x[-1])
    else:
        ax_kde.set_xlim(0, kde_x[-1])
    ax_kde.set_ylim(bottom=0)
    ax_kde.tick_params(labelsize=FONTSIZE_SMALL)
    ax_kde.spines['top'].set_visible(False)
    ax_kde.spines['right'].set_visible(False)

    # ── CDF rows (one row per metric, one column per mag bin) ─────────
    for mi, (outcome_col, x_label, use_int) in enumerate(outcome_specs):
        all_vals = np.concatenate([d[outcome_col].values for d in all_data.values()])
        pos = all_vals[all_vals > 0]
        data_max = np.percentile(pos, 99.9) if len(pos) > 0 else 1
        x_grid = (np.arange(0, int(data_max) + 2) if use_int
                  else np.linspace(0, data_max * 1.05, 400))

        # Compute conditional curves per (mag_bin, dataset)
        row_curves = {}
        y_max = 0.0
        for col, (mlo, mhi, _) in enumerate(MAG_BINS):
            for did in DATASETS:
                df = all_data[did]
                mask = (df['magnitude'] >= mlo) & (df['magnitude'] < mhi)
                vals = df.loc[mask, outcome_col].values
                if len(vals) < 10:
                    continue
                exc = compute_exceedance(vals, x_grid)
                nz = np.where(exc > 0)[0]
                if len(nz) == 0:
                    continue
                end = min(nz[-1] + 2, len(x_grid))
                row_curves[(col, did)] = (x_grid[:end], exc[:end])
                y_max = max(y_max, exc[0] * 100)

        # Compute joint marginal curves: P(outcome >= x AND mag in bin)
        # = (count in bin with outcome >= x) / (total events)
        marginal_curves = {}  # (col, did) -> (x_trunc, joint_prob_trunc)
        if show_marginal:
            for col, (mlo, mhi, _) in enumerate(MAG_BINS):
                for did in DATASETS:
                    df = all_data[did]
                    n_total = len(df)
                    mask = (df['magnitude'] >= mlo) & (df['magnitude'] < mhi)
                    vals = df.loc[mask, outcome_col].values
                    if len(vals) < 10:
                        continue
                    # Joint probability: events in this bin exceeding x / total
                    joint = np.array([np.sum(vals >= x) / n_total for x in x_grid])
                    nz = np.where(joint > 0)[0]
                    if len(nz) == 0:
                        continue
                    end = min(nz[-1] + 2, len(x_grid))
                    marginal_curves[(col, did)] = (x_grid[:end], joint[:end])

        for col, (mlo, mhi, mag_label) in enumerate(MAG_BINS):
            ax = fig.add_subplot(gs[1 + mi, col])

            cell_x_max = 0

            # Joint marginal P(outcome, mag bin) — dashed lines
            if show_marginal:
                for did in DATASETS:
                    mkey = (col, did)
                    if mkey not in marginal_curves:
                        continue
                    xc, yc = marginal_curves[mkey]
                    ax.plot(xc, yc * 100,
                            color=DATASET_COLORS.get(did),
                            linestyle='--', linewidth=1.0, alpha=0.5)
                    cell_x_max = max(cell_x_max, xc[-1])

            # Conditional P(outcome | mag bin) — solid lines
            for did in DATASETS:
                key = (col, did)
                if key not in row_curves:
                    continue
                xc, yc = row_curves[key]
                ax.plot(xc, yc * 100,
                        color=DATASET_COLORS.get(did),
                        linestyle='-',
                        linewidth=2.2, alpha=0.85)
                cell_x_max = max(cell_x_max, xc[-1])

            # 1964 drought reference line (only in matching mag bin)
            if col == recon_mag_bin and outcome_col in recon_1964:
                ref_val = recon_1964[outcome_col]
                ax.axvline(ref_val, color='red', ls='-', lw=2, alpha=0.8, zorder=6)
                ax.text(ref_val, y_max * 1.1, '1960s',
                        fontsize=7.5, color='red', ha='center', va='bottom',
                        fontweight='bold')

            ax.set_yscale('log')
            ax.set_ylim(0.05, y_max * 1.5)
            ax.set_xlim(0, cell_x_max * 1.1 if cell_x_max > 0 else 1)
            ax.grid(True, which='both', alpha=0.15, ls='--')
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=FONTSIZE_SMALL)

            # Integer tick labels instead of exponents
            from matplotlib.ticker import FuncFormatter
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda v, _: f'{v:g}'))
            ax.yaxis.set_minor_formatter(
                FuncFormatter(lambda v, _: ''))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Subplot letter
            letter = chr(ord('b') + mi * n_mag + col)
            ax.text(0.03, 0.95, f'({letter})', transform=ax.transAxes,
                    fontsize=13, va='top', ha='left')

            # Column title on first metric row only
            if mi == 0:
                ax.set_title(f'Magnitude {mag_label}', fontsize=13)
            # x-label on every row
            ax.set_xlabel(x_label, fontsize=13)
            # y-label on left column only
            if col == 0:
                ax.set_ylabel(r'P(outcome $\geq$ x)  [%]', fontsize=13)
            # Row label on right edge
            if col == n_mag - 1:
                ax.text(1.04, 0.5, x_label,
                        transform=ax.transAxes, fontsize=10,
                        va='center', ha='left', rotation=-90)

    # ── legend ────────────────────────────────────────────────────────
    ds_handles = [
        Line2D([0], [0], color=DATASET_COLORS.get(did),
               linestyle='-', lw=3,
               label=DATASET_LABELS.get(did, did))
        for did in DATASETS
    ]
    if show_marginal:
        ds_handles.append(
            Line2D([0], [0], color='grey', linestyle='--',
                   lw=1.0, alpha=0.5,
                   label='P(outcome, mag bin)  [joint]'))
    fig.legend(
        handles=ds_handles, loc='lower center',
        ncol=len(ds_handles),
        fontsize=11, frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


# ── main ──────────────────────────────────────────────────────────────

def main():
    ssi_window = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"Fnew: Conditional Outcome Distributions (SSI-{ssi_window})")
    print(f"  Severity range: [{MIN_SEVERITY}, {MAX_SEVERITY}]")

    # ── load ──────────────────────────────────────────────────────────
    all_data = {}
    for did in DATASETS:
        df = load_events(did, ssi_window)
        all_data[did] = df
        print(f"  {DATASET_LABELS.get(did, did)}: {len(df)} events")

    baseline = all_data['stationary_ensemble']

    # Tight axis limits
    all_sev = np.concatenate([all_data[d]['severity'].values for d in DATASETS])
    all_mag = np.concatenate([all_data[d]['magnitude'].values for d in DATASETS])
    sev_limit = min(np.percentile(all_sev, 99.9) * 1.15, MAX_SEVERITY)
    mag_limit = np.percentile(all_mag, 99.9) * 1.15
    print(f"  Axis limits: severity {MIN_SEVERITY}\u2013{sev_limit:.1f}, "
          f"magnitude 0\u2013{mag_limit:.0f}")

    # Magnitude bin summary
    print(f"\n  Magnitude bins (baseline, tercile-based):")
    for lo, hi, label in MAG_BINS:
        m = (baseline['magnitude'] >= lo) & (baseline['magnitude'] < hi)
        n = m.sum()
        print(f"    mag {label:>6s}: {n:>6,} ({100*n/len(baseline):.1f}%)")

    # ── heatmap grid ─────────────────────────────────────────────────
    sev_edges = np.linspace(MIN_SEVERITY, sev_limit, N_GRID + 1)
    mag_edges = np.linspace(0, mag_limit, N_GRID + 1)

    # Define heatmap variants: (col, label, stat_funcs, cmap, vmin, vmax,
    #                           delta_cmap, cb_label, cb_delta_label, fname_tag)
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

    # ── figure 2: conditional outcome CDFs ────────────────────────────
    outcome_specs = [
        ('max_consec_montague_days',
         'Max Consec. Montague Shortage Days', True),
        ('total_montague_shortage_mg',
         'Total Montague Shortage Volume (MG)', False),
    ]

    print(f"\n--- Conditional Outcome CDFs ({len(outcome_specs)} metrics) ---")
    plot_outcome_cdfs(
        ssi_window, all_data,
        outcome_specs=outcome_specs,
        show_marginal=False,
        show_1960s_ref=False,
        log_kde_xaxis=True,
        fname=f"{FIG_OUTPUT_DIR}/cdf_montague_outcomes_ssi{ssi_window}.png",
    )

    print("\nDone.")


if __name__ == '__main__':
    main()
