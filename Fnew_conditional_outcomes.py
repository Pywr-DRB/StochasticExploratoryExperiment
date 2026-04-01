"""
Fnew: Conditional Outcome Distributions — KDE/CDF Figure

Multipanel figure showing conditional exceedance CDFs of drought outcomes
(Montague shortage days and volume) by magnitude bin, with a KDE reference
panel showing the magnitude distribution across climate scenarios.

  Top row:  KDE of drought magnitude across datasets
  Lower rows: P(outcome >= x | magnitude bin) for each metric × bin

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
import warnings
warnings.filterwarnings("ignore")

from methods.config import ROOT_DIR, FIG_DIR, EVENT_METRICS_DIR
from methods.plotting.styles import (
    DATASET_COLORS, DATASET_LABELS,
    FONTSIZE_SMALL,
    DPI_HIGH, apply_publication_style,
)

FIG_OUTPUT_DIR = f"{FIG_DIR}/Fnew_conditional_outcomes"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

MIN_DURATION = 30
MIN_SEVERITY = 1.0
MAX_SEVERITY = 6.2
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
        f'{EVENT_METRICS_DIR}/'
        f'{dataset_id}_ssi{ssi_window}_event_metrics.csv'
    )
    df = df[df['duration_days'] >= MIN_DURATION].copy()
    df['severity'] = df['severity'].abs()
    df['magnitude'] = df['magnitude'].abs()
    df = df[(df['severity'] >= MIN_SEVERITY) & (df['severity'] <= MAX_SEVERITY)]
    return df


def compute_exceedance(vals, x_grid):
    vals_sorted = np.sort(vals)
    counts = len(vals_sorted) - np.searchsorted(vals_sorted, x_grid, side='left')
    return counts / len(vals_sorted)



# ── figure: NxN outcome grids ─────────────────────────────────────────

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
                f'{EVENT_METRICS_DIR}/reconstruction_ssi{ssi_window}_event_metrics.csv')
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
    print(f"Fnew: Conditional Outcome CDFs (SSI-{ssi_window})")
    print(f"  Severity range: [{MIN_SEVERITY}, {MAX_SEVERITY}]")

    # ── load ──────────────────────────────────────────────────────────
    all_data = {}
    for did in DATASETS:
        df = load_events(did, ssi_window)
        all_data[did] = df
        print(f"  {DATASET_LABELS.get(did, did)}: {len(df)} events")

    baseline = all_data['stationary_ensemble']

    # Magnitude bin summary
    print(f"\n  Magnitude bins (baseline, tercile-based):")
    for lo, hi, label in MAG_BINS:
        m = (baseline['magnitude'] >= lo) & (baseline['magnitude'] < hi)
        n = m.sum()
        print(f"    mag {label:>6s}: {n:>6,} ({100*n/len(baseline):.1f}%)")

    # ── conditional outcome CDFs ──────────────────────────────────────
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
