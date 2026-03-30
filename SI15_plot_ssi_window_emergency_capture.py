"""
SI15: SSI Window Drought Emergency Capture Rate

For each SSI window (3, 6, 12), shows what fraction of Drought Emergency (DE)
zone events are captured (i.e., overlap with a detected SSI drought) versus
missed entirely.  One grouped bar chart with all datasets side by side.

Usage:
    python SI15_plot_ssi_window_emergency_capture.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings("ignore")

from methods.config import ROOT_DIR, FIG_DIR, SSI_WINDOWS
from methods.plotting.styles import (
    DATASET_LABELS, DATASET_COLORS,
    FONTSIZE_SMALL, FONTSIZE_MEDIUM, FONTSIZE_LABEL,
    DPI_HIGH, apply_publication_style,
)

FIG_OUTPUT_DIR = f"{FIG_DIR}/SI15_ssi_emergency_capture"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

DATASETS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']
PERF_DIR = f"{ROOT_DIR}/pywrdrb/performance_metrics"
DROUGHT_DIR = f"{ROOT_DIR}/pywrdrb/drought_metrics"

# Emergency zone threshold (max_zone == 6 in zone_duration_events)
EMERGENCY_ZONE = 6


# ── data helpers ─────────────────────────────────────────────────────

def load_zone_events(dataset_id):
    """Load zone duration events, filter to Emergency."""
    df = pd.read_csv(f"{PERF_DIR}/{dataset_id}_zone_duration_events.csv")
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    return df[df['max_zone'] >= EMERGENCY_ZONE].copy()


def load_drought_events(dataset_id, ssi_window):
    """Load SSI drought events."""
    df = pd.read_csv(
        f"{DROUGHT_DIR}/{dataset_id}_ssi{ssi_window}_drought_events.csv")
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    return df


def _overlaps(a_start, a_end, b_start, b_end):
    """True if intervals [a_start, a_end] and [b_start, b_end] overlap."""
    return a_start <= b_end and b_start <= a_end


def compute_capture_rate(de_events, drought_events):
    """Fraction of DE events that overlap with at least one drought event.

    Also returns counts: (n_captured, n_missed, n_total).
    """
    if len(de_events) == 0:
        return np.nan, 0, 0, 0

    # Group drought events by realization for fast lookup
    drought_by_r = drought_events.groupby('realization_id')

    captured = 0
    for _, de in de_events.iterrows():
        rid = de['realization_id']
        if rid not in drought_by_r.groups:
            continue  # no droughts at all for this realization → missed

        dr = drought_by_r.get_group(rid)
        # Check if any drought event overlaps this DE event
        hit = False
        for _, d in dr.iterrows():
            if _overlaps(de['start_date'], de['end_date'],
                         d['start'], d['end']):
                hit = True
                break
        if hit:
            captured += 1

    n_total = len(de_events)
    n_missed = n_total - captured
    return captured / n_total, captured, n_missed, n_total


# ── compute all rates ────────────────────────────────────────────────

def compute_all_rates():
    """Return DataFrame: dataset, ssi_window, capture_rate, captured, missed, total."""
    rows = []
    for did in DATASETS:
        de = load_zone_events(did)
        print(f"  {DATASET_LABELS.get(did, did)}: {len(de)} DE events")

        for w in SSI_WINDOWS:
            dr = load_drought_events(did, w)
            rate, cap, mis, tot = compute_capture_rate(de, dr)
            rows.append({
                'dataset': did,
                'ssi_window': w,
                'capture_rate': rate,
                'captured': cap,
                'missed': mis,
                'total': tot,
            })
            print(f"    SSI-{w:>2d}: {cap}/{tot} captured "
                  f"({100*rate:.1f}%), {mis} missed")

    return pd.DataFrame(rows)


# ── figure ───────────────────────────────────────────────────────────

def plot_capture_rates(rates_df):
    apply_publication_style()

    windows = sorted(rates_df['ssi_window'].unique())
    n_win = len(windows)
    n_ds = len(DATASETS)

    group_width = 0.70
    bw = group_width / n_ds

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for d_idx, did in enumerate(DATASETS):
        sub = rates_df[rates_df['dataset'] == did].set_index('ssi_window')
        color = DATASET_COLORS.get(did, f'C{d_idx}')
        label = DATASET_LABELS.get(did, did)

        positions = np.arange(n_win) + (d_idx - (n_ds - 1) / 2) * bw
        values = [sub.loc[w, 'capture_rate'] * 100 for w in windows]

        bars = ax.bar(positions, values, width=bw * 0.88,
                      color=color, alpha=0.80, edgecolor='black',
                      linewidth=0.5, label=label, zorder=3)

        # Annotate capture / total counts inside each bar
        for pos, w in zip(positions, windows):
            row = sub.loc[w]
            ax.text(pos, row['capture_rate'] * 100 - 3,
                    f"{int(row['captured'])}/{int(row['total'])}",
                    ha='center', va='top', fontsize=FONTSIZE_SMALL - 1,
                    fontweight='bold', color='white')

    ax.set_xticks(np.arange(n_win))
    ax.set_xticklabels([f'SSI-{w}' for w in windows], fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('DE Events Captured by SSI Drought (%)',
                  fontsize=FONTSIZE_MEDIUM)
    ax.set_xlabel('SSI Window', fontsize=FONTSIZE_MEDIUM)
    ax.set_ylim(0, 105)
    ax.axhline(90, color='grey', linewidth=1, linestyle='--', alpha=0.5)
    ax.text(n_win - 0.5, 91, '90%', fontsize=FONTSIZE_SMALL,
            color='grey', ha='right')

    ax.legend(fontsize=FONTSIZE_SMALL, loc='lower right',
              frameon=True, fancybox=True)
    ax.set_title('Fraction of Drought Emergency Events\n'
                 'Overlapping an SSI-Detected Drought',
                 fontsize=FONTSIZE_LABEL, pad=10)

    fig.tight_layout()
    fname = f"{FIG_OUTPUT_DIR}/SI15_ssi_emergency_capture.png"
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close(fig)


# ── missed DE events: duration distribution ──────────────────────────

def plot_missed_duration_distribution(rates_df):
    """Supplementary panel: duration of captured vs missed DE events."""
    apply_publication_style()

    fig, axes = plt.subplots(1, len(SSI_WINDOWS), figsize=(5 * len(SSI_WINDOWS), 4.5),
                             sharey=True)
    if len(SSI_WINDOWS) == 1:
        axes = [axes]

    panel_letters = 'abcdefghij'

    for w_idx, w in enumerate(SSI_WINDOWS):
        ax = axes[w_idx]

        for d_idx, did in enumerate(DATASETS):
            de = load_zone_events(did)
            dr = load_drought_events(did, w)
            color = DATASET_COLORS.get(did, f'C{d_idx}')

            # Tag each DE event as captured or missed
            drought_by_r = dr.groupby('realization_id')
            captured_dur = []
            missed_dur = []

            for _, de_row in de.iterrows():
                rid = de_row['realization_id']
                dur = de_row['duration_days']
                hit = False
                if rid in drought_by_r.groups:
                    for _, d in drought_by_r.get_group(rid).iterrows():
                        if _overlaps(de_row['start_date'], de_row['end_date'],
                                     d['start'], d['end']):
                            hit = True
                            break
                if hit:
                    captured_dur.append(dur)
                else:
                    missed_dur.append(dur)

            # Box for missed durations only (if any)
            if missed_dur:
                pos = d_idx
                bp = ax.boxplot(
                    [missed_dur],
                    positions=[pos],
                    widths=0.6,
                    patch_artist=True,
                    showfliers=True,
                    whis=(5, 95),
                    medianprops=dict(color='black', linewidth=1.5),
                    boxprops=dict(facecolor=color, alpha=0.7,
                                  edgecolor='black', linewidth=0.5),
                    whiskerprops=dict(color='black', linewidth=0.8),
                    capprops=dict(color='black', linewidth=0.8),
                    flierprops=dict(marker='.', markersize=2,
                                    alpha=0.3, color=color),
                )

        ax.set_xticks(range(len(DATASETS)))
        ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in DATASETS],
                           fontsize=FONTSIZE_SMALL - 1, rotation=20, ha='right')
        ax.set_title(f'({panel_letters[w_idx]})  SSI-{w}',
                     fontsize=FONTSIZE_MEDIUM)
        if w_idx == 0:
            ax.set_ylabel('Duration of Missed DE Events (days)',
                          fontsize=FONTSIZE_MEDIUM)

    fig.suptitle('Duration of Drought Emergency Events\nNOT Captured by SSI Drought',
                 fontsize=FONTSIZE_LABEL, y=1.02)
    fig.tight_layout()
    fname = f"{FIG_OUTPUT_DIR}/SI15_missed_de_duration.png"
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────

def main():
    print("SI15: SSI Window Drought Emergency Capture Rate")

    rates = compute_all_rates()
    plot_capture_rates(rates)
    plot_missed_duration_distribution(rates)

    print("Done.")


if __name__ == '__main__':
    main()
