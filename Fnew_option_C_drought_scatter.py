"""
Figure Option C: Drought Event Analysis.

2x2 panel showing antecedent conditions, hazard, system response, and
outcomes for SSI-6 drought events. Uses FFMP dynamic zone classification
at the date of minimum storage as the outcome metric.

Usage:
    python Fnew_option_C_drought_scatter.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

from methods.config import ROOT_DIR, FIG_DIR
from methods.plotting.styles import (
    DATASET_COLORS, DATASET_LABELS,
    FFMP_ZONE_COLORS,
    FONTSIZE_LABEL, FONTSIZE_MEDIUM,
    DPI_HIGH, apply_publication_style,
)

FIG_OUTPUT_DIR = f"{FIG_DIR}/Fnew_drought_scatter"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

SSI_WINDOW = 3
MIN_DURATION = 30

FFMP_ZONE_ORDER = ['Normal', 'Watch', 'Warning', 'Emergency']

PEAK_MARKERS = {
    'Winter': ('D', [12, 1, 2]),
    'Spring': ('^', [3, 4, 5]),
    'Summer': ('o', [6, 7, 8]),
    'Fall': ('s', [9, 10, 11]),
}


def load_events(dataset_id):
    df = pd.read_csv(
        f'{ROOT_DIR}/pywrdrb/event_metrics/{dataset_id}_ssi{SSI_WINDOW}_event_metrics.csv')
    df = df[df['duration_days'] >= MIN_DURATION].copy()
    df['severity'] = df['severity'].abs()
    df['magnitude'] = df['magnitude'].abs()
    df['contrib_pct'] = (df['contribution_ratio'] * 100).clip(0, 250)
    return df


def plot_figure():
    apply_publication_style()
    plt.rcParams.update({'font.size': 10.5, 'axes.labelsize': 11})

    df = load_events('stationary_ensemble')
    print(f"Loaded {len(df)} events (SSI-{SSI_WINDOW})")
    print(f"  FFMP zones: {df['ffmp_zone_at_min'].value_counts().to_dict()}")

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.28, wspace=0.30,
                            left=0.06, right=0.97, top=0.96, bottom=0.06,
                            width_ratios=[1, 1, 1])

    # ================================================================
    # (a) Antecedent Storage vs Hazard Magnitude
    #     Color = FFMP zone at min storage
    #     Size = duration
    #     Shape = peak severity season
    # ================================================================
    ax = fig.add_subplot(gs[0, 0])
    sizes_a = 12 + 0.12 * df['duration_days']

    for season, (marker, months) in PEAK_MARKERS.items():
        for zone in FFMP_ZONE_ORDER:
            m = df['peak_severity_month'].isin(months) & (df['ffmp_zone_at_min'] == zone)
            if m.sum() == 0:
                continue
            ax.scatter(df.loc[m, 'storage_at_start_pct'], df.loc[m, 'magnitude'],
                       s=sizes_a[m], c=FFMP_ZONE_COLORS[zone], alpha=0.7, marker=marker,
                       edgecolors='black', linewidths=0.4, zorder=3)

    # Season shape legend
    for season, (marker, _) in PEAK_MARKERS.items():
        ax.scatter([], [], marker=marker, c='grey', s=30, edgecolors='black',
                   linewidths=0.3, label=f'{season} peak')
    # Size legend
    for dur, lab in [(90, '90d'), (500, '500d'), (1200, '3yr')]:
        s = 12 + 0.12 * dur
        ax.scatter([], [], s=s, marker='o', c='grey', alpha=0.4, edgecolors='black',
                   linewidths=0.3, label=lab)
    ax.legend(fontsize=6.5, framealpha=0.9, edgecolor='#ccc', ncol=2, loc='upper right',
              title='Peak Season / Duration', title_fontsize=7)

    ax.set_xlabel('Storage at Drought Start (%)')
    ax.set_ylabel(f'Drought Magnitude (SSI-{SSI_WINDOW})')
    ax.grid(alpha=0.12, linestyle='--')
    ax.set_axisbelow(True)
    ax.text(0.02, 0.97, '(a) Antecedent × Hazard', transform=ax.transAxes,
            fontsize=10.5, va='top', fontweight='bold')

    # ================================================================
    # (b) Peak Severity Month vs Min Storage
    #     Size = magnitude
    #     Color = contribution ratio
    # ================================================================
    ax = fig.add_subplot(gs[0, 1])
    cmap_contrib = cm.YlOrRd
    norm_contrib = plt.Normalize(vmin=0, vmax=50)
    sizes_b = 12 + 8 * df['magnitude']

    sc = ax.scatter(df['peak_severity_month'], df['event_min_storage_pct'],
                    s=sizes_b, c=df['contrib_pct'].clip(0, 50),
                    cmap=cmap_contrib, norm=norm_contrib, alpha=0.7,
                    edgecolors='black', linewidths=0.4, zorder=3)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label('Contribution / Inflow (%)', fontsize=10)

    # FFMP dynamic thresholds as horizontal bands (approximate seasonal range)
    ax.axhspan(0, 26, color=FFMP_ZONE_COLORS['Emergency'], alpha=0.06)
    ax.axhspan(26, 40, color=FFMP_ZONE_COLORS['Warning'], alpha=0.04)
    ax.text(0.3, 20, 'Emergency zone', fontsize=7, color=FFMP_ZONE_COLORS['Emergency'], alpha=0.6)
    ax.text(0.3, 32, 'Warning zone', fontsize=7, color=FFMP_ZONE_COLORS['Warning'], alpha=0.6)

    for mag, lab in [(3, '3'), (12, '12'), (25, '25')]:
        s = 12 + 8 * mag
        ax.scatter([], [], s=s, c='grey', alpha=0.4, edgecolors='black',
                   linewidths=0.3, label=f'Mag={lab}')
    ax.legend(fontsize=7, framealpha=0.9, edgecolor='#ccc', loc='lower left')

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'], fontsize=8)
    ax.set_xlabel('Peak Severity Month')
    ax.set_ylabel('Min Storage During Event (%)')
    ax.set_ylim(bottom=10)
    ax.grid(alpha=0.12, linestyle='--')
    ax.set_axisbelow(True)
    ax.text(0.02, 0.97, '(b) Peak Timing \u2192 Outcome', transform=ax.transAxes,
            fontsize=10.5, va='top', fontweight='bold')

    # ================================================================
    # (c) Start month bar chart (stacked by FFMP zone)
    # ================================================================
    ax = fig.add_subplot(gs[0, 2])
    months = range(1, 13)
    month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

    bottom = np.zeros(12)
    for zone in FFMP_ZONE_ORDER:
        counts = df[df['ffmp_zone_at_min'] == zone].groupby('start_month').size()
        counts = counts.reindex(months, fill_value=0).values
        ax.bar(list(months), counts, bottom=bottom, width=0.7,
               color=FFMP_ZONE_COLORS[zone], alpha=0.8,
               edgecolor='black', linewidth=0.3, label=zone)
        bottom += counts

    ax.set_xticks(list(months))
    ax.set_xticklabels(month_labels, fontsize=9)
    ax.set_xlabel('Drought Start Month')
    ax.set_ylabel('Number of Events')
    ax.legend(fontsize=7.5, framealpha=0.9, edgecolor='#ccc', loc='upper left')
    ax.grid(axis='y', alpha=0.12, linestyle='--')
    ax.set_axisbelow(True)
    ax.text(0.02, 0.97, '(c) Start Month \u2192 Outcome', transform=ax.transAxes,
            fontsize=10.5, va='top', fontweight='bold')

    # ================================================================
    # (d) Duration vs Magnitude (was c)
    #     Color = FFMP zone
    #     Size = contribution ratio
    # ================================================================
    ax = fig.add_subplot(gs[1, 0])
    sizes_c = np.where(df['contrib_pct'] < 5, 10,
                        10 + 3.5 * (df['contrib_pct'] - 5))

    for zone in FFMP_ZONE_ORDER:
        m = df['ffmp_zone_at_min'] == zone
        if m.sum() == 0:
            continue
        ax.scatter(df.loc[m, 'duration_days'], df.loc[m, 'magnitude'],
                   s=sizes_c[m], c=FFMP_ZONE_COLORS[zone], alpha=0.7,
                   edgecolors='black', linewidths=0.4, zorder=3, label=zone)

    # Size legend
    for pct, lab in [(5, '<5%'), (20, '20%'), (50, '50%')]:
        s = 10 if pct < 5 else 10 + 3.5 * (pct - 5)
        ax.scatter([], [], s=s, c='grey', alpha=0.4, edgecolors='black',
                   linewidths=0.3, label=f'{lab} contrib')
    ax.legend(fontsize=7, framealpha=0.9, edgecolor='#ccc', loc='upper left',
              ncol=2, columnspacing=0.8)

    ax.set_xlabel('Duration (days)')
    ax.set_ylabel(f'Drought Magnitude (SSI-{SSI_WINDOW})')
    ax.grid(alpha=0.12, linestyle='--')
    ax.set_axisbelow(True)
    ax.text(0.35, 0.04, '(d) Duration \u00d7 Magnitude', transform=ax.transAxes,
            fontsize=10.5, va='bottom', fontweight='bold')

    # ================================================================
    # (e) Response: Contribution vs Magnitude by Start Season
    #     Color = FFMP zone
    #     Size = duration
    #     Shape = start season
    # ================================================================
    ax = fig.add_subplot(gs[1, 1])
    start_markers = {
        'Winter': ('D', [12, 1, 2]),
        'Spring': ('^', [3, 4, 5]),
        'Summer': ('o', [6, 7, 8]),
        'Fall': ('s', [9, 10, 11]),
    }
    sizes_d = 12 + 0.12 * df['duration_days']

    for season, (marker, months) in start_markers.items():
        for zone in FFMP_ZONE_ORDER:
            m = df['start_month'].isin(months) & (df['ffmp_zone_at_min'] == zone)
            if m.sum() == 0:
                continue
            ax.scatter(df.loc[m, 'magnitude'], df.loc[m, 'contrib_pct'],
                       s=sizes_d[m], c=FFMP_ZONE_COLORS[zone], alpha=0.7,
                       marker=marker, edgecolors='black', linewidths=0.4, zorder=3)

    # Season legend
    for season, (marker, _) in start_markers.items():
        ax.scatter([], [], marker=marker, c='grey', s=30, edgecolors='black',
                   linewidths=0.3, label=f'{season} start')
    ax.legend(fontsize=7.5, framealpha=0.9, edgecolor='#ccc', loc='upper right')

    ax.set_xlabel(f'Drought Magnitude (SSI-{SSI_WINDOW})')
    ax.set_ylabel('NYC Contribution / Inflow (%)')
    ax.grid(alpha=0.12, linestyle='--')
    ax.set_axisbelow(True)
    ax.text(0.02, 0.97, '(e) Response by Start Season', transform=ax.transAxes,
            fontsize=10.5, va='top', fontweight='bold')

    # ================================================================
    # (f) Start storage vs min storage, color=FFMP zone, shape=start season
    # ================================================================
    ax = fig.add_subplot(gs[1, 2])
    start_markers_f = {
        'Winter': ('D', [12, 1, 2]),
        'Spring': ('^', [3, 4, 5]),
        'Summer': ('o', [6, 7, 8]),
        'Fall': ('s', [9, 10, 11]),
    }
    sizes_f = 12 + 8 * df['magnitude']

    for season, (marker, months_list) in start_markers_f.items():
        for zone in FFMP_ZONE_ORDER:
            m = df['start_month'].isin(months_list) & (df['ffmp_zone_at_min'] == zone)
            if m.sum() == 0:
                continue
            ax.scatter(df.loc[m, 'storage_at_start_pct'], df.loc[m, 'event_min_storage_pct'],
                       s=sizes_f[m], c=FFMP_ZONE_COLORS[zone], alpha=0.7, marker=marker,
                       edgecolors='black', linewidths=0.4, zorder=3)

    # 1:1 line
    ax.plot([30, 100], [30, 100], 'k--', alpha=0.25, linewidth=1)
    # Season legend
    for season, (marker, _) in start_markers_f.items():
        ax.scatter([], [], marker=marker, c='grey', s=30, edgecolors='black',
                   linewidths=0.3, label=f'{season} start')
    ax.legend(fontsize=7.5, framealpha=0.9, edgecolor='#ccc', loc='lower right')

    ax.set_xlabel('Storage at Drought Start (%)')
    ax.set_ylabel('Min Storage During Event (%)')
    ax.grid(alpha=0.12, linestyle='--')
    ax.set_axisbelow(True)
    ax.text(0.02, 0.97, '(f) Antecedent \u2192 Outcome', transform=ax.transAxes,
            fontsize=10.5, va='top', fontweight='bold')

    # ================================================================
    # Shared FFMP zone color legend
    # ================================================================
    zone_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=FFMP_ZONE_COLORS[z],
               markeredgecolor='black', markeredgewidth=0.4, markersize=8, label=z)
        for z in FFMP_ZONE_ORDER
    ]
    fig.legend(handles=zone_handles, loc='lower center', ncol=4,
               fontsize=10, frameon=True, framealpha=0.9, edgecolor='#ccc',
               bbox_to_anchor=(0.5, -0.01), title='FFMP Zone at Min Storage',
               title_fontsize=10)

    fname = f"{FIG_OUTPUT_DIR}/option_C_drought_scatter.png"
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close(fig)


if __name__ == '__main__':
    plot_figure()
