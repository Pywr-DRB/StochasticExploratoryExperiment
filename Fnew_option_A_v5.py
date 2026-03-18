"""
Figure Option A v5: Seasonal Storage Dynamics and Drought Zone Exposure

Key changes from v4:
  - All panels clearly labeled with explicit metric definitions
  - Panel (b) y-axis: "P(FFMP Drought Zone)" with annotation defining it as
    P(Watch OR Warning OR Emergency)
  - Panel (c) y-axis: explicitly states "vs. Baseline Ensemble"
  - Added methodology note in panel (b) explaining the calculation
  - Improved FFMP threshold labels in panel (a)

Usage:
    python Fnew_option_A_v5.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings("ignore")

from methods.config import ROOT_DIR, FIG_DIR
from methods.plotting.styles import (
    DATASET_COLORS, DATASET_LABELS, DATASET_LINESTYLES,
    FONTSIZE_SMALL, FONTSIZE_MEDIUM, FONTSIZE_LARGE, FONTSIZE_LABEL,
    LINEWIDTH_MEDIUM, LINEWIDTH_THICK,
    DPI_HIGH, apply_publication_style,
)

FIG_OUTPUT_DIR = f"{FIG_DIR}/Fnew_seasonal_dynamics"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

DATASETS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']

WY_MONTH_STARTS = [1, 5, 9, 14, 18, 23, 27, 32, 36, 40, 45, 49]
WY_MONTH_LABELS = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',
                    'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']


def reindex_to_wy(df):
    wy_order = list(range(23, 54)) + list(range(1, 23))
    wy_order = [w for w in wy_order if w in df.index]
    df_wy = df.loc[wy_order].copy()
    df_wy.index = range(1, len(df_wy) + 1)
    return df_wy


def load_storage_percentiles(did):
    return reindex_to_wy(pd.read_csv(
        f'{ROOT_DIR}/pywrdrb/zone_probabilities/{did}_storage_percentiles_weekly.csv',
        index_col='period'))


def load_zone_probs(did):
    return reindex_to_wy(pd.read_csv(
        f'{ROOT_DIR}/pywrdrb/zone_probabilities/{did}_zone_probs_weekly.csv',
        index_col='period'))


def smooth(series, window=3):
    return series.rolling(window, center=True, min_periods=1).mean()


def format_wy_xaxis(ax, show_labels=True):
    ax.set_xticks(WY_MONTH_STARTS)
    ax.set_xticklabels(WY_MONTH_LABELS if show_labels else [],
                       fontsize=FONTSIZE_MEDIUM)
    ax.set_xlim(0.5, 52.5)


def plot_figure():
    apply_publication_style()
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12})

    storage = {d: load_storage_percentiles(d) for d in DATASETS}
    zones = {d: load_zone_probs(d) for d in DATASETS}

    # P(Drought Zone) = P(Zone 4 OR Zone 5 OR Zone 6), smoothed
    p_drought = {}
    for d in DATASETS:
        zp = zones[d]
        raw = zp['zone_4'] + zp['zone_5'] + zp['zone_6']
        p_drought[d] = smooth(raw, window=3)

    # ====================================================================
    fig = plt.figure(figsize=(10, 10.5))
    gs = gridspec.GridSpec(
        3, 1,
        height_ratios=[1.2, 0.9, 0.9],
        hspace=0.07,
        left=0.14, right=0.94, top=0.97, bottom=0.06,
    )

    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1], sharex=ax_a)
    ax_c = fig.add_subplot(gs[2], sharex=ax_a)

    plt.setp(ax_a.get_xticklabels(), visible=False)
    plt.setp(ax_b.get_xticklabels(), visible=False)

    # ====================================================================
    # PANEL (a): Storage percentile fans
    # ====================================================================
    for did in DATASETS:
        sp = storage[did]
        w = sp.index.values
        color = DATASET_COLORS[did]
        ls = DATASET_LINESTYLES.get(did, '-')

        # Very light 5-95 shading
        ax_a.fill_between(w, sp['p5'], sp['p95'],
                          color=color, alpha=0.07, linewidth=0)

        # Median (solid, thick)
        ax_a.plot(w, sp['p50'], color=color, linewidth=2.5,
                  linestyle=ls, alpha=0.95, zorder=4)

        # 5th percentile (dashed)
        ax_a.plot(w, sp['p5'], color=color, linewidth=1.5,
                  linestyle='--', alpha=0.6, zorder=3)

    # Seasonal context
    ax_a.axvspan(1, 26, color='#FFF9C4', alpha=0.08, zorder=0)
    ax_a.axvspan(26, 52, color='#E3F2FD', alpha=0.08, zorder=0)
    ax_a.text(13, 2, 'Drawdown Season', fontsize=9, ha='center',
              color='#5D4037', alpha=0.55, style='italic')
    ax_a.text(39, 2, 'Refill Season', fontsize=9, ha='center',
              color='#1565C0', alpha=0.50, style='italic')

    # FFMP thresholds — clearer labels
    ax_a.axhline(25, color='#c62828', linewidth=1.0, linestyle=':', alpha=0.50)
    ax_a.axhline(40, color='#e65100', linewidth=1.0, linestyle=':', alpha=0.45)
    ax_a.text(52.8, 25, 'FFMP\nEmergency', fontsize=7.5, color='#c62828',
              va='center', ha='left', alpha=0.70, linespacing=0.9)
    ax_a.text(52.8, 40, 'FFMP\nWarning', fontsize=7.5, color='#e65100',
              va='center', ha='left', alpha=0.65, linespacing=0.9)

    ax_a.set_ylabel('Combined NYC Reservoir\nStorage (% of capacity)', fontsize=FONTSIZE_LABEL)
    ax_a.set_ylim(0, 105)
    ax_a.grid(True, alpha=0.12, linestyle='--')
    ax_a.set_axisbelow(True)
    ax_a.text(0.015, 0.97, '(a)', transform=ax_a.transAxes,
              fontsize=14, va='top', fontweight='bold')

    # ====================================================================
    # PANEL (b): P(Drought Zone) — clearly defined
    # ====================================================================
    for did in DATASETS:
        w = p_drought[did].index.values
        color = DATASET_COLORS[did]
        ls = DATASET_LINESTYLES.get(did, '-')

        ax_b.plot(w, p_drought[did].values, color=color,
                  linewidth=2.5, linestyle=ls, alpha=0.90, zorder=3)

    # Fill between scenarios to highlight divergence
    stat_vals = p_drought['stationary_ensemble'].values
    for did in ['climate_adjusted_low', 'climate_adjusted_high']:
        scenario_vals = p_drought[did].values
        color = DATASET_COLORS[did]
        ax_b.fill_between(
            p_drought[did].index.values,
            stat_vals, scenario_vals,
            where=scenario_vals > stat_vals,
            color=color, alpha=0.12, linewidth=0,
        )

    # Annotate refill period
    ax_b.axvspan(27, 32, color='#E3F2FD', alpha=0.20, zorder=0)
    ax_b.text(29.5, 8, 'Reservoirs\nat capacity', fontsize=8, ha='center',
              color='#1565C0', alpha=0.6, style='italic')

    # Explicit y-axis label defining the metric
    ax_b.set_ylabel('P(FFMP Drought Zone)\n(%)', fontsize=FONTSIZE_LABEL)
    ax_b.set_ylim(0, 100)
    ax_b.grid(True, alpha=0.12, linestyle='--')
    ax_b.set_axisbelow(True)
    ax_b.text(0.015, 0.97, '(b)', transform=ax_b.transAxes,
              fontsize=14, va='top', fontweight='bold')

    # Methodology annotation inside panel
    ax_b.text(0.98, 0.97,
              'P(Watch OR Warning OR Emergency)\nper week across all ensemble years',
              transform=ax_b.transAxes, fontsize=8, va='top', ha='right',
              color='#555555', style='italic',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        alpha=0.85, edgecolor='#cccccc'))

    # ====================================================================
    # PANEL (c): Change in P(Drought Zone) relative to baseline
    # ====================================================================
    baseline = p_drought['stationary_ensemble']

    for did in ['climate_adjusted_low', 'climate_adjusted_high']:
        delta = p_drought[did] - baseline
        w = delta.index.values
        color = DATASET_COLORS[did]
        ls = DATASET_LINESTYLES.get(did, '-')

        ax_c.fill_between(w, 0, delta.values,
                          where=delta.values >= 0,
                          color=color, alpha=0.20)
        ax_c.fill_between(w, 0, delta.values,
                          where=delta.values < 0,
                          color=color, alpha=0.10)
        ax_c.plot(w, delta.values, color=color,
                  linewidth=2.5, linestyle=ls, alpha=0.90, zorder=3)

        # Peak annotation — clearly separated
        peak_w = delta.idxmax()
        peak_v = delta.max()
        short_label = DATASET_LABELS[did]
        if did == 'climate_adjusted_high':
            x_off, y_off = 6, 4
        else:
            x_off, y_off = -7, 3
        ax_c.annotate(
            f'{short_label}\n+{peak_v:.0f} pp',
            xy=(peak_w, peak_v),
            xytext=(peak_w + x_off, peak_v + y_off),
            fontsize=9, color=color, fontweight='bold',
            ha='center', linespacing=0.9,
            arrowprops=dict(arrowstyle='->', color=color,
                            lw=1.0, alpha=0.6),
        )

    ax_c.axhline(0, color='black', linewidth=0.8, alpha=0.35)
    ax_c.set_ylabel('Change in P(FFMP Drought\nZone) vs. Baseline (pp)', fontsize=FONTSIZE_LABEL)
    ax_c.grid(True, alpha=0.12, linestyle='--')
    ax_c.set_axisbelow(True)
    ax_c.text(0.015, 0.97, '(c)', transform=ax_c.transAxes,
              fontsize=14, va='top', fontweight='bold')

    format_wy_xaxis(ax_c)
    ax_c.set_xlabel('Month (Water Year)', fontsize=FONTSIZE_LABEL)

    # ====================================================================
    # Legend
    # ====================================================================
    handles = []
    for did in DATASETS:
        handles.append(
            Line2D([0], [0], color=DATASET_COLORS[did], linewidth=2.5,
                   linestyle=DATASET_LINESTYLES.get(did, '-'),
                   label=DATASET_LABELS[did]))
    handles.append(
        Line2D([0], [0], color='grey', linewidth=2.5, linestyle='-',
               alpha=0.9, label='Median'))
    handles.append(
        Line2D([0], [0], color='grey', linewidth=1.5, linestyle='--',
               alpha=0.6, label='5th Percentile'))

    fig.legend(handles=handles, loc='lower center', ncol=5,
               fontsize=10, frameon=False, bbox_to_anchor=(0.54, -0.01))

    fname = f"{FIG_OUTPUT_DIR}/option_A_v5.png"
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close(fig)


if __name__ == '__main__':
    plot_figure()
