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


def load_storage_percentiles(did):
    """Load storage percentiles — CSV period is already WY-ordered (period 1 = June)."""
    return pd.read_csv(
        f'{ROOT_DIR}/pywrdrb/zone_probabilities/{did}_storage_percentiles_weekly.csv',
        index_col='period')


def load_zone_probs(did):
    """Load zone probabilities — CSV period is already WY-ordered (period 1 = June)."""
    return pd.read_csv(
        f'{ROOT_DIR}/pywrdrb/zone_probabilities/{did}_zone_probs_weekly.csv',
        index_col='period')


def load_ffmp_boundaries_wy():
    """Load FFMP zone boundaries as weekly water-year-ordered curves.

    Returns dict with keys 'emergency' (level5), 'warning' (level4),
    'watch' (level3), 'normal' (level2), each a Series indexed 1..52.
    """
    import pywrdrb
    data = pywrdrb.Data(results_sets=['ffmp_level_boundaries'])
    data.load_output(output_filenames=[f'{ROOT_DIR}/pywrdrb/outputs/reconstruction.hdf5'])
    ffmp = data.ffmp_level_boundaries['reconstruction'][0] * 100  # to %

    # Median by day-of-year
    ffmp['doy'] = ffmp.index.dayofyear
    num_cols = ['level5', 'level4', 'level3', 'level2']
    daily_med = ffmp.groupby('doy')[num_cols].median()

    # Reorder to water year (June 1 = WY day 1)
    june1_doy = 152
    wy_days = list(range(june1_doy, 366)) + list(range(1, june1_doy))
    wy_daily = daily_med.loc[wy_days].copy()
    wy_daily.index = range(1, len(wy_daily) + 1)

    # Aggregate to weekly
    wy_weekly = wy_daily.groupby((wy_daily.index - 1) // 7 + 1).mean()

    return {
        'emergency': wy_weekly['level5'],  # lower bound of emergency
        'warning': wy_weekly['level4'],    # lower bound of warning
        'watch': wy_weekly['level3'],      # lower bound of watch
        'normal': wy_weekly['level2'],     # lower bound of normal
    }


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
    ffmp = load_ffmp_boundaries_wy()

    # P(FFMP Drought Zone) = P(zone_0 + zone_1 + zone_2)
    # zone_0 = Emergency (storage < level5)
    # zone_1 = Warning   (level5 <= storage < level4)
    # zone_2 = Watch     (level4 <= storage < level3)
    # Note: zone_5/zone_6 are HIGH storage zones (near/at capacity), NOT drought zones.
    p_drought = {}
    for d in DATASETS:
        zp = zones[d]
        raw = zp['zone_0'] + zp['zone_1'] + zp['zone_2']
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
    # PANEL (a): FFMP zone bands (background) + Storage percentile fans
    # ====================================================================
    # Plot FFMP zone bands FIRST as background so storage lines overlay
    w_ffmp = ffmp['emergency'].index.values[:52]
    emerg_vals = ffmp['emergency'].values[:52]
    warn_vals = ffmp['warning'].values[:52]
    watch_vals = ffmp['watch'].values[:52]
    normal_vals = ffmp['normal'].values[:52]

    # Zone colors (soft, distinct)
    zone_colors = {
        'emergency': '#d32f2f',   # red
        'warning':   '#ef6c00',   # orange
        'watch':     '#f9a825',   # amber
        'normal':    '#66bb6a',   # green
    }

    # Zone boundary lines only (no fill)
    ax_a.plot(w_ffmp, emerg_vals, color=zone_colors['emergency'],
              linewidth=1.8, alpha=0.75, zorder=5)
    ax_a.plot(w_ffmp, warn_vals, color=zone_colors['warning'],
              linewidth=1.8, alpha=0.70, zorder=5)
    ax_a.plot(w_ffmp, watch_vals, color=zone_colors['watch'],
              linewidth=1.5, alpha=0.60, zorder=5)
    ax_a.plot(w_ffmp, normal_vals, color=zone_colors['normal'],
              linewidth=1.5, alpha=0.55, zorder=5)

    # Zone labels at right edge (stagger to avoid overlap)
    right_x = 52.8
    ax_a.text(right_x, emerg_vals[-1] - 3, 'Emergency',
              fontsize=7, color=zone_colors['emergency'], va='center', ha='left',
              fontweight='bold', alpha=0.85)
    ax_a.text(right_x, (emerg_vals[-1] + warn_vals[-1]) / 2, 'Warning',
              fontsize=7, color=zone_colors['warning'], va='center', ha='left',
              fontweight='bold', alpha=0.80)
    ax_a.text(right_x, (warn_vals[-1] + watch_vals[-1]) / 2, 'Watch',
              fontsize=7, color=zone_colors['watch'], va='center', ha='left',
              fontweight='bold', alpha=0.75)
    ax_a.text(right_x, (watch_vals[-1] + normal_vals[-1]) / 2, 'Normal',
              fontsize=7, color=zone_colors['normal'], va='center', ha='left',
              fontweight='bold', alpha=0.70)

    # Storage percentile fans on top
    for did in DATASETS:
        sp = storage[did]
        w = sp.index.values
        color = DATASET_COLORS[did]
        ls = DATASET_LINESTYLES.get(did, '-')

        # Very light 1-99 shading
        ax_a.fill_between(w, sp['p1'], sp['p99'],
                          color=color, alpha=0.07, linewidth=0)

        # Median (solid, thick)
        ax_a.plot(w, sp['p50'], color=color, linewidth=2.5,
                  linestyle=ls, alpha=0.95, zorder=6)

        # 1st percentile (dashed)
        ax_a.plot(w, sp['p1'], color=color, linewidth=1.5,
                  linestyle='--', alpha=0.6, zorder=5)

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

    # Explicit y-axis label defining the metric
    ax_b.set_ylabel('P(FFMP Drought Zone)\n(%)', fontsize=FONTSIZE_LABEL)
    ax_b.set_ylim(0, None)  # auto upper limit
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
               alpha=0.6, label='1st Percentile'))

    fig.legend(handles=handles, loc='lower center', ncol=5,
               fontsize=10, frameon=False, bbox_to_anchor=(0.54, -0.01))

    fname = f"{FIG_OUTPUT_DIR}/option_A_v5.png"
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close(fig)


if __name__ == '__main__':
    plot_figure()
