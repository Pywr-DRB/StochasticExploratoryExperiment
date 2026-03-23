"""
Drought Duration vs Hazard scatter plots.

Three versions of a 4x3 (season x dataset) log-log scatter:
  1. Duration vs Magnitude
  2. Duration vs Severity (min SSI)
  3. Duration vs Average Severity (magnitude / duration)

All versions use:
  - Marker shape = peak SSI season (one row per season)
  - Marker size  = contribution ratio (NYC releases / inflow)
  - Marker color = FFMP zone at min storage

Based on panel (d) of Fnew_option_C_drought_scatter.py.

Usage:
    python Fnew_duration_magnitude_scatter.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

from methods.config import ROOT_DIR, FIG_DIR
from methods.plotting.styles import (
    FFMP_ZONE_COLORS, DATASET_LABELS,
    FONTSIZE_LABEL, FONTSIZE_MEDIUM, FONTSIZE_SMALL,
    DPI_HIGH, apply_publication_style,
)

FIG_OUTPUT_DIR = f"{FIG_DIR}/SI12_duration_magnitude"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

SSI_WINDOW = 3
MIN_DURATION = 30

FFMP_ZONE_ORDER = ['Normal', 'Watch', 'Warning', 'Emergency']

SEASON_MARKERS = {
    'Winter': ('D', [12, 1, 2]),
    'Spring': ('^', [3, 4, 5]),
    'Summer': ('o', [6, 7, 8]),
    'Fall':   ('s', [9, 10, 11]),
}

DATASETS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']


def load_events(dataset_id):
    df = pd.read_csv(
        f'{ROOT_DIR}/pywrdrb/event_metrics/{dataset_id}_ssi{SSI_WINDOW}_event_metrics.csv')
    df = df[df['duration_days'] >= MIN_DURATION].copy()
    df['magnitude'] = df['magnitude'].abs()
    df['severity_abs'] = df['severity'].abs()
    df['avg_severity'] = df['magnitude'] / (df['duration_days'] / 30.44)  # per month
    df['contrib_pct'] = (df['contribution_ratio'] * 100).clip(0, 250)
    return df


# Figure versions: (x_col, x_label, x_log, y_col, y_label, y_log, filename)
FIGURE_VERSIONS = [
    ('duration_days', 'Duration (days)', True, 'magnitude', 'Magnitude', True, 'duration_magnitude'),
    ('duration_days', 'Duration (days)', True, 'severity_abs', 'Severity (min SSI)', False, 'duration_severity'),
    ('duration_days', 'Duration (days)', True, 'avg_severity', 'Avg Severity (mag/dur)', False, 'duration_avg_severity'),
    ('magnitude', 'Magnitude', False, 'severity_abs', 'Severity (min SSI)', False, 'magnitude_severity'),
]


def plot_version(all_dfs, x_key, x_label, x_log, y_key, y_label, y_log, output_fname):
    """Plot one 4x3 figure for a given x/y-axis metric pair."""
    n_rows = len(SEASON_MARKERS)
    n_cols = len(DATASETS)
    season_list = list(SEASON_MARKERS.keys())

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4 * n_rows),
                             sharex=True, sharey=True)

    for row, season in enumerate(season_list):
        marker, months = SEASON_MARKERS[season]

        for col, dataset_id in enumerate(DATASETS):
            ax = axes[row, col]
            df = all_dfs[dataset_id]

            # Filter to this season
            df_s = df[df['peak_severity_month'].isin(months)]

            # Marker size from contribution ratio
            sizes = np.where(df_s['contrib_pct'] < 5, 15,
                             15 + 3.5 * (df_s['contrib_pct'] - 5))

            # Plot each zone
            for zone in FFMP_ZONE_ORDER:
                m = df_s['ffmp_zone_at_min'] == zone
                if m.sum() == 0:
                    continue
                ax.scatter(df_s.loc[m, x_key], df_s.loc[m, y_key],
                           s=sizes[m.values], c=FFMP_ZONE_COLORS[zone], alpha=0.7,
                           marker=marker, edgecolors='black', linewidths=0.4, zorder=3)

            if x_log:
                ax.set_xscale('log')
            if y_log:
                ax.set_yscale('log')
            ax.grid(alpha=0.15, linestyle='--')
            ax.set_axisbelow(True)

            # Column titles on top row
            if row == 0:
                ax.set_title(DATASET_LABELS.get(dataset_id, dataset_id),
                             fontsize=FONTSIZE_LABEL, fontweight='bold')

            # Row labels on left column
            if col == 0:
                ax.set_ylabel(f'{season} Peak\n\n{y_label}')

            # X-axis label on bottom row only
            if row == n_rows - 1:
                ax.set_xlabel(x_label)

            # Event count annotation
            ax.text(0.97, 0.03, f'n={len(df_s)}', transform=ax.transAxes,
                    fontsize=FONTSIZE_SMALL, ha='right', va='bottom',
                    fontstyle='italic', alpha=0.7)

    # --- Shared legend ---
    size_handles = []
    for pct, lab in [(5, '<5%'), (25, '25%'), (50, '50%')]:
        s = 15 if pct < 5 else 15 + 3.5 * (pct - 5)
        size_handles.append(
            Line2D([0], [0], marker='o', color='none', markerfacecolor='grey',
                   markeredgecolor='black', markeredgewidth=0.3,
                   markersize=np.sqrt(s), alpha=0.5, label=f'{lab} contrib'))

    zone_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=FFMP_ZONE_COLORS[z],
               markeredgecolor='black', markeredgewidth=0.4, markersize=8, label=z)
        for z in FFMP_ZONE_ORDER
    ]

    all_handles = size_handles + zone_handles
    fig.legend(handles=all_handles, loc='lower center', ncol=len(all_handles),
               fontsize=9, frameon=True, framealpha=0.9, edgecolor='#ccc',
               bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    fig.savefig(output_fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {output_fname}")
    plt.close(fig)


def plot_figure():
    apply_publication_style()
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12})

    # Load all datasets once
    all_dfs = {}
    for dataset_id in DATASETS:
        all_dfs[dataset_id] = load_events(dataset_id)
        print(f"{dataset_id}: {len(all_dfs[dataset_id])} events")

    # Generate all versions
    for x_col, x_label, x_log, y_col, y_label, y_log, name in FIGURE_VERSIONS:
        fname = f"{FIG_OUTPUT_DIR}/{name}_scatter.png"
        print(f"\nGenerating: {name}")
        plot_version(all_dfs, x_col, x_label, x_log, y_col, y_label, y_log, fname)


if __name__ == '__main__':
    plot_figure()
