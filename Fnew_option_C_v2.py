"""
Figure Option C v2: Drought Event Severity-Outcome Relationships

Key changes from v1:
  - Uses SSI-3 events for higher data density (~170-210 events/scenario)
  - 2x2 layout instead of 1x3 (taller panels, more readable)
  - Added contribution_ratio panel (pre-computed, not raw volume)
  - Trend lines labeled as "2nd-order polynomial fit" in legend
  - SSI window clearly labeled
  - Markers per scenario match existing styles (DATASET_MARKERS)
  - Classification panel removed (sparse categories); replaced with
    diversion satisfaction panel

Panels:
  (a) Severity vs. Minimum Storage During Drought
  (b) Severity vs. NYC Contribution / Inflow Ratio
  (c) Severity vs. Max Consecutive Montague Shortage Days
  (d) Duration vs. Minimum Storage During Drought

Usage:
    python Fnew_option_C_v2.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

from methods.config import ROOT_DIR, FIG_DIR
from methods.plotting.styles import (
    DATASET_COLORS, DATASET_LABELS, DATASET_MARKERS, DATASET_LINESTYLES,
    FONTSIZE_SMALL, FONTSIZE_MEDIUM, FONTSIZE_LARGE, FONTSIZE_LABEL,
    LINEWIDTH_MEDIUM, LINEWIDTH_THICK,
    DPI_HIGH, apply_publication_style,
)

FIG_OUTPUT_DIR = f"{FIG_DIR}/Fnew_event_severity_outcomes"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

DATASETS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']
SSI_WINDOW = 3  # Using SSI-3 for maximum event count


def load_event_metrics(dataset_id, ssi_window=SSI_WINDOW):
    fname = f'{ROOT_DIR}/pywrdrb/event_metrics/{dataset_id}_ssi{ssi_window}_event_metrics.csv'
    return pd.read_csv(fname)


def add_poly_fit(ax, x, y, color, degree=2):
    """Add polynomial regression line with R annotation."""
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 10:
        return
    xv, yv = x[valid], y[valid]
    z = np.polyfit(xv, yv, degree)
    p = np.poly1d(z)
    x_line = np.linspace(np.nanmin(xv), np.nanmax(xv), 100)
    ax.plot(x_line, p(x_line), color=color,
            linewidth=LINEWIDTH_THICK, alpha=0.85, zorder=5)

    r, _ = pearsonr(xv, yv)
    return r


def make_scatter_panel(ax, events_dict, x_col, y_col, xlabel, ylabel,
                       panel_label, ylim=None, ref_line=None, ref_label=None,
                       x_transform=None, y_transform=None, show_fit=True,
                       show_r=True):
    """Create a scatter panel with polynomial fit lines per scenario."""
    r_values = {}

    for did in DATASETS:
        df = events_dict[did]
        color = DATASET_COLORS[did]
        marker = DATASET_MARKERS.get(did, 'o')

        x = df[x_col].values.copy()
        y = df[y_col].values.copy()
        if x_transform:
            x = x_transform(x)
        if y_transform:
            y = y_transform(y)

        ax.scatter(x, y, c=color, marker=marker, s=25, alpha=0.45,
                   edgecolors='white', linewidths=0.3, zorder=3)

        if show_fit:
            r = add_poly_fit(ax, x, y, color)
            if r is not None:
                r_values[did] = r

    # Show R values
    if show_r and r_values:
        r_text_parts = []
        for did in DATASETS:
            if did in r_values:
                short = DATASET_LABELS[did].split()[0]  # First word
                r_text_parts.append(f'{short}: r={r_values[did]:.2f}')
        r_text = '\n'.join(r_text_parts)
        ax.text(0.97, 0.03, r_text, transform=ax.transAxes,
                fontsize=8, va='bottom', ha='right', color='#444444',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.85, edgecolor='#cccccc'))

    if ref_line is not None:
        ax.axhline(ref_line, color='#c62828', linewidth=1.0,
                   linestyle='--', alpha=0.5)
        if ref_label:
            ax.text(ax.get_xlim()[0] + 0.05, ref_line + 1, ref_label,
                    fontsize=8, color='#c62828', alpha=0.6)

    ax.set_xlabel(xlabel, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL)
    ax.grid(True, alpha=0.12, linestyle='--')
    ax.set_axisbelow(True)
    ax.text(0.03, 0.97, panel_label, transform=ax.transAxes,
            fontsize=14, va='top', fontweight='bold')
    if ylim:
        ax.set_ylim(ylim)


def plot_figure():
    apply_publication_style()
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12})

    events = {d: load_event_metrics(d) for d in DATASETS}

    # Print summary
    for did in DATASETS:
        n = len(events[did])
        print(f'{DATASET_LABELS[did]}: {n} SSI-{SSI_WINDOW} drought events')

    # ====================================================================
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(
        2, 2,
        hspace=0.30, wspace=0.28,
        left=0.09, right=0.97, top=0.94, bottom=0.09,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # ====================================================================
    # Panel (a): Magnitude vs Min Storage
    # ====================================================================
    make_scatter_panel(
        ax_a, events,
        x_col='magnitude', y_col='event_min_storage_pct',
        xlabel=f'Drought Magnitude (SSI-{SSI_WINDOW})',
        ylabel='Minimum NYC Storage\nDuring Drought (%)',
        panel_label='(a)',
        ylim=(0, 105),
        ref_line=20, ref_label='Emergency threshold',
    )

    # ====================================================================
    # Panel (b): Magnitude vs Contribution Ratio
    # ====================================================================
    make_scatter_panel(
        ax_b, events,
        x_col='magnitude', y_col='contribution_ratio',
        xlabel=f'Drought Magnitude (SSI-{SSI_WINDOW})',
        ylabel='NYC Contribution /\nInflow Ratio',
        panel_label='(b)',
        ylim=(0, 1.0),
    )

    # ====================================================================
    # Panel (c): Magnitude vs Montague Shortage Days
    # ====================================================================
    make_scatter_panel(
        ax_c, events,
        x_col='magnitude', y_col='max_consec_montague_days',
        xlabel=f'Drought Magnitude (SSI-{SSI_WINDOW})',
        ylabel='Max Consecutive Montague\nShortage Days',
        panel_label='(c)',
        show_fit=True,
    )

    # ====================================================================
    # Panel (d): Duration vs Min Storage
    # ====================================================================
    make_scatter_panel(
        ax_d, events,
        x_col='duration_days', y_col='event_min_storage_pct',
        xlabel=f'Drought Duration (days)',
        ylabel='Minimum NYC Storage\nDuring Drought (%)',
        panel_label='(d)',
        ylim=(0, 105),
        ref_line=20, ref_label='Emergency threshold',
    )

    # ====================================================================
    # SSI window label
    # ====================================================================
    fig.text(0.5, 0.98,
             f'SSI-{SSI_WINDOW} Drought Events',
             ha='center', fontsize=13, fontweight='bold')

    # ====================================================================
    # Legend
    # ====================================================================
    handles = []
    for did in DATASETS:
        handles.append(
            Line2D([0], [0],
                   marker=DATASET_MARKERS.get(did, 'o'),
                   color=DATASET_COLORS[did],
                   linewidth=LINEWIDTH_THICK,
                   markersize=6,
                   markerfacecolor=DATASET_COLORS[did],
                   label=f'{DATASET_LABELS[did]} (n={len(events[did])})'))
    handles.append(
        Line2D([0], [0], color='grey', linewidth=LINEWIDTH_THICK,
               alpha=0.85, label='Polynomial fit (degree 2)'))

    fig.legend(handles=handles, loc='lower center', ncol=4,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.01))

    fname = f"{FIG_OUTPUT_DIR}/option_C_v2_ssi{SSI_WINDOW}.png"
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close(fig)


if __name__ == '__main__':
    plot_figure()
