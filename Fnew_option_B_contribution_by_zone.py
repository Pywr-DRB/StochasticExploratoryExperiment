"""
Figure Option B: NYC Contribution Ratio by FFMP Drought Zone.

Supports two modes (set via MODE constant):
  - 'annual'     : 270-day contribution ratio grouped by worst FFMP zone per water year.
  - 'ssi_events' : Per-event contribution ratio during SSI drought events,
                    grouped by FFMP zone at minimum storage. Events filtered
                    by MIN_DURATION.

Box plots with underlying scatter, all 3 scenarios side-by-side.

Usage:
    python Fnew_option_B_contribution_by_zone.py
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
    DATASET_COLORS, DATASET_LABELS, DATASET_LINESTYLES,
    FONTSIZE_SMALL, FONTSIZE_MEDIUM, FONTSIZE_LABEL,
    DPI_HIGH, apply_publication_style,
)

FIG_OUTPUT_DIR = f"{FIG_DIR}/Fnew_contribution_by_zone"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

DATASETS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']

ZONE_LABELS = {
    3: 'Normal',
    4: 'Drought\nWatch',
    5: 'Drought\nWarning',
    6: 'Drought\nEmergency',
}

ZONES_TO_PLOT = [3, 4, 5, 6]

WINDOW = '270d'  # 9-month window prior to min storage date

# ── Mode switch ──────────────────────────────────────────────────────
# 'annual'     : original behavior (contribution_ratio_270d grouped by FFMP zone)
# 'ssi_events' : SSI drought event contribution_ratio grouped by FFMP zone at min storage
MODE = 'ssi_events'

# ── SSI event mode settings ──────────────────────────────────────────
SSI_WINDOW = 6
MIN_DURATION = 30

EVENT_ZONE_ORDER = ['Normal', 'Watch', 'Warning', 'Emergency']
EVENT_ZONE_LABELS = {
    'Normal': 'Normal',
    'Watch': 'Drought\nWatch',
    'Warning': 'Drought\nWarning',
    'Emergency': 'Drought\nEmergency',
}


def load_contribution_metrics(dataset_id):
    """Load pre-computed annual contribution metrics."""
    df = pd.read_csv(f'{ROOT_DIR}/pywrdrb/performance_metrics/{dataset_id}_contribution_metrics.csv')
    df = df.dropna(subset=[f'contribution_ratio_{WINDOW}', 'annual_max_zone'])
    df['zone'] = df['annual_max_zone'].astype(int)
    df['ratio'] = df[f'contribution_ratio_{WINDOW}']
    return df


def load_event_metrics(dataset_id):
    """Load SSI drought event metrics, filter by duration, assign FFMP zone group."""
    df = pd.read_csv(
        f'{ROOT_DIR}/pywrdrb/event_metrics/{dataset_id}_ssi{SSI_WINDOW}_event_metrics.csv'
    )
    df = df[df['duration_days'] >= MIN_DURATION].copy()
    df['ratio'] = df['contribution_ratio'] * 100.0
    df['zone'] = df['ffmp_zone_at_min']
    df = df.dropna(subset=['zone', 'ratio'])
    return df


def plot_figure():
    apply_publication_style()
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12})

    # ── Mode-specific setup ────────────────────────────────────────────
    if MODE == 'ssi_events':
        data = {did: load_event_metrics(did) for did in DATASETS}
        groups_to_plot = EVENT_ZONE_ORDER
        group_labels = EVENT_ZONE_LABELS
        xlabel = f'FFMP Zone at Min Storage (SSI-{SSI_WINDOW} Drought Events)'
        ylabel = 'NYC Contribution / Total Inflow (%)\n(per drought event)'
        output_fname = f"{FIG_OUTPUT_DIR}/option_B_contribution_by_event_zone.png"
    else:
        data = {did: load_contribution_metrics(did) for did in DATASETS}
        groups_to_plot = ZONES_TO_PLOT
        group_labels = ZONE_LABELS
        xlabel = 'Worst FFMP Zone Reached (Water Year)'
        ylabel = 'NYC Contribution / Total Inflow (%)\n(270-day window prior to min storage)'
        output_fname = f"{FIG_OUTPUT_DIR}/option_B_contribution_by_zone.png"

    n_groups = len(groups_to_plot)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Layout
    n_scenarios = len(DATASETS)
    box_width = 0.22
    group_gap = 0.35
    group_width = n_scenarios * box_width + group_gap

    # Compute positions
    positions = {}
    for s_idx, did in enumerate(DATASETS):
        positions[did] = []
        for g_idx in range(n_groups):
            center = g_idx * group_width
            offset = (s_idx - (n_scenarios - 1) / 2) * (box_width + 0.03)
            positions[did].append(center + offset)

    # Plot scatter + box for each scenario
    for s_idx, did in enumerate(DATASETS):
        color = DATASET_COLORS[did]
        box_data = []

        for g_idx, group in enumerate(groups_to_plot):
            group_vals = data[did][data[did]['zone'] == group]['ratio'].values
            box_data.append(group_vals)

            # Scatter points (jittered)
            if len(group_vals) > 0:
                jitter = np.random.default_rng(42 + s_idx * 100 + g_idx).uniform(
                    -box_width * 0.35, box_width * 0.35, size=len(group_vals))
                ax.scatter(
                    positions[did][g_idx] + jitter,
                    group_vals,
                    color=color, alpha=0.25, s=12, zorder=2,
                    edgecolors='none',
                )

        # Box plots on top
        bp = ax.boxplot(
            box_data,
            positions=positions[did],
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color='black', linewidth=1.8),
            whiskerprops=dict(color=color, linewidth=1.3, alpha=0.8),
            capprops=dict(color=color, linewidth=1.3, alpha=0.8),
            boxprops=dict(facecolor=color, alpha=0.30, edgecolor=color, linewidth=1.3),
            zorder=3,
        )

    # X-axis
    group_centers = [g_idx * group_width for g_idx in range(n_groups)]
    ax.set_xticks(group_centers)
    ax.set_xticklabels([group_labels[g] for g in groups_to_plot], fontsize=FONTSIZE_MEDIUM)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE_LABEL)

    # Y-axis
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL)
    ax.set_ylim(bottom=-1)
    if MODE == 'ssi_events':
        all_ratios = pd.concat([d['ratio'] for d in data.values()])
        upper = min(all_ratios.quantile(0.98) * 1.15, 150)
        ax.set_ylim(top=upper)
    ax.grid(True, axis='y', alpha=0.15, linestyle='--')
    ax.set_axisbelow(True)

    # Sample size annotations
    for g_idx, group in enumerate(groups_to_plot):
        counts = []
        for did in DATASETS:
            n = len(data[did][data[did]['zone'] == group])
            counts.append(str(n))
        label = f'n = {"/".join(counts)}'
        ax.text(group_centers[g_idx], ax.get_ylim()[1] * 0.97, label,
                ha='center', va='top', fontsize=7.5, color='#666666', style='italic')

    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=DATASET_COLORS[did], alpha=0.30,
                       edgecolor=DATASET_COLORS[did], linewidth=1.3,
                       label=DATASET_LABELS[did])
        for did in DATASETS
    ]
    ax.legend(handles=handles, loc='upper left', fontsize=10, frameon=True,
              framealpha=0.9, edgecolor='#cccccc')

    plt.tight_layout()
    fig.savefig(output_fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {output_fname}")
    plt.close(fig)


if __name__ == '__main__':
    plot_figure()
