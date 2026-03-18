"""
Figure Option B: NYC Contribution Ratio by FFMP Drought Zone.

Box plots with underlying scatter showing annual NYC contribution / total inflow
(270-day window prior to min storage) grouped by the worst FFMP zone reached
that water year. All 3 scenarios side-by-side.

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


def load_contribution_metrics(dataset_id):
    """Load pre-computed annual contribution metrics."""
    df = pd.read_csv(f'{ROOT_DIR}/pywrdrb/performance_metrics/{dataset_id}_contribution_metrics.csv')
    df = df.dropna(subset=[f'contribution_ratio_{WINDOW}', 'annual_max_zone'])
    df['zone'] = df['annual_max_zone'].astype(int)
    df['ratio'] = df[f'contribution_ratio_{WINDOW}']
    return df


def plot_figure():
    apply_publication_style()
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12})

    # Load data
    data = {}
    for did in DATASETS:
        data[did] = load_contribution_metrics(did)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Layout
    n_scenarios = len(DATASETS)
    n_zones = len(ZONES_TO_PLOT)
    box_width = 0.22
    group_gap = 0.35
    group_width = n_scenarios * box_width + group_gap

    # Compute positions
    positions = {}
    for s_idx, did in enumerate(DATASETS):
        positions[did] = []
        for z_idx in range(n_zones):
            center = z_idx * group_width
            offset = (s_idx - (n_scenarios - 1) / 2) * (box_width + 0.03)
            positions[did].append(center + offset)

    # Plot scatter + box for each scenario
    for s_idx, did in enumerate(DATASETS):
        color = DATASET_COLORS[did]
        box_data = []

        for z_idx, zone in enumerate(ZONES_TO_PLOT):
            zone_vals = data[did][data[did]['zone'] == zone]['ratio'].values
            box_data.append(zone_vals)

            # Scatter points (jittered)
            if len(zone_vals) > 0:
                jitter = np.random.default_rng(42 + s_idx * 100 + z_idx).uniform(
                    -box_width * 0.35, box_width * 0.35, size=len(zone_vals))
                ax.scatter(
                    positions[did][z_idx] + jitter,
                    zone_vals,
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
    zone_centers = [z_idx * group_width for z_idx in range(n_zones)]
    ax.set_xticks(zone_centers)
    ax.set_xticklabels([ZONE_LABELS[z] for z in ZONES_TO_PLOT], fontsize=FONTSIZE_MEDIUM)
    ax.set_xlabel('Worst FFMP Zone Reached (Water Year)', fontsize=FONTSIZE_LABEL)

    # Y-axis
    ax.set_ylabel('NYC Contribution / Total Inflow (%)\n(270-day window prior to min storage)',
                   fontsize=FONTSIZE_LABEL)
    ax.set_ylim(bottom=-1)
    ax.grid(True, axis='y', alpha=0.15, linestyle='--')
    ax.set_axisbelow(True)

    # Sample size annotations
    for z_idx, zone in enumerate(ZONES_TO_PLOT):
        counts = []
        for did in DATASETS:
            n = len(data[did][data[did]['zone'] == zone])
            counts.append(str(n))
        label = f'n = {"/".join(counts)}'
        ax.text(zone_centers[z_idx], ax.get_ylim()[1] * 0.97, label,
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
    fname = f"{FIG_OUTPUT_DIR}/option_B_contribution_by_zone.png"
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close(fig)


if __name__ == '__main__':
    plot_figure()
