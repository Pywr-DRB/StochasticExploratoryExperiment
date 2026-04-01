"""
F3 (Alternative): Drought zone occurrence figure.

3-panel layout focused on drought zone occurrence:
  Left:  A  (Temporal P(FFMP Drought Zone) by water-year week)
  Right: B1 (frequency box plots) | B2 (duration box plots)

Panel A shows weekly probability of being in any FFMP drought zone
(Emergency OR Warning OR Watch) across the water year, for each
climate scenario.  Panels B1/B2 are identical to the original F3.

Usage:
    python F3_plot_drought_zone_occurrence.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings("ignore")

from methods.config import FIG_DIR, ZONE_PROB_DIR
from methods.plotting.styles import (
    DPI_HIGH,
    DATASET_COLORS, DATASET_LABELS,
    FONTSIZE_SMALL, FONTSIZE_LABEL, FONTSIZE_MEDIUM,
    apply_publication_style, label_panel,
)

# Drought zone boxplots (modular, in methods/plotting/)
from methods.plotting.drought_zone_boxplots import (
    plot_frequency_boxplot,
    plot_duration_boxplot,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

SCENARIOS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']

FIG_OUTPUT_DIR = f"{FIG_DIR}/F3_zone_occurence"

# Water-year axis constants (Jun-May)
WY_MONTH_STARTS = [1, 5, 9, 14, 18, 23, 27, 32, 36, 40, 45, 49]
WY_MONTH_LABELS = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',
                    'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']


# ============================================================================
# DATA LOADING
# ============================================================================

def load_zone_probs(dataset_id):
    """Load zone probabilities CSV (period is WY-ordered, period 1 = June)."""
    return pd.read_csv(
        f'{ZONE_PROB_DIR}/{dataset_id}_zone_probs_weekly.csv',
        index_col='period')


def smooth(series, window=3):
    """Rolling mean smoother."""
    return series.rolling(window, center=True, min_periods=1).mean()


def format_wy_xaxis(ax):
    """Format x-axis as water year months (Jun-May)."""
    ax.set_xticks(WY_MONTH_STARTS)
    ax.set_xticklabels(WY_MONTH_LABELS, fontsize=FONTSIZE_MEDIUM)
    ax.set_xlim(0.5, 52.5)


# ============================================================================
# PANEL A: Temporal P(FFMP Drought Zone)
# ============================================================================

def plot_panel_A_zone_probability(ax):
    """
    Plot temporal P(FFMP Drought Zone) across the water year.

    Loads zone probability data for all scenarios, computes
    P(drought) = zone_0 + zone_1 + zone_2 (Emergency + Warning + Watch),
    smooths with 3-week rolling mean, and plots one line per scenario.
    """
    zones = {d: load_zone_probs(d) for d in SCENARIOS}

    # P(drought) = P(Emergency) + P(Warning) + P(Watch)
    p_drought = {}
    for d in SCENARIOS:
        zp = zones[d]
        raw = zp['zone_0'] + zp['zone_1'] + zp['zone_2']
        p_drought[d] = smooth(raw, window=3)

    # Plot lines
    for did in SCENARIOS:
        w = p_drought[did].index.values
        color = DATASET_COLORS[did]
        ax.plot(w, p_drought[did].values, color=color,
                linewidth=2.5, linestyle='-', alpha=0.90, zorder=3)

    # Axis formatting
    ax.set_ylabel('Probability NYC Reservoirs are Within\nDrought Watch, Warning or Emergency Zones (%)', fontsize=FONTSIZE_LABEL)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    format_wy_xaxis(ax)
    ax.set_xlabel('Month', fontsize=FONTSIZE_LABEL)

    # Panel label (inside axes, matching F4 style)
    label_panel(ax, 'a')



# ============================================================================
# LEGEND
# ============================================================================

def add_combined_legend(fig, show_historic=False):
    """
    Shared legend for all three panels.

    Includes:
    - Scenario lines (Panel A)
    - Dataset box patches (Panels B1/B2)
    - Mean marker and Historic marker (Panels B1/B2)
    """
    legend_elements = []

    # Scenario lines (for Panel A)
    for dataset_id in SCENARIOS:
        legend_elements.append(
            Line2D([0], [0], color=DATASET_COLORS[dataset_id],
                   linewidth=2.5, linestyle='-',
                   label=DATASET_LABELS[dataset_id]))

    # Dataset box patches (for Panels B1/B2)
    for dataset_id in SCENARIOS:
        legend_elements.append(
            Patch(facecolor=DATASET_COLORS[dataset_id], alpha=0.7,
                  edgecolor='black', linewidth=1.2,
                  label=DATASET_LABELS[dataset_id]))

    # Mean marker
    legend_elements.append(
        Line2D([0], [0], color='gray', marker='o', linestyle='None',
               markersize=6, markeredgecolor='white',
               markeredgewidth=0.8, label='Mean'))

    # Historic marker
    if show_historic:
        legend_elements.append(
            Line2D([0], [0], color='black', marker='^', linestyle='None',
                markersize=8, label='Historic'))

    fig.legend(handles=legend_elements, loc='lower center',
               ncol=4, fontsize=FONTSIZE_SMALL,
               frameon=False,
               bbox_to_anchor=(0.5, -0.04))


# ============================================================================
# FIGURE ASSEMBLY
# ============================================================================

def create_figure(show_historic=False):
    """
    Create the 3-panel drought zone occurrence figure.

    Layout (matching original F3 GridSpec):
      A (P(drought zone), left column) | B1 (frequency, top right)
                                       | B2 (duration, bottom right)
    """
    os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1, 1],
        width_ratios=[1.5, 1],
        hspace=0.25, wspace=0.35,
        left=0.10, right=0.95, top=0.92, bottom=0.18,
    )

    ax_A = fig.add_subplot(gs[0:2, 0])    # Zone prob spans left column
    ax_B1 = fig.add_subplot(gs[0, 1])     # Frequency (top right)
    ax_B2 = fig.add_subplot(gs[1, 1])     # Duration (bottom right)

    # Panel A: P(FFMP Drought Zone) temporal plot
    plot_panel_A_zone_probability(ax_A)

    # Panel B1: Frequency boxplot
    plot_frequency_boxplot(ax_B1, panel_label='b)', show_historic=show_historic)

    # Panel B2: Duration boxplot
    plot_duration_boxplot(ax_B2, panel_label='c)', show_historic=show_historic)

    # Shared x-axis label for right-side panels
    ax_B2.set_xlabel('NYC Reservoir Storage Zone', fontsize=FONTSIZE_LABEL)

    # Align y-axis labels for right-side panels
    label_x = -0.2
    for ax in [ax_B1, ax_B2]:
        ax.yaxis.set_label_coords(label_x, 0.5)

    # When historic markers are shown, expand y-axis limits to include
    # the full historic data range (boxplot functions may set tight limits).
    if show_historic:
        for ax in [ax_B1, ax_B2]:
            all_y = []
            for coll in ax.collections:
                offsets = coll.get_offsets()
                if len(offsets) > 0:
                    all_y.extend(offsets[:, 1].tolist())
            if all_y:
                current_top = ax.get_ylim()[1]
                data_max = max(all_y)
                if data_max > current_top:
                    ax.set_ylim(bottom=0, top=data_max * 1.1)

    # Combined legend
    add_combined_legend(fig, show_historic=show_historic)

    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    apply_publication_style()
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
    })

    print("F3 (alt): Drought zone occurrence figure")
    print("=" * 70)

    # Version without historic data
    fig = create_figure(show_historic=False)
    fname = f"{FIG_OUTPUT_DIR}/F3_zone_occurrence.png"
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close(fig)

    # Version with historic data
    fig = create_figure(show_historic=True)
    fname = f"{FIG_OUTPUT_DIR}/F3_zone_occurrence_with_historic.png"
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close(fig)

    print("\n" + "=" * 70)
    print("F3 zone occurrence figures generated successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
