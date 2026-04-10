"""
F4 (Alternative): NYC contribution / inflow ratio — KDE by SSI-3 drought events.

3 rows (one per FFMP zone group reached during SSI-3 drought events):
  Row 1: Normal           (ffmp_zone_at_min == 'Normal')
  Row 2: Watch / Warning  (ffmp_zone_at_min in ['Watch', 'Warning'])
  Row 3: Emergency        (ffmp_zone_at_min == 'Emergency')

Each subplot overlays filled KDEs for all three datasets (scenarios),
matching the visual design of F4alt_3x3_kde.py but using SSI-3 drought
event periods as the unit of analysis instead of water years.

Usage:
    python F4alt_kde_by_ssi3_events.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings("ignore")

from methods.config import FIG_DIR, ROOT_DIR
from methods.plotting.styles import (
    DPI_HIGH,
    DATASET_COLORS, DATASET_LABELS,
    FONTSIZE_LABEL, FONTSIZE_MEDIUM, FONTSIZE_SMALL,
    apply_publication_style,
)
from methods.plotting.water_balance_by_drought_zone import (
    calculate_reconstruction_contribution_ratio,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

SCENARIOS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']
SSI_WINDOW = 3
MIN_DURATION_DAYS = 30
MIN_INFLOW_THRESHOLD = 1000  # MG

FIG_OUTPUT_DIR = f"{FIG_DIR}/F4alt_kde"
EVENT_METRICS_DIR = os.path.join(ROOT_DIR, 'archive', 'pywrdrb', 'event_metrics')

# Row definitions: key -> list of ffmp_zone_at_min values to include
ROW_ZONES = ['normal', 'watch_warning', 'emergency']
ROW_ZONE_FILTER = {
    'normal':        ['Normal'],
    'watch_warning': ['Watch', 'Warning'],
    'emergency':     ['Emergency'],
}
ROW_LABELS = {
    'normal':        'Normal or Flood',
    'watch_warning': 'Drought Watch\nor Warning',
    'emergency':     'Drought\nEmergency',
}

N_KDE_POINTS = 500
XLIM = (0, 500)  # contribution ratio in %

# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_event_metrics():
    """Load SSI-3 event metrics for all scenarios from archive."""
    all_metrics = {}
    for sc in SCENARIOS:
        fname = os.path.join(EVENT_METRICS_DIR, f'{sc}_ssi{SSI_WINDOW}_event_metrics.csv')
        df = pd.read_csv(fname)
        df = df[df['duration_days'] >= MIN_DURATION_DAYS].copy()
        df['severity'] = df['severity'].abs()
        df['magnitude'] = df['magnitude'].abs()
        all_metrics[sc] = df
    return all_metrics


def _get_ratios(df, zone_labels):
    """Return contribution ratio (%) for events matching given ffmp_zone_at_min values."""
    mask = df['ffmp_zone_at_min'].isin(zone_labels)
    filtered = df[mask]
    if 'total_inflow_mg' in filtered.columns:
        filtered = filtered[filtered['total_inflow_mg'] > MIN_INFLOW_THRESHOLD]
    if len(filtered) < 2:
        return None
    ratio = filtered['contribution_ratio'] * 100.0  # fraction -> %
    return ratio.replace([np.inf, -np.inf], np.nan).dropna()


def _kde(data, x_grid):
    if data is None or len(data) < 2:
        return np.zeros_like(x_grid)
    return gaussian_kde(data.values)(x_grid)


# ============================================================================
# FIGURE
# ============================================================================

def create_figure(all_metrics, recon_ratio):
    """3-row KDE grid. Each row = FFMP zone group, all datasets overlaid."""
    x_grid = np.linspace(XLIM[0], XLIM[1], N_KDE_POINTS)

    # Precompute KDEs
    kdes = {}
    for sc in SCENARIOS:
        for row in ROW_ZONES:
            r = _get_ratios(all_metrics[sc], ROW_ZONE_FILTER[row])
            kdes[(sc, row)] = _kde(r, x_grid)

    apply_publication_style()
    fig, axes = plt.subplots(
        3, 1,
        figsize=(5, 9),
        sharex=False,
        constrained_layout=True,
    )

    for row_i, row in enumerate(ROW_ZONES):
        ax = axes[row_i]

        for sc in SCENARIOS:
            color = DATASET_COLORS[sc]
            y = kdes[(sc, row)]
            ax.fill_between(x_grid, y, alpha=0.4, color=color)
            ax.plot(x_grid, y, color=color, linewidth=1.5)

        # 1964 reconstruction tick on emergency row only
        if row == 'emergency' and recon_ratio is not None and recon_ratio <= XLIM[1]:
            peak = max(kdes[(sc, row)].max() for sc in SCENARIOS)
            if peak > 0:
                ax.vlines(recon_ratio, 0, peak * 0.15, color='black', linewidth=3.0, zorder=5)

        ax.set_yticks([])
        # ax.set_xlim(*XLIM)
        ax.set_title(ROW_LABELS[row], fontsize=FONTSIZE_MEDIUM, pad=4)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color('#444444')

    axes[-1].set_xlabel(
        'NYC contribution / inflow (event period, %)',
        fontsize=FONTSIZE_LABEL,
    )

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=DATASET_COLORS[sc], linewidth=2.0, label=DATASET_LABELS[sc])
        for sc in SCENARIOS
    ]
    axes[0].legend(handles=handles, fontsize=FONTSIZE_SMALL, frameon=False,
                   loc='upper right')

    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    apply_publication_style()
    os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

    print("F4 alt SSI-3 events: NYC contribution/inflow KDE by drought event")
    print("=" * 70)

    all_metrics = load_all_event_metrics()

    for sc in SCENARIOS:
        n = len(all_metrics[sc])
        zone_counts = all_metrics[sc]['ffmp_zone_at_min'].value_counts()
        print(f"  {sc}: {n} events — {zone_counts.to_dict()}")

    recon_ratio = calculate_reconstruction_contribution_ratio()

    fig = create_figure(all_metrics, recon_ratio)
    fname = f"{FIG_OUTPUT_DIR}/F4alt_kde_by_ssi3_events.png"
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"\n  Saved: {fname}")
    plt.close(fig)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == '__main__':
    main()
