"""
F4 (Alternative): NYC contribution / inflow ratio — 3x3 KDE grid.

3 rows (one per dataset) × 3 columns (storage zone groups):
  Col 1: Normal or Flood        (zones 0-3)
  Col 2: Drought Watch/Warning  (zones 4-5)
  Col 3: Drought Emergency      (zone 6)

Each subplot shows a filled KDE of the NYC Montague contribution-to-inflow
ratio, coloured by dataset.  One figure is produced per aggregation window.

Usage:
    python F4alt_3x3_kde.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from methods.config import FIG_DIR, OUTPUT_DIR, verify_dataset_id
from methods.plotting.styles import (
    DPI_HIGH,
    DATASET_COLORS, DATASET_LABELS,
    FONTSIZE_LABEL, FONTSIZE_MEDIUM, FONTSIZE_SMALL,
    apply_publication_style,
)
import methods.plotting.water_balance_by_drought_zone as F4_module
from methods.plotting.water_balance_by_drought_zone import (
    aggregate_across_realizations,
    categorize_by_drought_zone,
    calculate_reconstruction_contribution_ratio,
    MIN_INFLOW_THRESHOLD,
)

# ============================================================================
# CONFIGURATION
# ============================================================================
xlims = {3: (0, 500), 6: (0, 300), 9: (0, 100)}

SCENARIOS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']
WINDOW_MONTHS = [3, 6, 9]
FIG_OUTPUT_DIR = f"{FIG_DIR}/F4alt_kde"

# Column zone groupings
COL_ZONES  = ['other', 'watch_warning', 'emergency']
COL_LABELS = {
    'other':         'Normal or Flood',
    'watch_warning': 'Drought Watch\nor Warning',
    'emergency':     'Drought\nEmergency',
}

# Original zone categories passed to categorize_by_zone
ZONE_CATEGORIES = {
    'emergency':    [6],
    'watch':        [5],
    'warning':      [4],
    'other':        [0, 1, 2, 3],
}

N_KDE_POINTS = 500

# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_data():
    all_data = {}
    for sc in SCENARIOS:
        verify_dataset_id(sc)
        fname = f'{OUTPUT_DIR}/{sc}_with_postprocessing.hdf5'
        data = pywrdrb.Data()
        data.load_from_export(fname, results_sets=[
            'res_level', 'inflow', 'contribution',
            'res_storage', 'ibt_diversions', 'ibt_demands',
        ])
        all_data[sc] = data
    return all_data


def categorize_all_scenarios(all_data, n_months_prior):
    F4_module.N_MONTHS_PRIOR = n_months_prior
    all_categorized = {}
    for sc in SCENARIOS:
        agg = aggregate_across_realizations(all_data[sc], sc)
        all_categorized[sc] = categorize_by_drought_zone(agg)
    return all_categorized


# ============================================================================
# DATA HELPERS
# ============================================================================

def _get_ratios(categorized, zone):
    """Return contribution / inflow ratio (%) for a single zone, filtered."""
    df = categorized[zone]
    df_f = df[df['inflow_total'] > MIN_INFLOW_THRESHOLD].copy()
    if len(df_f) < 2:
        return None
    ratio = (100.0 * df_f['contribution_total'] / df_f['inflow_total'])
    return ratio.replace([np.inf, -np.inf], np.nan).dropna()


def _get_col_ratios(categorized, col_zone):
    """Return combined ratios for a column zone group."""
    if col_zone == 'watch_warning':
        parts = []
        for z in ('watch', 'warning'):
            r = _get_ratios(categorized, z)
            if r is not None:
                parts.append(r)
        if not parts:
            return None
        import pandas as pd
        return pd.concat(parts, ignore_index=True)
    else:
        return _get_ratios(categorized, col_zone)


def _kde(data, x_grid):
    if data is None or len(data) < 2:
        return np.zeros_like(x_grid)
    return gaussian_kde(data.values)(x_grid)


# ============================================================================
# FIGURE
# ============================================================================

def create_figure(all_categorized, n_months_prior, recon_ratio):
    """
    3-row × 1-col KDE grid.  Each row = zone group, all datasets overlaid.
    """
    xmax = xlims.get(n_months_prior, 100)[1]
    x_grid = np.linspace(0, xmax, N_KDE_POINTS)

    # Precompute KDEs for all (scenario, zone) combinations
    kdes = {}
    for sc in SCENARIOS:
        for col in COL_ZONES:
            r = _get_col_ratios(all_categorized[sc], col)
            kdes[(sc, col)] = _kde(r, x_grid)

    apply_publication_style()
    fig, axes = plt.subplots(
        3, 1,
        figsize=(5, 9),
        sharex=True,
        constrained_layout=True,
    )

    for row_i, col in enumerate(COL_ZONES):
        ax = axes[row_i]

        for sc in SCENARIOS:
            color = DATASET_COLORS[sc]
            y = kdes[(sc, col)]
            ax.fill_between(x_grid, y, alpha=0.4, color=color)
            ax.plot(x_grid, y, color=color, linewidth=1.5)

        # 1964 reconstruction tick on emergency row only
        if col == 'emergency' and recon_ratio is not None and recon_ratio <= 100:
            peak = max(kdes[(sc, col)].max() for sc in SCENARIOS)
            ax.vlines(recon_ratio, 0, peak * 0.15, color='black', linewidth=3.0, zorder=5)

        ax.set_yticks([])
        
        
        xmax = xlims.get(n_months_prior, 100)[1]
        
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 0.12)
        ax.set_title(COL_LABELS[col], fontsize=FONTSIZE_MEDIUM, pad=4)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color('#444444')

    axes[-1].set_xlabel(
        f'NYC contribution / inflow ({n_months_prior}-mo, %)',
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

    print("F4 alt 3x3: NYC contribution/inflow KDE grid")
    print("=" * 70)

    use_cached = True
    try:
        from methods.load import load_contribution_metrics
        from methods.metrics.contribution import get_metrics_for_window, categorize_by_zone
        metrics_cache = {sc: load_contribution_metrics(sc) for sc in SCENARIOS}
    except (ImportError, FileNotFoundError):
        use_cached = False

    if not use_cached:
        all_data = load_all_data()

    for n_mo in WINDOW_MONTHS:
        print(f"\n  {n_mo}-month window ...")
        F4_module.N_MONTHS_PRIOR = n_mo

        if use_cached:
            window_days = n_mo * 30
            col_map = {
                f'contribution_total_{window_days}d': 'contribution_total',
                f'contribution_ratio_{window_days}d': 'contribution_ratio',
                f'inflow_total_{window_days}d':       'inflow_total',
                f'demand_satisfaction_{window_days}d': 'demand_satisfaction',
                f'worst_1mo_demand_sat_{window_days}d': 'worst_1mo_demand_sat',
            }
            all_categorized = {}
            for sc in SCENARIOS:
                df = get_metrics_for_window(metrics_cache[sc], window_days)
                df = df.rename(columns=col_map)
                all_categorized[sc] = categorize_by_zone(df, ZONE_CATEGORIES)
        else:
            all_categorized = categorize_all_scenarios(all_data, n_mo)

        recon_ratio = calculate_reconstruction_contribution_ratio()

        fig = create_figure(all_categorized, n_mo, recon_ratio)
        fname = f"{FIG_OUTPUT_DIR}/F4alt_3x3_kde_{n_mo}mo.png"
        fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
        print(f"    Saved: {fname}")
        plt.close(fig)

    print("\n" + "=" * 70)
    print("F4 alt 3x3 figures generated successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
