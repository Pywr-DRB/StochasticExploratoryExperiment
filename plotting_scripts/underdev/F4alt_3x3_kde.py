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

SCENARIOS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']
WINDOW_MONTHS = [9]
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
    3-row × 3-col KDE grid.  Each row = dataset, each col = zone group.
    """
    # Per-column x_max: 95th percentile of emergency data (col 2), then match cols 0-1
    emg_vals = []
    for sc in SCENARIOS:
        r = _get_col_ratios(all_categorized[sc], 'emergency')
        if r is not None:
            emg_vals.extend(r.values)
    x_max_emg = float(np.percentile(emg_vals, 95)) if emg_vals else 100.0
    x_max_emg = max(x_max_emg, 100.0)

    # Each column gets its own x_max based on 99th pct of that group's data
    col_x_max = {}
    for col in COL_ZONES:
        vals = []
        for sc in SCENARIOS:
            r = _get_col_ratios(all_categorized[sc], col)
            if r is not None:
                vals.extend(r.values)
        col_x_max[col] = float(np.percentile(vals, 99)) if vals else x_max_emg

    # Precompute KDE grids
    x_grids = {col: np.linspace(0, col_x_max[col], N_KDE_POINTS) for col in COL_ZONES}
    kdes = {}
    for sc in SCENARIOS:
        for col in COL_ZONES:
            r = _get_col_ratios(all_categorized[sc], col)
            kdes[(sc, col)] = _kde(r, x_grids[col])

    apply_publication_style()
    fig, axes = plt.subplots(
        3, 3,
        figsize=(9, 9),
        sharey=True,
        sharex=True,
        constrained_layout=True,
    )

    for row_i, sc in enumerate(SCENARIOS):
        color = DATASET_COLORS[sc]
        for col_i, col in enumerate(COL_ZONES):
            ax = axes[row_i, col_i]
            x = x_grids[col]
            y = kdes[(sc, col)]

            ax.fill_between(x, y, alpha=0.75, color=color)
            ax.plot(x, y, color='black', linewidth=1.2)

            # 1964 reconstruction tick on emergency column only
            if col == 'emergency' and recon_ratio is not None and recon_ratio <= col_x_max[col]:
                ylim = ax.get_ylim()
                tick_h = (y.max()) * 0.12
                ax.vlines(recon_ratio, 0, tick_h, color='black', linewidth=3.0, zorder=5)

            ax.set_yticks([])
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 0.12)
            ax.set_aspect(100 / 0.12)  # square subplot
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_color('#444444')

            # Column headers on top row
            if row_i == 0:
                ax.set_title(COL_LABELS[col], fontsize=FONTSIZE_MEDIUM, pad=6)

            # x-axis label on bottom row only
            if row_i == len(SCENARIOS) - 1:
                ax.set_xlabel(
                    f'NYC contribution / inflow ({n_months_prior}-mo, %)',
                    fontsize=FONTSIZE_SMALL,
                )
            else:
                ax.set_xlabel('')
                ax.tick_params(axis='x', labelbottom=False)

    # Row labels (dataset names) on left
    for row_i, sc in enumerate(SCENARIOS):
        axes[row_i, 0].set_ylabel(
            DATASET_LABELS[sc],
            fontsize=FONTSIZE_MEDIUM,
            rotation=90,
            labelpad=8,
        )

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
