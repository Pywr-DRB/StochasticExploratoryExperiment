"""
F4 (Alternative): NYC contribution / inflow ratio KDE by drought zone.

Single-panel figure showing KDE distributions of the NYC Montague
contribution-to-inflow ratio, coloured by the worst FFMP drought zone
reached in each water year.  One figure is produced per aggregation
window (3, 6, 9 months prior to the annual minimum-zone date).

Usage:
    python F4alt_nyc_contribution_kdes.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from methods.config import FIG_DIR, OUTPUT_DIR, verify_dataset_id
from methods.plotting.styles import (
    DPI_HIGH,
    apply_publication_style,
)

# Data-processing helpers (shared with F4 / SI composite scripts)
import methods.plotting.water_balance_by_drought_zone as F4_module
from methods.plotting.water_balance_by_drought_zone import (
    aggregate_across_realizations,
    categorize_by_drought_zone,
)

# Reusable KDE plotting function
from methods.plotting.contribution_kde import plot_kde_panel

# ============================================================================
# CONFIGURATION
# ============================================================================

SCENARIOS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']

WINDOW_MONTHS = [3, 6, 9]

FIG_OUTPUT_DIR = f"{FIG_DIR}/Fig7_kde"

# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_data():
    """Load pywrdrb.Data for all three scenarios."""
    all_data = {}
    results_sets = [
        'res_level', 'inflow', 'contribution',
        'res_storage', 'ibt_diversions', 'ibt_demands',
    ]
    for dataset_id in SCENARIOS:
        verify_dataset_id(dataset_id)
        fname = f'{OUTPUT_DIR}/{dataset_id}_with_postprocessing.hdf5'
        data = pywrdrb.Data()
        data.load_from_export(fname, results_sets=results_sets)
        all_data[dataset_id] = data
    return all_data


def categorize_all_scenarios(all_data, n_months_prior):
    """Aggregate and categorize by drought zone for all scenarios."""
    F4_module.N_MONTHS_PRIOR = n_months_prior
    all_categorized = {}
    for dataset_id in SCENARIOS:
        agg = aggregate_across_realizations(all_data[dataset_id], dataset_id)
        all_categorized[dataset_id] = categorize_by_drought_zone(agg)
    return all_categorized


# ============================================================================
# FIGURE
# ============================================================================

def create_figure(categorized_data, n_months_prior):
    """Create a single-panel KDE figure for the stationary ensemble."""
    fig, ax = plt.subplots(figsize=(7, 5))

    plot_kde_panel(ax, categorized_data, n_months_prior=n_months_prior,
                   panel_label='')

    fig.tight_layout()
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

    os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

    print("F4 alt: NYC contribution/inflow KDE by drought zone")
    print("=" * 70)

    # --- Try cached metrics first (fast path) ---
    use_cached = True
    try:
        from methods.load import load_contribution_metrics
        from methods.metrics.contribution import get_metrics_for_window, categorize_by_zone

        metrics_cache = {}
        for scenario in SCENARIOS:
            metrics_cache[scenario] = load_contribution_metrics(scenario)
    except (ImportError, FileNotFoundError):
        use_cached = False

    if not use_cached:
        all_data = load_all_data()

    for n_mo in WINDOW_MONTHS:
        print(f"\n  {n_mo}-month window ...")
        F4_module.N_MONTHS_PRIOR = n_mo

        if use_cached:
            window_days = n_mo * 30
            column_rename_map = {
                f'contribution_total_{window_days}d': 'contribution_total',
                f'contribution_ratio_{window_days}d': 'contribution_ratio',
                f'inflow_total_{window_days}d': 'inflow_total',
                f'demand_satisfaction_{window_days}d': 'demand_satisfaction',
                f'worst_1mo_demand_sat_{window_days}d': 'worst_1mo_demand_sat',
            }
            zone_categories = {
                'emergency': [6],
                'watch': [5],
                'warning': [4],
                'other': [0, 1, 2, 3],
            }
            window_df = get_metrics_for_window(
                metrics_cache['stationary_ensemble'], window_days)
            window_df = window_df.rename(columns=column_rename_map)
            categorized = categorize_by_zone(window_df, zone_categories)
        else:
            all_categorized = categorize_all_scenarios(all_data, n_mo)
            categorized = all_categorized['stationary_ensemble']

        fig = create_figure(categorized, n_mo)
        fname = f"{FIG_OUTPUT_DIR}/Fig7_kde_{n_mo}mo.png"
        fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
        print(f"    Saved: {fname}")
        plt.close(fig)

    print("\n" + "=" * 70)
    print("F4 alt KDE figures generated successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
