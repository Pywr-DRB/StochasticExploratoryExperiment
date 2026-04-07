"""
F4 (Alternative v2): NYC contribution / inflow ratio — joy-division ridgeline.

4-row ridgeline layout (seaborn FacetGrid) where each row shows the
distribution of the NYC Montague contribution-to-inflow ratio for one
FFMP drought zone category, with one KDE line per climate scenario.

  Row 1 (top): Normal / Flood
  Row 2:       Drought Warning
  Row 3:       Drought Watch
  Row 4 (bottom): Drought Emergency

Usage:
    python F4alt_nyc_contribution_kdes_v2.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from methods.config import FIG_DIR, OUTPUT_DIR, verify_dataset_id
from methods.plotting.styles import (
    DPI_HIGH,
    DATASET_COLORS,
    FONTSIZE_LABEL,
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
SCENARIO = 'stationary_ensemble'
WINDOW_MONTHS = [3, 6, 9]
FIG_OUTPUT_DIR = f"{FIG_DIR}/F4alt_kde"

# Zone display order: least severe → most severe (top → bottom)
ZONE_ORDER = ['other', 'warning', 'watch', 'emergency']
ZONE_LABELS = {
    'other':     'Normal / Flood',
    'warning':   'Drought Warning',
    'watch':     'Drought Watch',
    'emergency': 'Drought Emergency',
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_data():
    """Load pywrdrb.Data for all scenarios."""
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
    """Aggregate and categorize by drought zone for all scenarios."""
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
    """Return contribution / inflow ratio (%) for a zone, filtered."""
    df = categorized[zone]
    df_f = df[df['inflow_total'] > MIN_INFLOW_THRESHOLD].copy()
    if len(df_f) < 2:
        return None
    ratio = (100.0 * df_f['contribution_total'] / df_f['inflow_total'])
    return ratio.replace([np.inf, -np.inf], np.nan).dropna()


def build_long_df(all_categorized):
    """Build long-form DataFrame with columns [zone, ratio] for stationary ensemble."""
    rows = []
    for zone in ZONE_ORDER:
        r = _get_ratios(all_categorized[SCENARIO], zone)
        if r is not None and len(r) > 0:
            for val in r.values:
                rows.append({'zone': zone, 'ratio': val})
    return pd.DataFrame(rows)


# ============================================================================
# RIDGELINE FIGURE
# ============================================================================

def create_ridgeline_figure(all_categorized, n_months_prior, recon_ratio):
    """
    Build a joy-division style ridgeline using seaborn FacetGrid.
    One KDE line per scenario, no fill, dataset colours.

    Parameters
    ----------
    all_categorized : dict
        {scenario: {zone: DataFrame}}
    n_months_prior : int
        Aggregation window length (for x-axis label).
    recon_ratio : float or None
        1964 reconstruction contribution ratio (%).
    """
    # x range: 95th percentile of emergency zone data across all scenarios
    emergency_vals = []
    for sc in SCENARIOS:
        r = _get_ratios(all_categorized[sc], 'emergency')
        if r is not None and len(r) > 0:
            emergency_vals.extend(r.values)
    x_max = float(np.percentile(emergency_vals, 95)) if emergency_vals else 100.0
    x_max = max(x_max, 100.0)
    if recon_ratio is not None and recon_ratio <= x_max * 1.1:
        x_max = max(x_max, recon_ratio)

    df = build_long_df(all_categorized)

    sns.set_theme(
        style="white",
        rc={"axes.facecolor": (0, 0, 0, 0), "axes.linewidth": 1.5},
    )

    sc_color = DATASET_COLORS[SCENARIO]

    g = sns.FacetGrid(
        df,
        row="zone",
        hue="zone",
        row_order=ZONE_ORDER,
        hue_order=ZONE_ORDER,
        aspect=9,
        height=1.2,
        sharey=False,
    )

    # Filled KDE using the stationary ensemble color, black outline on top
    g.map_dataframe(sns.kdeplot, x="ratio", fill=True, color=sc_color, alpha=1, clip=(0, x_max))
    g.map_dataframe(sns.kdeplot, x="ratio", fill=False, color="black", linewidth=1.5, clip=(0, x_max))

    g.figure.subplots_adjust(hspace=-0.1)
    g.set_titles("")
    g.set(
        yticks=[],
        ylabel="",
        xlim=(0, x_max),
        xlabel=f"NYC contributions / total inflow  ({n_months_prior}-month window prior to min zone, %)",
    )
    g.despine(left=True)

    # Per-zone max data value for trimming the bottom spine
    zone_x_max = {}
    for zone in ZONE_ORDER:
        vals = []
        for sc in SCENARIOS:
            r = _get_ratios(all_categorized[sc], zone)
            if r is not None and len(r) > 0:
                vals.extend(r.values)
        zone_x_max[zone] = float(np.percentile(vals, 99)) if vals else x_max

    for ax, zone in zip(g.axes.flat, ZONE_ORDER):
        ax.set_ylabel("")
        ax.text(0.01, 0.25, ZONE_LABELS[zone], color="black",
                fontsize=FONTSIZE_LABEL, ha="left", va="center",
                transform=ax.transAxes)
        # Trim the bottom spine to the zone's max data extent
        ax.spines["bottom"].set_bounds(0, zone_x_max[zone])

    g.axes[-1, 0].xaxis.label.set_fontsize(FONTSIZE_LABEL)

    # 1964 reconstruction: bold black tick on the emergency (bottom) row baseline
    if recon_ratio is not None and recon_ratio <= x_max:
        ax_emg = g.axes[-1, 0]
        ylim = ax_emg.get_ylim()
        tick_height = (ylim[1] - ylim[0]) * 0.07
        ax_emg.vlines(recon_ratio, ylim[0], ylim[0] + tick_height,
                      color="black", linewidth=3.5, zorder=5)

    return g.figure


# ============================================================================
# MAIN
# ============================================================================

def main():
    apply_publication_style()

    os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

    print("F4 alt v2: NYC contribution/inflow ridgeline KDE by drought zone")
    print("=" * 70)

    # --- try cached metrics first ---
    use_cached = True
    try:
        from methods.load import load_contribution_metrics
        from methods.metrics.contribution import get_metrics_for_window, categorize_by_zone
        metrics_cache = {sc: load_contribution_metrics(sc) for sc in SCENARIOS}
    except (ImportError, FileNotFoundError):
        use_cached = False

    if not use_cached:
        all_data = load_all_data()

    zone_categories = {
        'emergency': [6],
        'watch':     [5],
        'warning':   [4],
        'other':     [0, 1, 2, 3],
    }

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
                all_categorized[sc] = categorize_by_zone(df, zone_categories)
        else:
            all_categorized = categorize_all_scenarios(all_data, n_mo)

        recon_ratio = calculate_reconstruction_contribution_ratio()

        fig = create_ridgeline_figure(all_categorized, n_mo, recon_ratio)
        fname = f"{FIG_OUTPUT_DIR}/F4alt_kde_v2_{n_mo}mo.png"
        fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
        print(f"    Saved: {fname}")
        plt.close(fig)

    print("\n" + "=" * 70)
    print("F4 alt v2 ridgeline figures generated successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
