"""
F3: Composite drought contribution and storage analysis figure (Version 3).

Simplified 3-panel layout:
  Left: A (KDE of contribution ratio by drought zone)
  Right: B1 (frequency box plots) | B2 (duration box plots)

Key features:
  - A: KDE showing NYC contribution/inflow ratio for different drought zones
  - B1: Box plots showing distribution of fraction of years in each drought zone across realizations
  - B2: Box plots showing distribution of mean event duration across realizations
  - Uses correct colorblind-friendly dataset colors from methods.plotting.styles

Changes from v2:
  - Replaced stacked bar charts with box plots for B1 and B2
  - Fixed dataset colors to use DATASET_COLORS from styles module
  - Focus on simplified 3-panel figure (removed full version with scatter plots)
  - Box plots show distribution across ensemble realizations (not stacked percentiles)

Usage:
    python F3_plot_drought_contribution_composite.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from methods.config import (
    NYC_TOTAL_CAPACITY, NYC_RESERVOIRS,
    FIG_DIR,
    verify_dataset_id,
)
from methods.plotting.styles import (
    DPI_HIGH, DATASET_COLORS, DATASET_LABELS,
    FFMP_ZONE_COLORS, FFMP_ZONE_COLORS_INT,
    FONTSIZE_SMALL, FONTSIZE_LABEL, FONTSIZE_MEDIUM,
    ALPHA_LINE,
    apply_publication_style,
)
from methods.load import load_ffmp_boundaries, load_annual_metrics, load_contribution_metrics

# Reuse data-processing functions
import methods.plotting.water_balance_by_drought_zone as F4_module
from methods.plotting.water_balance_by_drought_zone import (
    classify_years_by_max_zone,
    aggregate_across_realizations,
    categorize_by_drought_zone,
    calculate_reconstruction_contribution_ratio,
    DROUGHT_CATEGORIES,
    N_MONTHS_PRIOR,
    MIN_INFLOW_THRESHOLD,
    XLIM_MAX_MANUAL,
)

# KDE plotting (extracted to reusable module)
from methods.plotting.contribution_kde import plot_kde_panel, KDE_CATEGORIES

# ============================================================================
# CONFIGURATION
# ============================================================================

SCENARIOS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']

# Window lengths (months prior to min-zone date) to generate figures for.
WINDOW_MONTHS = [3, 6, 9]

FIG_OUTPUT_DIR = f"{FIG_DIR}/F3_composite_figures"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

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
        fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
        data = pywrdrb.Data()
        data.load_from_export(fname, results_sets=results_sets)
        all_data[dataset_id] = data

    return all_data


def categorize_all_scenarios(all_data, n_months_prior):
    """
    Aggregate and categorize by drought zone for all scenarios,
    using the specified window length.

    Sets F4_module.N_MONTHS_PRIOR before calling F4 functions so
    the aggregation window is consistent across panels.
    """
    F4_module.N_MONTHS_PRIOR = n_months_prior

    all_categorized = {}
    for dataset_id in SCENARIOS:
        agg = aggregate_across_realizations(all_data[dataset_id], dataset_id)
        all_categorized[dataset_id] = categorize_by_drought_zone(agg)
    return all_categorized


# ============================================================================
# DROUGHT METRICS PER YEAR (Panels B1 / B2)
# ============================================================================

def calculate_drought_metrics_per_year(data, dataset_id, n_months_prior=None):
    """
    For each realization-year, compute:
      - contribution_total: NYC Montague contribution (MG) in n-month window
      - contribution_ratio: contribution_total / NYC inflow over same window (%)
      - annual_min_storage_pct: annual minimum combined NYC storage %
      - demand_satisfaction: volumetric diversion / demand over window (clipped <=1)
      - worst_1mo_demand_sat: worst 1-month rolling demand satisfaction (%) during
        the n-month window. Computed as min of 30-day rolling sum(diversion)/sum(demand).

    Parameters
    ----------
    n_months_prior : int, optional
        Number of months before min-zone date used as the aggregation window.
        Defaults to N_MONTHS_PRIOR from F4 config.

    Returns pd.DataFrame.
    """
    if n_months_prior is None:
        n_months_prior = N_MONTHS_PRIOR

    realizations = sorted(data.res_level[dataset_id].keys())
    records = []

    for r in realizations:
        res_level_df = data.res_level[dataset_id][r]
        year_classifications = classify_years_by_max_zone(res_level_df)

        nyc_contributions = data.contribution[dataset_id][r]['mrf_montagueTrenton_nyc']
        nyc_inflow = data.inflow[dataset_id][r][NYC_RESERVOIRS].sum(axis=1)
        nyc_storage = data.res_storage[dataset_id][r][NYC_RESERVOIRS].sum(axis=1)
        nyc_storage_pct = 100.0 * nyc_storage / NYC_TOTAL_CAPACITY
        nyc_diversion = data.ibt_diversions[dataset_id][r]['delivery_nyc']
        nyc_demand = data.ibt_demands[dataset_id][r]['demand_nyc']

        for year, info in year_classifications.items():
            max_zone_date = info['max_zone_date']
            start_date = max_zone_date - pd.DateOffset(months=n_months_prior)

            # contribution total
            mask = (nyc_contributions.index >= start_date) & (nyc_contributions.index <= max_zone_date)
            contribution_total = nyc_contributions[mask].sum()

            # inflow total over same window
            inflow_mask = (nyc_inflow.index >= start_date) & (nyc_inflow.index <= max_zone_date)
            inflow_total = nyc_inflow[inflow_mask].sum()

            # contribution / inflow ratio (%)
            contribution_ratio = (100.0 * contribution_total / inflow_total
                                  if inflow_total > 0 else np.nan)

            # min storage %
            year_mask = nyc_storage_pct.index.year == year
            min_storage = nyc_storage_pct[year_mask].min()

            # demand satisfaction over full window
            div_mask = (nyc_diversion.index >= start_date) & (nyc_diversion.index <= max_zone_date)
            total_div = nyc_diversion[div_mask].sum()
            total_dem = nyc_demand[div_mask].sum()
            demand_sat = min(total_div / total_dem, 1.0) if total_dem > 0 else 1.0

            # Worst 1-month (30-day rolling) demand satisfaction during the window
            window_div = nyc_diversion[div_mask]
            window_dem = nyc_demand[div_mask]
            if len(window_div) >= 30:
                rolling_div = window_div.rolling(30, min_periods=30).sum()
                rolling_dem = window_dem.rolling(30, min_periods=30).sum()
                rolling_sat = (rolling_div / rolling_dem).clip(upper=1.0)
                worst_1mo = 100.0 * rolling_sat.min()
            elif len(window_div) > 0:
                # Window shorter than 30 days: use entire window
                worst_1mo = 100.0 * demand_sat
            else:
                worst_1mo = np.nan

            records.append({
                'year': year,
                'realization_id': r,
                'max_zone': info['max_zone'],
                'contribution_total': contribution_total,
                'contribution_ratio': contribution_ratio,
                'annual_min_storage_pct': min_storage,
                'demand_satisfaction': demand_sat,
                'worst_1mo_demand_sat': worst_1mo,
            })

    return pd.DataFrame(records)


# ============================================================================
# FFMP ZONE BANDS (Panels B1 / B2)
# ============================================================================

def get_ffmp_zone_medians():
    """
    Compute median of each FFMP boundary across all days-of-year.

    Returns list of dicts with keys: zone_name, color, median
    for the boundaries between Warning/Watch/Emergency zones.
    """
    ffmp = load_ffmp_boundaries()
    # Columns ordered ascending by median: L1a < L1b < ... < L5
    # Zone 6 (emergency) is triggered when storage < L1a (lowest threshold).
    # Zone 5 (watch): L1a <= storage < L1b
    # Zone 4 (warning): L1b <= storage < L2

    num_cols = ffmp.select_dtypes(include=[np.number]).columns.tolist()
    medians = ffmp[num_cols].median().sort_values()
    ordered_cols = list(medians.index)

    lines = []
    # Upper boundary of emergency zone (lower boundary of watch)
    if len(ordered_cols) >= 1:
        lines.append({
            'zone_name': 'emergency',
            'color': DROUGHT_CATEGORIES['emergency']['color'],
            'median': medians[ordered_cols[0]],
        })
    # Upper boundary of watch zone (lower boundary of warning)
    if len(ordered_cols) >= 2:
        lines.append({
            'zone_name': 'watch',
            'color': DROUGHT_CATEGORIES['watch']['color'],
            'median': medians[ordered_cols[1]],
        })
    # Upper boundary of warning zone (lower boundary of normal)
    if len(ordered_cols) >= 3:
        lines.append({
            'zone_name': 'warning',
            'color': DROUGHT_CATEGORIES['warning']['color'],
            'median': medians[ordered_cols[2]],
        })

    return lines


# Drought zone boxplots (modular, in methods/plotting/)
from methods.plotting.drought_zone_boxplots import (
    plot_frequency_boxplot,
    plot_duration_boxplot,
    style_boxplot,
)


def add_boxplot_legend(fig):
    """
    Add comprehensive shared legend for all figure elements.

    Includes:
    - KDE drought zone colors (Panel A)
    - Mean/1964 lines (Panel A)
    - Dataset colors (Panels B1/B2 box plots)
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = []

    # Row 1: KDE drought zone lines (Panel A)
    legend_elements.append(Line2D([0], [0], color=DROUGHT_CATEGORIES['emergency']['color'],
                                   linewidth=2.5, label='Emergency'))
    legend_elements.append(Line2D([0], [0], color=DROUGHT_CATEGORIES['watch']['color'],
                                   linewidth=2.5, label='Watch'))
    legend_elements.append(Line2D([0], [0], color=DROUGHT_CATEGORIES['warning']['color'],
                                   linewidth=2.5, label='Warning'))
    legend_elements.append(Line2D([0], [0], color=DROUGHT_CATEGORIES['other']['color'],
                                   linewidth=2.5, label='Normal'))

    # Row 2: Special lines (Panel A)
    legend_elements.append(Line2D([0], [0], color='gray', linestyle='--',
                                   linewidth=1.5, label='Mean'))
    legend_elements.append(Line2D([0], [0], color='black', linestyle='-',
                                   linewidth=2.5, label='1964 Drought'))

    # Row 3: Dataset patches (Panels B1/B2 box plots)
    for dataset_id in SCENARIOS:
        legend_elements.append(Patch(facecolor=DATASET_COLORS[dataset_id], alpha=0.7,
                                      edgecolor='black', linewidth=1.2,
                                      label=DATASET_LABELS[dataset_id]))

    # Mean marker
    legend_elements.append(Line2D([0], [0], color='gray', marker='o', linestyle='None',
                                   markersize=6, markeredgecolor='white',
                                   markeredgewidth=0.8, label='Mean'))

    # Historic marker
    legend_elements.append(Line2D([0], [0], color='black', marker='^', linestyle='None',
                                   markersize=8, label='Historic'))

    # Add legend at bottom center with simple styling
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=5, fontsize=FONTSIZE_SMALL,
               frameon=False,
               bbox_to_anchor=(0.5, -0.04))


def plot_scatter_panel_simple(ax, metrics_df, dataset_id, ffmp_lines,
                              x_key, x_label, panel_label,
                              vmin=None, vmax=None, show_xlabel=True):
    """
    Plot scatter panel for a single dataset (for C1-C3).

    Similar to plot_panel_B but simplified for bottom row.
    """
    df = metrics_df.dropna(subset=[x_key, 'annual_min_storage_pct', 'worst_1mo_demand_sat'])
    if len(df) == 0:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
        return None

    x = df[x_key].values
    y = df['annual_min_storage_pct'].values
    c = df['worst_1mo_demand_sat'].values

    if vmin is None:
        vmin = np.nanpercentile(c, 5)
    if vmax is None:
        vmax = np.nanpercentile(c, 95)

    sc = ax.scatter(x, y, c=c, cmap='viridis', s=8, alpha=0.4,
                    edgecolors='none', vmin=vmin, vmax=vmax, rasterized=True)

    # Regression line
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() > 10:
        z = np.polyfit(x[valid], y[valid], 1)
        p = np.poly1d(z)
        x_line = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(x_line, p(x_line), 'k--', linewidth=1.2, alpha=0.6)
        r = np.corrcoef(x[valid], y[valid])[0, 1]
        ax.text(0.05, 0.95, f'r = {r:.2f}', transform=ax.transAxes,
                fontsize=FONTSIZE_SMALL, va='top')

    # FFMP zone median lines
    for line_info in ffmp_lines:
        ax.axhline(line_info['median'], color=line_info['color'],
                   linestyle='--', linewidth=0.8, alpha=0.6)

    ax.set_ylim(0, 100)
    ax.set_ylabel('Min NYC storage (%)', fontsize=FONTSIZE_SMALL)
    if show_xlabel:
        ax.set_xlabel(x_label, fontsize=FONTSIZE_SMALL)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=FONTSIZE_SMALL-1)

    # Panel label and title
    ax.text(-0.12, 1.02, panel_label, transform=ax.transAxes,
            fontsize=12, va='bottom', ha='right')
    ax.set_title(DATASET_LABELS.get(dataset_id, dataset_id),
                 fontsize=FONTSIZE_SMALL, pad=3)

    return sc


# ============================================================================
# MAIN
# ============================================================================

def create_figure_simplified(n_mo, all_categorized,
                             show_historic=False):
    """
    Create simplified figure with only A, B1, B2.

    Layout:
      A (KDE) | B1/B2 (frequency and duration bar charts)
    """
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1, 1],
        width_ratios=[1.5, 1],
        hspace=0.25, wspace=0.35,
        left=0.10, right=0.95, top=0.92, bottom=0.18,  # Increased bottom margin for legend
    )

    ax_A = fig.add_subplot(gs[0:2, 0])   # KDE spans left column
    ax_B1 = fig.add_subplot(gs[0, 1])    # Frequency bar chart (top right)
    ax_B2 = fig.add_subplot(gs[1, 1])    # Duration bar chart (bottom right)

    # Panel A: KDE
    plot_kde_panel(ax_A, all_categorized['stationary_ensemble'], n_months_prior=n_mo)

    # Panel B1: Frequency box plot
    plot_frequency_boxplot(ax_B1, panel_label='b)', show_historic=show_historic)

    # Panel B2: Duration box plot
    plot_duration_boxplot(ax_B2, panel_label='c)', show_historic=show_historic)

    # Shared x-axis label for right-side panels
    ax_B2.set_xlabel('NYC Storage Zone', fontsize=FONTSIZE_LABEL)

    # Align y-axis labels for right-side panels
    label_x = -0.2
    for ax in [ax_B1, ax_B2]:
        ax.yaxis.set_label_coords(label_x, 0.5)

    # Shared legend for box plots
    add_boxplot_legend(fig)

    return fig


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

    print("F3 v2: Composite drought contribution figure")
    print("=" * 70)

    # Try loading pre-computed metrics first (FAST PATH)
    use_cached = True
    try:
        from methods.load import load_contribution_metrics
        from methods.metrics.contribution import get_metrics_for_window, categorize_by_zone

        metrics_cache = {}
        for scenario in SCENARIOS:
            metrics_cache[scenario] = load_contribution_metrics(scenario)
        use_cached = True

    except (ImportError, FileNotFoundError):
        use_cached = False

    if not use_cached:
        all_data = load_all_data()

    ffmp_lines = get_ffmp_zone_medians()

    # Generate figures for each window length
    for n_mo in WINDOW_MONTHS:
        print(f"\nProcessing {n_mo}-month window...")

        # Set F4 module variable so Panel A KDE + 1964 line use same window
        F4_module.N_MONTHS_PRIOR = n_mo

        if use_cached:
            # FAST PATH: Use pre-computed metrics
            window_days = n_mo * 30

            column_rename_map = {
                f'contribution_total_{window_days}d': 'contribution_total',
                f'contribution_ratio_{window_days}d': 'contribution_ratio',
                f'inflow_total_{window_days}d': 'inflow_total',
                f'demand_satisfaction_{window_days}d': 'demand_satisfaction',
                f'worst_1mo_demand_sat_{window_days}d': 'worst_1mo_demand_sat'
            }

            all_categorized = {}
            for scenario in SCENARIOS:
                window_df = get_metrics_for_window(metrics_cache[scenario], window_days)
                window_df = window_df.rename(columns=column_rename_map)
                zone_categories = {
                    'emergency': [6],
                    'watch': [5],
                    'warning': [4],
                    'other': [0, 1, 2, 3]
                }
                all_categorized[scenario] = categorize_by_zone(window_df, zone_categories)

            metrics_stat = get_metrics_for_window(metrics_cache['stationary_ensemble'], window_days)
            metrics_low = get_metrics_for_window(metrics_cache['climate_adjusted_low'], window_days)
            metrics_high = get_metrics_for_window(metrics_cache['climate_adjusted_high'], window_days)
            metrics_stat = metrics_stat.rename(columns=column_rename_map)
            metrics_low = metrics_low.rename(columns=column_rename_map)
            metrics_high = metrics_high.rename(columns=column_rename_map)

        else:
            # FALLBACK: Original calculation
            all_categorized = categorize_all_scenarios(all_data, n_mo)
            metrics_stat = calculate_drought_metrics_per_year(
                all_data['stationary_ensemble'], 'stationary_ensemble', n_months_prior=n_mo)
            metrics_low = calculate_drought_metrics_per_year(
                all_data['climate_adjusted_low'], 'climate_adjusted_low', n_months_prior=n_mo)
            metrics_high = calculate_drought_metrics_per_year(
                all_data['climate_adjusted_high'], 'climate_adjusted_high', n_months_prior=n_mo)

        # Common color range for worst 1-month demand satisfaction across all scenarios
        all_ds = pd.concat([
            metrics_stat['worst_1mo_demand_sat'],
            metrics_low['worst_1mo_demand_sat'],
            metrics_high['worst_1mo_demand_sat']
        ]).dropna()
        ds_vmin = np.nanpercentile(all_ds, 5)
        ds_vmax = np.nanpercentile(all_ds, 95)

        suffix = f'contribution_ratio_{n_mo}mo'

        # Generate simplified version (main figure)
        print(f"  Creating simplified 3-panel figure (A + B1 + B2)...")
        fig_simple = create_figure_simplified(n_mo, all_categorized)
        fname_simple = f"{FIG_OUTPUT_DIR}/F3_composite_{suffix}.png"
        fig_simple.savefig(fname_simple, dpi=DPI_HIGH, bbox_inches='tight')
        print(f"    Saved: {fname_simple}")
        plt.close(fig_simple)

    print("\n" + "=" * 70)
    print("All F3 figures generated successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
