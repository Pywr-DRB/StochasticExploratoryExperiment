"""
F3: Composite drought contribution and storage analysis figure.

Multi-panel figure combining:
  A) KDE of NYC contributions/inflow ratio by drought storage zone (stationary)
  B1) Scatter: x-metric vs min storage % for climate_adjusted_low
  B2) Scatter: x-metric vs min storage % for climate_adjusted_high
  C) Bar chart: drought zone frequency across scenarios

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
from matplotlib.ticker import MaxNLocator
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from methods.config import (
    NYC_TOTAL_CAPACITY, NYC_RESERVOIRS,
    FIG_DIR, N_YEARS,
    verify_dataset_id,
)
from methods.plotting.styles import (
    DPI_HIGH, DATASET_COLORS, DATASET_LABELS,
    FONTSIZE_LABEL, FONTSIZE_MEDIUM,
    ALPHA_LINE,
    apply_publication_style,
)
from methods.load import load_ffmp_boundaries, load_performance_metrics

# Reuse data-processing functions
import methods.plotting.water_balance_by_drought_zone as F4_module
from methods.plotting.water_balance_by_drought_zone import (
    classify_years_by_min_zone,
    aggregate_across_realizations,
    categorize_by_drought_zone,
    calculate_reconstruction_contribution_ratio,
    DROUGHT_CATEGORIES,
    N_MONTHS_PRIOR,
    MIN_INFLOW_THRESHOLD,
    XLIM_MAX_MANUAL,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

SCENARIOS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']

# Window lengths (months prior to min-zone date) to generate figures for.
WINDOW_MONTHS = [3, 6, 9]

FIG_OUTPUT_DIR = f"{FIG_DIR}/F3_composite_figures"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

# KDE categories to plot (excluding 'other' / Normal keeps Panel A focused on drought)
KDE_CATEGORIES = ['emergency', 'watch', 'warning', 'other']


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
      - min_storage_pct: annual minimum combined NYC storage %
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
        year_classifications = classify_years_by_min_zone(res_level_df)

        nyc_contributions = data.contribution[dataset_id][r]['mrf_montagueTrenton_nyc']
        nyc_inflow = data.inflow[dataset_id][r][NYC_RESERVOIRS].sum(axis=1)
        nyc_storage = data.res_storage[dataset_id][r][NYC_RESERVOIRS].sum(axis=1)
        nyc_storage_pct = 100.0 * nyc_storage / NYC_TOTAL_CAPACITY
        nyc_diversion = data.ibt_diversions[dataset_id][r]['delivery_nyc']
        nyc_demand = data.ibt_demands[dataset_id][r]['demand_nyc']

        for year, info in year_classifications.items():
            min_zone_date = info['min_zone_date']
            start_date = min_zone_date - pd.DateOffset(months=n_months_prior)

            # contribution total
            mask = (nyc_contributions.index >= start_date) & (nyc_contributions.index <= min_zone_date)
            contribution_total = nyc_contributions[mask].sum()

            # inflow total over same window
            inflow_mask = (nyc_inflow.index >= start_date) & (nyc_inflow.index <= min_zone_date)
            inflow_total = nyc_inflow[inflow_mask].sum()

            # contribution / inflow ratio (%)
            contribution_ratio = (100.0 * contribution_total / inflow_total
                                  if inflow_total > 0 else np.nan)

            # min storage %
            year_mask = nyc_storage_pct.index.year == year
            min_storage = nyc_storage_pct[year_mask].min()

            # demand satisfaction over full window
            div_mask = (nyc_diversion.index >= start_date) & (nyc_diversion.index <= min_zone_date)
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
                'min_zone': info['min_zone'],
                'contribution_total': contribution_total,
                'contribution_ratio': contribution_ratio,
                'min_storage_pct': min_storage,
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


# ============================================================================
# PANEL A: KDE of contribution ratio by drought zone
# ============================================================================

def plot_panel_A(ax, categorized_data, n_months_prior=None):
    """
    KDE of NYC contributions / inflow ratio by drought zone (stationary ensemble).

    Returns dict of legend handles.
    """
    if n_months_prior is None:
        n_months_prior = N_MONTHS_PRIOR
    categories = KDE_CATEGORIES
    category_data = {}

    for cat in categories:
        cat_info = DROUGHT_CATEGORIES[cat]
        df = categorized_data[cat].copy()
        if len(df) == 0:
            continue
        df_filtered = df[df['inflow_total'] > MIN_INFLOW_THRESHOLD]
        if len(df_filtered) == 0:
            continue
        ratio = 100.0 * df_filtered['contribution_total'] / df_filtered['inflow_total']
        category_data[cat] = {'ratio': ratio, 'n': len(df_filtered)}

    # Determine x-axis max
    xlim_max = XLIM_MAX_MANUAL if XLIM_MAX_MANUAL is not None else 100

    # Plot KDEs
    for cat in categories:
        if cat not in category_data:
            continue
        cat_info = DROUGHT_CATEGORIES[cat]
        ratio = category_data[cat]['ratio']
        n = category_data[cat]['n']

        label = f"{cat_info['label']} (n={n})" if cat != 'other' else f"Normal or Above (n={n})"
        ratio.plot.kde(ax=ax, color=cat_info['color'], linewidth=2.5, alpha=ALPHA_LINE, label=label)
        mean_val = ratio.mean()
        ax.axvline(mean_val, color=cat_info['color'], linestyle='--', linewidth=1.5, alpha=0.7)

    # Dummy for mean legend entry
    ax.axvline(np.nan, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Mean')

    # 1964 reconstruction
    reconstruction_ratio = calculate_reconstruction_contribution_ratio()
    if reconstruction_ratio is not None:
        ax.axvline(reconstruction_ratio, color='black', linestyle='-', linewidth=2.5, alpha=0.9, label='1964 Drought')
        if reconstruction_ratio > xlim_max:
            xlim_max = reconstruction_ratio * 1.1

    xlabel = f'NYC contributions / total inflow\n({n_months_prior}-mo prior to min zone, %)'
    ax.set_xlabel(xlabel, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Density', fontsize=FONTSIZE_LABEL)
    ax.set_xlim(left=0, right=xlim_max)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Panel label
    ax.text(-0.05, 1.02, 'a)', transform=ax.transAxes, fontsize=14, va='bottom', ha='right')

    return ax.get_legend_handles_labels()


# ============================================================================
# PANEL B: Scatter with regression + FFMP bands
# ============================================================================

def plot_panel_B(ax, metrics_df, ffmp_lines, x_key, x_label,
                 vmin=None, vmax=None, panel_label='b)', show_xlabel=True):
    """
    Scatter of x_key vs min_storage_pct, colored by demand_satisfaction.

    Returns the scatter PathCollection (for colorbar).
    """
    df = metrics_df.dropna(subset=[x_key, 'min_storage_pct', 'worst_1mo_demand_sat'])
    if len(df) == 0:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
        return None

    x = df[x_key].values
    y = df['min_storage_pct'].values
    c = df['worst_1mo_demand_sat'].values

    if vmin is None:
        vmin = np.nanpercentile(c, 5)
    if vmax is None:
        vmax = np.nanpercentile(c, 95)

    sc = ax.scatter(x, y, c=c, cmap='viridis', s=12, alpha=0.5,
                    edgecolors='none', vmin=vmin, vmax=vmax, rasterized=True)

    # Regression line
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() > 10:
        z = np.polyfit(x[valid], y[valid], 1)
        p = np.poly1d(z)
        x_line = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(x_line, p(x_line), 'k--', linewidth=1.5, alpha=0.6)
        r = np.corrcoef(x[valid], y[valid])[0, 1]
        ax.text(0.05, 0.95, f'r = {r:.2f}', transform=ax.transAxes,
                fontsize=FONTSIZE_LABEL, va='top')

    # FFMP zone median lines
    for line_info in ffmp_lines:
        ax.axhline(line_info['median'], color=line_info['color'],
                   linestyle='--', linewidth=1.0, alpha=0.7)

    ax.set_ylim(0, 100)
    ax.set_ylabel('Min NYC storage (%)', fontsize=FONTSIZE_LABEL)
    if show_xlabel:
        ax.set_xlabel(x_label, fontsize=FONTSIZE_LABEL)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Clean up x-axis ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))
    ax.tick_params(axis='x', labelsize=FONTSIZE_MEDIUM, rotation=0)

    # Panel label
    ax.text(-0.08, 1.02, panel_label, transform=ax.transAxes,
            fontsize=14, va='bottom', ha='right')

    return sc


# ============================================================================
# PANEL C: Bar chart of drought zone frequency
# ============================================================================

def plot_panel_C(ax):
    """Grouped bar chart of drought zone frequency across scenarios."""
    zone_keys = ['years_exactly_warning', 'years_exactly_watch', 'years_exactly_emergency']
    zone_labels = ['Warning', 'Watch', 'Emergency']

    x = np.arange(len(zone_keys))
    width = 0.25
    n_scenarios = len(SCENARIOS)

    for i, dataset_id in enumerate(SCENARIOS):
        metrics = load_performance_metrics(dataset_id)
        freqs = []
        err_lo = []
        err_hi = []
        for zk in zone_keys:
            vals = metrics[zk] / N_YEARS
            mean_val = vals.mean()
            freqs.append(mean_val)
            err_lo.append(mean_val - vals.quantile(0.1))
            err_hi.append(vals.quantile(0.9) - mean_val)

        offset = (i - (n_scenarios - 1) / 2) * width
        ax.bar(x + offset, freqs, width,
               color=DATASET_COLORS[dataset_id], alpha=0.8,
               label=DATASET_LABELS.get(dataset_id, dataset_id))
        ax.errorbar(x + offset, freqs,
                    yerr=[err_lo, err_hi],
                    fmt='none', color='black', capsize=3, linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(zone_labels, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Fraction of years', fontsize=FONTSIZE_LABEL)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Panel label
    ax.text(-0.08, 1.02, 'd)', transform=ax.transAxes, fontsize=14, va='bottom', ha='right')


# ============================================================================
# SHARED LEGEND
# ============================================================================

def create_shared_legend(ax_legend, kde_handles, kde_labels):
    """
    Assemble a clean shared legend from all panels.
    """
    ax_legend.axis('off')

    elements = []
    labels = []

    # (1) Drought zone KDE lines (from Panel A)
    # Reorder: Normal, Warning, Watch, Emergency
    desired_order = ['Normal or Above', 'Drought Warning', 'Drought Watch', 'Drought Emergency']
    for keyword in desired_order:
        for idx, lbl in enumerate(kde_labels):
            if keyword in lbl:
                elements.append(kde_handles[idx])
                labels.append(lbl)
                break

    # (2) Mean line and 1964 Drought
    for keyword in ['Mean', '1964']:
        for idx, lbl in enumerate(kde_labels):
            if keyword in lbl:
                elements.append(kde_handles[idx])
                labels.append(lbl)
                break

    # (3) Scenario colors (from Panel C)
    for dataset_id in SCENARIOS:
        elements.append(Patch(facecolor=DATASET_COLORS[dataset_id], alpha=0.8))
        labels.append(DATASET_LABELS.get(dataset_id, dataset_id))

    # (4) FFMP zone median line indicator
    elements.append(Line2D([0], [0], color='gray', linestyle='--', linewidth=1.0, alpha=0.7))
    labels.append('FFMP zone boundary (median)')

    ax_legend.legend(elements, labels, loc='center', ncol=2,
                     fontsize=11, frameon=False,
                     handlelength=2.5, columnspacing=1.5)


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

    print("F3: Composite drought contribution figure")

    # Try loading pre-computed metrics first (FAST PATH)
    use_cached = True
    try:
        from methods.load_contribution_metrics import (
            load_contribution_metrics, get_metrics_for_window, categorize_by_zone
        )

        metrics_cache = {}
        for scenario in SCENARIOS:
            metrics_cache[scenario] = load_contribution_metrics(scenario)
        use_cached = True

    except (ImportError, FileNotFoundError):
        use_cached = False

    if not use_cached:
        all_data = load_all_data()

    ffmp_lines = get_ffmp_zone_medians()

    # Generate one figure per window length
    for n_mo in WINDOW_MONTHS:
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

            metrics_low = get_metrics_for_window(metrics_cache['climate_adjusted_low'], window_days)
            metrics_high = get_metrics_for_window(metrics_cache['climate_adjusted_high'], window_days)
            metrics_low = metrics_low.rename(columns=column_rename_map)
            metrics_high = metrics_high.rename(columns=column_rename_map)

        else:
            # FALLBACK: Original calculation
            all_categorized = categorize_all_scenarios(all_data, n_mo)
            metrics_low = calculate_drought_metrics_per_year(
                all_data['climate_adjusted_low'], 'climate_adjusted_low', n_months_prior=n_mo)
            metrics_high = calculate_drought_metrics_per_year(
                all_data['climate_adjusted_high'], 'climate_adjusted_high', n_months_prior=n_mo)

        # Common color range for worst 1-month demand satisfaction across both scenarios
        all_ds = pd.concat([metrics_low['worst_1mo_demand_sat'], metrics_high['worst_1mo_demand_sat']]).dropna()
        ds_vmin = np.nanpercentile(all_ds, 5)
        ds_vmax = np.nanpercentile(all_ds, 95)

        x_key = 'contribution_ratio'
        x_label = f'NYC contribution / inflow\n({n_mo}-mo prior, %)'
        suffix = f'contribution_ratio_{n_mo}mo'

        fig = plt.figure(figsize=(14, 12))
        gs = fig.add_gridspec(
            3, 2,
            height_ratios=[1, 1, 0.8],
            width_ratios=[1.2, 1],
            hspace=0.30, wspace=0.30,
            left=0.08, right=0.92, top=0.95, bottom=0.06,
        )

        ax_A = fig.add_subplot(gs[0:2, 0])
        ax_B1 = fig.add_subplot(gs[0, 1])   # climate_adjusted_low (top)
        ax_B2 = fig.add_subplot(gs[1, 1])   # climate_adjusted_high (bottom)
        ax_C = fig.add_subplot(gs[2, 0])
        ax_legend = fig.add_subplot(gs[2, 1])

        # Panel A
        kde_handles, kde_labels = plot_panel_A(ax_A, all_categorized['stationary_ensemble'], n_months_prior=n_mo)

        # Panel B1 (climate_adjusted_low - top)
        sc1 = plot_panel_B(ax_B1, metrics_low, ffmp_lines,
                           x_key, x_label, vmin=ds_vmin, vmax=ds_vmax,
                           panel_label='b)', show_xlabel=False)

        # Panel B2 (climate_adjusted_high - bottom)
        sc2 = plot_panel_B(ax_B2, metrics_high, ffmp_lines,
                           x_key, x_label, vmin=ds_vmin, vmax=ds_vmax,
                           panel_label='c)', show_xlabel=True)

        # Sync x-limits across B1/B2 and suppress B1 x-tick labels
        xlim_lo = min(ax_B1.get_xlim()[0], ax_B2.get_xlim()[0])
        xlim_hi = max(ax_B1.get_xlim()[1], ax_B2.get_xlim()[1])
        xlim_lo = 0.0
        ax_B1.set_xlim(xlim_lo, xlim_hi)
        ax_B2.set_xlim(xlim_lo, xlim_hi)
        ax_B1.set_xticklabels([])

        # Shared colorbar for B1/B2
        sc_ref = sc1 if sc1 is not None else sc2
        if sc_ref is not None:
            cbar = fig.colorbar(sc_ref, ax=[ax_B1, ax_B2], shrink=0.6,
                                pad=0.03, aspect=30)
            cbar.set_label('Worst 1-mo demand satisfaction (%)', fontsize=FONTSIZE_LABEL)
            cbar.ax.tick_params(labelsize=FONTSIZE_MEDIUM)

        # Panel C
        plot_panel_C(ax_C)

        # Shared legend
        create_shared_legend(ax_legend, kde_handles, kde_labels)

        # Save
        fname_base = f"{FIG_OUTPUT_DIR}/F3_drought_contribution_composite_{suffix}"
        fig.savefig(f"{fname_base}.png", dpi=DPI_HIGH, bbox_inches='tight')
        print(f"Saved: {fname_base}.png")
        plt.close(fig)


if __name__ == '__main__':
    main()
