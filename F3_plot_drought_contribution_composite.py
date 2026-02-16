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
    FONTSIZE_SMALL, FONTSIZE_LABEL, FONTSIZE_MEDIUM,
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
# PANEL C: Bar chart of drought zone frequency (DEPRECATED - replaced by B1/B2)
# ============================================================================

def plot_panel_C(ax):
    """Grouped bar chart of drought zone frequency across scenarios (deprecated)."""
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
# PANEL B1/B2: New bar charts with stacked percentiles
# ============================================================================

# Percentiles for stacked bars (user-configurable)
PERCENTILES = [5, 50, 95]

def plot_frequency_boxplot(ax, panel_label='b)'):
    """
    Plot Panel B1: Frequency box plot showing distribution across realizations.

    X-axis groups: Warning, Watch, Emergency (3 groups)
    Per group: 3 side-by-side box plots (Stationary, Climate Low, Climate High)
    Box plots show the distribution of frequency values across ensemble realizations
    """
    from methods.load import load_performance_metrics

    # Drought zones and order (left to right on x-axis)
    zone_keys = ['years_exactly_warning', 'years_exactly_watch', 'years_exactly_emergency']
    zone_labels = ['Warning', 'Watch', 'Emergency']

    # Load performance metrics for all datasets
    all_freq_data = {}
    for dataset_id in SCENARIOS:
        metrics = load_performance_metrics(dataset_id)
        # Convert counts to fractions
        freq_data = {}
        for zk in zone_keys:
            freq_data[zk] = metrics[zk].values / N_YEARS
        all_freq_data[dataset_id] = freq_data

    # Set up grouped box plot positions
    n_zones = len(zone_keys)
    n_datasets = len(SCENARIOS)
    positions_all = []
    colors_all = []
    data_all = []

    group_width = 0.8
    box_width = group_width / (n_datasets + 0.5)  # Add spacing between groups

    for zone_idx, zk in enumerate(zone_keys):
        for ds_idx, dataset_id in enumerate(SCENARIOS):
            # Calculate position for this box
            x_pos = zone_idx + (ds_idx - 1) * box_width
            positions_all.append(x_pos)
            colors_all.append(DATASET_COLORS[dataset_id])
            data_all.append(all_freq_data[dataset_id][zk])

    # Create box plots
    bp = ax.boxplot(data_all, positions=positions_all, widths=box_width * 0.8,
                    patch_artist=True, showfliers=True,
                    boxprops=dict(linewidth=1.2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    medianprops=dict(linewidth=1.5, color='black'),
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))

    # Color the boxes
    for patch, color in zip(bp['boxes'], colors_all):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Format axes
    ax.set_xticks(range(n_zones))
    ax.set_xticklabels(zone_labels, fontsize=FONTSIZE_SMALL)
    ax.set_ylabel('Fraction of years', fontsize=FONTSIZE_LABEL)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    ax.text(-0.08, 1.02, panel_label, transform=ax.transAxes,
            fontsize=14, va='bottom', ha='right')


def plot_duration_boxplot(ax, panel_label='c)'):
    """
    Plot Panel B2: Duration box plot showing distribution across realizations.

    X-axis groups: Warning, Watch, Emergency (3 groups)
    Per group: 3 side-by-side box plots (Stationary, Climate Low, Climate High)
    Box plots show the distribution of mean duration per realization
    """
    from methods.zone_duration_metrics import calculate_zone_events
    import pywrdrb

    # Drought zones and order (left to right on x-axis)
    zone_order = [4, 5, 6]  # Warning, Watch, Emergency
    zone_labels_map = {4: 'Warning', 5: 'Watch', 6: 'Emergency'}

    # Calculate duration data for all datasets
    all_duration_data = {}
    for dataset_id in SCENARIOS:
        fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
        data = pywrdrb.Data()
        data.load_from_export(fname, results_sets=['res_level'])

        dataset_durations = {}
        for zone_num in zone_order:
            realization_means = []
            realizations = sorted(data.res_level[dataset_id].keys())

            for r in realizations:
                zone_series = data.res_level[dataset_id][r]['nyc']
                events = calculate_zone_events(zone_series, zone_num, min_end_days=7)
                durations = [e['duration_days'] for e in events]

                # Calculate mean duration for this realization
                if len(durations) > 0:
                    realization_means.append(np.mean(durations))
                else:
                    realization_means.append(0)

            dataset_durations[zone_num] = realization_means

        all_duration_data[dataset_id] = dataset_durations

    # Set up grouped box plot positions
    n_zones = len(zone_order)
    n_datasets = len(SCENARIOS)
    positions_all = []
    colors_all = []
    data_all = []

    group_width = 0.8
    box_width = group_width / (n_datasets + 0.5)  # Add spacing between groups

    for zone_idx, zone_num in enumerate(zone_order):
        for ds_idx, dataset_id in enumerate(SCENARIOS):
            # Calculate position for this box
            x_pos = zone_idx + (ds_idx - 1) * box_width
            positions_all.append(x_pos)
            colors_all.append(DATASET_COLORS[dataset_id])

            # Get mean durations for this zone and dataset (per realization)
            zone_durations = all_duration_data[dataset_id][zone_num]
            data_all.append(zone_durations)

    # Create box plots
    bp = ax.boxplot(data_all, positions=positions_all, widths=box_width * 0.8,
                    patch_artist=True, showfliers=True,
                    boxprops=dict(linewidth=1.2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    medianprops=dict(linewidth=1.5, color='black'),
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))

    # Color the boxes
    for patch, color in zip(bp['boxes'], colors_all):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Format axes
    ax.set_xticks(range(n_zones))
    ax.set_xticklabels([zone_labels_map[z] for z in zone_order], fontsize=FONTSIZE_SMALL)
    ax.set_ylabel('Mean event duration (days)', fontsize=FONTSIZE_LABEL)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    ax.text(-0.08, 1.02, panel_label, transform=ax.transAxes,
            fontsize=14, va='bottom', ha='right')


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

    # Add legend at bottom center with simple styling
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=5, fontsize=FONTSIZE_SMALL,
               frameon=True, framealpha=1.0, edgecolor='black',
               fancybox=False,
               bbox_to_anchor=(0.5, -0.01))


def plot_scatter_panel_simple(ax, metrics_df, dataset_id, ffmp_lines,
                              x_key, x_label, panel_label,
                              vmin=None, vmax=None, show_xlabel=True):
    """
    Plot scatter panel for a single dataset (for C1-C3).

    Similar to plot_panel_B but simplified for bottom row.
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

def create_figure_full(n_mo, all_categorized, metrics_stat, metrics_low, metrics_high,
                       ffmp_lines, ds_vmin, ds_vmax):
    """
    Create full figure with all panels (A, B1, B2, C1, C2, C3).

    Layout:
      Top row: A (KDE) | B1/B2 (frequency and duration bar charts)
      Bottom row: C1, C2, C3 (scatter plots)
    """
    x_key = 'contribution_ratio'
    x_label = f'NYC contribution / inflow\n({n_mo}-mo prior, %)'

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(
        3, 4,
        height_ratios=[1.2, 1.2, 1],
        width_ratios=[1, 1, 0.8, 0.8],
        hspace=0.35, wspace=0.35,
        left=0.08, right=0.95, top=0.95, bottom=0.12,  # Increased bottom margin for legend
    )

    # Top row
    ax_A = fig.add_subplot(gs[0:2, 0:2])  # KDE spans left
    ax_B1 = fig.add_subplot(gs[0, 2:4])    # Frequency bar chart (top right)
    ax_B2 = fig.add_subplot(gs[1, 2:4])    # Duration bar chart (bottom right)

    # Bottom row - scatter plots
    ax_C1 = fig.add_subplot(gs[2, 0])
    ax_C2 = fig.add_subplot(gs[2, 1])
    ax_C3 = fig.add_subplot(gs[2, 2])

    # Panel A: KDE
    kde_handles, kde_labels = plot_panel_A(ax_A, all_categorized['stationary_ensemble'],
                                           n_months_prior=n_mo)

    # Panel B1: Frequency box plot
    plot_frequency_boxplot(ax_B1, panel_label='b)')

    # Panel B2: Duration box plot
    plot_duration_boxplot(ax_B2, panel_label='c)')

    # Panel C1-C3: Scatter plots
    metrics_dict = {
        'stationary_ensemble': metrics_stat,
        'climate_adjusted_low': metrics_low,
        'climate_adjusted_high': metrics_high,
    }
    panel_labels = ['d)', 'e)', 'f)']

    scatter_axes = [ax_C1, ax_C2, ax_C3]
    scatter_objs = []

    for idx, (ax_c, dataset_id) in enumerate(zip(scatter_axes, SCENARIOS)):
        sc = plot_scatter_panel_simple(
            ax_c, metrics_dict[dataset_id], dataset_id, ffmp_lines,
            x_key, x_label, panel_labels[idx],
            vmin=ds_vmin, vmax=ds_vmax,
            show_xlabel=(idx == 1)  # Only middle plot shows xlabel
        )
        scatter_objs.append(sc)

    # Sync x-limits across C1-C3
    xlim_lo = min(ax.get_xlim()[0] for ax in scatter_axes)
    xlim_hi = max(ax.get_xlim()[1] for ax in scatter_axes)
    for ax in scatter_axes:
        ax.set_xlim(xlim_lo, xlim_hi)

    # Shared colorbar for scatter plots
    sc_ref = next((s for s in scatter_objs if s is not None), None)
    if sc_ref is not None:
        cbar_ax = fig.add_axes([0.08, 0.02, 0.5, 0.015])
        cbar = fig.colorbar(sc_ref, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Worst 1-mo demand satisfaction (%)', fontsize=FONTSIZE_SMALL)
        cbar.ax.tick_params(labelsize=FONTSIZE_SMALL-1)

    # Shared legend for box plots
    add_boxplot_legend(fig)

    return fig


def create_figure_simplified(n_mo, all_categorized):
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
    plot_panel_A(ax_A, all_categorized['stationary_ensemble'], n_months_prior=n_mo)

    # Panel B1: Frequency box plot
    plot_frequency_boxplot(ax_B1, panel_label='b)')

    # Panel B2: Duration box plot
    plot_duration_boxplot(ax_B2, panel_label='c)')

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

        # Full version with scatter plots (DEPRECATED - uncomment if needed)
        # print(f"  Creating full version (A + B1 + B2 + C1 + C2 + C3)...")
        # fig_full = create_figure_full(n_mo, all_categorized, metrics_stat, metrics_low,
        #                               metrics_high, ffmp_lines, ds_vmin, ds_vmax)
        # fname_full = f"{FIG_OUTPUT_DIR}/F3_composite_full_{suffix}.png"
        # fig_full.savefig(fname_full, dpi=DPI_HIGH, bbox_inches='tight')
        # print(f"    Saved: {fname_full}")
        # plt.close(fig_full)

    print("\n" + "=" * 70)
    print("All F3 figures generated successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
