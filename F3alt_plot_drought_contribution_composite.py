"""
F3alt: Alternative composite drought contribution figure.

Layout (5 panels, 2 columns):
  Left column:   a) Fraction of years per drought zone (box plots, all 3 scenarios)
                 b) Time in storage zone (box plots, all 3 scenarios)
  Right column:  c) Baseline KDE of contribution ratio by drought zone (stationary)
                 d) Delta KDE for climate_adjusted_low  (climate - baseline density)
                 e) Delta KDE for climate_adjusted_high (climate - baseline density)

The right-column delta panels follow the visual idiom of F2 and F4, showing
how the contribution-ratio distribution shifts under each climate scenario
relative to the stationary baseline.

Usage:
    python F3alt_plot_drought_contribution_composite.py
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

from methods.config import FIG_DIR
from methods.plotting.styles import (
    DPI_HIGH, DATASET_COLORS, DATASET_LABELS,
    FONTSIZE_SMALL, FONTSIZE_LABEL, FONTSIZE_MEDIUM,
    ALPHA_LINE,
    apply_publication_style,
)
from methods.load import load_contribution_metrics

# Reuse from F3
from F3_plot_drought_contribution_composite import (
    SCENARIOS, WINDOW_MONTHS, KDE_CATEGORIES,
    load_all_data, categorize_all_scenarios,
    plot_frequency_boxplot, plot_duration_boxplot,
)

# Reuse from water_balance_by_drought_zone (F4 module)
import methods.plotting.water_balance_by_drought_zone as F4_module
from methods.plotting.water_balance_by_drought_zone import (
    compute_kde_on_grid,
    calculate_reconstruction_contribution_ratio,
    DROUGHT_CATEGORIES,
    MIN_INFLOW_THRESHOLD,
    XLIM_MAX_MANUAL,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

FIG_OUTPUT_DIR = f"{FIG_DIR}/F3_composite_figures"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

N_KDE_POINTS = 500  # Resolution of shared x-grid for KDE evaluation


# ============================================================================
# HELPERS
# ============================================================================

def extract_ratio_data(categorized_data, categories=None):
    """
    Extract filtered contribution ratio arrays from categorized data.

    Parameters
    ----------
    categorized_data : dict
        Output from categorize_by_drought_zone() or categorize_by_zone().
    categories : list of str, optional
        Categories to extract. Default: KDE_CATEGORIES.

    Returns
    -------
    dict
        {cat: {'ratio': np.ndarray, 'n': int}} for categories with data.
    """
    if categories is None:
        categories = KDE_CATEGORIES

    result = {}
    for cat in categories:
        df = categorized_data[cat].copy()
        if len(df) == 0:
            continue
        df = df[df['inflow_total'] > MIN_INFLOW_THRESHOLD]
        if len(df) < 2:
            continue
        ratio = (100.0 * df['contribution_total'] / df['inflow_total']).values
        result[cat] = {'ratio': ratio, 'n': len(ratio)}
    return result


def compute_kde_delta(baseline_categorized, climate_categorized, x_grid,
                      categories=None):
    """
    Compute KDE delta (climate minus baseline density) per drought zone.

    Parameters
    ----------
    baseline_categorized : dict
        Categorized data for the stationary ensemble.
    climate_categorized : dict
        Categorized data for a climate scenario.
    x_grid : np.ndarray
        Shared x values for KDE evaluation.
    categories : list of str, optional
        Drought zone categories to process.

    Returns
    -------
    dict
        {cat: {'delta': array, 'baseline': array, 'climate': array,
               'n_baseline': int, 'n_climate': int}}
    """
    if categories is None:
        categories = KDE_CATEGORIES

    baseline_data = extract_ratio_data(baseline_categorized, categories)
    climate_data = extract_ratio_data(climate_categorized, categories)

    results = {}
    for cat in categories:
        if cat not in baseline_data or cat not in climate_data:
            continue
        base_kde = compute_kde_on_grid(baseline_data[cat]['ratio'], x_grid)
        clim_kde = compute_kde_on_grid(climate_data[cat]['ratio'], x_grid)
        results[cat] = {
            'delta': clim_kde - base_kde,
            'baseline': base_kde,
            'climate': clim_kde,
            'n_baseline': baseline_data[cat]['n'],
            'n_climate': climate_data[cat]['n'],
        }
    return results


# ============================================================================
# PANEL PLOTTING
# ============================================================================

def plot_panel_kde_baseline(ax, categorized_data, n_months_prior,
                            panel_label='c)'):
    """
    Plot baseline (absolute) KDE of contribution ratio by drought zone.

    Adapted from F3 plot_panel_A with configurable panel label.
    """
    ratio_data = extract_ratio_data(categorized_data)
    xlim_max = XLIM_MAX_MANUAL if XLIM_MAX_MANUAL is not None else 100

    for cat in KDE_CATEGORIES:
        if cat not in ratio_data:
            continue
        cat_info = DROUGHT_CATEGORIES[cat]
        ratio = pd.Series(ratio_data[cat]['ratio'])
        n = ratio_data[cat]['n']

        label = (f"{cat_info['label']} (n={n})" if cat != 'other'
                 else f"Normal or Above (n={n})")
        ratio.plot.kde(ax=ax, color=cat_info['color'], linewidth=2.5,
                       alpha=ALPHA_LINE, label=label)
        ax.axvline(ratio.mean(), color=cat_info['color'], linestyle='--',
                   linewidth=1.5, alpha=0.7)

    # Dummy for mean legend entry
    ax.axvline(np.nan, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
               label='Mean')

    # 1964 reconstruction
    reconstruction_ratio = calculate_reconstruction_contribution_ratio()
    if reconstruction_ratio is not None:
        ax.axvline(reconstruction_ratio, color='black', linestyle='-',
                   linewidth=2.5, alpha=0.9, label='1964 Drought')
        if reconstruction_ratio > xlim_max:
            xlim_max = reconstruction_ratio * 1.1

    xlabel = (f'NYC contributions / total inflow\n'
              f'({n_months_prior}-mo prior to min zone, %)')
    ax.set_xlabel(xlabel, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Density', fontsize=FONTSIZE_LABEL)
    ax.set_xlim(left=0, right=xlim_max)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    ax.text(-0.05, 1.02, panel_label, transform=ax.transAxes,
            fontsize=14, va='bottom', ha='right')


def plot_panel_kde_delta(ax, delta_data, x_grid, scenario_label,
                         n_months_prior, panel_label='d)'):
    """
    Plot delta KDE (climate minus baseline density) per drought zone.

    Parameters
    ----------
    delta_data : dict
        Output from compute_kde_delta for one scenario.
    x_grid : np.ndarray
        Shared x-grid.
    scenario_label : str
        Display label for the scenario (used as title).
    n_months_prior : int
        Window length (for x-axis label).
    panel_label : str
        Panel identifier.
    """
    for cat in KDE_CATEGORIES:
        if cat not in delta_data:
            continue
        cat_info = DROUGHT_CATEGORIES[cat]
        delta = delta_data[cat]['delta']
        n_clim = delta_data[cat]['n_climate']
        label = (f"{cat_info['label']} (n={n_clim})" if cat != 'other'
                 else f"Normal or Above (n={n_clim})")
        ax.plot(x_grid, delta, color=cat_info['color'], linewidth=2.0,
                alpha=ALPHA_LINE, label=label)
        # Light fill to zero
        ax.fill_between(x_grid, 0, delta, color=cat_info['color'], alpha=0.15)

    # Reference line
    ax.axhline(0, color='gray', linestyle='-', linewidth=1.0, alpha=0.5)

    ax.set_title(scenario_label, fontsize=FONTSIZE_MEDIUM)
    xlabel = (f'NYC contributions / total inflow\n'
              f'({n_months_prior}-mo prior to min zone, %)')
    ax.set_xlabel(xlabel, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('\u0394 Density', fontsize=FONTSIZE_LABEL)

    xlim_max = XLIM_MAX_MANUAL if XLIM_MAX_MANUAL is not None else 100
    ax.set_xlim(left=0, right=xlim_max)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    ax.text(-0.05, 1.02, panel_label, transform=ax.transAxes,
            fontsize=14, va='bottom', ha='right')


# ============================================================================
# LEGEND
# ============================================================================

def add_combined_legend(fig):
    """
    Combined legend for KDE zone colors, reference lines, and dataset patches.
    """
    elements = []

    # Zone-color lines
    for cat in KDE_CATEGORIES:
        cat_info = DROUGHT_CATEGORIES[cat]
        label = cat_info['label'] if cat != 'other' else 'Normal'
        elements.append(Line2D([0], [0], color=cat_info['color'],
                               linewidth=2.5, label=label))

    # Reference lines
    elements.append(Line2D([0], [0], color='gray', linestyle='--',
                           linewidth=1.5, label='Mean'))
    elements.append(Line2D([0], [0], color='black', linestyle='-',
                           linewidth=2.5, label='1964 Drought'))

    # Dataset patches (for box plots)
    for dataset_id in SCENARIOS:
        elements.append(Patch(facecolor=DATASET_COLORS[dataset_id], alpha=0.7,
                              edgecolor='black', linewidth=1.2,
                              label=DATASET_LABELS[dataset_id]))

    # Historic marker
    elements.append(Line2D([0], [0], color='black', marker='^',
                           linestyle='None', markersize=8, label='Historic'))

    fig.legend(handles=elements, loc='lower center',
               ncol=5, fontsize=FONTSIZE_SMALL,
               frameon=False, bbox_to_anchor=(0.5, -0.03))


# ============================================================================
# FIGURE LAYOUT
# ============================================================================

def create_figure_alt(n_mo, all_categorized, show_historic=False):
    """
    Create alternative 5-panel composite figure.

    Left:  a) frequency box plots, b) duration box plots
    Right: c) baseline KDE, d) delta KDE low, e) delta KDE high
    """
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(
        6, 2,
        width_ratios=[1, 1.3],
        hspace=0.55, wspace=0.35,
        left=0.08, right=0.95, top=0.95, bottom=0.08,
    )

    # Left column: box plots
    ax_B1 = fig.add_subplot(gs[0:3, 0])
    ax_B2 = fig.add_subplot(gs[3:6, 0])

    # Right column: KDE panels (shared x-axis)
    ax_kde_base = fig.add_subplot(gs[0:2, 1])
    ax_kde_low = fig.add_subplot(gs[2:4, 1], sharex=ax_kde_base)
    ax_kde_high = fig.add_subplot(gs[4:6, 1], sharex=ax_kde_base)

    # --- Left panels ---
    plot_frequency_boxplot(ax_B1, panel_label='a)', show_historic=show_historic)
    plot_duration_boxplot(ax_B2, panel_label='b)', show_historic=show_historic)
    ax_B2.set_xlabel('NYC Storage Zone', fontsize=FONTSIZE_LABEL)

    label_x = -0.15
    for ax in [ax_B1, ax_B2]:
        ax.yaxis.set_label_coords(label_x, 0.5)

    # --- Right panels ---
    # Shared x-grid for KDE evaluation
    xlim_max = XLIM_MAX_MANUAL if XLIM_MAX_MANUAL is not None else 100
    x_grid = np.linspace(0, xlim_max, N_KDE_POINTS)

    # c) Baseline KDE
    plot_panel_kde_baseline(ax_kde_base, all_categorized['stationary_ensemble'],
                            n_months_prior=n_mo, panel_label='c)')

    # d) Delta KDE — climate low
    delta_low = compute_kde_delta(
        all_categorized['stationary_ensemble'],
        all_categorized['climate_adjusted_low'],
        x_grid,
    )
    plot_panel_kde_delta(ax_kde_low, delta_low, x_grid,
                         DATASET_LABELS['climate_adjusted_low'],
                         n_months_prior=n_mo, panel_label='d)')

    # e) Delta KDE — climate high
    delta_high = compute_kde_delta(
        all_categorized['stationary_ensemble'],
        all_categorized['climate_adjusted_high'],
        x_grid,
    )
    plot_panel_kde_delta(ax_kde_high, delta_high, x_grid,
                         DATASET_LABELS['climate_adjusted_high'],
                         n_months_prior=n_mo, panel_label='e)')

    # Sync y-limits between delta panels (symmetric)
    ymin = min(ax_kde_low.get_ylim()[0], ax_kde_high.get_ylim()[0])
    ymax = max(ax_kde_low.get_ylim()[1], ax_kde_high.get_ylim()[1])
    ylim = max(abs(ymin), abs(ymax))
    ax_kde_low.set_ylim(-ylim, ylim)
    ax_kde_high.set_ylim(-ylim, ylim)

    # Hide x-tick labels on top two right panels
    plt.setp(ax_kde_base.get_xticklabels(), visible=False)
    ax_kde_base.set_xlabel('')
    plt.setp(ax_kde_low.get_xticklabels(), visible=False)
    ax_kde_low.set_xlabel('')

    # Legend
    add_combined_legend(fig)

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

    print("F3alt: Alternative composite drought contribution figure")
    print("=" * 70)

    # Try cached fast path
    use_cached = True
    try:
        from methods.metrics.contribution import get_metrics_for_window, categorize_by_zone

        metrics_cache = {}
        for scenario in SCENARIOS:
            metrics_cache[scenario] = load_contribution_metrics(scenario)

    except (ImportError, FileNotFoundError):
        use_cached = False

    if not use_cached:
        all_data = load_all_data()

    for n_mo in WINDOW_MONTHS:
        print(f"\nProcessing {n_mo}-month window...")

        F4_module.N_MONTHS_PRIOR = n_mo

        if use_cached:
            from methods.metrics.contribution import get_metrics_for_window, categorize_by_zone

            window_days = n_mo * 30
            column_rename_map = {
                f'contribution_total_{window_days}d': 'contribution_total',
                f'contribution_ratio_{window_days}d': 'contribution_ratio',
                f'inflow_total_{window_days}d': 'inflow_total',
                f'demand_satisfaction_{window_days}d': 'demand_satisfaction',
                f'worst_1mo_demand_sat_{window_days}d': 'worst_1mo_demand_sat',
            }

            all_categorized = {}
            zone_categories = {
                'emergency': [6],
                'watch': [5],
                'warning': [4],
                'other': [0, 1, 2, 3],
            }
            for scenario in SCENARIOS:
                window_df = get_metrics_for_window(metrics_cache[scenario], window_days)
                window_df = window_df.rename(columns=column_rename_map)
                all_categorized[scenario] = categorize_by_zone(window_df, zone_categories)

        else:
            all_categorized = categorize_all_scenarios(all_data, n_mo)

        fig = create_figure_alt(n_mo, all_categorized)
        suffix = f'contribution_ratio_{n_mo}mo'
        fname = f"{FIG_OUTPUT_DIR}/F3alt_composite_{suffix}.png"
        fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
        print(f"  Saved: {fname}")
        plt.close(fig)

    print("\n" + "=" * 70)
    print("All F3alt figures generated successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
