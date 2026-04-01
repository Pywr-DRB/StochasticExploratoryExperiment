"""
F4: NYC Contribution Distribution Timeseries

Modes:
  1. Single dataset: Contribution/Montague distribution for one ensemble
  2. Comparison: Difference between two ensembles
  3. --montague / --multipanel: 3-panel stacked Montague contribution
  4. --ratio: 3-panel stacked contribution/inflow ratio (rolling window)
  5. --combined: 6-panel combined (both metrics side by side)

Usage:
    python F4_plot_contribution_distributions.py <dataset_id>
    python F4_plot_contribution_distributions.py --comparison <baseline_id> <comparison_id>
    python F4_plot_contribution_distributions.py --montague
    python F4_plot_contribution_distributions.py --ratio [--window 90]
    python F4_plot_contribution_distributions.py --combined [--window 90]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from methods.config import FIG_DIR, DATASET_CONFIGS, NYC_RESERVOIRS, verify_dataset_id
from methods.plotting.styles import (
    DPI_HIGH, DATASET_COLORS, DATASET_LABELS,
    FONTSIZE_LABEL, FONTSIZE_MEDIUM,
    apply_publication_style, label_panel,
)
from methods.contribution import (
    calculate_daily_contribution_percentage,
    calculate_daily_contribution_ratio,
    find_representative_year_for_zone,
    get_zone_filter_label,
    DEFAULT_WINDOW_DAYS,
)
from methods.plotting.percentile_bands import (
    calculate_percentiles,
    calculate_pairwise_difference_percentiles,
    format_xaxis_water_year,
    plot_bands,
    plot_difference_bands,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

FIG_OUTPUT_DIR = f"{FIG_DIR}/F4_contribution_timeseries"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

SCENARIOS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']

# Drought zone filtering (None = all years)
FILTER_BY_ZONES = None

# Show representative year trace in single-dataset mode
SHOW_REPRESENTATIVE_YEAR = True


# ============================================================================
# DATA LOADING HELPERS
# ============================================================================

def _load_dataset(dataset_id, results_sets):
    """Load a single pywrdrb dataset."""
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Data not found: {fname}")
    data = pywrdrb.Data()
    data.load_from_export(fname, results_sets=results_sets)
    return data


def _load_all_scenarios(results_sets):
    """Load data for all three scenarios."""
    all_data = {}
    for did in SCENARIOS:
        print(f"  Loading {did}...")
        all_data[did] = _load_dataset(did, results_sets)
    return all_data


# ============================================================================
# SHARED LEGEND
# ============================================================================

def _add_shared_legend(fig, alpha_outer=0.15, alpha_inner=0.35):
    """Add a shared grey-scale legend at the bottom of the figure."""
    legend_elements = [
        Patch(facecolor='grey', alpha=alpha_outer, edgecolor='none',
              label='1st–99th percentile'),
        Patch(facecolor='grey', alpha=alpha_inner, edgecolor='none',
              label='25th–75th percentile'),
        Line2D([0], [0], color='grey', linewidth=1.8, label='Median'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=3, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, -0.02))


# ============================================================================
# SHARED: 3-panel stacked helper
# ============================================================================

def _plot_stacked_3panel(stat_pcts, diff_low, diff_high,
                         stat_ylabel, diff_ylabel,
                         figsize=(12, 10)):
    """
    Create 3-panel stacked figure: stationary + 2 climate diffs.

    Returns fig.
    """
    apply_publication_style()

    fig, (ax_stat, ax_low, ax_high) = plt.subplots(
        3, 1, figsize=figsize, sharex=True,
        gridspec_kw={'hspace': 0.12,
                     'left': 0.08, 'right': 0.95,
                     'top': 0.96, 'bottom': 0.08},
    )
    plt.setp(ax_stat.get_xticklabels(), visible=False)
    plt.setp(ax_low.get_xticklabels(), visible=False)

    # (a) Stationary
    plot_bands(ax_stat, stat_pcts, DATASET_COLORS['stationary_ensemble'])
    format_xaxis_water_year(ax_stat)
    ax_stat.set_ylabel(stat_ylabel, fontsize=FONTSIZE_LABEL)
    ax_stat.set_ylim(bottom=0)
    ax_stat.grid(axis='y', alpha=0.3, linestyle='--')
    ax_stat.set_axisbelow(True)
    label_panel(ax_stat, 'a', 'stationary_ensemble')

    # (b) Climate low diff
    plot_difference_bands(ax_low, diff_low, DATASET_COLORS['climate_adjusted_low'])
    ax_low.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    ax_low.set_ylabel(diff_ylabel, fontsize=FONTSIZE_LABEL)
    ax_low.grid(axis='y', alpha=0.3, linestyle='--')
    ax_low.set_axisbelow(True)
    label_panel(ax_low, 'b', 'climate_adjusted_low')

    # (c) Climate high diff
    plot_difference_bands(ax_high, diff_high, DATASET_COLORS['climate_adjusted_high'])
    ax_high.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    format_xaxis_water_year(ax_high)
    ax_high.set_xlabel('Month', fontsize=FONTSIZE_LABEL)
    ax_high.set_ylabel(diff_ylabel, fontsize=FONTSIZE_LABEL)
    ax_high.grid(axis='y', alpha=0.3, linestyle='--')
    ax_high.set_axisbelow(True)
    label_panel(ax_high, 'c', 'climate_adjusted_high')

    # Match y-limits for difference panels
    y_lo = min(ax_low.get_ylim()[0], ax_high.get_ylim()[0])
    y_hi = max(ax_low.get_ylim()[1], ax_high.get_ylim()[1])
    ax_low.set_ylim(y_lo, y_hi)
    ax_high.set_ylim(y_lo, y_hi)

    _add_shared_legend(fig)

    return fig


def _plot_side_by_side_3panel(stat_pcts, diff_low, diff_high,
                               stat_ylabel, diff_ylabel,
                               figsize=(12, 6)):
    """
    Create 3-panel side-by-side figure: stationary on left, diffs stacked right.

    Returns fig.
    """
    apply_publication_style()

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
                          hspace=0.08, wspace=0.25,
                          left=0.08, right=0.95, top=0.95, bottom=0.12)
    ax_stat = fig.add_subplot(gs[:, 0])
    ax_low = fig.add_subplot(gs[0, 1])
    ax_high = fig.add_subplot(gs[1, 1], sharex=ax_low)
    plt.setp(ax_low.get_xticklabels(), visible=False)

    # (a) Stationary distribution
    plot_bands(ax_stat, stat_pcts, DATASET_COLORS['stationary_ensemble'],
               alpha_outer=0.2, alpha_inner=0.4)
    format_xaxis_water_year(ax_stat)
    ax_stat.set_xlabel('Month', fontsize=FONTSIZE_LABEL)
    ax_stat.set_ylabel(stat_ylabel, fontsize=FONTSIZE_LABEL)
    ax_stat.set_ylim(0, 100)
    ax_stat.grid(axis='y', alpha=0.3, linestyle='--')
    ax_stat.set_axisbelow(True)
    label_panel(ax_stat, 'a', 'stationary_ensemble')

    # (b) Climate low difference
    plot_difference_bands(ax_low, diff_low, DATASET_COLORS['climate_adjusted_low'],
                          alpha_outer=0.2, alpha_inner=0.4)
    ax_low.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    ax_low.set_ylabel('Change vs. Baseline\n(pp)', fontsize=10)
    ax_low.grid(axis='y', alpha=0.3, linestyle='--')
    ax_low.set_axisbelow(True)
    label_panel(ax_low, 'b', 'climate_adjusted_low')

    # (c) Climate high difference
    plot_difference_bands(ax_high, diff_high, DATASET_COLORS['climate_adjusted_high'],
                          alpha_outer=0.2, alpha_inner=0.4)
    ax_high.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    format_xaxis_water_year(ax_high)
    ax_high.set_xlabel('Month', fontsize=FONTSIZE_LABEL)
    ax_high.set_ylabel('Change vs. Baseline\n(pp)', fontsize=10)
    ax_high.grid(axis='y', alpha=0.3, linestyle='--')
    ax_high.set_axisbelow(True)
    label_panel(ax_high, 'c', 'climate_adjusted_high')

    # Match y-limits for difference panels
    y_lo = min(ax_low.get_ylim()[0], ax_high.get_ylim()[0])
    y_hi = max(ax_low.get_ylim()[1], ax_high.get_ylim()[1])
    ax_low.set_ylim(y_lo, y_hi)
    ax_high.set_ylim(y_lo, y_hi)

    _add_shared_legend(fig, alpha_outer=0.2, alpha_inner=0.4)

    return fig


# ============================================================================
# MODE: Single dataset
# ============================================================================

def plot_single_dataset(dataset_id):
    """Plot contribution/Montague distribution for a single dataset."""
    verify_dataset_id(dataset_id)
    print(f"F4: NYC contribution timeseries — {dataset_id}")

    results_sets = ['contribution', 'major_flow']
    if FILTER_BY_ZONES is not None or SHOW_REPRESENTATIVE_YEAR:
        results_sets.append('res_level')
    if SHOW_REPRESENTATIVE_YEAR:
        results_sets.append('inflow')

    data = _load_dataset(dataset_id, results_sets)

    contrib_df, n_total, n_filtered = calculate_daily_contribution_percentage(
        data, dataset_id, zone_filter=FILTER_BY_ZONES)

    representative_year = None
    if SHOW_REPRESENTATIVE_YEAR:
        representative_year = find_representative_year_for_zone(
            data, dataset_id, zone_filter=FILTER_BY_ZONES)

    pcts = calculate_percentiles(contrib_df)

    _, ax = plt.subplots(1, 1, figsize=(12, 5))
    plot_bands(ax, pcts, color='steelblue', alpha_outer=0.2, alpha_inner=0.4,
               label_prefix='', representative_year=representative_year)

    format_xaxis_water_year(ax)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('NYC Contribution to Montague Flow (%)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=9, frameon=True, fancybox=True)

    zone_label = get_zone_filter_label(FILTER_BY_ZONES)
    annotation = f'{zone_label}\nn = {n_filtered} / {n_total} water year-realizations'
    ax.text(0.02, 0.98, annotation, transform=ax.transAxes, fontsize=9,
            va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    zone_suffix = '_zones_' + '_'.join(map(str, sorted(FILTER_BY_ZONES, reverse=True))) if FILTER_BY_ZONES else ''
    fname = f"{FIG_OUTPUT_DIR}/F4_{dataset_id}_contribution{zone_suffix}.png"
    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close()


# ============================================================================
# MODE: Comparison
# ============================================================================

def plot_comparison(baseline_id, comparison_id):
    """Plot difference between two ensembles."""
    verify_dataset_id(baseline_id)
    verify_dataset_id(comparison_id)
    print(f"F4: Contribution comparison — {comparison_id} vs {baseline_id}")

    results_sets = ['contribution', 'major_flow']
    if FILTER_BY_ZONES is not None:
        results_sets.append('res_level')

    baseline_data = _load_dataset(baseline_id, results_sets)
    comparison_data = _load_dataset(comparison_id, results_sets)

    baseline_df, _, _ = calculate_daily_contribution_percentage(
        baseline_data, baseline_id, zone_filter=FILTER_BY_ZONES)
    comparison_df, _, _ = calculate_daily_contribution_percentage(
        comparison_data, comparison_id, zone_filter=FILTER_BY_ZONES)

    diff_pcts = calculate_pairwise_difference_percentiles(baseline_df, comparison_df)

    _, ax = plt.subplots(1, 1, figsize=(12, 5))
    plot_difference_bands(ax, diff_pcts, color='steelblue',
                          alpha_outer=0.2, alpha_inner=0.4, label_prefix='')

    ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    format_xaxis_water_year(ax)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Contribution Change (% points)', fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    zone_label = get_zone_filter_label(FILTER_BY_ZONES)
    annotation = f'Change: {comparison_id}\nrelative to {baseline_id}\n{zone_label}'
    ax.text(0.02, 0.98, annotation, transform=ax.transAxes, fontsize=9,
            va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    zone_suffix = '_zones_' + '_'.join(map(str, sorted(FILTER_BY_ZONES, reverse=True))) if FILTER_BY_ZONES else ''
    fname = f"{FIG_OUTPUT_DIR}/F4_{comparison_id}_vs_{baseline_id}_comparison{zone_suffix}.png"
    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close()


# ============================================================================
# MODE: --montague  (3-panel Montague contribution)
# ============================================================================

def plot_montague_multipanel(zone_filter=None, layout='stacked'):
    """3-panel Montague contribution: stacked or side_by_side."""
    print(f"F4: Multi-panel Montague contribution comparison (layout={layout})")

    results_sets = ['contribution', 'major_flow']
    if zone_filter is not None:
        results_sets.append('res_level')

    all_data = _load_all_scenarios(results_sets)

    all_dfs = {}
    for did in SCENARIOS:
        all_dfs[did], _, _ = calculate_daily_contribution_percentage(
            all_data[did], did, zone_filter=zone_filter)

    stat_pcts = calculate_percentiles(all_dfs['stationary_ensemble'])
    diff_low = calculate_pairwise_difference_percentiles(
        all_dfs['stationary_ensemble'], all_dfs['climate_adjusted_low'])
    diff_high = calculate_pairwise_difference_percentiles(
        all_dfs['stationary_ensemble'], all_dfs['climate_adjusted_high'])

    if layout == 'side_by_side':
        fig = _plot_side_by_side_3panel(
            stat_pcts, diff_low, diff_high,
            stat_ylabel='NYC Contribution to Montague Flow (%)',
            diff_ylabel='Change vs. Baseline (pp)',
        )
    else:
        fig = _plot_stacked_3panel(
            stat_pcts, diff_low, diff_high,
            stat_ylabel='NYC Contribution to Montague Flow (%)',
            diff_ylabel='Change in Distribution\nvs. Baseline (pp)',
        )

    zone_suffix = '_zones_' + '_'.join(map(str, sorted(zone_filter, reverse=True))) if zone_filter else ''
    layout_suffix = f'_{layout}' if layout != 'side_by_side' else ''
    fname = f"{FIG_OUTPUT_DIR}/F4_multipanel_contribution_comparison{layout_suffix}{zone_suffix}.png"
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close(fig)


# ============================================================================
# MODE: --ratio  (3-panel contribution/inflow ratio)
# ============================================================================

def plot_ratio_multipanel(window=None):
    """3-panel stacked: stationary ratio distribution + 2 climate diffs."""
    if window is None:
        window = DEFAULT_WINDOW_DAYS
    window_months = window // 30

    print(f"F4: Multi-panel contribution ratio — {window}-day window")

    all_data = _load_all_scenarios(['contribution', 'inflow'])

    all_dfs = {}
    for did in SCENARIOS:
        all_dfs[did] = calculate_daily_contribution_ratio(
            all_data[did], did, window=window)

    stat_pcts = calculate_percentiles(all_dfs['stationary_ensemble'])
    diff_low = calculate_pairwise_difference_percentiles(
        all_dfs['stationary_ensemble'], all_dfs['climate_adjusted_low'])
    diff_high = calculate_pairwise_difference_percentiles(
        all_dfs['stationary_ensemble'], all_dfs['climate_adjusted_high'])

    fig = _plot_stacked_3panel(
        stat_pcts, diff_low, diff_high,
        stat_ylabel=f'NYC Contribution / NYC Inflow\n({window_months}-month rolling window, %)',
        diff_ylabel='Change in Contribution Ratio\nvs. Baseline (pp)',
    )

    fname = f"{FIG_OUTPUT_DIR}/F4_contribution_ratio_{window}d_stacked.png"
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close(fig)


# ============================================================================
# MODE: --combined  (6-panel)
# ============================================================================

def plot_combined(window=None):
    """6-panel (3×2): ratio on left, Montague on right."""
    if window is None:
        window = DEFAULT_WINDOW_DAYS
    window_months = window // 30

    print(f"F4: Combined contribution figure — {window}-day window")

    apply_publication_style()

    # Load data (need contribution, inflow, major_flow)
    all_data = _load_all_scenarios(['contribution', 'inflow', 'major_flow'])

    # Compute both metrics
    ratio_dfs = {}
    montague_dfs = {}
    for did in SCENARIOS:
        ratio_dfs[did] = calculate_daily_contribution_ratio(
            all_data[did], did, window=window)
        montague_dfs[did], _, _ = calculate_daily_contribution_percentage(
            all_data[did], did, zone_filter=None)

    # Percentiles and diffs
    ratio_stat = calculate_percentiles(ratio_dfs['stationary_ensemble'])
    mont_stat = calculate_percentiles(montague_dfs['stationary_ensemble'])

    ratio_diff, mont_diff = {}, {}
    for did in ['climate_adjusted_low', 'climate_adjusted_high']:
        ratio_diff[did] = calculate_pairwise_difference_percentiles(
            ratio_dfs['stationary_ensemble'], ratio_dfs[did])
        mont_diff[did] = calculate_pairwise_difference_percentiles(
            montague_dfs['stationary_ensemble'], montague_dfs[did])

    # Build figure: 3 rows × 2 columns
    print("Plotting...")
    fig = plt.figure(figsize=(14, 9.5))
    gs = gridspec.GridSpec(
        3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1],
        hspace=0.08, wspace=0.22,
        left=0.08, right=0.96, top=0.93, bottom=0.09,
    )

    axes = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(3)]

    # Format x-axis on all panels
    for r in range(3):
        for c in range(2):
            format_xaxis_water_year(axes[r][c])
            if r < 2:
                plt.setp(axes[r][c].get_xticklabels(), visible=False)

    # Column headers
    axes[0][0].set_title(
        f'NYC Contribution / NYC Inflow\n({window_months}-mo rolling window)',
        fontsize=12, pad=8)
    axes[0][1].set_title(
        'NYC Contribution / Montague Flow\n(daily)', fontsize=12, pad=8)

    # Row 0: Stationary
    plot_bands(axes[0][0], ratio_stat, DATASET_COLORS['stationary_ensemble'])
    plot_bands(axes[0][1], mont_stat, DATASET_COLORS['stationary_ensemble'])
    for c in range(2):
        axes[0][c].set_ylim(bottom=0)
        axes[0][c].grid(axis='y', alpha=0.2, linestyle='--')
        axes[0][c].set_axisbelow(True)
    axes[0][0].set_ylabel(
        f'{DATASET_LABELS["stationary_ensemble"]}\n\nContribution Ratio (%)',
        fontsize=FONTSIZE_LABEL)

    # Row 1: Climate low diff
    did = 'climate_adjusted_low'
    plot_bands(axes[1][0], ratio_diff[did], DATASET_COLORS[did])
    plot_bands(axes[1][1], mont_diff[did], DATASET_COLORS[did])
    for c in range(2):
        axes[1][c].axhline(0, color='black', linewidth=0.7, alpha=0.4)
        axes[1][c].grid(axis='y', alpha=0.2, linestyle='--')
        axes[1][c].set_axisbelow(True)
    axes[1][0].set_ylabel(
        f'{DATASET_LABELS[did]}\n\nChange vs. Baseline (pp)',
        fontsize=FONTSIZE_LABEL)

    # Row 2: Climate high diff
    did = 'climate_adjusted_high'
    plot_bands(axes[2][0], ratio_diff[did], DATASET_COLORS[did])
    plot_bands(axes[2][1], mont_diff[did], DATASET_COLORS[did])
    for c in range(2):
        axes[2][c].axhline(0, color='black', linewidth=0.7, alpha=0.4)
        axes[2][c].grid(axis='y', alpha=0.2, linestyle='--')
        axes[2][c].set_axisbelow(True)
    axes[2][0].set_ylabel(
        f'{DATASET_LABELS[did]}\n\nChange vs. Baseline (pp)',
        fontsize=FONTSIZE_LABEL)
    axes[2][0].set_xlabel('Month', fontsize=FONTSIZE_LABEL)
    axes[2][1].set_xlabel('Month', fontsize=FONTSIZE_LABEL)

    # Match y-limits within each column for diff panels
    for c in range(2):
        y_lo = min(axes[1][c].get_ylim()[0], axes[2][c].get_ylim()[0])
        y_hi = max(axes[1][c].get_ylim()[1], axes[2][c].get_ylim()[1])
        axes[1][c].set_ylim(y_lo, y_hi)
        axes[2][c].set_ylim(y_lo, y_hi)

    # Panel labels (inside axes, with dataset label)
    panel_info = [
        [('a', 'stationary_ensemble'), ('b', 'stationary_ensemble')],
        [('c', 'climate_adjusted_low'), ('d', 'climate_adjusted_low')],
        [('e', 'climate_adjusted_high'), ('f', 'climate_adjusted_high')],
    ]
    for r in range(3):
        for c in range(2):
            letter, did = panel_info[r][c]
            label_panel(axes[r][c], letter, did)

    _add_shared_legend(fig)

    fname = f"{FIG_OUTPUT_DIR}/F4_contribution_combined_{window}d.png"
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close(fig)


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='NYC Contribution Timeseries Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python F4_plot_contribution_distributions.py stationary_ensemble
  python F4_plot_contribution_distributions.py --comparison stationary_ensemble climate_adjusted_low
  python F4_plot_contribution_distributions.py --montague
  python F4_plot_contribution_distributions.py --ratio --window 90
  python F4_plot_contribution_distributions.py --combined --window 90

Available datasets: {list(DATASET_CONFIGS.keys())}
        """
    )
    parser.add_argument('dataset_id', nargs='?', help='Single dataset mode')
    parser.add_argument('--comparison', '-c', nargs=2,
                        metavar=('BASELINE', 'COMPARISON'),
                        help='Comparison: differences between two ensembles')
    parser.add_argument('--montague', action='store_true',
                        help='3-panel stacked Montague contribution')
    parser.add_argument('--multipanel', '-m', action='store_true',
                        help='Alias for --montague (backward compat)')
    parser.add_argument('--ratio', action='store_true',
                        help='3-panel stacked contribution/inflow ratio')
    parser.add_argument('--combined', action='store_true',
                        help='6-panel combined (both metrics)')
    parser.add_argument('--window', '-w', type=int, default=DEFAULT_WINDOW_DAYS,
                        help=f'Rolling window in days (default {DEFAULT_WINDOW_DAYS})')
    parser.add_argument('--layout', choices=['stacked', 'side_by_side'],
                        default='stacked',
                        help='Layout for --montague mode (default: stacked)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.combined:
        plot_combined(window=args.window)
    elif args.ratio:
        plot_ratio_multipanel(window=args.window)
    elif args.montague or args.multipanel:
        plot_montague_multipanel(zone_filter=FILTER_BY_ZONES, layout=args.layout)
    elif args.comparison:
        plot_comparison(*args.comparison)
    elif args.dataset_id:
        plot_single_dataset(args.dataset_id)
    else:
        print(__doc__)
        print(f"\nAvailable datasets: {list(DATASET_CONFIGS.keys())}")
