"""
SI9: Comprehensive performance metric distribution overview.

Multi-panel figure with one subplot per metric. Each subplot shows
per-realization empirical CDFs of annual water-year values, summarized
as median + min/max envelope across ~2000 realizations.

For each realization, a CDF is built from its ~70 annual values. All
realization CDFs are evaluated on a shared x-grid, then the median (p50),
min (p0), and max (p100) CDF lines are plotted — matching the style of
F2_plot_drought_metric_distribution.py.

Usage:
    python SI9_plot_metric_distributions.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerTuple
import warnings
warnings.filterwarnings("ignore")

from methods.config import FIG_DIR, DATASET_CONFIGS
from methods.plotting.styles import (
    DATASET_COLORS, DATASET_LABELS,
    DPI_HIGH, LINEWIDTH_MEDIUM,
    FONTSIZE_SMALL, FONTSIZE_MEDIUM,
    DATASET_LINESTYLES,
    apply_publication_style,
)
from methods.load import load_annual_metrics, load_hashimoto_metrics

# ============================================================================
# CONFIG
# ============================================================================
FIG_OUTPUT_DIR = f"{FIG_DIR}/SI9_metric_distributions"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

DATASETS = list(DATASET_CONFIGS.keys())

# Each metric: (column_name, subplot_title, x_axis_label)
ANNUAL_METRICS = [
    # --- Montague flow target ---
    ('montague_reliability',
     'Montague: Weekly Reliability (Weeks Without 3+ Deficit Days)',
     'Fraction of Weeks Without Failure'),

    ('montague_shortage_mg',
     'Montague: Annual Shortage Volume',
     'Shortage (MG/yr)'),

    ('montague_max_consec_shortage_days',
     'Montague: Max Consecutive Shortage Days per Year',
     'Consecutive Days'),

    ('montague_max_1day_shortage_mg',
     'Montague: Peak Single-Day Shortage per Year',
     'Single-Day Shortage (MG)'),

    # --- Trenton flow target ---
    ('trenton_reliability',
     'Trenton: Weekly Reliability (Weeks Without 3+ Deficit Days)',
     'Fraction of Weeks Without Failure'),

    ('trenton_shortage_mg',
     'Trenton: Annual Shortage Volume',
     'Shortage (MG/yr)'),

    ('trenton_max_consec_shortage_days',
     'Trenton: Max Consecutive Shortage Days per Year',
     'Consecutive Days'),

    ('trenton_max_1day_shortage_mg',
     'Trenton: Peak Single-Day Shortage per Year',
     'Single-Day Shortage (MG)'),

    # --- NYC diversion target ---
    ('nyc_reliability',
     'NYC: Weekly Diversion Reliability (Weeks Without 3+ Deficit Days)',
     'Fraction of Weeks Without Failure'),

    ('nyc_shortage_mg',
     'NYC: Annual Diversion Shortage Volume',
     'Shortage (MG/yr)'),

    ('nyc_max_consec_shortage_days',
     'NYC: Max Consecutive Diversion Shortage Days per Year',
     'Consecutive Days'),

    ('nyc_max_1day_shortage_mg',
     'NYC: Peak Single-Day Diversion Shortage per Year',
     'Single-Day Shortage (MG)'),

    # --- NYC reservoir storage ---
    ('nyc_min_storage_pct',
     'NYC: Annual Minimum Combined Reservoir Storage',
     'Storage (% of Capacity)'),

    ('june1_storage_pct',
     'NYC: June 1 Combined Reservoir Storage',
     'Storage (% of Capacity)'),

    ('sept1_storage_pct',
     'NYC: September 1 Combined Reservoir Storage',
     'Storage (% of Capacity)'),

    ('ndays_storage_below_20pct',
     'NYC: Days per Year with Storage Below 20%',
     'Days per Water Year'),

    ('ndays_storage_below_30pct',
     'NYC: Days per Year with Storage Below 30%',
     'Days per Water Year'),

    # --- System-level ---
    ('nyc_contribution_mg',
     'NYC: Annual Downstream Flow Contribution',
     'Release Volume (MG/yr)'),

    ('ndays_combined_stress',
     'Days per Year with Simultaneous Montague + NYC Shortage',
     'Days per Water Year'),

    ('max_zone',
     'NYC: Maximum Drought Zone Reached per Year',
     'Zone Level (1=Normal, 5=Emergency)'),
]

# Hashimoto metrics are simulation-level (1 value per realization, no annual CDF)
# They are plotted as a simple CDF across realizations.
HASHIMOTO_METRICS = [
    ('hashimoto_reliability_montague',
     'Hashimoto: Montague Reliability (70-yr)',
     'Fraction of Days Without Deficit'),

    ('hashimoto_resiliency_montague',
     'Hashimoto: Montague Resiliency (70-yr)',
     'P(Recovery | Deficit Day)'),

    ('hashimoto_reliability_trenton',
     'Hashimoto: Trenton Reliability (70-yr)',
     'Fraction of Days Without Deficit'),

    ('hashimoto_resiliency_trenton',
     'Hashimoto: Trenton Resiliency (70-yr)',
     'P(Recovery | Deficit Day)'),
]

HASHIMOTO_NAMES = {m[0] for m in HASHIMOTO_METRICS}

# Direction: True = higher is better (right = better by default)
HIGHER_IS_BETTER = {
    'montague_reliability': True,
    'trenton_reliability': True,
    'nyc_reliability': True,
    'montague_shortage_mg': False,
    'trenton_shortage_mg': False,
    'nyc_shortage_mg': False,
    'montague_max_consec_shortage_days': False,
    'trenton_max_consec_shortage_days': False,
    'nyc_max_consec_shortage_days': False,
    'montague_max_1day_shortage_mg': False,
    'trenton_max_1day_shortage_mg': False,
    'nyc_max_1day_shortage_mg': False,
    'nyc_min_storage_pct': True,
    'june1_storage_pct': True,
    'sept1_storage_pct': True,
    'ndays_storage_below_20pct': False,
    'ndays_storage_below_30pct': False,
    'nyc_contribution_mg': False,
    'ndays_combined_stress': False,
    'max_zone': False,
    'hashimoto_reliability_montague': True,
    'hashimoto_resiliency_montague': True,
    'hashimoto_reliability_trenton': True,
    'hashimoto_resiliency_trenton': True,
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_data():
    """Load annual metrics and hashimoto metrics for all datasets."""
    annual = {}
    hashimoto = {}
    for did in DATASETS:
        annual[did] = load_annual_metrics(did)
        try:
            hashimoto[did] = load_hashimoto_metrics(did)
        except FileNotFoundError as e:
            print(f"  Warning: Hashimoto metrics not found for {did}: {e}")
            hashimoto[did] = None
    return annual, hashimoto


# ============================================================================
# PER-REALIZATION CDF BANDS (F2-style)
# ============================================================================

def compute_cdf_bands(df, metric_name, period='all', n_grid=300,
                      percentiles=(0, 50, 100)):
    """Build per-realization CDFs and return percentile envelopes.

    For each realization, builds an empirical CDF from its ~70 annual
    values, evaluates all CDFs on a shared x-grid, then computes
    percentile bands across realizations.

    Parameters
    ----------
    df : pd.DataFrame
        Annual metrics with columns: realization_id, period, metric_name.
    metric_name : str
        Column to build CDFs from.
    period : str
        Period filter ('all', 'drought', 'nondrought').
    n_grid : int
        Resolution of the shared x-grid.
    percentiles : tuple of int
        Percentiles to compute across realization CDFs.

    Returns
    -------
    x_grid : np.ndarray, shape (n_grid,)
    bands : dict {percentile: np.ndarray of shape (n_grid,)}
    """
    subset = df[df['period'] == period]
    if metric_name not in subset.columns:
        return None, None

    all_vals = subset[metric_name].dropna().values
    if len(all_vals) == 0:
        return None, None

    x_min, x_max = np.nanmin(all_vals), np.nanmax(all_vals)
    # Add small padding to avoid edge effects
    pad = (x_max - x_min) * 0.01 if x_max > x_min else 0.5
    x_grid = np.linspace(x_min - pad, x_max + pad, n_grid)

    realization_ids = sorted(subset['realization_id'].unique())
    curves = np.zeros((len(realization_ids), n_grid))

    for i, rid in enumerate(realization_ids):
        vals = np.sort(
            subset.loc[subset['realization_id'] == rid, metric_name].dropna().values
        )
        if len(vals) == 0:
            curves[i, :] = np.nan
            continue
        # CDF: fraction of annual values <= x
        # Using searchsorted: count of vals <= x = searchsorted(vals, x, side='right')
        curves[i, :] = np.searchsorted(vals, x_grid, side='right') / len(vals)

    bands = {}
    for p in percentiles:
        bands[p] = np.nanpercentile(curves, p, axis=0)

    return x_grid, bands


# ============================================================================
# MAIN FIGURE
# ============================================================================

def make_distribution_figure(annual, hashimoto, period='all'):
    """Create multi-panel figure with per-realization CDF bands."""

    all_metrics = list(ANNUAL_METRICS) + list(HASHIMOTO_METRICS)
    n_metrics = len(all_metrics)
    ncols = 4
    nrows = int(np.ceil(n_metrics / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    for i, (metric_name, title, xlabel) in enumerate(all_metrics):
        ax = axes[i]
        has_data = False

        for did in DATASETS:
            color = DATASET_COLORS[did]
            ls = DATASET_LINESTYLES.get(did, '-')

            if metric_name in HASHIMOTO_NAMES:
                # Hashimoto: simulation-level, plot simple CDF across realizations
                df_h = hashimoto.get(did)
                if df_h is None or metric_name not in df_h.columns:
                    continue
                vals = np.sort(df_h[metric_name].dropna().values)
                if len(vals) == 0:
                    continue
                cdf = np.arange(1, len(vals) + 1) / len(vals)
                ax.plot(vals, cdf, color=color, linestyle=ls,
                        linewidth=LINEWIDTH_MEDIUM,
                        label=DATASET_LABELS.get(did, did))
                has_data = True
            else:
                # Annual metric: per-realization CDF bands
                x_grid, bands = compute_cdf_bands(
                    annual[did], metric_name, period=period
                )
                if x_grid is None:
                    continue

                # Shaded band: min–max across realizations
                ax.fill_between(x_grid, bands[0], bands[100],
                                color=color, alpha=0.15, zorder=3)
                # Median realization CDF
                ax.plot(x_grid, bands[50], color=color, linestyle=ls,
                        linewidth=LINEWIDTH_MEDIUM, zorder=5,
                        label=DATASET_LABELS.get(did, did))
                has_data = True

        if not has_data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=FONTSIZE_SMALL)

        # Title and labels
        ax.set_title(title, fontsize=FONTSIZE_SMALL, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=FONTSIZE_SMALL - 1)

        # Invert x-axis so "better" is always to the right
        if not HIGHER_IS_BETTER.get(metric_name, True):
            ax.invert_xaxis()

        ax.annotate('better →', xy=(0.97, 0.03), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=FONTSIZE_SMALL - 2,
                    color='0.4', fontstyle='italic')

        if metric_name in HASHIMOTO_NAMES:
            ax.set_ylabel('CDF across\nrealizations', fontsize=FONTSIZE_SMALL - 1)
        else:
            ax.set_ylabel('CDF of annual\nvalues', fontsize=FONTSIZE_SMALL - 1)
        if metric_name.endswith('_reliability'):
            ax.set_ylim(-0.02, 0.4)
        else:
            ax.set_ylim(-0.02, 1.02)
        ax.tick_params(labelsize=FONTSIZE_SMALL - 1)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)

    # Hide unused axes
    for j in range(n_metrics, len(axes)):
        axes[j].set_visible(False)

    # Legend at top — use combined patch+line handles like F2
    legend_handles = []
    legend_labels = []
    for did in DATASETS:
        color = DATASET_COLORS[did]
        patch = mpatches.Patch(facecolor=color, alpha=0.15)
        line = mlines.Line2D([], [], color=color,
                             linestyle=DATASET_LINESTYLES.get(did, '-'),
                             linewidth=LINEWIDTH_MEDIUM)
        legend_handles.append((patch, line))
        legend_labels.append(
            f'{DATASET_LABELS.get(did, did)} (median & range)')

    fig.legend(legend_handles, legend_labels, loc='upper center',
               ncol=len(DATASETS), fontsize=FONTSIZE_MEDIUM,
               bbox_to_anchor=(0.5, 1.0), frameon=False,
               handler_map={tuple: HandlerTuple(ndivide=1)},
               handleheight=1.5)

    period_label = {
        'all': 'All Days',
        'drought': 'Drought Days Only',
        'nondrought': 'Non-Drought Days Only',
    }
    fig.suptitle(
        f'Performance Metric Distributions — {period_label.get(period, period)}',
        fontsize=FONTSIZE_MEDIUM + 2, fontweight='bold', y=1.03,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ============================================================================
# 2×3 RELIABILITY / RESILIENCY FIGURE
# ============================================================================

# Layout:
#   Columns: Montague, Trenton, NYC
#   Row 1: Weekly reliability (annual, per-realization CDF bands)
#   Row 2: Hashimoto resiliency (simulation-level, simple CDF)

RR_GRID = [
    # (row, col, metric, is_annual, title, xlabel)
    (0, 0, 'montague_reliability', True,
     'Montague', 'Fraction of Weeks Without Failure'),
    (0, 1, 'trenton_reliability', True,
     'Trenton', 'Fraction of Weeks Without Failure'),
    (0, 2, 'nyc_reliability', True,
     'NYC', 'Fraction of Weeks Without Failure'),

    (1, 0, 'hashimoto_resiliency_montague', False,
     'Montague', 'P(Recovery | Deficit Day)'),
    (1, 1, 'hashimoto_resiliency_trenton', False,
     'Trenton', 'P(Recovery | Deficit Day)'),
    (1, 2, 'nyc_reliability', False,  # placeholder — NYC resiliency not available
     'NYC', ''),
]

ROW_LABELS = [
    'Weekly Reliability\n(Weeks Without 3+ Deficit Days)',
    'Hashimoto Resiliency\nP(Recovery | Deficit Day)',
]


def make_reliability_resiliency_figure(annual, hashimoto, period='all'):
    """Create focused 2×3 reliability/resiliency comparison figure."""

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.0 * nrows))

    for row, col, metric, is_annual, col_title, xlabel in RR_GRID:
        ax = axes[row, col]

        # NYC resiliency doesn't exist in Hashimoto (only Montague/Trenton)
        if row == 1 and col == 2:
            ax.text(0.5, 0.5, 'Not applicable\n(NYC shortage uses\ndiversion deficit,\nnot flow target)',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=FONTSIZE_SMALL, color='0.5')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(col_title, fontsize=FONTSIZE_MEDIUM, fontweight='bold')
            continue

        has_data = False
        for did in DATASETS:
            color = DATASET_COLORS[did]
            ls = DATASET_LINESTYLES.get(did, '-')

            if is_annual:
                x_grid, bands = compute_cdf_bands(
                    annual[did], metric, period=period
                )
                if x_grid is None:
                    continue
                ax.fill_between(x_grid, bands[0], bands[100],
                                color=color, alpha=0.15, zorder=3)
                ax.plot(x_grid, bands[50], color=color, linestyle=ls,
                        linewidth=LINEWIDTH_MEDIUM, zorder=5,
                        label=DATASET_LABELS.get(did, did))
                has_data = True
            else:
                df_h = hashimoto.get(did)
                if df_h is None or metric not in df_h.columns:
                    continue
                vals = np.sort(df_h[metric].dropna().values)
                if len(vals) == 0:
                    continue
                cdf = np.arange(1, len(vals) + 1) / len(vals)
                ax.plot(vals, cdf, color=color, linestyle=ls,
                        linewidth=LINEWIDTH_MEDIUM,
                        label=DATASET_LABELS.get(did, did))
                has_data = True

        if not has_data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=FONTSIZE_SMALL)

        # Column title on top row only
        if row == 0:
            ax.set_title(col_title, fontsize=FONTSIZE_MEDIUM, fontweight='bold')

        ax.set_xlabel(xlabel, fontsize=FONTSIZE_SMALL)

        # "better →" always to the right
        higher_better = HIGHER_IS_BETTER.get(metric, True)
        if not higher_better:
            ax.invert_xaxis()
        ax.annotate('better →', xy=(0.97, 0.03), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=FONTSIZE_SMALL - 2,
                    color='0.4', fontstyle='italic')

        if is_annual:
            ax.set_ylabel('CDF of annual values', fontsize=FONTSIZE_SMALL)
        else:
            ax.set_ylabel('CDF across realizations', fontsize=FONTSIZE_SMALL)
        if metric.endswith('_reliability'):
            ax.set_ylim(-0.02, 0.4)
        else:
            ax.set_ylim(-0.02, 1.02)
        ax.tick_params(labelsize=FONTSIZE_SMALL - 1)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)

    # Row labels on the left
    for r, label in enumerate(ROW_LABELS):
        axes[r, 0].annotate(
            label, xy=(-0.35, 0.5), xycoords='axes fraction',
            ha='center', va='center', fontsize=FONTSIZE_MEDIUM,
            fontweight='bold', rotation=90,
        )

    # Legend at top
    legend_handles = []
    legend_labels = []
    for did in DATASETS:
        color = DATASET_COLORS[did]
        patch = mpatches.Patch(facecolor=color, alpha=0.15)
        line = mlines.Line2D([], [], color=color,
                             linestyle=DATASET_LINESTYLES.get(did, '-'),
                             linewidth=LINEWIDTH_MEDIUM)
        legend_handles.append((patch, line))
        legend_labels.append(
            f'{DATASET_LABELS.get(did, did)} (median & range)')

    fig.legend(legend_handles, legend_labels, loc='upper center',
               ncol=len(DATASETS), fontsize=FONTSIZE_MEDIUM,
               bbox_to_anchor=(0.5, 1.0), frameon=False,
               handler_map={tuple: HandlerTuple(ndivide=1)},
               handleheight=1.5)

    period_label = {
        'all': 'All Days',
        'drought': 'Drought Days Only',
        'nondrought': 'Non-Drought Days Only',
    }
    fig.suptitle(
        f'Shortage Reliability & Resiliency — {period_label.get(period, period)}',
        fontsize=FONTSIZE_MEDIUM + 2, fontweight='bold', y=1.03,
    )

    fig.tight_layout(rect=[0.05, 0, 1, 0.97])
    return fig


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    apply_publication_style()

    print("Loading data...")
    annual, hashimoto = load_all_data()

    # Full metric overview (24-panel)
    for period in ['all', 'drought', 'nondrought']:
        print(f"Plotting {period} period (full overview)...")
        fig = make_distribution_figure(annual, hashimoto, period=period)

        fname = f"{FIG_OUTPUT_DIR}/metric_cdfs_{period}.png"
        fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
        print(f"  Saved: {fname}")
        plt.close(fig)

    # Focused reliability/resiliency (2×3)
    for period in ['all', 'drought', 'nondrought']:
        print(f"Plotting {period} period (reliability & resiliency)...")
        fig = make_reliability_resiliency_figure(annual, hashimoto, period=period)

        fname = f"{FIG_OUTPUT_DIR}/reliability_resiliency_{period}.png"
        fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
        print(f"  Saved: {fname}")
        plt.close(fig)

    print("Done.")
