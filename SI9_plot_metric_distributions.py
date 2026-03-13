"""
SI9: Comprehensive metric distribution overview.

Multi-panel figure with one subplot per metric showing empirical CDFs
for all three datasets. Used for exploratory analysis to identify
the most interesting/discriminating metrics.

Usage:
    python SI9_plot_metric_distributions.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from methods.config import FIG_DIR, DATASET_CONFIGS
from methods.plotting.styles import (
    DATASET_COLORS, DATASET_LABELS,
    METRIC_DISPLAY_NAMES, METRIC_UNITS,
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

# Annual metrics to plot — 20 from annual_metrics, grouped by category.
# Each tuple: (column_name, aggregation_method)
#   'per_year' = plot each water-year value (full distribution across years × realizations)
#   'per_real_max' = aggregate to max across years per realization, then plot distribution
#   'per_real_mean' = aggregate to mean across years per realization
#   'per_real_min' = aggregate to min across years per realization
ANNUAL_METRICS = [
    # Shortage reliability (fraction 0–1, per year)
    ('montague_reliability', 'per_year'),
    ('trenton_reliability', 'per_year'),
    ('nyc_reliability', 'per_year'),

    # Total shortage volume (MG, per year)
    ('montague_shortage_mg', 'per_year'),
    ('trenton_shortage_mg', 'per_year'),
    ('nyc_shortage_mg', 'per_year'),

    # Max consecutive shortage days (per year)
    ('montague_max_consec_shortage_days', 'per_year'),
    ('trenton_max_consec_shortage_days', 'per_year'),
    ('nyc_max_consec_shortage_days', 'per_year'),

    # Max single-day shortage (MG, per year)
    ('montague_max_1day_shortage_mg', 'per_year'),
    ('trenton_max_1day_shortage_mg', 'per_year'),
    ('nyc_max_1day_shortage_mg', 'per_year'),

    # NYC storage (per year)
    ('nyc_min_storage_pct', 'per_year'),
    ('june1_storage_pct', 'per_year'),
    ('sept1_storage_pct', 'per_year'),
    ('ndays_storage_below_20pct', 'per_year'),
    ('ndays_storage_below_30pct', 'per_year'),

    # System-level (per year)
    ('nyc_contribution_mg', 'per_year'),
    ('ndays_combined_stress', 'per_year'),
    ('max_zone', 'per_year'),
]

# Hashimoto simulation-level metrics (1 value per realization)
HASHIMOTO_METRICS = [
    'hashimoto_reliability_montague',
    'hashimoto_resiliency_montague',
    'hashimoto_reliability_trenton',
    'hashimoto_resiliency_trenton',
]

# Direction convention: True = higher values are better (right = better by default).
# Metrics where lower is better get their x-axis inverted so "better" is always right.
HIGHER_IS_BETTER = {
    # Reliability: higher = fewer shortages relative to target
    'montague_reliability': True,
    'trenton_reliability': True,
    'nyc_reliability': True,

    # Shortage volumes: lower = less unmet demand
    'montague_shortage_mg': False,
    'trenton_shortage_mg': False,
    'nyc_shortage_mg': False,

    # Consecutive shortage days: lower = shorter disruptions
    'montague_max_consec_shortage_days': False,
    'trenton_max_consec_shortage_days': False,
    'nyc_max_consec_shortage_days': False,

    # Peak single-day shortage: lower = less severe peak deficit
    'montague_max_1day_shortage_mg': False,
    'trenton_max_1day_shortage_mg': False,
    'nyc_max_1day_shortage_mg': False,

    # Storage levels: higher = more water in reserve
    'nyc_min_storage_pct': True,
    'june1_storage_pct': True,
    'sept1_storage_pct': True,

    # Days below storage thresholds: lower = less time in critical storage
    'ndays_storage_below_20pct': False,
    'ndays_storage_below_30pct': False,

    # NYC contribution: lower = less NYC release needed to meet downstream targets
    'nyc_contribution_mg': False,

    # Combined stress days: lower = fewer days with simultaneous shortages
    'ndays_combined_stress': False,

    # Max drought zone: lower = less severe drought classification
    'max_zone': False,

    # Hashimoto: higher = more reliable / more resilient
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


def get_metric_values(annual, hashimoto, metric_name, agg, period='all'):
    """
    Extract values for a single metric across all datasets.

    Returns dict: {dataset_id: 1-D numpy array of values}
    """
    result = {}

    for did in DATASETS:
        if metric_name in HASHIMOTO_METRICS:
            df = hashimoto.get(did)
            if df is None or metric_name not in df.columns:
                continue
            vals = df[metric_name].dropna().values
        else:
            df = annual[did]
            # Filter to period='all' for annual metrics
            subset = df[df['period'] == period].copy()

            if metric_name not in subset.columns:
                continue

            if agg == 'per_year':
                vals = subset[metric_name].dropna().values
            elif agg == 'per_real_max':
                vals = subset.groupby('realization_id')[metric_name].max().dropna().values
            elif agg == 'per_real_min':
                vals = subset.groupby('realization_id')[metric_name].min().dropna().values
            elif agg == 'per_real_mean':
                vals = subset.groupby('realization_id')[metric_name].mean().dropna().values
            else:
                vals = subset[metric_name].dropna().values

        if len(vals) > 0:
            result[did] = vals

    return result


def plot_ecdf(ax, values, color, linestyle, label, linewidth=LINEWIDTH_MEDIUM):
    """Plot empirical CDF on axis."""
    sorted_vals = np.sort(values)
    ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax.plot(sorted_vals, ecdf, color=color, linestyle=linestyle,
            linewidth=linewidth, label=label)


# ============================================================================
# MAIN FIGURE
# ============================================================================

def make_distribution_figure(annual, hashimoto, period='all'):
    """Create multi-panel CDF figure for all metrics."""

    all_metrics = [(m, a) for m, a in ANNUAL_METRICS]
    all_metrics += [(m, 'per_real') for m in HASHIMOTO_METRICS]

    n_metrics = len(all_metrics)
    ncols = 4
    nrows = int(np.ceil(n_metrics / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows))
    axes = axes.flatten()

    for i, (metric_name, agg) in enumerate(all_metrics):
        ax = axes[i]

        if metric_name in HASHIMOTO_METRICS:
            data_by_ds = get_metric_values(annual, hashimoto, metric_name, agg, period)
        else:
            data_by_ds = get_metric_values(annual, hashimoto, metric_name, agg, period)

        if not data_by_ds:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=FONTSIZE_SMALL)
            ax.set_title(metric_name, fontsize=FONTSIZE_SMALL)
            continue

        for did in DATASETS:
            if did not in data_by_ds:
                continue
            vals = data_by_ds[did]
            plot_ecdf(
                ax, vals,
                color=DATASET_COLORS[did],
                linestyle=DATASET_LINESTYLES.get(did, '-'),
                label=DATASET_LABELS.get(did, did),
            )

        # Title
        display_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
        # Clean up multiline display names for subplot titles
        title = display_name.replace('\n', ' ')
        if agg not in ('per_year', 'per_real'):
            title += f' ({agg.replace("per_real_", "")})'
        ax.set_title(title, fontsize=FONTSIZE_SMALL, fontweight='bold')

        # Invert x-axis so "better" is always to the right
        higher_better = HIGHER_IS_BETTER.get(metric_name, True)
        if not higher_better:
            ax.invert_xaxis()

        # Arrow annotation showing "better" direction
        ax.annotate('better →', xy=(0.97, 0.03), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=FONTSIZE_SMALL - 2,
                    color='0.4', fontstyle='italic')

        # Axis labels
        ax.set_ylabel('CDF', fontsize=FONTSIZE_SMALL - 1)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(labelsize=FONTSIZE_SMALL - 1)

        # Light grid
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)

    # Hide unused axes
    for j in range(n_metrics, len(axes)):
        axes[j].set_visible(False)

    # Single legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center',
                   ncol=len(DATASETS), fontsize=FONTSIZE_MEDIUM,
                   bbox_to_anchor=(0.5, 1.0), frameon=False)

    period_label = {'all': 'All Days', 'drought': 'Drought Days', 'nondrought': 'Non-Drought Days'}
    fig.suptitle(f'Metric Distributions — {period_label.get(period, period)}',
                 fontsize=FONTSIZE_MEDIUM + 2, fontweight='bold', y=1.02)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    apply_publication_style()

    print("Loading data...")
    annual, hashimoto = load_all_data()

    for period in ['all', 'drought', 'nondrought']:
        print(f"Plotting {period} period...")
        fig = make_distribution_figure(annual, hashimoto, period=period)

        fname = f"{FIG_OUTPUT_DIR}/metric_cdfs_{period}.png"
        fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
        print(f"  Saved: {fname}")
        plt.close(fig)

    print("Done.")
