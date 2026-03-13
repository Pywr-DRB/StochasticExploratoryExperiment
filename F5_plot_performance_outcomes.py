"""
F5: Performance Outcomes — three visualization options.

Three panels:
  A – Max consecutive Montague shortage days per realization
  B – Mean annual NYC demand satisfaction (%) per realization
  C – Joint satisficing rate (fraction of simulation years satisficing) per realization

Three layout prototypes:
  Option 1  Complementary CDF (exceedance curves)
  Option 2  Ridgeline / joy plot (stacked KDEs)
  Option 3  Quantile strip plot (jittered points + percentile markers)

Usage:
    python F5_plot_performance_outcomes.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from methods.config import FIG_DIR, DATASET_CONFIGS, N_YEARS
from methods.plotting.styles import (
    DATASET_COLORS, DATASET_LABELS,
    DPI_HIGH, ALPHA_LINE, ALPHA_FILL, ALPHA_SCATTER,
    LINEWIDTH_MEDIUM, LINEWIDTH_THICK,
    FONTSIZE_SMALL, FONTSIZE_MEDIUM, FONTSIZE_LARGE, FONTSIZE_TITLE,
    DATASET_LINESTYLES,
    apply_publication_style,
)
from methods.load import load_annual_metrics, load_annual_satisficing

# ============================================================================
# CONFIG
# ============================================================================
FIG_OUTPUT_DIR = f"{FIG_DIR}/F5_performance_outcomes"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

DATASETS = list(DATASET_CONFIGS.keys())
SSI_WINDOW = 12  # for satisficing calculation

# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_data():
    """Load annual metrics and satisficing results for all datasets."""
    perf = {}
    satis = {}

    for did in DATASETS:
        perf[did] = load_annual_metrics(did)

        # Load annual satisficing results
        try:
            satis[did] = load_annual_satisficing(did, SSI_WINDOW)
        except (FileNotFoundError, KeyError) as e:
            print(f"  Warning: satisficing results not found for {did}: {e}")
            satis[did] = None

    return perf, satis


def compute_panel_data(perf, satis):
    """
    Compute the three panel metrics per realization per dataset.

    Returns dict: {dataset_id: DataFrame with columns
        [max_consec_shortage_days, mean_nyc_satisfaction_pct, satisficing_rate]}
    """
    panel_data = {}

    for did in DATASETS:
        df = perf[did]
        # Filter to period='all' for realization-level aggregation
        df_all = df[df['period'] == 'all'].copy()

        # Aggregate annual metrics to per-realization values
        by_r = df_all.groupby('realization_id')

        # Panel A: max consecutive Montague shortage days across all water years
        panel_a = by_r['montague_max_consec_shortage_days'].max()

        # Panel B: mean NYC reliability across water years (as %)
        panel_b = by_r['nyc_reliability'].mean() * 100.0

        # Panel C: satisficing rate (fraction of years satisficing per realization)
        # Satisficing = (nyc_min_storage_pct >= 20) & (montague_max_consec_shortage_days <= 3)
        df_all['satisficing'] = (
            (df_all['nyc_min_storage_pct'] >= 20.0) &
            (df_all['montague_max_consec_shortage_days'] <= 3)
        ).astype(float)
        panel_c = df_all.groupby('realization_id')['satisficing'].mean() * 100.0

        # Combine into a single DataFrame
        combined = pd.DataFrame({
            'max_consec_shortage_days': panel_a,
            'mean_nyc_satisfaction_pct': panel_b,
            'satisficing_rate_pct': panel_c,
        })
        panel_data[did] = combined

    return panel_data


# ============================================================================
# PANEL METADATA
# ============================================================================
PANEL_CONFIG = [
    {
        'key': 'max_consec_shortage_days',
        'title': 'Max Consecutive Montague\nShortage Days',
        'xlabel': 'Days',
        'invert_good': True,  # lower is better
    },
    {
        'key': 'mean_nyc_satisfaction_pct',
        'title': 'NYC Water Supply\nDemand Satisfaction',
        'xlabel': '% of Demand Met',
        'invert_good': False,  # higher is better
    },
    {
        'key': 'satisficing_rate_pct',
        'title': 'Joint Satisficing Rate\n(Storage + Montague Criteria)',
        'xlabel': '% of Years Satisficing',
        'invert_good': False,  # higher is better
    },
]


# ============================================================================
# OPTION 1: COMPLEMENTARY CDF (EXCEEDANCE CURVES)
# ============================================================================

def plot_exceedance_curves(panel_data):
    """
    For each metric, plot P(X >= x) with one curve per scenario.
    Semi-log y-axis for tail behaviour.
    """
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, cfg in zip(axes, PANEL_CONFIG):
        key = cfg['key']

        for i, did in enumerate(DATASETS):
            values = panel_data[did][key].dropna().values
            if len(values) == 0:
                continue

            sorted_vals = np.sort(values)
            exceedance = 1.0 - np.arange(1, len(sorted_vals) + 1) / (len(sorted_vals) + 1)

            color = DATASET_COLORS.get(did, f'C{i}')
            label = DATASET_LABELS.get(did, did)
            ls = DATASET_LINESTYLES.get(did, '-')

            ax.step(sorted_vals, exceedance, where='post',
                    color=color, label=label, linewidth=LINEWIDTH_MEDIUM,
                    linestyle=ls, alpha=ALPHA_LINE)

        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-2, top=1.1)
        ax.set_ylabel('Exceedance Probability', fontsize=FONTSIZE_MEDIUM)
        ax.set_xlabel(cfg['xlabel'], fontsize=FONTSIZE_MEDIUM)
        ax.set_title(cfg['title'], fontsize=FONTSIZE_LARGE, pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=FONTSIZE_SMALL)

        # Add reference lines at key exceedance levels
        for p in [0.5, 0.1, 0.05]:
            ax.axhline(p, color='grey', linewidth=0.5, linestyle=':', alpha=0.4)

    axes[0].legend(fontsize=FONTSIZE_SMALL, loc='upper right',
                   frameon=True, fancybox=True)

    fig.suptitle('Performance Outcomes: Exceedance Curves',
                 fontsize=FONTSIZE_TITLE, fontweight='bold', y=1.02)
    plt.tight_layout()

    fname = f"{FIG_OUTPUT_DIR}/F5_option1_exceedance_curves.png"
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    return fig


# ============================================================================
# OPTION 2: RIDGELINE (JOY PLOT)
# ============================================================================

def plot_ridgeline(panel_data):
    """
    Stacked KDE distributions, one row per scenario, one column per metric.
    """
    apply_publication_style()

    n_datasets = len(DATASETS)
    n_panels = len(PANEL_CONFIG)
    row_height = 1.4
    overlap = 0.6

    fig, axes = plt.subplots(n_datasets, n_panels,
                             figsize=(16, n_datasets * row_height + 2),
                             sharex='col')

    for col, cfg in enumerate(PANEL_CONFIG):
        key = cfg['key']

        # Get global x range across datasets for this metric
        all_vals = []
        for did in DATASETS:
            v = panel_data[did][key].dropna().values
            if len(v) > 0:
                all_vals.extend(v)
        if len(all_vals) == 0:
            continue

        xmin = min(all_vals)
        xmax = max(all_vals)
        pad = 0.1 * (xmax - xmin) if xmax > xmin else 1
        x_grid = np.linspace(xmin - pad, xmax + pad, 300)

        for row, did in enumerate(DATASETS):
            ax = axes[row, col] if n_datasets > 1 else axes[col]
            values = panel_data[did][key].dropna().values

            if len(values) < 2:
                ax.set_visible(False)
                continue

            color = DATASET_COLORS.get(did, f'C{row}')
            label = DATASET_LABELS.get(did, did)

            try:
                kde = stats.gaussian_kde(values)
                density = kde(x_grid)
            except np.linalg.LinAlgError:
                density = np.zeros_like(x_grid)

            ax.fill_between(x_grid, density, alpha=ALPHA_FILL + 0.1, color=color)
            ax.plot(x_grid, density, color=color, linewidth=LINEWIDTH_MEDIUM,
                    alpha=ALPHA_LINE)

            # Add median line
            med = np.median(values)
            ax.axvline(med, color=color, linewidth=1.5, linestyle='--', alpha=0.7)

            # Add percentile annotations
            p5 = np.percentile(values, 5)
            p95 = np.percentile(values, 95)
            ax.axvline(p5, color=color, linewidth=0.8, linestyle=':', alpha=0.5)
            ax.axvline(p95, color=color, linewidth=0.8, linestyle=':', alpha=0.5)

            # Clean up axes
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

            if col == 0:
                ax.set_ylabel(label, fontsize=FONTSIZE_MEDIUM,
                              rotation=0, ha='right', va='center',
                              labelpad=10)

            if row == 0:
                ax.set_title(cfg['title'], fontsize=FONTSIZE_LARGE, pad=10)

            if row == n_datasets - 1:
                ax.set_xlabel(cfg['xlabel'], fontsize=FONTSIZE_MEDIUM)
                ax.tick_params(axis='x', labelsize=FONTSIZE_SMALL)
            else:
                ax.tick_params(axis='x', labelbottom=False)

    fig.suptitle('Performance Outcomes: Ridgeline Distributions',
                 fontsize=FONTSIZE_TITLE, fontweight='bold', y=1.02)
    plt.tight_layout()

    fname = f"{FIG_OUTPUT_DIR}/F5_option2_ridgeline.png"
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    return fig


# ============================================================================
# OPTION 3: QUANTILE STRIP PLOT
# ============================================================================

def plot_quantile_strip(panel_data):
    """
    Each realization as a semi-transparent point (jittered), with horizontal
    markers at p5, p50, p95.
    """
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    n_datasets = len(DATASETS)

    for ax, cfg in zip(axes, PANEL_CONFIG):
        key = cfg['key']

        y_positions = np.arange(n_datasets)

        for i, did in enumerate(DATASETS):
            values = panel_data[did][key].dropna().values
            if len(values) == 0:
                continue

            color = DATASET_COLORS.get(did, f'C{i}')
            label = DATASET_LABELS.get(did, did)

            # Jittered strip
            jitter = np.random.default_rng(42).uniform(-0.2, 0.2, size=len(values))
            ax.scatter(values, y_positions[i] + jitter,
                       color=color, alpha=ALPHA_SCATTER * 0.4,
                       s=12, edgecolors='none', zorder=2)

            # Percentile markers
            p5 = np.percentile(values, 5)
            p50 = np.percentile(values, 50)
            p95 = np.percentile(values, 95)

            # Horizontal line from p5 to p95
            ax.plot([p5, p95], [y_positions[i], y_positions[i]],
                    color=color, linewidth=LINEWIDTH_THICK, alpha=0.9, zorder=3)

            # p5 and p95 markers (vertical ticks)
            for px in [p5, p95]:
                ax.plot([px, px], [y_positions[i] - 0.12, y_positions[i] + 0.12],
                        color=color, linewidth=LINEWIDTH_MEDIUM, alpha=0.9, zorder=3)

            # Median marker (diamond)
            ax.scatter([p50], [y_positions[i]], color=color,
                       marker='D', s=60, edgecolors='black', linewidths=0.8,
                       zorder=4, label=label if cfg == PANEL_CONFIG[0] else None)

            # Annotate percentiles
            ax.annotate(f'{p5:.0f}', (p5, y_positions[i] + 0.25),
                        fontsize=7, ha='center', color=color, alpha=0.8)
            ax.annotate(f'{p50:.0f}', (p50, y_positions[i] - 0.35),
                        fontsize=8, ha='center', fontweight='bold', color=color)
            ax.annotate(f'{p95:.0f}', (p95, y_positions[i] + 0.25),
                        fontsize=7, ha='center', color=color, alpha=0.8)

        ax.set_yticks(y_positions)
        ax.set_yticklabels([DATASET_LABELS.get(d, d) for d in DATASETS],
                           fontsize=FONTSIZE_MEDIUM)
        ax.set_xlabel(cfg['xlabel'], fontsize=FONTSIZE_MEDIUM)
        ax.set_title(cfg['title'], fontsize=FONTSIZE_LARGE, pad=10)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.tick_params(axis='x', labelsize=FONTSIZE_SMALL)
        ax.set_ylim(-0.6, n_datasets - 0.4)
        ax.invert_yaxis()

    # Add legend explaining markers
    legend_elements = [
        Line2D([0], [0], marker='D', color='grey', markerfacecolor='grey',
               markersize=8, linewidth=0, label='Median (p50)'),
        Line2D([0], [0], color='grey', linewidth=LINEWIDTH_THICK,
               label='p5–p95 range'),
        Line2D([0], [0], marker='o', color='grey', markerfacecolor='grey',
               markersize=4, linewidth=0, alpha=0.4, label='Individual realizations'),
    ]
    axes[0].legend(handles=legend_elements, fontsize=FONTSIZE_SMALL - 1,
                   loc='lower left', frameon=True, fancybox=True)

    fig.suptitle('Performance Outcomes: Quantile Strip Plots',
                 fontsize=FONTSIZE_TITLE, fontweight='bold', y=1.02)
    plt.tight_layout()

    fname = f"{FIG_OUTPUT_DIR}/F5_option3_quantile_strip.png"
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    return fig


# ============================================================================
# SUMMARY TABLE
# ============================================================================

def print_summary_table(panel_data):
    """Print key percentiles for comparison across options."""
    print("\n" + "=" * 90)
    print("F5 PERFORMANCE OUTCOMES — SUMMARY STATISTICS")
    print("=" * 90)

    for cfg in PANEL_CONFIG:
        key = cfg['key']
        print(f"\n  {cfg['title'].replace(chr(10), ' ')}")
        print(f"  {'Dataset':<25} {'p5':>10} {'p50':>10} {'p95':>10} {'mean':>10}")
        print(f"  {'-'*65}")

        for did in DATASETS:
            values = panel_data[did][key].dropna().values
            if len(values) == 0:
                continue

            label = DATASET_LABELS.get(did, did)
            p5 = np.percentile(values, 5)
            p50 = np.percentile(values, 50)
            p95 = np.percentile(values, 95)
            mean = np.mean(values)
            print(f"  {label:<25} {p5:>10.1f} {p50:>10.1f} {p95:>10.1f} {mean:>10.1f}")

    print("\n" + "=" * 90)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("F5: Performance Outcomes")
    print("Loading data...")

    perf, satis = load_all_data()
    panel_data = compute_panel_data(perf, satis)

    print_summary_table(panel_data)

    print("\nGenerating Option 1: Exceedance Curves...")
    plot_exceedance_curves(panel_data)

    print("Generating Option 2: Ridgeline...")
    plot_ridgeline(panel_data)

    print("Generating Option 3: Quantile Strip Plot...")
    plot_quantile_strip(panel_data)

    print(f"\nAll figures saved to: {FIG_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
