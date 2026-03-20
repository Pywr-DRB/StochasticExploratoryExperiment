"""
F6b: Continuous Parallel Coordinates Figure

Creates a parallel coordinates figure with continuous (non-binned) axes
stacked vertically. Each line represents one drought event, colored by
satisficing classification. Non-satisficing outcomes are highlighted
over a grey background of passing events.

Axes (top to bottom):
  1. Drought start month
  2. Storage % at start of drought
  3. Drought magnitude
  4. Drought severity
  5. Drought duration (days)
  6. NYC contribution ratio (releases to Montague / inflow)
  7. NYC diversion / inflow ratio
  8. Min storage during drought (%)
  9. NYC diversion shortage (% of demand)

Pre-calculated event metrics are loaded from pywrdrb/event_metrics/.
Run 06_calculate_performance_metrics.py first to generate these.

Usage:
    python F6b_plot_parallel_coordinates.py
    python F6b_plot_parallel_coordinates.py --ssi_window 6
    python F6b_plot_parallel_coordinates.py --datasets stationary_ensemble
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from methods.config import ROOT_DIR, FIG_DIR, SSI_WINDOWS, DATASET_CONFIGS
from methods.plotting.styles import (
    DATASET_LABELS, apply_publication_style,
    FONTSIZE_SMALL, FONTSIZE_MEDIUM, FONTSIZE_LABEL, FONTSIZE_TITLE,
    DPI_HIGH,
)

# Storage threshold for highlighting (%)
STORAGE_FAIL_THRESHOLD = 20.0

# Colors: pass vs storage failure only
HIGHLIGHT_COLORS = {
    'pass': '#888888',
    'storage_fail': '#ff7f0e',
}

HIGHLIGHT_LABELS = {
    'pass': 'Storage > 20%',
    'storage_fail': 'Storage < 20%',
}

# Draw order: pass first (background), failures on top
HIGHLIGHT_DRAW_ORDER = ['pass', 'storage_fail']

# =============================================================================
# OUTPUT / INPUT DIRECTORIES
# =============================================================================

FIG_OUTPUT_DIR = os.path.join(FIG_DIR, 'F6b_parallel_coordinates')
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

EVENT_METRICS_DIR = os.path.join(ROOT_DIR, 'pywrdrb', 'event_metrics')


# =============================================================================
# AXIS DEFINITIONS
# =============================================================================

AXIS_DEFS = [
    {
        'metric': 'start_month',
        'label': 'Drought\nStart Month',
        'ticks': [1, 4, 7, 10, 12],
        'tick_labels': ['Jan', 'Apr', 'Jul', 'Oct', 'Dec'],
        'range': (0.5, 12.5),
    },
    {
        'metric': 'storage_at_start_pct',
        'label': 'Storage at\nDrought Start (%)',
        'range': (0, 100),
    },
    {
        'metric': 'magnitude',
        'label': 'Drought\nMagnitude',
        'clip_percentile': (0, 99),  # clip outliers
    },
    {
        'metric': 'severity',
        'label': 'Drought\nSeverity (SSI)',
        'clip_percentile': (0, 99),
    },
    {
        'metric': 'duration_days',
        'label': 'Drought\nDuration (days)',
        'range_min': 0,
        'clip_percentile': (None, 99),
    },
    {
        'metric': 'contribution_ratio',
        'label': 'NYC Release /\nInflow Ratio',
        'range_min': 0,
        'clip_percentile': (None, 95),
    },
    {
        'metric': 'nyc_diversion_inflow_ratio',
        'label': 'NYC Diversion /\nInflow Ratio',
        'range_min': 0,
        'clip_percentile': (None, 95),
    },
    {
        'metric': 'event_min_storage_pct',
        'label': 'Min Storage\nDuring Drought (%)',
        'range': (0, 100),
    },
    {
        'metric': 'nyc_shortage_pct',
        'label': 'NYC Shortage\n(% of Demand)',
        'range_min': 0,
    },
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_event_metrics(dataset_id, ssi_window):
    """Load pre-calculated event metrics CSV."""
    fname = os.path.join(EVENT_METRICS_DIR,
                         f'{dataset_id}_ssi{ssi_window}_event_metrics.csv')
    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Event metrics not found: {fname}\n"
            "Run 06_calculate_performance_metrics.py first!"
        )
    df = pd.read_csv(fname)
    print(f"  Loaded {len(df)} events from {os.path.basename(fname)}")
    return df


# =============================================================================
# PLOTTING
# =============================================================================

def plot_parallel_coordinates(metrics_df, dataset_id, ssi_window, output_path=None):
    """
    Generate continuous parallel coordinates figure.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Event metrics with columns matching AXIS_DEFS metrics
        and a 'classification' column.
    dataset_id : str
    ssi_window : int
    output_path : str, optional
    """
    apply_publication_style()
    plt.rcParams['axes.grid'] = False  # no gridlines for this figure

    n_axes = len(AXIS_DEFS)
    n_events = len(metrics_df)

    # Binary classification: storage < 20% or not
    is_fail = metrics_df['event_min_storage_pct'] < STORAGE_FAIL_THRESHOLD
    highlight = np.where(is_fail, 'storage_fail', 'pass')

    # --- Resolve axis ranges (with optional percentile clipping) ---
    axis_ranges = []
    for adef in AXIS_DEFS:
        if 'range' in adef:
            axis_ranges.append(adef['range'])
        else:
            vals = metrics_df[adef['metric']]
            clip_pct = adef.get('clip_percentile', (None, None))
            lo_pct, hi_pct = clip_pct if clip_pct else (None, None)
            lo = vals.quantile(lo_pct / 100) if lo_pct is not None else vals.min()
            hi = vals.quantile(hi_pct / 100) if hi_pct is not None else vals.max()
            # Override with explicit min/max if given
            if 'range_min' in adef:
                lo = adef['range_min']
            if 'range_max' in adef:
                hi = adef['range_max']
            pad = max((hi - lo) * 0.03, 1e-6)
            axis_ranges.append((lo - pad, hi + pad))

    # --- Normalise each metric to [0, 1] for plotting ---
    def normalise(val, ax_idx):
        lo, hi = axis_ranges[ax_idx]
        if hi == lo:
            return 0.5
        return (val - lo) / (hi - lo)

    # --- Figure layout ---
    fig_width = 14
    ax_height = 0.55          # inches per axis strip
    gap_height = 0.70         # inches between axes
    top_margin = 1.0
    bot_margin = 0.7
    label_margin = 1.7        # inches reserved on left for labels
    fig_height = top_margin + bot_margin + n_axes * ax_height + (n_axes - 1) * gap_height

    fig, axes = plt.subplots(
        n_axes, 1,
        figsize=(fig_width, fig_height),
        gridspec_kw={'hspace': 0},
    )
    fig.subplots_adjust(
        left=label_margin / fig_width, right=0.98,
        top=1 - top_margin / fig_height,
        bottom=bot_margin / fig_height,
        hspace=gap_height / ax_height,
    )

    # --- Pre-compute normalised x positions for every event on every axis ---
    # Clip to [0, 1] so outliers beyond axis range pin to the edges
    norm_x = np.empty((n_events, n_axes))
    for j, adef in enumerate(AXIS_DEFS):
        vals = metrics_df[adef['metric']].values
        lo, hi = axis_ranges[j]
        denom = hi - lo if hi != lo else 1.0
        norm_x[:, j] = np.clip((vals - lo) / denom, 0, 1)

    # --- Draw lines: pass first (background), then failures on top ---
    line_alpha_pass = 0.08
    line_alpha_fail = 0.45
    linewidth_pass = 0.5
    linewidth_fail = 1.0

    for cls in HIGHLIGHT_DRAW_ORDER:
        mask = highlight == cls
        if not mask.any():
            continue

        is_pass = (cls == 'pass')
        color = HIGHLIGHT_COLORS[cls]
        alpha = line_alpha_pass if is_pass else line_alpha_fail
        lw = linewidth_pass if is_pass else linewidth_fail

        idxs = np.where(mask)[0]

        # Draw segment between each pair of adjacent axes
        for j in range(n_axes - 1):
            ax = axes[j]
            x_top = norm_x[idxs, j]
            x_bot = norm_x[idxs, j + 1]
            for k in range(len(idxs)):
                ax.plot(
                    [x_top[k], x_bot[k]], [1, 0],
                    color=color, alpha=alpha, lw=lw,
                    solid_capstyle='round',
                    zorder=2 if is_pass else 3,
                )

    # --- Configure each axis strip ---
    for j, (ax, adef) in enumerate(zip(axes, AXIS_DEFS)):
        lo, hi = axis_ranges[j]

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Horizontal axis line at top of each strip
        ax.axhline(y=1.0, color='k', lw=1.2, zorder=5)

        # --- Ticks ---
        if 'ticks' in adef:
            tick_vals = np.array(adef['ticks'], dtype=float)
            tick_labels = adef.get('tick_labels', [str(t) for t in tick_vals])
        else:
            # Use MaxNLocator to get nice round tick values within the range
            from matplotlib.ticker import MaxNLocator
            locator = MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10])
            tick_vals = np.array([t for t in locator.tick_values(lo, hi)
                                  if lo <= t <= hi])
            # Format labels — choose precision based on span
            span = hi - lo
            if span >= 100:
                tick_labels = [f'{t:.0f}' for t in tick_vals]
            elif span >= 1:
                tick_labels = [f'{t:.1f}' for t in tick_vals]
            else:
                tick_labels = [f'{t:.2f}' for t in tick_vals]
            # Strip trailing zeros for cleanliness
            tick_labels = [s.rstrip('0').rstrip('.') if '.' in s else s
                           for s in tick_labels]

        # Normalise tick positions to [0, 1]
        denom = hi - lo if hi != lo else 1.0
        tick_norm = [(t - lo) / denom for t in tick_vals]

        ax.set_xticks(tick_norm)
        ax.set_xticklabels(tick_labels, fontsize=FONTSIZE_SMALL + 1)
        ax.xaxis.set_ticks_position('top')
        ax.tick_params(axis='x', which='major', direction='out',
                       pad=3, length=5, width=1.0, top=True, bottom=False)

        # Remove y axis and all spines except top
        ax.set_yticks([])
        for spine in ['left', 'right', 'bottom']:
            ax.spines[spine].set_visible(False)
        ax.spines['top'].set_visible(True)
        ax.spines['top'].set_linewidth(1.2)

        # --- Axis label — placed to the left, vertically aligned with axis line ---
        ax.text(
            -0.02, 1.0, adef['label'],
            transform=ax.transAxes,
            fontsize=FONTSIZE_MEDIUM, fontweight='bold',
            ha='right', va='center',
        )

    # --- Title ---
    dataset_label = DATASET_LABELS.get(dataset_id, dataset_id)
    fig.suptitle(
        f"Drought Event Parallel Coordinates — {dataset_label} (SSI-{ssi_window})",
        fontsize=FONTSIZE_TITLE, fontweight='bold',
        y=1 - 0.25 / fig_height,
    )

    # --- Legend ---
    legend_handles = []
    for cls in HIGHLIGHT_DRAW_ORDER:
        if cls in highlight:
            color = HIGHLIGHT_COLORS[cls]
            label = HIGHLIGHT_LABELS[cls]
            alpha = line_alpha_pass + 0.15 if cls == 'pass' else line_alpha_fail + 0.15
            patch = mpatches.Patch(facecolor=color, edgecolor='none',
                                   alpha=min(alpha, 1.0), label=label)
            legend_handles.append(patch)

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc='lower center',
            fontsize=FONTSIZE_MEDIUM,
            frameon=True, fancybox=True, framealpha=0.9,
            ncol=len(legend_handles),
            bbox_to_anchor=(0.55, 0.01),
        )

    # --- Sample count ---
    n_fail = is_fail.sum()
    axes[0].text(
        0.0, 1.15,
        f"N = {n_events} drought events  ({n_fail} with storage < {STORAGE_FAIL_THRESHOLD:.0f}%)",
        transform=axes[0].transAxes,
        fontsize=FONTSIZE_SMALL, fontstyle='italic',
        va='bottom', ha='left',
    )

    if output_path:
        fig.savefig(output_path, dpi=DPI_HIGH, bbox_inches='tight',
                    facecolor='white')
        print(f"  Saved: {output_path}")

    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate continuous parallel coordinates figures'
    )
    parser.add_argument('--ssi_window', type=int, default=3,
                        help='SSI window (default: 3)')
    parser.add_argument('--datasets', nargs='+', type=str,
                        default=list(DATASET_CONFIGS.keys()),
                        help='Dataset IDs to process')
    parser.add_argument('--all_ssi', action='store_true',
                        help='Generate for all SSI windows (3, 6, 12)')
    args = parser.parse_args()

    ssi_windows = list(SSI_WINDOWS) if args.all_ssi else [args.ssi_window]

    print(f"Datasets: {args.datasets}")
    print(f"SSI windows: {ssi_windows}")
    print(f"Output: {FIG_OUTPUT_DIR}")

    generated = []
    for dataset_id in args.datasets:
        for ssi_window in ssi_windows:
            try:
                metrics_df = load_event_metrics(dataset_id, ssi_window)
                if len(metrics_df) == 0:
                    print("  No events. Skipping.")
                    continue

                output_path = os.path.join(
                    FIG_OUTPUT_DIR,
                    f'{dataset_id}_ssi{ssi_window}_parallel_coordinates.png'
                )

                fig = plot_parallel_coordinates(
                    metrics_df, dataset_id, ssi_window,
                    output_path=output_path,
                )
                plt.close(fig)
                generated.append(output_path)

            except FileNotFoundError as e:
                print(f"  Skipping: {e}")
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()

    print(f"\nGenerated {len(generated)} figures in {FIG_OUTPUT_DIR}")


if __name__ == '__main__':
    main()
