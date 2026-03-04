"""
F6: Sankey-Parallel Coordinate Hybrid Figure

Creates a hybrid Sankey-Parallel Coordinate figure showing how drought
hazard characteristics cascade through system actions to outcomes.

Each sample is one SSI-defined drought event. Multiple horizontal axes
represent different metrics, with bin widths proportional to sample count.
Sankey-style flows between adjacent axes are colored by satisficing
classification (pass, storage_fail, montague_fail, both_fail).

Generates THREE versions per dataset:
  1. Fixed-threshold bins (physically meaningful breakpoints)
  2. Quantile-based bins (data-driven tertiles)
  3. CART-based bins (data-driven thresholds separating satisficing outcomes)

Pre-calculated event metrics are loaded from pywrdrb/event_metrics/.
Run 07_calculate_event_metrics.py first to generate these.

Usage:
    python F6_plot_sankey_parallel_coordinates.py
    python F6_plot_sankey_parallel_coordinates.py --ssi_window 6
    python F6_plot_sankey_parallel_coordinates.py --datasets stationary_ensemble
    python F6_plot_sankey_parallel_coordinates.py --versions default quantile cart
"""

import os
import sys
import argparse
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from methods.config import ROOT_DIR, FIG_DIR, SSI_WINDOWS, DATASET_CONFIGS
from methods.plotting.styles import DATASET_LABELS
from methods.plotting.sankey_parallel import (
    AxisConfig, SankeyFigureConfig, plot_sankey_parallel,
)
from methods.metrics.cart_bin_selection import (
    compute_cart_bin_edges, cart_results_to_axis_configs, print_cart_summary,
)


# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

FIG_OUTPUT_DIR = os.path.join(FIG_DIR, 'F6_sankey_parallel')
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

# Input data directory (from 07_calculate_event_metrics.py)
EVENT_METRICS_DIR = os.path.join(ROOT_DIR, 'pywrdrb', 'event_metrics')


# =============================================================================
# AXIS CONFIGURATIONS
# =============================================================================

def get_default_axes():
    """
    Fixed-threshold axis configuration with physically meaningful bin edges.

    Vertical ordering (top to bottom):
      HAZARDS       -> drought timing, severity, duration
      SYSTEM STATE  -> NYC contribution
      OUTCOMES      -> Montague violations, min storage, diversion satisfaction
    """
    return [
        # === HAZARDS (top) ===
        AxisConfig(
            metric='start_month',
            label='Drought Start',
            bin_edges=[0.5, 3.5, 7.5, 12.5],
            bin_labels=['Jan-Mar', 'Apr-Jul', 'Aug-Dec'],
        ),
        AxisConfig(
            metric='severity',
            label='Drought Severity',
            bin_edges=[0, 1.2, 2.0, 99],
            bin_labels=['SSI < 1.2', '1.2 < SSI < 2.0', 'SSI > 2.0'],
        ),
        AxisConfig(
            metric='duration_days',
            label='Drought Duration',
            bin_edges=[0, 90, 365, 1e5],
            bin_labels=['< 3 mo', '3-12 mo', '> 1 yr'],
        ),

        # === SYSTEM STATE/ACTIONS (middle) ===
        AxisConfig(
            metric='total_nyc_contribution_mg',
            label='NYC Contribution\nto Montague',
            bin_edges=[0, 10000, 50000, 1e9],
            bin_labels=['< 10 BG', '10-50 BG', '> 50 BG'],
        ),

        # === OUTCOMES (bottom) ===
        AxisConfig(
            metric='max_consec_montague_days',
            label='Montague\nViolations',
            bin_edges=[-0.5, 0.5, 3.5, 999],
            bin_labels=['0 days', '1-3 days', '> 3 days'],
        ),
        AxisConfig(
            metric='min_storage_pct',
            label='Min Storage\nDuring Drought',
            bin_edges=[0, 20, 50, 100.1],
            bin_labels=['< 20%', '20-50%', '> 50%'],
        ),
        AxisConfig(
            metric='nyc_diversion_sat_ratio',
            label='NYC Diversion\nSatisfaction',
            bin_edges=[0, 0.85, 0.95, 1.001],
            bin_labels=['< 85%', '85-95%', '> 95%'],
        ),
    ]


def get_cart_axes(metrics_df, max_depth=2, n_bootstrap=200):
    """
    CART-based axis configuration. Bin edges are derived from classification
    trees that find thresholds maximally separating satisficing from
    non-satisficing drought events.

    Uses class_weight='balanced' to handle severe class imbalance.
    Bootstrap stability analysis flags unreliable thresholds.

    Metrics with unstable CART splits fall back to quantile bins.
    start_month always uses fixed seasonal bins (not amenable to CART).

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Event metrics with classification column (needed for CART fitting).
    max_depth : int
        Maximum CART depth per metric (1=2 bins, 2=up to 3 bins).
    n_bootstrap : int
        Bootstrap resamples for stability assessment.

    Returns
    -------
    list of AxisConfig
    """
    # Metrics to derive CART bins for (same ordering as default/quantile)
    cart_metrics = [
        'severity',
        'duration_days',
        'total_nyc_contribution_mg',
        'max_consec_montague_days',
        'min_storage_pct',
        'nyc_diversion_sat_ratio',
    ]

    # Compute CART bin edges
    cart_results = compute_cart_bin_edges(
        metrics_df, cart_metrics,
        target_col='classification',
        max_depth=max_depth,
        n_bootstrap=n_bootstrap,
    )

    # Print summary to console
    print_cart_summary(cart_results)

    # Fixed axes that should not use CART
    fixed_axes = {
        'start_month': {
            'label': 'Drought Start',
            'bin_edges': [0.5, 3.5, 7.5, 12.5],
            'bin_labels': ['Jan-Mar', 'Apr-Jul', 'Aug-Dec'],
        },
    }

    # Desired axis order (top to bottom)
    axis_order = [
        'start_month',
        'severity',
        'duration_days',
        'total_nyc_contribution_mg',
        'max_consec_montague_days',
        'min_storage_pct',
        'nyc_diversion_sat_ratio',
    ]

    return cart_results_to_axis_configs(cart_results, axis_order, fixed_axes)


def get_quantile_axes():
    """
    Quantile-based axis configuration. All bins are data-driven tertiles.

    Same axis ordering as get_default_axes() but every axis uses 'quantile'
    for automatic tertile edge computation from the data.
    """
    return [
        # === HAZARDS (top) ===
        AxisConfig(
            metric='start_month',
            label='Drought Start',
            bin_edges=[0.5, 3.5, 7.5, 12.5],
            bin_labels=['Jan-Mar', 'Apr-Jul', 'Aug-Dec'],
        ),
        AxisConfig(
            metric='severity',
            label='Drought Severity',
            bin_edges='quantile',
        ),
        AxisConfig(
            metric='duration_days',
            label='Drought Duration',
            bin_edges='quantile',
        ),

        # === SYSTEM STATE/ACTIONS (middle) ===
        AxisConfig(
            metric='total_nyc_contribution_mg',
            label='NYC Contribution\nto Montague',
            bin_edges='quantile',
        ),

        # === OUTCOMES (bottom) ===
        AxisConfig(
            metric='max_consec_montague_days',
            label='Montague\nViolations',
            bin_edges='quantile',
        ),
        AxisConfig(
            metric='min_storage_pct',
            label='Min Storage\nDuring Drought',
            bin_edges='quantile',
        ),
        AxisConfig(
            metric='nyc_diversion_sat_ratio',
            label='NYC Diversion\nSatisfaction',
            bin_edges='quantile',
        ),
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
            "Run 07_calculate_event_metrics.py first!"
        )
    df = pd.read_csv(fname)
    print(f"  Loaded {len(df)} events from {os.path.basename(fname)}")
    return df


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def generate_figure(dataset_id, ssi_window, axes_config, version_label,
                     fig_config_overrides=None):
    """
    Generate one Sankey-Parallel Coordinate figure.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    ssi_window : int
        SSI window (3, 6, or 12)
    axes_config : list of AxisConfig
        Axis configuration to use.
    version_label : str
        Version label for filename (e.g., 'default', 'quantile').
    fig_config_overrides : dict, optional
        Override SankeyFigureConfig fields.
    """
    print(f"\n{'='*60}")
    print(f"Generating: {dataset_id}, SSI-{ssi_window}, version={version_label}")
    print(f"{'='*60}")

    # Load data
    metrics_df = load_event_metrics(dataset_id, ssi_window)

    if len(metrics_df) == 0:
        print("  No events. Skipping.")
        return

    # Figure height scales with number of axes
    n_axes = len(axes_config)
    fig_height = max(14, 2.2 * n_axes)

    # Figure configuration
    fig_kwargs = {
        'axes': axes_config,
        'classification_col': 'classification',
        'figsize': (16, fig_height),
        'flow_alpha': 0.50,
        'min_bin_width_frac': 0.08,
    }
    if fig_config_overrides:
        fig_kwargs.update(fig_config_overrides)

    config = SankeyFigureConfig(**fig_kwargs)

    # Title
    dataset_label = DATASET_LABELS.get(dataset_id, dataset_id)
    bin_descs = {
        'default': 'Fixed Thresholds',
        'quantile': 'Quantile Bins',
        'cart': 'CART-Derived Bins',
    }
    bin_desc = bin_descs.get(version_label, version_label)
    title = f"Drought Event Outcomes - {dataset_label} (SSI-{ssi_window}, {bin_desc})"

    # Output path
    output_path = os.path.join(
        FIG_OUTPUT_DIR,
        f'{dataset_id}_ssi{ssi_window}_{version_label}_sankey_parallel.png'
    )

    # Generate
    fig = plot_sankey_parallel(
        metrics_df, config,
        output_path=output_path,
        show=False,
        title=title,
    )
    plt.close(fig)

    return output_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate Sankey-Parallel Coordinate figures'
    )
    parser.add_argument('--ssi_window', type=int, default=6,
                        help='SSI window (default: 6)')
    parser.add_argument('--datasets', nargs='+', type=str,
                        default=list(DATASET_CONFIGS.keys()),
                        help='Dataset IDs to process')
    parser.add_argument('--all_ssi', action='store_true',
                        help='Generate for all SSI windows (3, 6, 12)')
    parser.add_argument('--versions', nargs='+', type=str,
                        default=['default', 'quantile', 'cart'],
                        choices=['default', 'quantile', 'cart'],
                        help='Bin versions to generate (default: all three)')
    parser.add_argument('--cart_depth', type=int, default=2,
                        help='Max CART depth for cart version (default: 2)')
    args = parser.parse_args()

    ssi_windows = list(SSI_WINDOWS) if args.all_ssi else [args.ssi_window]

    print(f"Datasets: {args.datasets}")
    print(f"SSI windows: {ssi_windows}")
    print(f"Versions: {args.versions}")
    print(f"Output: {FIG_OUTPUT_DIR}")

    generated = []
    for dataset_id in args.datasets:
        for ssi_window in ssi_windows:
            # Version 1: Fixed-threshold bins
            if 'default' in args.versions:
                try:
                    path = generate_figure(
                        dataset_id, ssi_window,
                        axes_config=get_default_axes(),
                        version_label='default',
                    )
                    if path:
                        generated.append(path)
                except FileNotFoundError as e:
                    print(f"  Skipping: {e}")
                except Exception as e:
                    print(f"  Error (default): {e}")
                    import traceback
                    traceback.print_exc()

            # Version 2: Quantile bins
            if 'quantile' in args.versions:
                try:
                    path = generate_figure(
                        dataset_id, ssi_window,
                        axes_config=get_quantile_axes(),
                        version_label='quantile',
                    )
                    if path:
                        generated.append(path)
                except FileNotFoundError as e:
                    print(f"  Skipping: {e}")
                except Exception as e:
                    print(f"  Error (quantile): {e}")
                    import traceback
                    traceback.print_exc()

            # Version 3: CART-derived bins
            if 'cart' in args.versions:
                try:
                    # CART needs the data to fit trees
                    metrics_df = load_event_metrics(dataset_id, ssi_window)
                    if len(metrics_df) > 0:
                        cart_axes = get_cart_axes(
                            metrics_df,
                            max_depth=args.cart_depth,
                        )
                        path = generate_figure(
                            dataset_id, ssi_window,
                            axes_config=cart_axes,
                            version_label='cart',
                        )
                        if path:
                            generated.append(path)
                except FileNotFoundError as e:
                    print(f"  Skipping: {e}")
                except Exception as e:
                    print(f"  Error (cart): {e}")
                    import traceback
                    traceback.print_exc()

    print(f"\nGenerated {len(generated)} figures in {FIG_OUTPUT_DIR}")


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    main()
