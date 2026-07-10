"""
F4 (Alternative): NYC contribution & diversion KDEs — 2x3 grid by window.

2 rows × 3 columns (storage zone groups):
  Row 1: 3-month aggregation window
  Row 2: 9-month aggregation window

  Col 1: Normal or Flood        (zones 0-3)
  Col 2: Drought Watch/Warning  (zones 4-5)
  Col 3: Drought Emergency      (zone 6)

Each subplot overlays KDE lines for all three datasets (scenarios).
Solid lines = NYC contribution / inflow ratio.
Dashed lines = NYC diversion / inflow ratio.

Usage:
    python F4alt_2x3_kde_windows.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from methods.config import FIG_DIR, OUTPUT_DIR, RECONSTRUCTION_OUTPUT_FNAME, verify_dataset_id
from methods.plotting.styles import (
    DPI_HIGH,
    DATASET_COLORS, DATASET_LABELS,
    FONTSIZE_LABEL, FONTSIZE_MEDIUM, FONTSIZE_SMALL,
    apply_publication_style, label_panel,
)
import methods.plotting.water_balance_by_drought_zone as F4_module
from methods.plotting.water_balance_by_drought_zone import (
    aggregate_across_realizations,
    categorize_by_drought_zone,
    calculate_reconstruction_contribution_ratio,
    MIN_INFLOW_THRESHOLD,
    NYC_RESERVOIRS,
)


# ============================================================================
# RECONSTRUCTION HELPERS
# ============================================================================

def _load_reconstruction_1964():
    """Load reconstruction data and return (min_date, inflow, contribution, diversion) series."""
    if not os.path.exists(RECONSTRUCTION_OUTPUT_FNAME):
        return None

    try:
        data = pywrdrb.Data()
        data.load_output(
            output_filenames=[RECONSTRUCTION_OUTPUT_FNAME],
            results_sets=['res_storage', 'inflow', 'ibt_diversions', 'nyc_release_components'],
        )
        ds = list(data.res_storage.keys())[0]
        r = list(data.res_storage[ds].keys())[0]

        nyc_storage = data.res_storage[ds][r][NYC_RESERVOIRS].sum(axis=1)
        storage_1964 = nyc_storage[nyc_storage.index.year == 1964]
        if len(storage_1964) == 0:
            return None

        min_date = storage_1964.idxmin()
        nyc_inflow = data.inflow[ds][r][NYC_RESERVOIRS].sum(axis=1)
        nyc_diversion = data.ibt_diversions[ds][r]['delivery_nyc']

        contrib_cols = [f'mrf_montagueTrenton_{res}' for res in NYC_RESERVOIRS]
        nyc_contribution = data.nyc_release_components[ds][r][contrib_cols].sum(axis=1)

        return {
            'min_date': min_date,
            'inflow': nyc_inflow,
            'contribution': nyc_contribution,
            'diversion': nyc_diversion,
        }
    except Exception as e:
        print(f"  Warning: Error loading reconstruction: {e}")
        return None


def _recon_ratio(recon_data, n_months, numerator_key):
    """Compute a reconstruction ratio for a given window and numerator."""
    if recon_data is None:
        return None
    min_date = recon_data['min_date']
    start_date = min_date - pd.DateOffset(months=n_months)
    mask = lambda s: (s.index >= start_date) & (s.index <= min_date)

    inflow_total = recon_data['inflow'][mask(recon_data['inflow'])].sum()
    num_total = recon_data[numerator_key][mask(recon_data[numerator_key])].sum()
    if inflow_total <= 0:
        return None
    return 100.0 * num_total / inflow_total


# ============================================================================
# CONFIGURATION
# ============================================================================

SCENARIOS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']
ROW_WINDOWS = [3, 9]  # months: row 1 = 3-mo, row 2 = 9-mo
FIG_OUTPUT_DIR = f"{FIG_DIR}/Fig7_kdes"

COL_ZONES = ['other', 'watch_warning', 'emergency']
COL_LABELS = {
    'other':         'Min NYC storage in\nNormal or Flood zone',
    'watch_warning': 'Min NYC storage in\nWatch or Warning zone',
    'emergency':     'Min NYC storage in\nEmergency zone',
    'drought_all':   'Min NYC storage in\nWatch, Warning or Emergency zone',
}

# 2x2 layout options for the right column
LAYOUT_2X2_COLS = {
    'drought_all':    ['other', 'drought_all'],
    'emergency_only': ['other', 'emergency'],
}

# pywrdrb drought_level_agg_nyc index: 4=Watch, 5=Warning, 6=Emergency
ZONE_CATEGORIES = {
    'emergency':    [6],
    'watch':        [4],
    'warning':      [5],
    'other':        [0, 1, 2, 3],
}

METRICS = [
    {'numerator': 'contribution_total', 'linestyle': '-',  'label_suffix': 'contribution'},
    {'numerator': 'diversion_total',    'linestyle': '--', 'label_suffix': 'diversion'},
]

XLIM_QUANTILE = 0.99
N_KDE_POINTS = 500
KDE_LINEWIDTH = 3.5

PANEL_LETTERS = [
    ['a', 'b', 'c'],
    ['d', 'e', 'f'],
]

PANEL_LETTERS_2X2 = [
    ['a', 'b'],
    ['c', 'd'],
]


# ============================================================================
# DATA HELPERS
# ============================================================================

def _get_ratios(categorized, zone, numerator_col='contribution_total'):
    """Return numerator / inflow ratio (%) for a single zone, filtered."""
    df = categorized[zone]
    df_f = df[df['inflow_total'] > MIN_INFLOW_THRESHOLD].copy()
    if len(df_f) < 2:
        return None
    ratio = (100.0 * df_f[numerator_col] / df_f['inflow_total'])
    return ratio.replace([np.inf, -np.inf], np.nan).dropna()


def _get_col_ratios(categorized, col_zone, numerator_col='contribution_total'):
    """Return combined ratios for a column zone group."""
    if col_zone == 'watch_warning':
        sub_zones = ('watch', 'warning')
    elif col_zone == 'drought_all':
        sub_zones = ('watch', 'warning', 'emergency')
    else:
        return _get_ratios(categorized, col_zone, numerator_col)

    parts = []
    for z in sub_zones:
        r = _get_ratios(categorized, z, numerator_col)
        if r is not None:
            parts.append(r)
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def _kde_line(data, x_grid):
    """Compute KDE on x_grid, return array (zeros if insufficient data)."""
    if data is None or len(data) < 2:
        return np.zeros_like(x_grid)
    return gaussian_kde(data.values)(x_grid)


# ============================================================================
# FIGURE
# ============================================================================

def create_figure(all_categorized_by_window, recon_data,
                   col_zones=None, panel_letters=None):
    """
    2-row × N-col KDE grid (N = 2 or 3).
    Row 1: 3-mo window, Row 2: 9-mo window.
    Solid = contribution/inflow, dashed = diversion/inflow.
    """
    apply_publication_style()
    if col_zones is None:
        col_zones = COL_ZONES
    if panel_letters is None:
        panel_letters = PANEL_LETTERS if len(col_zones) == 3 else PANEL_LETTERS_2X2

    n_rows = 2
    n_cols = len(col_zones)
    size = 4

    fig = plt.figure(figsize=(size * n_cols + 1, size * n_rows + 1.0))

    axes = [[None] * n_cols for _ in range(n_rows)]
    for row_i in range(n_rows):
        for col_i in range(n_cols):
            idx = row_i * n_cols + col_i + 1
            sharey = None
            # In 3-col layout, col c shares y with col b (both drought zones)
            if n_cols == 3 and col_i == 2:
                sharey = axes[row_i][1]
            axes[row_i][col_i] = fig.add_subplot(n_rows, n_cols, idx, sharey=sharey)

    fig.subplots_adjust(bottom=0.12, left=0.07, right=0.97, top=0.94, hspace=0.32)

    # Custom horizontal spacing per row
    for row_i in range(n_rows):
        positions = [axes[row_i][c].get_position() for c in range(n_cols)]
        panel_w = positions[0].width
        if n_cols == 3:
            gaps = [0.09, 0.03]
        else:
            gaps = [0.10]
        x0 = positions[0].x0
        for col_i in range(n_cols):
            if col_i > 0:
                x0 = x0 + panel_w + gaps[col_i - 1]
            axes[row_i][col_i].set_position(
                [x0, positions[col_i].y0, panel_w, positions[col_i].height]
            )

    # Total years per scenario (same across windows)
    first_window = ROW_WINDOWS[0]
    total_years = {}
    for sc in SCENARIOS:
        total_years[sc] = sum(len(df) for df in all_categorized_by_window[first_window][sc].values())

    for row_i, n_mo in enumerate(ROW_WINDOWS):
        all_categorized = all_categorized_by_window[n_mo]

        for col_i, col in enumerate(col_zones):
            ax = axes[row_i][col_i]

            # Compute xlim across both metrics for this row/col
            all_vals = []
            for metric in METRICS:
                for sc in SCENARIOS:
                    r = _get_col_ratios(all_categorized[sc], col, metric['numerator'])
                    if r is not None and len(r) >= 2:
                        all_vals.append(r)

            if not all_vals:
                label_panel(ax, panel_letters[row_i][col_i],
                            label=COL_LABELS[col] if row_i == 0 else '')
                continue

            combined = pd.concat(all_vals, ignore_index=True)
            xmax = np.percentile(combined, XLIM_QUANTILE * 100)
            x_grid = np.linspace(0, xmax * 1.2, N_KDE_POINTS)

            # Plot KDEs for each metric × scenario
            for metric in METRICS:
                numerator_col = metric['numerator']
                ls = metric['linestyle']
                for sc in SCENARIOS:
                    r = _get_col_ratios(all_categorized[sc], col, numerator_col)
                    if r is None or len(r) < 2:
                        continue
                    color = DATASET_COLORS[sc]
                    y = _kde_line(r, x_grid)
                    ax.plot(x_grid, y, color=color, linewidth=KDE_LINEWIDTH,
                            linestyle=ls)

            # % of years annotation (row 1 only)
            if row_i == 0:
                y_text = 0.88
                for sc in SCENARIOS:
                    r = _get_col_ratios(all_categorized[sc], col, 'contribution_total')
                    n = len(r) if r is not None else 0
                    pct = 100.0 * n / total_years[sc] if total_years[sc] > 0 else 0
                    ax.text(0.97, y_text, f'{pct:.0f}% of years',
                            transform=ax.transAxes, fontsize=FONTSIZE_SMALL,
                            color=DATASET_COLORS[sc], ha='right', va='top',
                            fontweight='bold')
                    y_text -= 0.06

            # 1964 reconstruction ticks on emergency / drought_all columns
            if col in ('emergency', 'drought_all') and recon_data is not None:
                for metric in METRICS:
                    recon_key = 'contribution' if metric['numerator'] == 'contribution_total' else 'diversion'
                    recon_val = _recon_ratio(recon_data, n_mo, recon_key)
                    if recon_val is not None and recon_val <= xmax:
                        ymax = ax.get_ylim()[1]
                        ax.vlines(recon_val, 0, ymax * 0.15,
                                  color='black', linewidth=3.0, zorder=5,
                                  linestyles=metric['linestyle'])

            ax.set_xlim(0, xmax)
            ax.set_xlabel(
                f'Ratio to NYC inflow ({n_mo}-mo, %)',
                fontsize=FONTSIZE_LABEL,
            )

            # Panel label: zone title on row 1, just letter on row 2
            if row_i == 0:
                label_panel(ax, panel_letters[row_i][col_i], label=COL_LABELS[col])
            else:
                label_panel(ax, panel_letters[row_i][col_i])

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_color('#444444')

    # Breathing room and y-axis labels per row (ymin pinned to 0)
    for row_i in range(n_rows):
        for ax in axes[row_i]:
            _, y_hi = ax.get_ylim()
            ax.set_ylim(0, y_hi * 1.15)

        if n_cols == 3:
            axes[row_i][0].set_ylabel('Density', fontsize=FONTSIZE_LABEL)
            axes[row_i][1].set_ylabel('Density', fontsize=FONTSIZE_LABEL)
            axes[row_i][2].tick_params(labelleft=False)
            axes[row_i][2].set_ylabel('')
        else:
            for col_i in range(n_cols):
                axes[row_i][col_i].set_ylabel('Density', fontsize=FONTSIZE_LABEL)

    # Legend: two rows — scenarios on top, line styles below
    scenario_handles = [
        Line2D([0], [0], color=DATASET_COLORS[sc], linewidth=2.5,
               linestyle='-', label=DATASET_LABELS[sc])
        for sc in SCENARIOS
    ]
    style_handles = [
        Line2D([0], [0], color='gray', linewidth=2.5, linestyle='-',
               label='Contribution / inflow'),
        Line2D([0], [0], color='gray', linewidth=2.5, linestyle='--',
               label='Diversion / inflow'),
    ]
    all_handles = scenario_handles + style_handles
    fig.legend(handles=all_handles, loc='lower center',
               ncol=3, fontsize=FONTSIZE_SMALL, frameon=False,
               bbox_to_anchor=(0.5, -0.03))

    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    apply_publication_style()
    os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

    print("F4 alt 2x3 windows: NYC contribution & diversion KDE grid")
    print("=" * 70)

    from methods.load import load_contribution_metrics
    from methods.metrics.contribution import get_metrics_for_window, categorize_by_zone

    use_cached = True
    try:
        metrics_cache = {sc: load_contribution_metrics(sc) for sc in SCENARIOS}
    except (ImportError, FileNotFoundError):
        use_cached = False

    # Load categorized data for each window
    all_categorized_by_window = {}
    for n_mo in ROW_WINDOWS:
        print(f"\n  Loading {n_mo}-month window ...")
        F4_module.N_MONTHS_PRIOR = n_mo

        if use_cached:
            window_days = n_mo * 30
            base_metrics = ['contribution_total', 'contribution_ratio', 'inflow_total',
                            'demand_satisfaction', 'worst_1mo_demand_sat']
            sample_cols = metrics_cache[SCENARIOS[0]].columns
            if f'diversion_total_{window_days}d' in sample_cols:
                base_metrics += ['diversion_total', 'diversion_ratio']

            col_map = {f'{m}_{window_days}d': m for m in base_metrics}
            all_categorized = {}
            for sc in SCENARIOS:
                df = get_metrics_for_window(metrics_cache[sc], window_days, metrics=base_metrics)
                df = df.rename(columns=col_map)
                all_categorized[sc] = categorize_by_zone(df, ZONE_CATEGORIES)
        else:
            if not hasattr(main, '_all_data'):
                main._all_data = load_all_data()
            all_categorized = categorize_all_scenarios(main._all_data, n_mo)

        all_categorized_by_window[n_mo] = all_categorized

    # Load reconstruction data once
    print("\n  Loading reconstruction data...")
    recon_data = _load_reconstruction_1964()

    variants = [
        ('2x3',                None,                            'F4alt_2x3_kde_windows.png'),
        ('2x2_drought_all',    LAYOUT_2X2_COLS['drought_all'],    'F4alt_2x2_kde_drought_all.png'),
        ('2x2_emergency_only', LAYOUT_2X2_COLS['emergency_only'], 'F4alt_2x2_kde_emergency_only.png'),
    ]
    for tag, col_zones, basename in variants:
        fig = create_figure(all_categorized_by_window, recon_data, col_zones=col_zones)
        fname = f"{FIG_OUTPUT_DIR}/{basename}"
        fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
        fname_svg = fname.replace('.png', '.svg')
        fig.savefig(fname_svg, dpi=DPI_HIGH, bbox_inches='tight')
        print(f"  Saved [{tag}]: {fname}")
        plt.close(fig)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == '__main__':
    main()
