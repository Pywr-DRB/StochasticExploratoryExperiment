"""
F4 (Alternative): NYC contribution & diversion KDEs — 2x3 grid.

2 rows × 3 columns (storage zone groups):
  Row 1: NYC contribution / inflow ratio (%)
  Row 2: NYC diversion / inflow ratio (%)

  Col 1: Normal or Flood        (zones 0-3)
  Col 2: Drought Watch/Warning  (zones 4-5)
  Col 3: Drought Emergency      (zone 6)

Each subplot overlays KDE lines for all three datasets (scenarios).
One figure is produced per aggregation window.

Usage:
    python F4alt_2x3_kde.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from methods.config import FIG_DIR, OUTPUT_DIR, verify_dataset_id
from methods.plotting.styles import (
    DPI_HIGH,
    DATASET_COLORS, DATASET_LABELS,
    FONTSIZE_LABEL, FONTSIZE_MEDIUM, FONTSIZE_SMALL,
    apply_publication_style, label_panel,
)
import pywrdrb
import methods.plotting.water_balance_by_drought_zone as F4_module
from methods.plotting.water_balance_by_drought_zone import (
    aggregate_across_realizations,
    categorize_by_drought_zone,
    calculate_reconstruction_contribution_ratio,
    MIN_INFLOW_THRESHOLD,
    NYC_RESERVOIRS,
    AGGREGATION_METHOD, N_MONTHS_PRIOR,
)
from methods.config import RECONSTRUCTION_OUTPUT_FNAME

def calculate_reconstruction_diversion_ratio():
    """Calculate the diversion/inflow ratio for the 1964 reconstruction."""
    if not os.path.exists(RECONSTRUCTION_OUTPUT_FNAME):
        return None

    try:
        data = pywrdrb.Data()
        data.load_output(
            output_filenames=[RECONSTRUCTION_OUTPUT_FNAME],
            results_sets=['res_storage', 'inflow', 'ibt_diversions'],
        )
        ds = list(data.res_storage.keys())[0]
        r = list(data.res_storage[ds].keys())[0]

        nyc_storage = data.res_storage[ds][r][NYC_RESERVOIRS].sum(axis=1)
        storage_1964 = nyc_storage[nyc_storage.index.year == 1964]
        if len(storage_1964) == 0:
            return None

        min_date = storage_1964.idxmin()

        if F4_module.AGGREGATION_METHOD == 'n_months_prior':
            start_date = min_date - pd.DateOffset(months=F4_module.N_MONTHS_PRIOR)
        else:
            yr = min_date.year if min_date.month >= 6 else min_date.year - 1
            start_date = pd.Timestamp(year=yr, month=6, day=1)

        mask = lambda s: (s.index >= start_date) & (s.index <= min_date)

        nyc_inflow = data.inflow[ds][r][NYC_RESERVOIRS].sum(axis=1)
        nyc_diversion = data.ibt_diversions[ds][r]['delivery_nyc']

        inflow_total = nyc_inflow[mask(nyc_inflow)].sum()
        diversion_total = nyc_diversion[mask(nyc_diversion)].sum()

        if inflow_total <= 0:
            return None

        ratio = 100.0 * diversion_total / inflow_total
        print(f"  Reconstruction diversion ratio: {ratio:.1f}%")
        return ratio

    except Exception as e:
        print(f"  Warning: Error calculating reconstruction diversion ratio: {e}")
        return None


# ============================================================================
# CONFIGURATION
# ============================================================================

SCENARIOS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']
WINDOW_MONTHS = [3, 6, 9]
FIG_OUTPUT_DIR = f"{FIG_DIR}/F4alt_kde"

# Column zone groupings
COL_ZONES  = ['other', 'watch_warning', 'emergency']
COL_LABELS = {
    'other':         'Min NYC storage in\nNormal or Flood zone',
    'watch_warning': 'Min NYC storage in\nWatch or Warning zone',
    'emergency':     'Min NYC storage in\nEmergency zone',
}

# Zone categories passed to categorize_by_zone
ZONE_CATEGORIES = {
    'emergency':    [6],
    'watch':        [5],
    'warning':      [4],
    'other':        [0, 1, 2, 3],
}

# Row definitions
ROW_METRICS = [
    {
        'numerator': 'contribution_total',
        'xlabel': 'NYC contribution / inflow ({n_mo}-mo, %)',
    },
    {
        'numerator': 'diversion_total',
        'xlabel': 'NYC diversion / inflow ({n_mo}-mo, %)',
    },
]

XLIM_QUANTILE = 0.99

PANEL_LETTERS = [
    ['a', 'b', 'c'],  # row 1
    ['d', 'e', 'f'],  # row 2
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
        parts = []
        for z in ('watch', 'warning'):
            r = _get_ratios(categorized, z, numerator_col)
            if r is not None:
                parts.append(r)
        if not parts:
            return None
        return pd.concat(parts, ignore_index=True)
    else:
        return _get_ratios(categorized, col_zone, numerator_col)


# ============================================================================
# FIGURE
# ============================================================================

def create_figure(all_categorized, n_months_prior, recon_contribution, recon_diversion):
    """
    2-row × 3-col KDE grid.
    Row 1: contribution/inflow, Row 2: diversion/inflow.
    Each col = zone group, all datasets overlaid.
    """
    apply_publication_style()
    n_rows = 2
    n_cols = 3
    size = 4

    fig = plt.figure(figsize=(size * n_cols + 1, size * n_rows + 1.0))

    # Row 1: independent a, then b/c share y-axis
    r1_a = fig.add_subplot(n_rows, n_cols, 1)
    r1_b = fig.add_subplot(n_rows, n_cols, 2)
    r1_c = fig.add_subplot(n_rows, n_cols, 3, sharey=r1_b)

    # Row 2: sharex with row 1, b/c share y-axis within row
    r2_a = fig.add_subplot(n_rows, n_cols, 4, sharex=r1_a)
    r2_b = fig.add_subplot(n_rows, n_cols, 5, sharex=r1_b)
    r2_c = fig.add_subplot(n_rows, n_cols, 6, sharex=r1_c, sharey=r2_b)

    axes = [[r1_a, r1_b, r1_c], [r2_a, r2_b, r2_c]]

    fig.subplots_adjust(bottom=0.12, left=0.07, right=0.97, top=0.94, hspace=0.28)

    # Custom horizontal spacing per row
    for row_i in range(n_rows):
        pos_a = axes[row_i][0].get_position()
        pos_b = axes[row_i][1].get_position()
        pos_c = axes[row_i][2].get_position()
        panel_w = pos_a.width
        gap_ab = 0.09
        gap_bc = 0.03
        x0_a = pos_a.x0
        x0_b = x0_a + panel_w + gap_ab
        x0_c = x0_b + panel_w + gap_bc
        axes[row_i][0].set_position([x0_a, pos_a.y0, panel_w, pos_a.height])
        axes[row_i][1].set_position([x0_b, pos_b.y0, panel_w, pos_b.height])
        axes[row_i][2].set_position([x0_c, pos_c.y0, panel_w, pos_c.height])

    # Total years per scenario
    total_years = {}
    for sc in SCENARIOS:
        total_years[sc] = sum(len(df) for df in all_categorized[sc].values())

    # Compute shared xlim per column (max across both rows)
    col_xmax = {}
    for col_i, col in enumerate(COL_ZONES):
        all_vals = []
        for row_def in ROW_METRICS:
            for sc in SCENARIOS:
                r = _get_col_ratios(all_categorized[sc], col, row_def['numerator'])
                if r is not None and len(r) >= 2:
                    all_vals.append(r)
        if all_vals:
            combined = pd.concat(all_vals, ignore_index=True)
            col_xmax[col] = np.percentile(combined, XLIM_QUANTILE * 100)
        else:
            col_xmax[col] = 100

    for row_i, row_def in enumerate(ROW_METRICS):
        numerator_col = row_def['numerator']
        xlabel = row_def['xlabel'].format(n_mo=n_months_prior)

        for col_i, col in enumerate(COL_ZONES):
            ax = axes[row_i][col_i]
            xmax = col_xmax[col]

            # Collect ratios for this row/col
            all_ratios = []
            for sc in SCENARIOS:
                r = _get_col_ratios(all_categorized[sc], col, numerator_col)
                if r is not None and len(r) >= 2:
                    all_ratios.append(r)

            if not all_ratios:
                label_panel(ax, PANEL_LETTERS[row_i][col_i],
                            label=COL_LABELS[col] if row_i == 0 else '')
                continue

            # Plot KDEs
            for sc in SCENARIOS:
                r = _get_col_ratios(all_categorized[sc], col, numerator_col)
                if r is None or len(r) < 2:
                    continue
                color = DATASET_COLORS[sc]
                sns.kdeplot(
                    r, ax=ax, fill=False,
                    color=color, linewidth=4.5,
                    common_norm=False, clip=(0, xmax * 1.2),
                    label=DATASET_LABELS[sc],
                )

            # % of years annotation (row 1 only to avoid clutter)
            if row_i == 0:
                y_text = 0.88
                for sc in SCENARIOS:
                    r = _get_col_ratios(all_categorized[sc], col, numerator_col)
                    n = len(r) if r is not None else 0
                    pct = 100.0 * n / total_years[sc] if total_years[sc] > 0 else 0
                    ax.text(0.97, y_text, f'{pct:.0f}% of years',
                            transform=ax.transAxes, fontsize=FONTSIZE_SMALL,
                            color=DATASET_COLORS[sc], ha='right', va='top',
                            fontweight='bold')
                    y_text -= 0.06

            # 1964 reconstruction tick on emergency column
            recon_ratios = [recon_contribution, recon_diversion]
            recon_val = recon_ratios[row_i]
            if (col == 'emergency' and recon_val is not None and recon_val <= xmax):
                ymax = ax.get_ylim()[1]
                ax.vlines(recon_val, 0, ymax * 0.15,
                          color='black', linewidth=3.0, zorder=5)

            ax.set_xlim(0, xmax)
            ax.set_xlabel(xlabel, fontsize=FONTSIZE_LABEL)

            # Panel label: zone title on row 1, just letter on row 2
            if row_i == 0:
                label_panel(ax, PANEL_LETTERS[row_i][col_i], label=COL_LABELS[col])
            else:
                label_panel(ax, PANEL_LETTERS[row_i][col_i])

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_color('#444444')

    # Breathing room and y-axis labels per row
    for row_i in range(n_rows):
        for ax in axes[row_i]:
            y_lo, y_hi = ax.get_ylim()
            ax.set_ylim(y_lo, y_hi * 1.15)

        axes[row_i][0].set_ylabel('Density', fontsize=FONTSIZE_LABEL)
        axes[row_i][1].set_ylabel('Density', fontsize=FONTSIZE_LABEL)
        axes[row_i][2].tick_params(labelleft=False)
        axes[row_i][2].set_ylabel('')

    # Shared legend below the figure
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=DATASET_COLORS[sc], linewidth=2.5,
               label=DATASET_LABELS[sc])
        for sc in SCENARIOS
    ]
    fig.legend(handles=handles, loc='lower center', ncol=len(SCENARIOS),
               fontsize=FONTSIZE_SMALL, frameon=False,
               bbox_to_anchor=(0.5, -0.01))

    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    apply_publication_style()
    os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

    print("F4 alt 2x3: NYC contribution & diversion KDE grid")
    print("=" * 70)

    from methods.load import load_contribution_metrics
    from methods.metrics.contribution import get_metrics_for_window, categorize_by_zone

    use_cached = True
    try:
        metrics_cache = {sc: load_contribution_metrics(sc) for sc in SCENARIOS}
    except (ImportError, FileNotFoundError):
        use_cached = False

    if not use_cached:
        all_data = load_all_data()

    for n_mo in WINDOW_MONTHS:
        print(f"\n  {n_mo}-month window ...")
        F4_module.N_MONTHS_PRIOR = n_mo

        if use_cached:
            window_days = n_mo * 30
            col_map = {
                f'contribution_total_{window_days}d': 'contribution_total',
                f'contribution_ratio_{window_days}d': 'contribution_ratio',
                f'inflow_total_{window_days}d':       'inflow_total',
                f'diversion_total_{window_days}d':    'diversion_total',
                f'diversion_ratio_{window_days}d':    'diversion_ratio',
                f'demand_satisfaction_{window_days}d': 'demand_satisfaction',
                f'worst_1mo_demand_sat_{window_days}d': 'worst_1mo_demand_sat',
            }
            all_categorized = {}
            for sc in SCENARIOS:
                df = get_metrics_for_window(metrics_cache[sc], window_days)
                df = df.rename(columns=col_map)
                all_categorized[sc] = categorize_by_zone(df, ZONE_CATEGORIES)
        else:
            all_categorized = categorize_all_scenarios(all_data, n_mo)

        recon_contribution = calculate_reconstruction_contribution_ratio()
        recon_diversion = calculate_reconstruction_diversion_ratio()

        fig = create_figure(all_categorized, n_mo, recon_contribution, recon_diversion)
        fname = f"{FIG_OUTPUT_DIR}/F4alt_2x3_kde_{n_mo}mo.png"
        fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
        print(f"    Saved: {fname}")
        plt.close(fig)

    print("\n" + "=" * 70)
    print("F4 alt 2x3 figures generated successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
