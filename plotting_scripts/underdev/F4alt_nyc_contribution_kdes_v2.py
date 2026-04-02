"""
F4 (Alternative v2): NYC contribution / inflow ratio — ridgeline KDE design.

4-row ridgeline layout where each row shows the distribution of the NYC
Montague contribution-to-inflow ratio for one FFMP drought zone category:
  Row 1 (top): Normal / Above
  Row 2:       Drought Warning
  Row 3:       Drought Watch
  Row 4 (bottom): Drought Emergency

Three overlaid KDE curves per row, one per climate scenario, coloured by
DATASET_COLORS.  A dashed vertical line marks the 1964 reconstruction value.
One figure is produced per aggregation window (3, 6, 9 months).

Usage:
    python F4alt_nyc_contribution_kdes_v2.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from methods.config import FIG_DIR, OUTPUT_DIR, verify_dataset_id
from methods.plotting.styles import (
    DPI_HIGH,
    DATASET_COLORS, DATASET_LABELS,
    FONTSIZE_SMALL, FONTSIZE_LABEL, FONTSIZE_MEDIUM,
    apply_publication_style,
)
import methods.plotting.water_balance_by_drought_zone as F4_module
from methods.plotting.water_balance_by_drought_zone import (
    aggregate_across_realizations,
    categorize_by_drought_zone,
    calculate_reconstruction_contribution_ratio,
    MIN_INFLOW_THRESHOLD,
    XLIM_MAX_MANUAL,
)

# Import contribution_kde to pick up the FFMP-aligned zone color overrides
from methods.plotting.contribution_kde import DROUGHT_CATEGORIES

# ============================================================================
# CONFIGURATION
# ============================================================================

SCENARIOS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']
WINDOW_MONTHS = [3, 6, 9]
FIG_OUTPUT_DIR = f"{FIG_DIR}/F4alt_kde"

# Zone display order: least severe → most severe (top → bottom)
ZONE_ORDER  = ['other', 'warning', 'watch', 'emergency']
ZONE_LABELS = {
    'other':     'Normal or\nFlood',
    'warning':   'Drought\nWarning',
    'watch':     'Drought\nWatch',
    'emergency': 'Drought\nEmergency',
}

N_KDE_POINTS = 600

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
        fname = f'{OUTPUT_DIR}/{dataset_id}_with_postprocessing.hdf5'
        data = pywrdrb.Data()
        data.load_from_export(fname, results_sets=results_sets)
        all_data[dataset_id] = data
    return all_data


def categorize_all_scenarios(all_data, n_months_prior):
    """Aggregate and categorize by drought zone for all scenarios."""
    F4_module.N_MONTHS_PRIOR = n_months_prior
    all_categorized = {}
    for dataset_id in SCENARIOS:
        agg = aggregate_across_realizations(all_data[dataset_id], dataset_id)
        all_categorized[dataset_id] = categorize_by_drought_zone(agg)
    return all_categorized


# ============================================================================
# KDE HELPERS
# ============================================================================

def _get_ratios(categorized, zone):
    """Return contribution / inflow ratio (%) for a zone, filtered."""
    df = categorized[zone]
    df_f = df[df['inflow_total'] > MIN_INFLOW_THRESHOLD].copy()
    if len(df_f) < 2:
        return None
    ratio = (100.0 * df_f['contribution_total'] / df_f['inflow_total'])
    return ratio.replace([np.inf, -np.inf], np.nan).dropna()


def _kde_on_grid(data, x_grid):
    if data is None or len(data) < 2:
        return np.zeros_like(x_grid, dtype=float)
    return gaussian_kde(data.values)(x_grid)


# ============================================================================
# RIDGELINE FIGURE
# ============================================================================

def create_ridgeline_figure(all_categorized, n_months_prior, recon_ratio):
    """
    Build a 4-row ridgeline figure.

    Parameters
    ----------
    all_categorized : dict
        {scenario: {zone: DataFrame}} from categorize_by_drought_zone / cached path.
    n_months_prior : int
        Aggregation window length (for x-axis label).
    recon_ratio : float or None
        1964 reconstruction contribution ratio (%).
    """
    # --- shared x range ---
    all_vals = []
    for zone in ZONE_ORDER:
        for sc in SCENARIOS:
            r = _get_ratios(all_categorized[sc], zone)
            if r is not None and len(r) > 0:
                all_vals.extend(r.values)
    x_max = (XLIM_MAX_MANUAL if XLIM_MAX_MANUAL is not None
             else np.percentile(all_vals, 99.5))
    # Ensure the 1964 line is visible if within a reasonable range
    if recon_ratio is not None and recon_ratio <= x_max * 1.1:
        x_max = max(x_max, recon_ratio)
    x_grid = np.linspace(0, x_max, N_KDE_POINTS)

    # --- precompute all KDE curves ---
    kdes = {}
    for zone in ZONE_ORDER:
        for sc in SCENARIOS:
            r = _get_ratios(all_categorized[sc], zone)
            kdes[(zone, sc)] = _kde_on_grid(r, x_grid)

    # per-zone peak density (used to scale y-limits consistently within each row)
    zone_peak = {
        z: max(kdes[(z, sc)].max() for sc in SCENARIOS)
        for z in ZONE_ORDER
    }

    # -----------------------------------------------------------------------
    # Layout
    # Taller figure so that row centres are ~1.5 in apart, giving rotated
    # labels room to breathe.  hspace=-0.45 keeps the ridgeline aesthetic
    # while leaving enough visible band per row.
    # -----------------------------------------------------------------------
    n_rows = len(ZONE_ORDER)
    fig = plt.figure(figsize=(8.5, 9))
    gs = gridspec.GridSpec(
        n_rows, 1,
        hspace=-0.45,
        top=0.95, bottom=0.14,
        left=0.17, right=0.97,
    )
    axes = [fig.add_subplot(gs[i]) for i in range(n_rows)]

    # Shared x tick positions — drawn on every row
    X_TICKS = [0, 20, 40, 60, 80, 100]

    # KDE tail threshold — fraction of zone peak below which the line is
    # masked so curves end naturally rather than running flat to x_max.
    KDE_TAIL_THRESH = 0.003

    # Upper rows rendered on top — critical for white-background masking
    for i, ax in enumerate(axes):
        ax.set_zorder(n_rows - i)
        ax.patch.set_facecolor('white')
        ax.patch.set_alpha(1.0)

    # -----------------------------------------------------------------------
    # Draw each row
    # -----------------------------------------------------------------------
    for i, zone in enumerate(ZONE_ORDER):
        ax = axes[i]
        peak = zone_peak[zone]

        ax.set_ylim(-peak * 0.05, peak * 2.3)
        ax.set_xlim(0, x_max)

        # --- vertical grid lines clipped to visible KDE band only ---
        # Using vlines (not axvline) so they don't bleed into the overlap zone.
        for xg in X_TICKS:
            ax.vlines(xg, 0, peak, color='#e4e4e4', linewidth=0.6,
                      linestyle='-', zorder=0)

        # --- KDE lines (tails masked below threshold) ---
        for sc in SCENARIOS:
            kv    = kdes[(zone, sc)]
            r     = _get_ratios(all_categorized[sc], zone)
            color = DATASET_COLORS[sc]

            # Mask values below threshold so flat tails disappear naturally
            thresh = peak * KDE_TAIL_THRESH
            kv_plot = kv.copy().astype(float)
            kv_plot[kv_plot < thresh] = np.nan
            ax.plot(x_grid, kv_plot, color=color, linewidth=2.0,
                    alpha=0.90, zorder=3)

            # Mean: dotted vertical tick of constant height (= zone peak)
            if r is not None and len(r) > 0:
                mean_val = float(r.mean())
                ax.vlines(mean_val, 0, peak, color=color,
                          linewidth=1.2, linestyle=':', alpha=0.85, zorder=4)

        # --- 1964 reconstruction line: Emergency row only ---
        if zone == 'emergency' and recon_ratio is not None and recon_ratio <= x_max:
            ax.vlines(recon_ratio, 0, peak, color='#444444', linestyle='--',
                      linewidth=1.1, alpha=0.65, zorder=4)

        # --- thin baseline ---
        ax.axhline(0, color='#bbbbbb', linewidth=0.6, zorder=1)

        # --- spines ---
        ax.set_yticks([])
        for spine in ('left', 'top', 'right'):
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_visible(i == n_rows - 1)
        if i == n_rows - 1:
            ax.spines['bottom'].set_color('#888888')

        # --- x tick labels on ALL rows ---
        # Upper rows: negative pad floats labels up into the visible KDE band
        # so they aren't hidden under the overlapping row above.
        ax.set_xticks(X_TICKS)
        ax.set_xticklabels([str(x) for x in X_TICKS])
        if i < n_rows - 1:
            ax.tick_params(axis='x', labelsize=FONTSIZE_MEDIUM - 1,
                           colors='#666666', length=0, pad=-13)
        else:
            ax.tick_params(axis='x', labelsize=FONTSIZE_MEDIUM,
                           colors='#333333', length=3, pad=3)

    axes[-1].set_xlabel(
        f'NYC contributions / total inflow  ({n_months_prior}-month window prior to min zone, %)',
        fontsize=FONTSIZE_LABEL,
        labelpad=8,
    )

    # -----------------------------------------------------------------------
    # Zone labels and y-axis suptitle — via fig.text() in figure coordinates
    # so they are never obscured by overlapping axes' white patches.
    #
    # LABEL_FRAC: fraction of each subplot's HEIGHT where the visible KDE
    # band centre sits.  With ylim=(-peak*0.05, peak*2.3):
    #   baseline  → 0.05/2.35 ≈ 0.021 of axes height
    #   KDE peak  → 1.05/2.35 ≈ 0.447 of axes height
    #   mid-band  → ~0.23; use 0.26 to sit slightly above centre.
    # -----------------------------------------------------------------------
    fig.canvas.draw()

    LABEL_FRAC    = 0.26
    LABEL_X_OFFSET = 0.032   # figure-width units left of the plot left edge

    for i, zone in enumerate(ZONE_ORDER):
        pos   = axes[i].get_position()
        y_fig = pos.y0 + LABEL_FRAC * pos.height
        x_fig = pos.x0 - LABEL_X_OFFSET

        fig.text(
            x_fig, y_fig,
            ZONE_LABELS[zone],
            fontsize=FONTSIZE_MEDIUM,
            va='center', ha='center',
            color='#111111',
            fontweight='normal',
            rotation=90,
            clip_on=False,
        )

    # Suptitle: immediately left of zone labels, larger font
    top_pos  = axes[0].get_position()
    bot_pos  = axes[-1].get_position()
    y_center = (top_pos.y0 + LABEL_FRAC * top_pos.height +
                bot_pos.y0 + LABEL_FRAC * bot_pos.height) / 2.0

    # x position: just far enough left to not overlap the zone labels
    suptitle_x = axes[0].get_position().x0 - LABEL_X_OFFSET - 0.065

    fig.text(
        suptitle_x, y_center,
        'Water-years where minimum\nNYC reservoir storage zone is:',
        fontsize=FONTSIZE_LABEL + 3,
        va='center', ha='center',
        rotation=90,
        color='#444444',
        clip_on=False,
    )

    # -----------------------------------------------------------------------
    # Legend — bottom centre, 2 columns
    # -----------------------------------------------------------------------
    legend_handles = [
        Line2D([0], [0], color=DATASET_COLORS[sc], linewidth=2.0,
               label=DATASET_LABELS[sc])
        for sc in SCENARIOS
    ]
    legend_handles.append(
        Line2D([0], [0], color='#444444', linestyle=':',
               linewidth=1.2, label='Dataset mean')
    )
    if recon_ratio is not None:
        legend_handles.append(
            Line2D([0], [0], color='#444444', linestyle='--',
                   linewidth=1.1, label='1964 Drought (DE only)')
        )
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=2,
        fontsize=FONTSIZE_SMALL,
        frameon=False,
        bbox_to_anchor=(0.58, -0.01),
        columnspacing=1.4,
    )

    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    apply_publication_style()
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
    })

    os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

    print("F4 alt v2: NYC contribution/inflow ridgeline KDE by drought zone")
    print("=" * 70)

    # --- try cached metrics first ---
    use_cached = True
    try:
        from methods.load import load_contribution_metrics
        from methods.metrics.contribution import get_metrics_for_window, categorize_by_zone
        metrics_cache = {sc: load_contribution_metrics(sc) for sc in SCENARIOS}
    except (ImportError, FileNotFoundError):
        use_cached = False

    if not use_cached:
        all_data = load_all_data()

    zone_categories = {
        'emergency': [6],
        'watch':     [5],
        'warning':   [4],
        'other':     [0, 1, 2, 3],
    }

    for n_mo in WINDOW_MONTHS:
        print(f"\n  {n_mo}-month window ...")
        F4_module.N_MONTHS_PRIOR = n_mo

        if use_cached:
            window_days = n_mo * 30
            col_map = {
                f'contribution_total_{window_days}d': 'contribution_total',
                f'contribution_ratio_{window_days}d': 'contribution_ratio',
                f'inflow_total_{window_days}d':       'inflow_total',
                f'demand_satisfaction_{window_days}d': 'demand_satisfaction',
                f'worst_1mo_demand_sat_{window_days}d': 'worst_1mo_demand_sat',
            }
            all_categorized = {}
            for sc in SCENARIOS:
                df = get_metrics_for_window(metrics_cache[sc], window_days)
                df = df.rename(columns=col_map)
                all_categorized[sc] = categorize_by_zone(df, zone_categories)
        else:
            all_categorized = categorize_all_scenarios(all_data, n_mo)

        recon_ratio = calculate_reconstruction_contribution_ratio()

        fig = create_ridgeline_figure(all_categorized, n_mo, recon_ratio)
        fname = f"{FIG_OUTPUT_DIR}/F4alt_kde_v2_{n_mo}mo.png"
        fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
        print(f"    Saved: {fname}")
        plt.close(fig)

    print("\n" + "=" * 70)
    print("F4 alt v2 ridgeline figures generated successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
