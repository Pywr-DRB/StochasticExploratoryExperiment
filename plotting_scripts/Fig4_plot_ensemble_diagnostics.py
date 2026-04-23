"""
F1: Ensemble validation figure.

Four-panel figure showing:
- (a) Autocorrelation comparison for synthetic vs historic
- (b) Annual flow duration curve ranges for synthetic vs historic
- (c) Weekly streamflow percentile bands for synthetic vs historic
- (d) Wilcoxon rank-sum p-values by month
- (e) Levene test p-values by month

Usage:
    python Fig4_plot_ensemble_diagnostics.py <dataset_id>

Rev1 changes
------------
- Legend: "Reconstructed Historical" (was "Historical"), "Stationary Baseline"
  (was "Historic Baseline").
- Envelope label: "99% IQR (Q0.5-Q99.5)" everywhere.
- Panel letters: lowercase "a)"..."e)" via label_panel().
- Scope annotation in panel (c) lower-left via add_scope_annotation().
- Exports PNG + SVG + PDF via save_fig() with _rev1 suffix.
- Companion _rev1_smooth variant: 3-week rolling mean on Q0.5 / Q99.5
  bounds only in panel (c); median unsmoothed.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from methods.plotting.ensemble_summary import (
    plot_autocorrelation_comparison,
    plot_fdc_percentile_comparison,
    plot_weekly_streamflow_percentiles,
    plot_pvalue_comparison,
    _get_aggregate_flow,
    _pre_aggregate_synthetic,
    NYC_RESERVOIRS,
    MONTH_LABELS,
    MONTH_WEEK_STARTS,
    MGD_TO_MCM,
)
from methods.plotting.styles import (
    DATASET_COLORS, ALPHA_FILL, LINEWIDTH_MEDIUM, LINEWIDTH_THICK,
    DPI_PRINT, apply_publication_style,
    save_fig, add_scope_annotation, label_panel, IQR_LABELS,
)
from methods.load import load_baseline_historical_flow, load_and_combine_ensemble_sets
from methods.config import (
    FIG_DIR, DATASET_CONFIGS, BASELINE_DATASET,
    verify_dataset_id,
)
from methods.ensemble_utils import ENSEMBLE_SETS

# Output directory
FIG_OUTPUT_DIR = f"{FIG_DIR}/Fig4_ensemble_diagnostics"

# Rev1 legend labels
_RECONSTRUCTED_HIST_LABEL = 'Reconstructed Historical'
_STATIONARY_BASELINE_LABEL = 'Stationary Baseline'
_IQR_99_LABEL = IQR_LABELS['99']   # "99% IQR (Q0.5-Q99.5)"

# Smoothing window (weeks) for _rev1_smooth variant
_SMOOTH_WINDOW = 3


def _rolling_mean_1d(arr, window):
    """Centered rolling mean of a 1D array; pads edges with edge values."""
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(arr.astype(float), size=window, mode='nearest')


def _plot_weekly_smooth(
    hist_agg, syn_agg, ax,
    pct, smooth_window, synthetic_color, synthetic_label,
):
    """Panel-c variant with smoothed envelope bounds.

    Recomputes weekly Q0.5, Q50, Q99.5 across all realizations, then applies
    a 3-week centered rolling mean to Q0.5 and Q99.5 only; median is unchanged.
    Historic bands are also smoothed on the envelope bounds.
    """
    n_periods = 52
    periods = np.arange(1, n_periods + 1)

    # --- Historic ---
    Q_hist_weekly = hist_agg.resample('W').mean()
    hist_weeks = Q_hist_weekly.index.isocalendar().week.values.astype(int)
    hist_values = Q_hist_weekly.values

    hist_median = np.empty(n_periods)
    hist_p_low = np.empty(n_periods)
    hist_p_high = np.empty(n_periods)
    for w in periods:
        vals = hist_values[hist_weeks == w]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            hist_median[w - 1] = np.median(vals)
            hist_p_low[w - 1] = np.percentile(vals, pct[0])
            hist_p_high[w - 1] = np.percentile(vals, pct[1])
        else:
            hist_median[w - 1] = hist_p_low[w - 1] = hist_p_high[w - 1] = np.nan

    # --- Synthetic ---
    syn_weekly_by_week = {w: [] for w in periods}
    for flow_series in syn_agg.values():
        weekly = flow_series.resample('W').mean()
        w_arr = weekly.index.isocalendar().week.values.astype(int)
        v_arr = weekly.values
        for w in periods:
            vals = v_arr[w_arr == w]
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                syn_weekly_by_week[w].append(vals)

    syn_median = np.empty(n_periods)
    syn_p_low = np.empty(n_periods)
    syn_p_high = np.empty(n_periods)
    for w in periods:
        if syn_weekly_by_week[w]:
            all_vals = np.concatenate(syn_weekly_by_week[w])
            syn_median[w - 1] = np.median(all_vals)
            syn_p_low[w - 1] = np.percentile(all_vals, pct[0])
            syn_p_high[w - 1] = np.percentile(all_vals, pct[1])
        else:
            syn_median[w - 1] = syn_p_low[w - 1] = syn_p_high[w - 1] = np.nan

    # Convert units
    syn_p_low *= MGD_TO_MCM
    syn_p_high *= MGD_TO_MCM
    syn_median *= MGD_TO_MCM
    hist_p_low *= MGD_TO_MCM
    hist_p_high *= MGD_TO_MCM
    hist_median *= MGD_TO_MCM

    # Apply smoothing to envelope bounds only (not median)
    syn_p_low_s = _rolling_mean_1d(syn_p_low, smooth_window)
    syn_p_high_s = _rolling_mean_1d(syn_p_high, smooth_window)
    hist_p_low_s = _rolling_mean_1d(hist_p_low, smooth_window)
    hist_p_high_s = _rolling_mean_1d(hist_p_high, smooth_window)

    ax.fill_between(periods, syn_p_low_s, syn_p_high_s,
                    alpha=ALPHA_FILL, color=synthetic_color,
                    label=f'{synthetic_label} {IQR_LABELS["99"]} (smoothed)')
    ax.plot(periods, syn_median,
            color=synthetic_color, linewidth=LINEWIDTH_MEDIUM, linestyle='-',
            label=f'{synthetic_label} (median)')

    ax.fill_between(periods, hist_p_low_s, hist_p_high_s,
                    alpha=ALPHA_FILL * 0.7, color='black',
                    label=f'Reconstructed Historical {IQR_LABELS["99"]} (smoothed)')
    ax.plot(periods, hist_median,
            color='black', linewidth=LINEWIDTH_THICK, linestyle='--',
            label='Reconstructed Historical (median)')

    ax.set_xlabel('Week of Year', fontsize=10)
    ax.set_ylabel('Streamflow (MCM/day)', fontsize=10)
    ax.set_xlim(1, 52.85)
    ax.set_yscale('log')
    ax.set_xticks(MONTH_WEEK_STARTS)
    ax.set_xticklabels(MONTH_LABELS, fontsize=9)
    ax.grid(False)


def _build_ensemble_figure_rev1(
    Q_historic,
    Q_synthetic,
    dataset_id,
    smooth_envelope=False,
    percentiles=(0.5, 99.5),
    figsize=(9, 9),
):
    """Build the rev1 ensemble-diagnostic figure.

    Parameters
    ----------
    Q_historic : pd.DataFrame
    Q_synthetic : dict
    dataset_id : str
    smooth_envelope : bool
        If True, apply 3-week centred rolling mean to Q0.5/Q99.5 bounds
        in panel (c) only.  Median is always unsmoothed.
    percentiles : tuple
        (lower, upper) percentiles for envelope, e.g. (0.5, 99.5).
    figsize : tuple

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    apply_publication_style()

    sites = NYC_RESERVOIRS
    synthetic_color = DATASET_COLORS.get(dataset_id, DATASET_COLORS['stationary_ensemble'])
    synthetic_label = _STATIONARY_BASELINE_LABEL

    # Pre-aggregate ONCE
    hist_agg = _get_aggregate_flow(Q_historic, sites)
    syn_agg = _pre_aggregate_synthetic(Q_synthetic, sites)

    pct = percentiles
    stats_timescale = 'monthly'
    timescale = 'weekly'

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 0.22, 0.22])

    ax_autocorr = fig.add_subplot(gs[0, 0])
    ax_fdc = fig.add_subplot(gs[0, 1])
    ax_periodic = fig.add_subplot(gs[1, :])
    ax_wilcoxon = fig.add_subplot(gs[2, :])
    ax_levene = fig.add_subplot(gs[3, :])

    # ---------- Panel (a): Autocorrelation ----------
    plot_autocorrelation_comparison(
        Q_historic, Q_synthetic,
        ax=ax_autocorr, percentiles=pct,
        synthetic_color=synthetic_color, synthetic_label=synthetic_label,
        show_legend=False,
        _hist_agg=hist_agg, _syn_agg=syn_agg,
    )
    label_panel(ax_autocorr, 'a', fontweight='bold')

    # ---------- Panel (b): FDC ----------
    plot_fdc_percentile_comparison(
        Q_historic, Q_synthetic,
        ax=ax_fdc, percentiles=pct,
        synthetic_color=synthetic_color, synthetic_label=synthetic_label,
        show_legend=False,
        _hist_agg=hist_agg, _syn_agg=syn_agg,
    )
    label_panel(ax_fdc, 'b', fontweight='bold')

    # ---------- Panel (c): Weekly percentile bands ----------
    if not smooth_envelope:
        plot_weekly_streamflow_percentiles(
            Q_historic, Q_synthetic,
            ax=ax_periodic, timescale=timescale, percentiles=pct,
            synthetic_color=synthetic_color, synthetic_label=synthetic_label,
            show_legend=False,
            _hist_agg=hist_agg, _syn_agg=syn_agg,
        )
    else:
        _plot_weekly_smooth(
            hist_agg, syn_agg,
            ax=ax_periodic,
            pct=pct,
            smooth_window=_SMOOTH_WINDOW,
            synthetic_color=synthetic_color,
            synthetic_label=synthetic_label,
        )

    label_panel(ax_periodic, 'c', fontweight='bold')
    add_scope_annotation(
        ax_periodic,
        "Aggregate inflow to Cannonsville, Pepacton, Neversink",
        fontsize=8,
    )

    # ---------- Panel (d): Wilcoxon p-values ----------
    plot_pvalue_comparison(
        Q_historic, Q_synthetic,
        ax=ax_wilcoxon, which='wilcoxon', timescale=stats_timescale,
        show_xticklabels=False,
        show_legend=False,
        _hist_agg=hist_agg, _syn_agg=syn_agg,
    )
    label_panel(ax_wilcoxon, 'd', fontweight='bold', y=0.85)

    # ---------- Panel (e): Levene p-values ----------
    plot_pvalue_comparison(
        Q_historic, Q_synthetic,
        ax=ax_levene, which='levene', timescale=stats_timescale,
        show_xticklabels=True,
        show_legend=False,
        _hist_agg=hist_agg, _syn_agg=syn_agg,
    )
    label_panel(ax_levene, 'e', fontweight='bold', y=0.85)

    # ---------- Shared legend ----------
    iqr_label = _IQR_99_LABEL
    legend_handles = [
        Patch(facecolor=synthetic_color, alpha=ALPHA_FILL,
              label=f'{synthetic_label} {iqr_label}'),
        Line2D([0], [0], color=synthetic_color, linewidth=LINEWIDTH_MEDIUM,
               linestyle='-', label=f'{synthetic_label} (median)'),
        Patch(facecolor='black', alpha=ALPHA_FILL * 0.7,
              label=f'{_RECONSTRUCTED_HIST_LABEL} {iqr_label}'),
        Line2D([0], [0], color='black', linewidth=LINEWIDTH_THICK,
               linestyle='--', label=f'{_RECONSTRUCTED_HIST_LABEL} (median)'),
        Line2D([0], [0], color='k', linestyle='--', linewidth=1, label='p = 0.05'),
    ]

    fig.legend(
        handles=legend_handles,
        loc='lower center', ncol=3, frameon=False,
        bbox_to_anchor=(0.5, -0.04), fontsize=9,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.09)

    fig.align_ylabels([ax_autocorr, ax_periodic, ax_wilcoxon, ax_levene])

    # Reduce gap between panels d and e
    pos_w = ax_wilcoxon.get_position()
    pos_l = ax_levene.get_position()
    gap = pos_w.y0 - (pos_l.y0 + pos_l.height)
    shift = gap * 0.6
    ax_levene.set_position([pos_l.x0, pos_l.y0 + shift, pos_l.width, pos_l.height])

    return fig


def plot_manuscript_ensemble_figure(dataset_id):
    """Generate ensemble validation figure (rev1 + rev1_smooth)."""
    verify_dataset_id(dataset_id)
    ensemble_set_specs = ENSEMBLE_SETS[dataset_id]

    # Check if ensemble data exists
    missing_sets = [spec.set_id + 1 for spec in ensemble_set_specs
                    if not os.path.exists(spec.files['gage_flow'])]
    if missing_sets:
        print(f"ERROR: Missing ensemble sets: {missing_sets}")
        return False

    # Load data
    print("Loading data...")
    Q_historic = load_baseline_historical_flow(
        period='full', gage_flow=False, flowtype=BASELINE_DATASET)
    Q_historic.replace(0, np.nan, inplace=True)
    Q_historic.drop(columns=['delTrenton'], inplace=True, errors='ignore')

    syn_ensemble = load_and_combine_ensemble_sets(ensemble_set_specs, by_site=False)
    print(f"Loaded {len(syn_ensemble)} realizations")

    os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

    rev1_stem = f"{FIG_OUTPUT_DIR}/F1_{dataset_id}_ensemble_diagnostics_rev1"
    rev1_smooth_stem = (
        f"{FIG_OUTPUT_DIR}/F1_{dataset_id}_ensemble_diagnostics_rev1_smooth"
    )

    # --- Rev1 (unsmoothed) ---
    print("Building rev1 (unsmoothed) figure...")
    fig = _build_ensemble_figure_rev1(
        Q_historic, syn_ensemble, dataset_id,
        smooth_envelope=False,
        percentiles=(0.5, 99.5),
    )
    save_fig(fig, rev1_stem, dpi=600)
    plt.close(fig)

    # --- Rev1 smooth (smoothed envelope bounds) ---
    print("Building rev1_smooth (smoothed envelope) figure...")
    fig_smooth = _build_ensemble_figure_rev1(
        Q_historic, syn_ensemble, dataset_id,
        smooth_envelope=True,
        percentiles=(0.5, 99.5),
    )
    save_fig(fig_smooth, rev1_smooth_stem, dpi=600)
    plt.close(fig_smooth)

    return True


def main(dataset_id):
    """Main function."""
    print(f"F1: Ensemble validation - {dataset_id}")
    success = plot_manuscript_ensemble_figure(dataset_id)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Fig4_plot_ensemble_diagnostics.py <dataset_id>")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)

    dataset_id = sys.argv[1]
    verify_dataset_id(dataset_id)
    main(dataset_id)
