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
    DATASET_COLORS, DATASET_LABELS,
    HISTORIC_COLOR, RECONSTRUCTED_HIST_LABEL,
    ALPHA_BAND_OUTER, ALPHA_BAND_INNER,
    LINEWIDTH_MEDIUM, LINEWIDTH_THICK,
    DPI_PRINT, apply_publication_style,
    save_fig, label_panel,
)
from methods.plotting.legend import (
    IQRBandHandle, iqr_band_legend_kwargs, draw_iqr_anatomy,
)
from methods.load import load_baseline_historical_flow, load_and_combine_ensemble_sets
from methods.config import (
    FIG_DIR, DATASET_CONFIGS, BASELINE_DATASET,
    verify_dataset_id,
)
from methods.ensemble_utils import ENSEMBLE_SETS

# Output directory
FIG_OUTPUT_DIR = f"{FIG_DIR}/Fig4_ensemble_diagnostics"

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
    Q_hist_weekly = hist_agg.resample('W').sum()
    hist_weeks = Q_hist_weekly.index.isocalendar().week.values.astype(int)
    hist_values = Q_hist_weekly.values

    hist_median = np.empty(n_periods)
    hist_p_low = np.empty(n_periods)
    hist_p_high = np.empty(n_periods)
    hist_iq_low = np.empty(n_periods)
    hist_iq_high = np.empty(n_periods)
    for w in periods:
        vals = hist_values[hist_weeks == w]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            hist_median[w - 1] = np.median(vals)
            hist_p_low[w - 1] = np.percentile(vals, pct[0])
            hist_p_high[w - 1] = np.percentile(vals, pct[1])
            hist_iq_low[w - 1] = np.percentile(vals, 25)
            hist_iq_high[w - 1] = np.percentile(vals, 75)
        else:
            hist_median[w-1] = hist_p_low[w-1] = hist_p_high[w-1] = np.nan
            hist_iq_low[w-1] = hist_iq_high[w-1] = np.nan

    # --- Synthetic ---
    syn_weekly_by_week = {w: [] for w in periods}
    for flow_series in syn_agg.values():
        weekly = flow_series.resample('W').sum()
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
    syn_iq_low = np.empty(n_periods)
    syn_iq_high = np.empty(n_periods)
    for w in periods:
        if syn_weekly_by_week[w]:
            all_vals = np.concatenate(syn_weekly_by_week[w])
            syn_median[w - 1] = np.median(all_vals)
            syn_p_low[w - 1] = np.percentile(all_vals, pct[0])
            syn_p_high[w - 1] = np.percentile(all_vals, pct[1])
            syn_iq_low[w - 1] = np.percentile(all_vals, 25)
            syn_iq_high[w - 1] = np.percentile(all_vals, 75)
        else:
            syn_median[w-1] = syn_p_low[w-1] = syn_p_high[w-1] = np.nan
            syn_iq_low[w-1] = syn_iq_high[w-1] = np.nan

    # Convert MGD to MCM (weekly totals)
    syn_p_low   *= MGD_TO_MCM
    syn_p_high  *= MGD_TO_MCM
    syn_iq_low  *= MGD_TO_MCM
    syn_iq_high *= MGD_TO_MCM
    syn_median  *= MGD_TO_MCM
    hist_p_low  *= MGD_TO_MCM
    hist_p_high *= MGD_TO_MCM
    hist_iq_low  *= MGD_TO_MCM
    hist_iq_high *= MGD_TO_MCM
    hist_median *= MGD_TO_MCM

    # Apply smoothing to outer bounds only (not inner or median)
    syn_p_low_s = _rolling_mean_1d(syn_p_low, smooth_window)
    syn_p_high_s = _rolling_mean_1d(syn_p_high, smooth_window)
    hist_p_low_s = _rolling_mean_1d(hist_p_low, smooth_window)
    hist_p_high_s = _rolling_mean_1d(hist_p_high, smooth_window)

    # Layered from bottom: syn 99% → hist 99% → syn 50% → hist 50% → syn median → hist median
    ax.fill_between(periods, syn_p_low_s, syn_p_high_s,
                    alpha=ALPHA_BAND_OUTER, color=synthetic_color, linewidth=0,
                    zorder=1, label=f'{synthetic_label} 99% IQR (smoothed)')
    ax.fill_between(periods, hist_p_low_s, hist_p_high_s,
                    alpha=ALPHA_BAND_OUTER, color=HISTORIC_COLOR, linewidth=0,
                    zorder=2, label=f'{RECONSTRUCTED_HIST_LABEL} 99% IQR (smoothed)')
    ax.fill_between(periods, syn_iq_low, syn_iq_high,
                    alpha=ALPHA_BAND_INNER, color=synthetic_color, linewidth=0,
                    zorder=3, label=f'{synthetic_label} 50% IQR')
    ax.fill_between(periods, hist_iq_low, hist_iq_high,
                    alpha=ALPHA_BAND_INNER, color=HISTORIC_COLOR, linewidth=0,
                    zorder=4, label=f'{RECONSTRUCTED_HIST_LABEL} 50% IQR')
    ax.plot(periods, syn_median,
            color=synthetic_color, linewidth=LINEWIDTH_MEDIUM, linestyle='-',
            zorder=5, label=f'{synthetic_label} (median)')
    ax.plot(periods, hist_median,
            color=HISTORIC_COLOR, linewidth=LINEWIDTH_THICK, linestyle='--',
            zorder=6, label=f'{RECONSTRUCTED_HIST_LABEL} (median)')

    ax.set_xlabel('Week of Year', fontsize=10)
    ax.set_ylabel('Combined NYC Reservoir\nWeekly Total Inflow (MCM)', fontsize=10)
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
    figsize=(9, 10.5),
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
    synthetic_label = DATASET_LABELS.get(dataset_id, dataset_id)

    # Pre-aggregate ONCE
    hist_agg = _get_aggregate_flow(Q_historic, sites)
    syn_agg = _pre_aggregate_synthetic(Q_synthetic, sites)

    pct = percentiles
    stats_timescale = 'monthly'
    timescale = 'weekly'

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        5, 2, figure=fig, height_ratios=[1, 1, 0.22, 0.22, 0.4],
    )

    ax_autocorr = fig.add_subplot(gs[0, 0])
    ax_fdc = fig.add_subplot(gs[0, 1])
    ax_periodic = fig.add_subplot(gs[1, :])
    ax_wilcoxon = fig.add_subplot(gs[2, :])
    ax_levene = fig.add_subplot(gs[3, :])

    # Bottom row: composite legend strip — [markers | anatomy | ensembles]
    legend_gs = gridspec.GridSpecFromSubplotSpec(
        1, 3,
        subplot_spec=gs[4, :],
        width_ratios=[0.6, 1.0, 1.4],
        wspace=0.2,
    )
    ax_markers = fig.add_subplot(legend_gs[0, 0])
    ax_anatomy = fig.add_subplot(legend_gs[0, 1])
    ax_datasets = fig.add_subplot(legend_gs[0, 2])
    for _ax in (ax_markers, ax_anatomy, ax_datasets):
        _ax.set_axis_off()

    # ---------- Panel (a): Autocorrelation ----------
    plot_autocorrelation_comparison(
        Q_historic, Q_synthetic,
        ax=ax_autocorr, percentiles=pct,
        synthetic_color=synthetic_color, synthetic_label=synthetic_label,
        show_legend=False,
        _hist_agg=hist_agg, _syn_agg=syn_agg,
    )
    label_panel(ax_autocorr, 'a')

    # ---------- Panel (b): FDC ----------
    plot_fdc_percentile_comparison(
        Q_historic, Q_synthetic,
        ax=ax_fdc, percentiles=pct,
        synthetic_color=synthetic_color, synthetic_label=synthetic_label,
        show_legend=False,
        _hist_agg=hist_agg, _syn_agg=syn_agg,
    )
    label_panel(ax_fdc, 'b')

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

    label_panel(ax_periodic, 'c')

    # ---------- Panel (d): Wilcoxon p-values ----------
    plot_pvalue_comparison(
        Q_historic, Q_synthetic,
        ax=ax_wilcoxon, which='wilcoxon', timescale=stats_timescale,
        show_xticklabels=False,
        show_legend=False,
        _hist_agg=hist_agg, _syn_agg=syn_agg,
    )
    label_panel(ax_wilcoxon, 'd', y=0.85)

    # ---------- Panel (e): Levene p-values ----------
    plot_pvalue_comparison(
        Q_historic, Q_synthetic,
        ax=ax_levene, which='levene', timescale=stats_timescale,
        show_xticklabels=True,
        show_legend=False,
        _hist_agg=hist_agg, _syn_agg=syn_agg,
    )
    label_panel(ax_levene, 'e', y=0.85)

    # ---------- Composite legend: [markers | anatomy | ensembles] ----------
    # Panel 1 (left): other markers — just the p = 0.05 line
    ax_markers.legend(
        handles=[Line2D([0], [0], color='k', linestyle='--', linewidth=1)],
        labels=['p = 0.05'],
        loc='center', frameon=False, fontsize=9,
        handletextpad=1.0,
    )

    # Panel 2 (middle): anatomy teaching glyph
    draw_iqr_anatomy(ax_anatomy, fontsize=9)

    # Panel 3 (right): ensembles with IQR glyphs. Reconstructed historical
    # shares the alpha structure but uses a dashed thick median.
    dataset_handles = [
        IQRBandHandle(color=synthetic_color),
        IQRBandHandle(color=HISTORIC_COLOR, linestyle='--',
                      linewidth=LINEWIDTH_THICK),
    ]
    dataset_labels = [
        f'{synthetic_label} Ensemble',
        f'{RECONSTRUCTED_HIST_LABEL} Ensemble',
    ]
    ax_datasets.legend(
        handles=dataset_handles,
        labels=dataset_labels,
        loc='center left',
        bbox_to_anchor=(0.0, 0.5),
        frameon=False,
        fontsize=9,
        **iqr_band_legend_kwargs(
            handleheight=1.8, handlelength=2.4, labelspacing=0.7,
        ),
    )

    plt.tight_layout()

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
        period='full', 
        gage_flow=False, 
        flowtype=BASELINE_DATASET)
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
        percentiles=(2.5, 97.5),
    )
    save_fig(fig, rev1_stem, dpi=600)
    plt.close(fig)

    # --- Rev1 smooth (smoothed envelope bounds) ---
    print("Building rev1_smooth (smoothed envelope) figure...")
    fig_smooth = _build_ensemble_figure_rev1(
        Q_historic, syn_ensemble, dataset_id,
        smooth_envelope=True,
        percentiles=(2.5, 97.5),
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
