"""
F2: Drought metric distribution figure.

Left panel: hexbin of severity vs magnitude for the stationary ensemble.
Right panels (2x2): exceedance-rate CDFs per climate scenario (rows) and
drought metric (columns), with stationary ensemble shown as outline bands.

Usage:
  python F2_plot_drought_metric_distribution.py [ssi_window]
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import warnings
warnings.filterwarnings("ignore")

from methods.config import FIG_DIR, N_YEARS, TOTAL_REALIZATIONS, SSI_WINDOWS
from methods.load import load_drought_events
from methods.plotting.styles import (
    DATASET_COLORS, DATASET_LINESTYLES, DATASET_LABELS,
    HISTORIC_COLOR, HISTORIC_LABEL,
    LINEWIDTH_MEDIUM, CMAP_SEQUENTIAL, DPI_HIGH,
    FONTSIZE_SMALL, FONTSIZE_MEDIUM,
)

# Output directory
FIG_OUTPUT_DIR = f"{FIG_DIR}/F2_drought_distributions"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

# Axis labels
METRIC_AXIS_LABELS = {
    'severity': 'Severity (min SSI)',
    'magnitude': 'Magnitude (cumulative deficit)',
    'duration': 'Duration (months)',
}

PANEL_LETTERS = list('abcdefghij')

# Number of years for exceedance rate normalization
HISTORIC_N_YEARS = 77

# Climate scenario datasets for right-panel rows (low above high)
CLIMATE_SCENARIOS = ['climate_adjusted_low', 'climate_adjusted_high']


def _compute_realization_exceedance_bands(df, metric, n_years, n_grid=200,
                                           percentiles=(0, 50, 100)):
    """Compute exceedance-rate bands across realizations.

    For each realization, builds a step-function exceedance curve
    (count of events with metric >= x, divided by n_years), then
    evaluates all realizations on a shared x-grid and returns
    percentile envelopes.

    Parameters
    ----------
    df : pd.DataFrame
        Drought events with columns ``metric`` and ``realization_id``.
    metric : str
        Column name for the drought metric.
    n_years : int
        Number of simulation years *per realization* (for rate normalisation).
    n_grid : int
        Resolution of the common x-grid.
    percentiles : tuple of int
        Percentiles to compute across realizations.

    Returns
    -------
    x_grid : np.ndarray
        Common metric values.
    bands : dict
        ``{p: array}`` for each percentile *p*.
    """
    all_vals = df[metric].values
    x_min, x_max = np.nanmin(all_vals), np.nanmax(all_vals)
    x_grid = np.linspace(x_min, x_max, n_grid)

    realization_ids = sorted(df['realization_id'].unique())
    curves = np.zeros((len(realization_ids), n_grid))

    for i, rid in enumerate(realization_ids):
        vals = df.loc[df['realization_id'] == rid, metric].values
        for j, x in enumerate(x_grid):
            curves[i, j] = np.sum(vals >= x) / n_years

    bands = {}
    for p in percentiles:
        bands[p] = np.percentile(curves, p, axis=0)

    return x_grid, bands


def plot_drought_manuscript_figure(
    ssi_window=12,
    cdf_metrics=None,
    hexbin_dataset='stationary_ensemble',
    hexbin_x='severity',
    hexbin_y='magnitude',
    figsize=None,
    fname=None,
    log_magnitude=False,
    log_hexbin_counts=False,
    log_exceedance=True,
):
    """Create the multipanel drought distribution manuscript figure.

    Layout
    ------
    Left: hexbin of severity vs magnitude (stationary ensemble).
    Right: 2x2 grid — rows are climate scenarios (low, high),
           columns are metrics (severity, magnitude).
    Each right subplot shows the climate-adjusted realization bands (filled),
    the stationary ensemble as unfilled outline bands, and historic markers.

    Parameters
    ----------
    ssi_window : int
        SSI window (3, 6, or 12).
    cdf_metrics : list of str, optional
        Metrics for CDF columns. Default: ['severity', 'magnitude'].
    hexbin_dataset : str
        Dataset used for the hexbin panel.
    hexbin_x, hexbin_y : str
        Metrics for hexbin x/y axes.
    figsize : tuple, optional
        Figure size. Auto-computed if None.
    fname : str, optional
        Output path. Auto-generated if None.
    log_magnitude : bool
        If True, use log scale for magnitude axes (hexbin y-axis and
        magnitude CDF x-axis).
    log_hexbin_counts : bool
        If True, use log scale for hexbin colorbar counts. Default False.
    log_exceedance : bool
        If True, use log scale for the exceedance-rate y-axis on right
        panels. Default True.

    Returns
    -------
    fig, (ax_hex, cdf_axes)
    """
    if cdf_metrics is None:
        cdf_metrics = ['severity', 'magnitude']

    n_rows = len(CLIMATE_SCENARIOS)
    n_cols = len(cdf_metrics)

    if figsize is None:
        figsize = (11.5, 7.0)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    MAX_SEVERITY = 6.2

    all_dataset_ids = ['stationary_ensemble'] + CLIMATE_SCENARIOS
    obs_droughts = load_drought_events(all_dataset_ids[0], ssi_window, observed=True)
    obs_droughts = obs_droughts[obs_droughts['severity'] <= MAX_SEVERITY]

    ensemble_data = {}
    for did in all_dataset_ids:
        df = load_drought_events(did, ssi_window, observed=False)
        df = df[df['severity'] <= MAX_SEVERITY]
        ensemble_data[did] = {'droughts': df}

    hexbin_droughts = ensemble_data[hexbin_dataset]['droughts']

    # ------------------------------------------------------------------
    # Figure layout: 2 rows (main panels + bottom bar), 2 columns
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=figsize)

    outer_gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[1.3, 1.0],
        height_ratios=[1, 0.04],
        hspace=0.35, wspace=0.35,
    )

    # Top-left: hexbin
    ax_hex = fig.add_subplot(outer_gs[0, 0])

    # Top-right: 2x2 CDF grid
    inner_gs = gridspec.GridSpecFromSubplotSpec(
        n_rows, n_cols,
        subplot_spec=outer_gs[0, 1],
        hspace=0.4, wspace=0.35,
    )
    cdf_axes = np.empty((n_rows, n_cols), dtype=object)
    for r in range(n_rows):
        for c in range(n_cols):
            cdf_axes[r, c] = fig.add_subplot(inner_gs[r, c])

    # Bottom-left: colorbar axes
    ax_cbar = fig.add_subplot(outer_gs[1, 0])

    # Bottom-right: legend axes (invisible, used for anchor)
    ax_legend = fig.add_subplot(outer_gs[1, 1])
    ax_legend.set_axis_off()

    # ------------------------------------------------------------------
    # Left panel: hexbin
    # ------------------------------------------------------------------
    hexbin_kwargs = dict(
        gridsize=30,
        cmap=CMAP_SEQUENTIAL,
        mincnt=1,
    )
    if log_hexbin_counts:
        hexbin_kwargs['bins'] = 'log'
    if log_magnitude and hexbin_y == 'magnitude':
        hexbin_kwargs['yscale'] = 'log'
    hb = ax_hex.hexbin(
        hexbin_droughts[hexbin_x].values,
        hexbin_droughts[hexbin_y].values,
        **hexbin_kwargs,
    )

    # Overlay observed
    if len(obs_droughts) > 0:
        ax_hex.scatter(
            obs_droughts[hexbin_x].values,
            obs_droughts[hexbin_y].values,
            s=40, marker='^', c=HISTORIC_COLOR, edgecolors='white',
            linewidths=0.6, alpha=0.9, zorder=10,
        )

    ax_hex.set_xlabel(METRIC_AXIS_LABELS[hexbin_x], fontsize=FONTSIZE_MEDIUM)
    ax_hex.set_ylabel(METRIC_AXIS_LABELS[hexbin_y], fontsize=FONTSIZE_MEDIUM)
    ax_hex.tick_params(labelsize=FONTSIZE_SMALL)
    ax_hex.spines['top'].set_visible(False)
    ax_hex.spines['right'].set_visible(False)
    ax_hex.text(
        0.03, 0.97, f'({PANEL_LETTERS[0]})',
        transform=ax_hex.transAxes, fontsize=FONTSIZE_MEDIUM,
        va='top', ha='left',
    )

    # ------------------------------------------------------------------
    # Colorbar in bottom-left cell
    # ------------------------------------------------------------------
    cb = fig.colorbar(hb, cax=ax_cbar, orientation='horizontal')
    cb.set_label('Count', fontsize=FONTSIZE_SMALL)
    cb.ax.tick_params(labelsize=FONTSIZE_SMALL - 1)
    if log_hexbin_counts:
        vmin, vmax = hb.get_clim()
        log_ticks = [np.log10(10**e) for e in range(0, 7)
                     if vmin <= np.log10(10**e) <= vmax]
        log_labels = [f'{10**e:g}' for e in range(0, 7)
                      if vmin <= np.log10(10**e) <= vmax]
        if log_ticks:
            cb.set_ticks(log_ticks)
            cb.set_ticklabels(log_labels)

    # ------------------------------------------------------------------
    # Right panels: exceedance-rate CDFs (2x2)
    # ------------------------------------------------------------------
    # Pre-compute stationary bands for each metric
    stat_bands = {}
    stat_df = ensemble_data['stationary_ensemble']['droughts']
    for metric in cdf_metrics:
        stat_bands[metric] = _compute_realization_exceedance_bands(
            stat_df, metric, n_years=N_YEARS,
        )

    panel_idx = 1  # panel (a) is hexbin
    for r, scenario_id in enumerate(CLIMATE_SCENARIOS):
        for c, metric in enumerate(cdf_metrics):
            ax = cdf_axes[r, c]

            # --- Stationary ensemble: unfilled outline bands ---
            x_s, bands_s = stat_bands[metric]
            stat_color = DATASET_COLORS['stationary_ensemble']
            # Full range outline (0-100%)
            ax.fill_between(
                x_s, bands_s[0], bands_s[100],
                facecolor='none', edgecolor=stat_color,
                linewidth=0.6, alpha=0.5, zorder=3,
            )
            # Median line
            ax.plot(
                x_s, bands_s[50],
                color=stat_color, linestyle='-',
                linewidth=LINEWIDTH_MEDIUM, zorder=3.5,
            )

            # --- Climate-adjusted ensemble: filled bands ---
            clim_df = ensemble_data[scenario_id]['droughts']
            clim_color = DATASET_COLORS.get(scenario_id, '#808080')
            x_c, bands_c = _compute_realization_exceedance_bands(
                clim_df, metric, n_years=N_YEARS,
            )
            # Full range (0-100%)
            ax.fill_between(
                x_c, bands_c[0], bands_c[100],
                color=clim_color, alpha=0.15, zorder=4,
            )
            # Median line
            ax.plot(
                x_c, bands_c[50],
                color=clim_color, linestyle=DATASET_LINESTYLES.get(scenario_id, '--'),
                linewidth=LINEWIDTH_MEDIUM, zorder=5,
            )

            # --- Historic markers ---
            vals = np.sort(obs_droughts[metric].values)[::-1]
            exceedance = np.arange(1, len(vals) + 1) / HISTORIC_N_YEARS
            ax.scatter(
                vals, exceedance,
                color=HISTORIC_COLOR, marker='^', s=25,
                edgecolors='white', linewidths=0.4,
                zorder=6,
            )

            # --- Formatting ---
            if r == n_rows - 1:
                ax.set_xlabel(METRIC_AXIS_LABELS[metric], fontsize=FONTSIZE_MEDIUM)
            else:
                ax.set_xlabel('')
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel('Exceedance rate (yr$^{-1}$)', fontsize=FONTSIZE_MEDIUM)
            else:
                ax.set_ylabel('')
                ax.set_yticklabels([])

            if log_magnitude and metric == 'magnitude':
                ax.set_xscale('log')

            if log_exceedance:
                ax.set_yscale('log')
                # Set a reasonable lower bound to avoid log(0) issues
                ax.set_ylim(bottom=1e-3)

            ax.tick_params(labelsize=FONTSIZE_SMALL)
            ax.grid(True, which='both', color='gray', alpha=0.15,
                    linewidth=0.5, linestyle='--')
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.text(
                0.03, 0.97, f'({PANEL_LETTERS[panel_idx]})',
                transform=ax.transAxes, fontsize=FONTSIZE_MEDIUM,
                va='top', ha='left',
            )

            # Row label on right side of rightmost column
            if c == n_cols - 1:
                ax.text(
                    1.02, 0.5, DATASET_LABELS.get(scenario_id, scenario_id),
                    transform=ax.transAxes, fontsize=FONTSIZE_MEDIUM,
                    va='center', ha='left', rotation=-90,
                )

            panel_idx += 1

    # ------------------------------------------------------------------
    # Shared legend below figure
    # ------------------------------------------------------------------
    legend_handles = [
        mlines.Line2D([], [], color=HISTORIC_COLOR, marker='^',
                      linestyle='None', markersize=6, label=HISTORIC_LABEL),
        mlines.Line2D([], [], color=DATASET_COLORS['stationary_ensemble'],
                      linestyle='-', linewidth=LINEWIDTH_MEDIUM,
                      label='Stationary'),
        mlines.Line2D([], [], color=DATASET_COLORS['climate_adjusted_low'],
                      linestyle=DATASET_LINESTYLES['climate_adjusted_low'],
                      linewidth=LINEWIDTH_MEDIUM,
                      label=DATASET_LABELS['climate_adjusted_low']),
        mlines.Line2D([], [], color=DATASET_COLORS['climate_adjusted_high'],
                      linestyle=DATASET_LINESTYLES['climate_adjusted_high'],
                      linewidth=LINEWIDTH_MEDIUM,
                      label=DATASET_LABELS['climate_adjusted_high']),
    ]

    # Place legend in the bottom-right cell
    ax_legend.legend(
        handles=legend_handles,
        loc='center',
        ncol=2,
        frameon=False,
        fontsize=FONTSIZE_SMALL,
        columnspacing=1.5,
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if fname is None:
        fname = f"{FIG_OUTPUT_DIR}/F2_drought_distributions_ssi{ssi_window}.png"

    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")

    return fig, (ax_hex, cdf_axes)


def main():
    """Generate the F2 manuscript figure."""
    ssi_window = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    if ssi_window not in SSI_WINDOWS:
        print(f"ERROR: Invalid SSI window: {ssi_window}. Must be one of {SSI_WINDOWS}")
        sys.exit(1)

    print(f"F2: Drought metric distribution (SSI-{ssi_window})")

    plot_drought_manuscript_figure(ssi_window=ssi_window,
                                   log_magnitude=True,
                                   log_exceedance=False)
    plt.close('all')


if __name__ == "__main__":
    main()
