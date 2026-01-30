"""
Multipanel manuscript figure: drought metric distributions.

Left panel: hexbin of severity vs magnitude for the stationary ensemble.
Right panels (stacked): exceedance-rate CDFs of each drought metric across datasets.

Usage:
  python F2_plot_drought_metric_distribution.py [ssi_window]

Examples:
  python F2_plot_drought_metric_distribution.py
  python F2_plot_drought_metric_distribution.py 6
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from methods.config import *
from methods.load import load_drought_events
from methods.plotting.styles import (
    DATASET_COLORS, DATASET_LINESTYLES, DATASET_LABELS,
    HISTORIC_COLOR, HISTORIC_LABEL,
    LINEWIDTH_MEDIUM, CMAP_SEQUENTIAL, DPI_HIGH,
    FONTSIZE_SMALL, FONTSIZE_MEDIUM,
)

# Output directory
FIG_OUTPUT_DIR = f"{FIG_DIR}/drought_distributions"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

# Axis labels (clean for manuscript)
METRIC_AXIS_LABELS = {
    'severity': 'Severity (min SSI)',
    'magnitude': 'Magnitude (cumulative deficit)',
    'duration': 'Duration (months)',
}

# Panel labels
PANEL_LETTERS = list('abcdefghij')

# Number of years for exceedance rate normalization
HISTORIC_N_YEARS = 77  # 1945-2022
ENSEMBLE_N_YEARS = N_YEARS * TOTAL_REALIZATIONS


def plot_drought_manuscript_figure(
    ssi_window=12,
    cdf_metrics=None,
    dataset_ids=None,
    hexbin_dataset='stationary_ensemble',
    hexbin_x='severity',
    hexbin_y='magnitude',
    figsize=None,
    fname=None,
):
    """Create the multipanel drought distribution manuscript figure.

    Layout: left column is a single hexbin panel spanning all rows;
    right column has N stacked CDF panels (one per metric).

    Parameters
    ----------
    ssi_window : int
        SSI window (3, 6, or 12).
    cdf_metrics : list of str, optional
        Metrics for CDF panels. Default: ['severity', 'magnitude', 'duration'].
    dataset_ids : list of str, optional
        Ensemble dataset IDs. Default: all from DATASET_CONFIGS.
    hexbin_dataset : str
        Dataset used for the hexbin panel.
    hexbin_x, hexbin_y : str
        Metrics for hexbin x/y axes.
    figsize : tuple, optional
        Figure size. Auto-computed if None.
    fname : str, optional
        Output path. Auto-generated if None.

    Returns
    -------
    fig, (ax_hex, cdf_axes)
    """
    if cdf_metrics is None:
        cdf_metrics = ['severity', 'magnitude', 'duration']
    if dataset_ids is None:
        dataset_ids = list(DATASET_CONFIGS.keys())

    n_cdf = len(cdf_metrics)
    if figsize is None:
        panel_h = 2.5
        figsize = (3.0 + panel_h + 4.0, panel_h * n_cdf)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    MAX_SEVERITY = 6.2

    obs_droughts = load_drought_events(dataset_ids[0], ssi_window, observed=True)
    obs_droughts = obs_droughts[obs_droughts['severity'] <= MAX_SEVERITY]

    ensemble_data = {}
    for did in dataset_ids:
        df = load_drought_events(did, ssi_window, observed=False)
        df = df[df['severity'] <= MAX_SEVERITY]
        ensemble_data[did] = {'droughts': df}

    hexbin_droughts = ensemble_data[hexbin_dataset]['droughts']

    # ------------------------------------------------------------------
    # Create figure with gridspec: left col spans all rows, right col stacked
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        n_cdf, 2,
        width_ratios=[1.6, 1.0],
        hspace=0.35, wspace=0.4,
    )

    # Left panel: hexbin spanning all rows
    ax_hex = fig.add_subplot(gs[:, 0])

    # Right panels: one per CDF metric
    cdf_axes = [fig.add_subplot(gs[i, 1]) for i in range(n_cdf)]

    # ------------------------------------------------------------------
    # Left panel: hexbin
    # ------------------------------------------------------------------
    hb = ax_hex.hexbin(
        hexbin_droughts[hexbin_x].values,
        hexbin_droughts[hexbin_y].values,
        gridsize=30,
        cmap=CMAP_SEQUENTIAL,
        mincnt=1,
        bins='log',
    )

    # Colorbar below the hexbin panel
    import matplotlib.ticker as ticker
    cb = fig.colorbar(
        hb, ax=ax_hex, orientation='horizontal',
        shrink=0.8, pad=0.08, aspect=30,
    )
    cb.set_label('Count (log scale)', fontsize=FONTSIZE_SMALL)
    cb.ax.tick_params(labelsize=FONTSIZE_SMALL - 1)
    # Use clean log-spaced ticks: 1, 10, 100, 1000, ...
    cb.ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=6))
    cb.ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    cb.ax.xaxis.set_minor_locator(ticker.NullLocator())
    cb.update_ticks()

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
    # Right panels: exceedance-rate CDFs
    # ------------------------------------------------------------------
    legend_handles = []
    legend_labels = []

    for k, metric in enumerate(cdf_metrics):
        ax = cdf_axes[k]

        # --- Observed (markers only) ---
        vals = np.sort(obs_droughts[metric].values)[::-1]
        exceedance = np.arange(1, len(vals) + 1) / HISTORIC_N_YEARS
        scatter = ax.scatter(
            vals, exceedance,
            color=HISTORIC_COLOR, marker='^', s=25,
            edgecolors='white', linewidths=0.4,
            zorder=6,
        )
        if k == 0:
            legend_handles.append(scatter)
            legend_labels.append(HISTORIC_LABEL)

        # --- Ensemble datasets ---
        for did in dataset_ids:
            df = ensemble_data[did]['droughts']
            vals = np.sort(df[metric].values)[::-1]
            exceedance = np.arange(1, len(vals) + 1) / ENSEMBLE_N_YEARS
            line, = ax.plot(
                vals, exceedance,
                color=DATASET_COLORS.get(did, '#808080'),
                linestyle=DATASET_LINESTYLES.get(did, '-'),
                linewidth=LINEWIDTH_MEDIUM, zorder=5,
            )
            if k == 0:
                legend_handles.append(line)
                legend_labels.append(DATASET_LABELS.get(did, did))

        ax.set_xlabel(METRIC_AXIS_LABELS[metric], fontsize=FONTSIZE_MEDIUM)
        ax.set_ylabel('Exceedance rate (yr$^{-1}$)', fontsize=FONTSIZE_MEDIUM)
        # ax.set_yscale('log')
        ax.tick_params(labelsize=FONTSIZE_SMALL)
        ax.grid(True, which='both', color='gray', alpha=0.15,
                linewidth=0.5, linestyle='--')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.text(
            0.03, 0.97, f'({PANEL_LETTERS[1 + k]})',
            transform=ax.transAxes, fontsize=FONTSIZE_MEDIUM,
            va='top', ha='left',
        )

    # ------------------------------------------------------------------
    # Shared legend below figure
    # ------------------------------------------------------------------
    fig.legend(
        legend_handles, legend_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.04),
        ncol=len(legend_labels),
        frameon=False,
        fontsize=FONTSIZE_SMALL,
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

    print("=" * 60)
    print(f"F2: DROUGHT METRIC DISTRIBUTION (SSI-{ssi_window})")
    print("=" * 60)

    plot_drought_manuscript_figure(ssi_window=ssi_window,
                                   dataset_ids=['stationary_ensemble','climate_adjusted_low', 'climate_adjusted_high'],)
    plt.close('all')

    print("\nDone.")


if __name__ == "__main__":
    main()
