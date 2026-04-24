"""
F2: Drought metric distribution figure.

Left panel: hexbin of severity vs magnitude for the stationary ensemble.
Right panels (3x2): exceedance-rate CDFs per dataset (rows) and
drought metric (columns). Each row shows one dataset plus historic markers.

Usage:
  python F2_plot_drought_metric_distribution.py [ssi_window]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator
import warnings
warnings.filterwarnings("ignore")

from methods.config import FIG_DIR, N_YEARS, RECONSTRUCTION_N_YEARS, SSI_WINDOWS
from methods.load import load_drought_events
from methods.plotting.styles import (
    DATASET_COLORS, DATASET_LINESTYLES, DATASET_LABELS,
    HISTORIC_COLOR, HISTORIC_LABEL,
    ALPHA_BAND_OUTER, ALPHA_BAND_INNER,
    LINEWIDTH_MEDIUM, CMAP_SEQUENTIAL, DPI_HIGH,
    FONTSIZE_SMALL, FONTSIZE_MEDIUM,
)


HISTORIC_LABEL += " Droughts"

# Output directory
FIG_OUTPUT_DIR = f"{FIG_DIR}/Fig5_drought_distributions"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

# Axis labels
METRIC_AXIS_LABELS = {
    'severity': 'Severity (max deviation)',
    'magnitude': 'Magnitude (cumulative deficit, deficit-months)',
    'duration': 'Duration (months)',
}

# Multiline x-axis labels for the right-panel (CDF) bottom row
METRIC_CDF_AXIS_LABELS = {
    'severity': 'Severity\n(max deficit)',
    'magnitude': 'Magnitude\n(cumulative deficit, deficit-months)',
    'duration': 'Duration\n(months)',
}

PANEL_LETTERS = list('abcdefghij')

# Number of years for exceedance rate normalization (from config)
HISTORIC_N_YEARS = RECONSTRUCTION_N_YEARS

# All datasets for right-panel rows (stationary first, then low, then high)
ALL_DATASETS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']


def _compute_realization_exceedance_bands(df, metric, n_years, n_grid=200,
                                           percentiles=(1, 25, 50, 75, 99)):
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


def _compute_exceedance_on_grid(df, metric, n_years, x_grid, percentiles=(1, 25, 50, 75, 99)):
    """
    Compute exceedance-rate bands on a provided x_grid using vectorised searchsorted.

    Parameters
    ----------
    df : pd.DataFrame
        Drought events with columns ``metric`` and ``realization_id``.
    metric : str
        Column name for the drought metric.
    n_years : int
        Simulation years per realization for rate normalisation.
    x_grid : np.ndarray
        Pre-computed common grid of metric values.
    percentiles : tuple of int
        Percentiles to compute across realizations.

    Returns
    -------
    bands : dict
        ``{p: array}`` for each percentile *p*.
    """
    realization_ids = sorted(df['realization_id'].unique())
    curves = np.zeros((len(realization_ids), len(x_grid)))

    for i, rid in enumerate(realization_ids):
        vals = np.sort(df.loc[df['realization_id'] == rid, metric].dropna().values)
        counts = len(vals) - np.searchsorted(vals, x_grid, side='left')
        curves[i, :] = counts / n_years

    return {p: np.percentile(curves, p, axis=0) for p in percentiles}


def _compute_delta_bands(df_scenario, baseline_median, metric, n_years, x_grid,
                         percentiles=(1, 25, 50, 75, 99)):
    """
    Compute change in exceedance rate per realization vs. a fixed baseline curve.

    For each realization in *df_scenario*, computes the exceedance rate on *x_grid*
    then subtracts *baseline_median* element-wise.  Percentile bands of the resulting
    delta curves are returned.

    Parameters
    ----------
    df_scenario : pd.DataFrame
        Drought events with columns ``metric`` and ``realization_id``.
    baseline_median : np.ndarray, shape (len(x_grid),)
        Reference exceedance curve (stationary ensemble median).
    metric : str
        Column name for the drought metric.
    n_years : int
        Simulation years per realization.
    x_grid : np.ndarray
        Shared metric grid.
    percentiles : tuple of int
        Percentiles to compute across realization delta curves.

    Returns
    -------
    bands : dict
        ``{p: array}`` delta exceedance rate (yr⁻¹) at each *x_grid* point.
    """
    realization_ids = sorted(df_scenario['realization_id'].unique())
    delta_curves = np.zeros((len(realization_ids), len(x_grid)))

    for i, rid in enumerate(realization_ids):
        vals = np.sort(
            df_scenario.loc[df_scenario['realization_id'] == rid, metric].dropna().values
        )
        counts = len(vals) - np.searchsorted(vals, x_grid, side='left')
        delta_curves[i, :] = counts / n_years - baseline_median

    return {p: np.percentile(delta_curves, p, axis=0) for p in percentiles}


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
    plot_relative_change=True,
    gridshape='square',
):
    """Create the multipanel drought distribution manuscript figure.

    Layout
    ------
    Left: 2-D distribution of severity vs magnitude (stationary ensemble).
    Right: 3x2 grid - rows are datasets (stationary, low, high),
           columns are metrics (severity, magnitude).
    Each right subplot shows that dataset's realization bands plus historic markers.

    Parameters
    ----------
    ssi_window : int
        SSI window (3, 6, or 12).
    cdf_metrics : list of str, optional
        Metrics for CDF columns. Default: ['severity', 'magnitude'].
    hexbin_dataset : str
        Dataset used for the left panel.
    hexbin_x, hexbin_y : str
        Metrics for left-panel x/y axes.
    figsize : tuple, optional
        Figure size. Auto-computed if None.
    fname : str, optional
        Output path. Auto-generated if None.
    log_magnitude : bool
        If True, use log scale for magnitude axes.
    log_hexbin_counts : bool
        If True, use log scale for the left-panel count colorbar.
    log_exceedance : bool
        If True, use log scale for the exceedance-rate y-axis.
    plot_relative_change : bool
        If True, right panels show the change in exceedance rate relative to the
        stationary-ensemble median, rather than absolute exceedance rates.
        For each realization, Δ = exceedance_realization − median_stationary on a
        shared x-grid.  The filled band is the 0–100th percentile of Δ across
        realizations; the solid line is the median Δ.  A dashed y = 0 line marks
        the baseline.  Historic markers are expressed as Δ relative to the
        stationary median curve.  Default: False.
    gridshape : str
        Left-panel bin shape: 'square' (default, uses hist2d) or 'hex'
        (uses hexbin, legacy behaviour).

    Returns
    -------
    fig, (ax_hex, cdf_axes)
    """
    if cdf_metrics is None:
        cdf_metrics = ['severity', 'magnitude']

    n_rows = len(ALL_DATASETS)
    n_cols = len(cdf_metrics)

    if figsize is None:
        figsize = (11.5, 8) 
    
    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    MAX_SEVERITY = 6.2

    obs_droughts = load_drought_events(ALL_DATASETS[0], ssi_window, observed=True)
    obs_droughts = obs_droughts[obs_droughts['severity'] <= MAX_SEVERITY]

    ensemble_data = {}
    for did in ALL_DATASETS:
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
        hspace=0.25, wspace=0.35,
    )

    # Top-left: hexbin
    ax_hex = fig.add_subplot(outer_gs[0, 0])

    # Top-right: 3x2 CDF grid (one row per dataset)
    inner_gs = gridspec.GridSpecFromSubplotSpec(
        n_rows, n_cols,
        subplot_spec=outer_gs[0, 1],
        hspace=0.35, wspace=0.35,
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
    # Left panel: 2-D distribution (square bins or hexbin)
    # ------------------------------------------------------------------
    mag_lim = 100 if ssi_window == 3 else 200
    n_bins = 15
    
    if gridshape == 'hex':
        hexbin_kwargs = dict(gridsize=30, cmap=CMAP_SEQUENTIAL, mincnt=1)
        if log_hexbin_counts:
            hexbin_kwargs['bins'] = 'log'
        if log_magnitude and hexbin_y == 'magnitude':
            hexbin_kwargs['yscale'] = 'log'
        hb = ax_hex.hexbin(
            hexbin_droughts[hexbin_x].values,
            hexbin_droughts[hexbin_y].values,
            **hexbin_kwargs,
        )
        bin_min = int(hb.get_array().min()) if len(hb.get_array()) > 0 else 1
        bin_max = int(hb.get_array().max()) if len(hb.get_array()) > 0 else 1
    else:  # square
        x_data = hexbin_droughts[hexbin_x].values
        y_data = hexbin_droughts[hexbin_y].values

        x_max = mag_lim if hexbin_x == 'magnitude' else x_data.max()
        y_max = mag_lim if hexbin_y == 'magnitude' else y_data.max()

        if log_magnitude and hexbin_x == 'magnitude':
            x_min = x_data[x_data > 0].min() if np.any(x_data > 0) else 1e-3
            x_bins = np.logspace(np.log10(x_min), np.log10(x_max), n_bins + 1)
        else:
            x_bins = np.linspace(x_data.min(), x_max, n_bins + 1)

        if log_magnitude and hexbin_y == 'magnitude':
            y_min = y_data[y_data > 0].min() if np.any(y_data > 0) else 1e-3
            y_bins = np.logspace(np.log10(y_min), np.log10(y_max), n_bins + 1)
        else:
            y_bins = np.linspace(y_data.min(), y_max, n_bins + 1)

        # Pre-compute counts to find the true populated range, so the
        # norm / colorbar don't extend into empty decades (10^4, 10^5).
        counts_prelim, _, _ = np.histogram2d(x_data, y_data, bins=[x_bins, y_bins])
        bin_min = int(counts_prelim[counts_prelim >= 1].min()) if np.any(counts_prelim >= 1) else 1
        bin_max = int(counts_prelim.max()) if np.any(counts_prelim >= 1) else 1

        # Discrete half-decade log-spaced color bins. extend='both' puts
        # arrows on both colorbar ends that capture the tails.
        log_step = 0.5
        log_lo = np.floor(np.log10(max(bin_min, 1)) / log_step) * log_step
        log_hi = np.ceil(np.log10(max(bin_max, 1)) / log_step) * log_step
        cbar_boundaries = 10 ** np.arange(log_lo, log_hi + log_step / 2, log_step)
        # BoundaryNorm with extend='both' needs ncolors = (#intervals) + 2
        # so the two extension regions each get their own discrete color.
        n_data_bins = max(len(cbar_boundaries) - 1, 1)
        n_cbar_colors = n_data_bins + 2
        discrete_cmap = plt.get_cmap(CMAP_SEQUENTIAL, n_cbar_colors)
        discrete_norm = mcolors.BoundaryNorm(cbar_boundaries, n_cbar_colors,
                                             extend='both')

        hist2d_kwargs = dict(bins=[x_bins, y_bins], cmap=discrete_cmap, cmin=1,
                             norm=discrete_norm)
        _, _, _, hb = ax_hex.hist2d(x_data, y_data, **hist2d_kwargs)

        if log_magnitude and hexbin_y == 'magnitude':
            ax_hex.set_yscale('log')

    # Overlay observed
    if len(obs_droughts) > 0:
        ax_hex.scatter(
            obs_droughts[hexbin_x].values,
            obs_droughts[hexbin_y].values,
            s=80, marker='^', c=HISTORIC_COLOR, edgecolors='white',
            linewidths=0.6, alpha=0.9, zorder=10,
        )

    # Identify & highlight the 1960s drought (event active during Dec 1964)
    target_date = pd.Timestamp('1964-12-01')
    drought_1960s = obs_droughts[
        (pd.to_datetime(obs_droughts['start']) <= target_date) &
        (pd.to_datetime(obs_droughts['end']) >= target_date)
    ]
    if len(drought_1960s) > 0:
        row_1960s = drought_1960s.iloc[0]
        drought_1960s_start = pd.to_datetime(row_1960s['start']).strftime('%b %Y')
        drought_1960s_end = pd.to_datetime(row_1960s['end']).strftime('%b %Y')
        drought_1960s_label = f"1960s Drought ({drought_1960s_start}\u2013{drought_1960s_end})"
        ax_hex.scatter(
            drought_1960s[hexbin_x].values,
            drought_1960s[hexbin_y].values,
            s=100, marker='^', c='red', edgecolors='white',
            linewidths=0.6, alpha=0.9, zorder=11,
        )

    if hexbin_x == 'magnitude':
        ax_hex.set_xlim(right=mag_lim)
    if hexbin_y == 'magnitude':
        ax_hex.set_ylim(top=mag_lim)
    ax_hex.set_xlabel(METRIC_AXIS_LABELS[hexbin_x], fontsize=FONTSIZE_MEDIUM)
    ax_hex.set_ylabel(METRIC_AXIS_LABELS[hexbin_y], fontsize=FONTSIZE_MEDIUM)
    ax_hex.tick_params(labelsize=FONTSIZE_SMALL)
    ax_hex.text(
        0.03, 0.97, f'({PANEL_LETTERS[0]})',
        transform=ax_hex.transAxes, fontsize=FONTSIZE_MEDIUM,
        va='top', ha='left',
    )

    # -------- 1960s drought-of-record callout (arrow + label) --------
    if len(drought_1960s) > 0:
        start_year = pd.to_datetime(row_1960s['start']).year
        end_year = pd.to_datetime(row_1960s['end']).year
        short_1960s_label = f"{start_year}–{end_year} Drought"
        ax_hex.annotate(
            short_1960s_label,
            xy=(float(row_1960s[hexbin_x]), float(row_1960s[hexbin_y])),
            xytext=(-12, -14),
            textcoords='offset points',
            fontsize=FONTSIZE_SMALL - 2,
            color='red', fontweight='bold',
            ha='right', va='top',
            arrowprops=dict(arrowstyle='->', color='red', lw=0.7,
                            shrinkA=1, shrinkB=3),
            zorder=12,
        )

    # ------------------------------------------------------------------
    # Colorbar in bottom-left cell: discrete bins with end arrows
    # ------------------------------------------------------------------
    cb = fig.colorbar(hb, cax=ax_cbar, orientation='horizontal',
                      extend='both', spacing='uniform')
    cb.set_label('Drought event count per bin', fontsize=FONTSIZE_SMALL)
    cb.ax.tick_params(labelsize=FONTSIZE_SMALL - 1)
    if log_hexbin_counts and gridshape == 'hex':
        # hexbin stores log10(count) internally; legacy path (not discretized)
        vmin, vmax = hb.get_clim()
        log_ticks = [np.log10(10**e) for e in range(0, 7)
                     if vmin <= np.log10(10**e) <= vmax]
        log_labels = [f'{10**e:g}' for e in range(0, 7)
                      if vmin <= np.log10(10**e) <= vmax]
        if log_ticks:
            cb.set_ticks(log_ticks)
            cb.set_ticklabels(log_labels)
    else:
        # One tick per discrete boundary. If there are many bins, thin the
        # labels to only powers of 10 to avoid overlap.
        if len(cbar_boundaries) <= 8:
            tick_vals = cbar_boundaries
        else:
            tick_vals = np.array([b for b in cbar_boundaries
                                  if np.isclose(np.log10(b) % 1, 0, atol=1e-6)])
        cb.set_ticks(tick_vals)
        cb.set_ticklabels([f'{int(round(t))}' if t >= 1 else f'{t:.2g}'
                           for t in tick_vals])

    # ------------------------------------------------------------------
    # Right panels: exceedance-rate CDFs (3x2) - one dataset per row
    # ------------------------------------------------------------------
    # Pre-compute shared x-grids and stationary-median baselines
    # (used only when plot_relative_change=True)
    if plot_relative_change:
        shared_x_grids = {}
        baseline_medians = {}
        for metric in cdf_metrics:
            all_vals = np.concatenate([
                ensemble_data[did]['droughts'][metric].dropna().values
                for did in ALL_DATASETS
            ])
            x_grid_m = np.linspace(np.nanmin(all_vals), np.nanmax(all_vals), 200)
            shared_x_grids[metric] = x_grid_m
            baseline_bands_m = _compute_exceedance_on_grid(
                ensemble_data['stationary_ensemble']['droughts'],
                metric, N_YEARS, x_grid_m, percentiles=(50,),
            )
            baseline_medians[metric] = baseline_bands_m[50]

    panel_idx = 1  # panel (a) is hexbin
    for r, dataset_id in enumerate(ALL_DATASETS):
        is_baseline = (dataset_id == 'stationary_ensemble')

        for c, metric in enumerate(cdf_metrics):
            ax = cdf_axes[r, c]
            df = ensemble_data[dataset_id]['droughts']
            color = DATASET_COLORS.get(dataset_id, '#808080')

            if plot_relative_change and not is_baseline:
                # --- Relative-change mode (climate scenarios only) ---
                x = shared_x_grids[metric]
                baseline_med = baseline_medians[metric]

                delta_bands = _compute_delta_bands(
                    df, baseline_med, metric, N_YEARS, x,
                )

                # Grey zone: where scenario has no droughts but
                # baseline may still have events (Δ=0 is misleading)
                scenario_max = df[metric].max()
                baseline_df = ensemble_data['stationary_ensemble']['droughts']
                baseline_max = baseline_df[metric].max()

                # Truncate bands/line at scenario data extent
                if scenario_max < baseline_max:
                    valid_mask = x <= scenario_max
                    ax.axvspan(scenario_max, x[-1],
                               facecolor='#d0d0d0', alpha=0.35,
                               hatch='///', edgecolor='#333333',
                               linewidth=0.8, zorder=2)
                else:
                    valid_mask = np.ones(len(x), dtype=bool)

                xv = x[valid_mask]
                ax.fill_between(xv,
                                delta_bands[1][valid_mask],
                                delta_bands[99][valid_mask],
                                color=color, alpha=ALPHA_BAND_OUTER, linewidth=0, zorder=4)
                ax.fill_between(xv,
                                delta_bands[25][valid_mask],
                                delta_bands[75][valid_mask],
                                color=color, alpha=ALPHA_BAND_INNER, linewidth=0, zorder=4)
                ax.plot(xv, delta_bands[50][valid_mask],
                        color=color,
                        linestyle='-',
                        linewidth=LINEWIDTH_MEDIUM, zorder=5)

                ax.axhline(0, color='gray', linestyle='--', linewidth=0.8,
                           alpha=0.7, zorder=3)

                if c == 0:
                    ax.set_ylabel('Δ Exceedance\nrate (yr$^{-1}$)',
                                  fontsize=FONTSIZE_MEDIUM)
                else:
                    ax.set_ylabel('')
                    ax.set_yticklabels([])

            else:
                # --- Absolute exceedance mode ---
                if plot_relative_change:
                    # Baseline row in relative-change figure: use shared grid
                    x = shared_x_grids[metric]
                    bands = _compute_exceedance_on_grid(
                        df, metric, N_YEARS, x,
                    )
                else:
                    x, bands = _compute_realization_exceedance_bands(
                        df, metric, n_years=N_YEARS,
                    )

                ax.fill_between(x, bands[1], bands[99],
                                color=color, alpha=ALPHA_BAND_OUTER, linewidth=0, zorder=4)
                ax.fill_between(x, bands[25], bands[75],
                                color=color, alpha=ALPHA_BAND_INNER, linewidth=0, zorder=4)
                ax.plot(x, bands[50],
                        color=color,
                        linestyle='-',
                        linewidth=LINEWIDTH_MEDIUM, zorder=5)

                # Historic markers (all rows for absolute; baseline only for relative change)
                if not plot_relative_change or is_baseline:
                    vals = np.sort(obs_droughts[metric].values)[::-1]
                    exceedance = np.arange(1, len(vals) + 1) / HISTORIC_N_YEARS
                    ax.scatter(vals, exceedance,
                               color=HISTORIC_COLOR, marker='^', s=50,
                               edgecolors='white', linewidths=0.4, zorder=6)

                    # Highlight 1960s drought on exceedance plot
                    if len(drought_1960s) > 0:
                        val_1960s = row_1960s[metric]
                        exc_1960s = np.sum(obs_droughts[metric].values >= val_1960s) / HISTORIC_N_YEARS
                        ax.scatter([val_1960s], [exc_1960s],
                                   color='red', marker='^', s=70,
                                   edgecolors='white', linewidths=0.4, zorder=7)

                if c == 0:
                    ax.set_ylabel('Exceedance\nrate (yr$^{-1}$)',
                                  fontsize=FONTSIZE_MEDIUM)
                else:
                    ax.set_ylabel('')
                    ax.set_yticklabels([])

                if log_exceedance:
                    ax.set_yscale('log')
                    ax.set_ylim(bottom=1e-3)

            # --- Common formatting ---
            # Show numeric x-tick labels on every row (per reviewer feedback);
            # only the bottom row gets the full axis-label text.
            if r == n_rows - 1:
                ax.set_xlabel(METRIC_CDF_AXIS_LABELS[metric], fontsize=FONTSIZE_MEDIUM)
            else:
                ax.set_xlabel('')

            if log_magnitude and metric == 'magnitude':
                ax.set_xscale('log')
            if metric == 'magnitude':
                ax.set_xlim(right=100 if ssi_window == 3 else 200)
            if metric == 'severity':
                ax.xaxis.set_major_locator(MultipleLocator(1))

            ax.tick_params(labelsize=FONTSIZE_SMALL)
            ax.grid(True, which='both', color='gray', alpha=0.15,
                    linewidth=0.5, linestyle='--')
            ax.set_axisbelow(True)

            ax.text(
                0.03, 0.97, f'({PANEL_LETTERS[panel_idx]})',
                transform=ax.transAxes, fontsize=FONTSIZE_MEDIUM,
                va='top', ha='left',
            )

            # Row label on right side of rightmost column
            if c == n_cols - 1:
                ax.text(
                    1.12, 0.5, DATASET_LABELS.get(dataset_id, dataset_id),
                    transform=ax.transAxes, fontsize=FONTSIZE_MEDIUM,
                    va='center', ha='left', rotation=-90,
                )

            panel_idx += 1

    # ------------------------------------------------------------------
    # Synchronise y-axis limits across CDF subplots
    # ------------------------------------------------------------------
    if plot_relative_change:
        # Baseline row (absolute) shares one y-range; delta rows share another
        # Absolute row (r=0)
        abs_ylims = [cdf_axes[0, c].get_ylim() for c in range(n_cols)]
        shared_abs_ymin = min(yl[0] for yl in abs_ylims)
        shared_abs_ymax = max(yl[1] for yl in abs_ylims)
        for c in range(n_cols):
            cdf_axes[0, c].set_ylim(shared_abs_ymin, shared_abs_ymax)

        # Delta rows (r=1..n_rows-1)
        delta_ylims = [cdf_axes[r, c].get_ylim()
                       for r in range(1, n_rows) for c in range(n_cols)]
        shared_delta_ymin = min(yl[0] for yl in delta_ylims)
        shared_delta_ymax = max(yl[1] for yl in delta_ylims)
        for r in range(1, n_rows):
            for c in range(n_cols):
                cdf_axes[r, c].set_ylim(shared_delta_ymin, shared_delta_ymax)
    else:
        # All rows use absolute exceedance — share a single y-range
        all_ylims = [cdf_axes[r, c].get_ylim()
                     for r in range(n_rows) for c in range(n_cols)]
        shared_ymin = min(yl[0] for yl in all_ylims)
        shared_ymax = max(yl[1] for yl in all_ylims)
        for r in range(n_rows):
            for c in range(n_cols):
                cdf_axes[r, c].set_ylim(shared_ymin, shared_ymax)

    # ------------------------------------------------------------------
    # Align y-axis labels across right-panel rows
    # ------------------------------------------------------------------
    label_x = -0.3
    for r in range(n_rows):
        cdf_axes[r, 0].yaxis.set_label_coords(label_x, 0.5)

    # ------------------------------------------------------------------
    # Shared legend below figure
    # ------------------------------------------------------------------
    legend_handles = [
        mlines.Line2D([], [], color=HISTORIC_COLOR, marker='^',
                      linestyle='None', markersize=8, label=HISTORIC_LABEL),
    ]
    if len(drought_1960s) > 0:
        legend_handles.append(
            mlines.Line2D([], [], color='red', marker='^',
                          linestyle='None', markersize=8, label=drought_1960s_label),
        )
    # Add all datasets (combined range + median as single entry)
    from matplotlib.legend_handler import HandlerTuple
    legend_labels = [h.get_label() for h in legend_handles]
    for dataset_id in ALL_DATASETS:
        color = DATASET_COLORS[dataset_id]
        patch_outer = mpatches.Patch(facecolor=color, alpha=ALPHA_BAND_OUTER)
        patch_inner = mpatches.Patch(facecolor=color, alpha=ALPHA_BAND_INNER)
        line = mlines.Line2D([], [], color=color,
                             linestyle='-',
                             linewidth=LINEWIDTH_MEDIUM)
        legend_handles.append((patch_outer, patch_inner, line))
        legend_labels.append(
            f'{DATASET_LABELS.get(dataset_id, dataset_id)} '
            '(99% IQR, 50% IQR, median)'
        )
    if plot_relative_change:
        no_data_handle = mpatches.Patch(facecolor='#d0d0d0', alpha=0.35,
                                         hatch='///', edgecolor='#333333', linewidth=0.8)
        legend_handles.append(no_data_handle)
        legend_labels.append('No scenario droughts beyond this value')

    # Place legend in the bottom-right cell, shifted down to add whitespace above
    ax_legend.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.25),
        ncol=1,
        frameon=False,
        fontsize=FONTSIZE_SMALL,
        columnspacing=1.0,
        handler_map={tuple: HandlerTuple(ndivide=1)},
        handleheight=1.5,
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if fname is None:
        rc_suffix = '_relative_change' if plot_relative_change else ''
        metric_suffix = f"_{'_'.join(cdf_metrics)}" if cdf_metrics != ['severity', 'magnitude'] else ''
        fname = f"{FIG_OUTPUT_DIR}/Fig5_drought_distributions_ssi{ssi_window}{metric_suffix}{rc_suffix}.png"

    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")

    return fig, (ax_hex, cdf_axes)


def main():
    """Generate the F2 manuscript figure."""
    ssi_window = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    if ssi_window not in SSI_WINDOWS:
        print(f"ERROR: Invalid SSI window: {ssi_window}. Must be one of {SSI_WINDOWS}")
        sys.exit(1)

    log_magnitude = True
    
    # # --- Severity vs Magnitude (default) ---
    # print(f"F2: Drought metric distribution (SSI-{ssi_window})")
    # plot_drought_manuscript_figure(ssi_window=ssi_window,
    #                                log_magnitude=log_magnitude,
    #                                log_exceedance=False,
    #                                plot_relative_change=False)
    # plt.close('all')

    print(f"F2: Relative change in exceedance rates (SSI-{ssi_window})")
    plot_drought_manuscript_figure(ssi_window=ssi_window,
                                   log_magnitude=log_magnitude,
                                   log_exceedance=False,
                                   plot_relative_change=True)
    plt.close('all')

    # # --- Severity vs Duration variant ---
    # print(f"F2: Drought metric distribution - duration (SSI-{ssi_window})")
    # plot_drought_manuscript_figure(ssi_window=ssi_window,
    #                                cdf_metrics=['severity', 'duration'],
    #                                hexbin_y='duration',
    #                                log_magnitude=log_magnitude,
    #                                log_exceedance=False,
    #                                plot_relative_change=False)
    # plt.close('all')

    # print(f"F2: Relative change - duration (SSI-{ssi_window})")
    # plot_drought_manuscript_figure(ssi_window=ssi_window,
    #                                cdf_metrics=['severity', 'duration'],
    #                                hexbin_y='duration',
    #                                log_magnitude=False,
    #                                log_exceedance=False,
    #                                plot_relative_change=True)
    plt.close('all')


if __name__ == "__main__":
    main()
