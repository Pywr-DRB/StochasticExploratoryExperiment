"""
Fig9alt: Drought Satisficing Heatmaps with Multi-Metric Focal Region

Six-panel figure (3 rows x 2 columns).
  Rows    = climate scenarios (Baseline, Mixed Future, Wet Future)
  Col 1   = Joint exceedance rate (events/year) per (severity, magnitude) bin
  Col 2   = Fraction of events avoiding Drought Emergency

When SHOW_CHANGE = True (default), the baseline row shows absolute values
while the climate-scenario rows show the *change* relative to baseline,
plotted with divergent colourmaps centred on zero.

Usage:
    python Fig9_plot_drought_satisficing_heatmap_2col.py [ssi_window]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings("ignore")

from methods.config import FIG_DIR, N_YEARS
from methods.load import load_event_metrics
from methods.plotting.styles import (
    DATASET_LABELS, FONTSIZE_SMALL, FONTSIZE_MEDIUM,
    FONTSIZE_LABEL, DPI_HIGH,
    apply_publication_style, label_panel,
)
from methods.plotting.heatmap import (
    make_shared_edges_logmag, compute_min_storage_grid, compute_emergency_grid,
    compute_exceedance_rate_grid, identify_focal_region, draw_focal_boundary,
    GRID_N_BINS, WORST_STORAGE_THRESH, SATISFICING_THRESHOLD,
    FOCAL_FRAC_THRESH, FOCAL_RATE_THRESH,
)

# -- configuration -----------------------------------------------------------
FIG_OUTPUT_DIR = f"{FIG_DIR}/Fig9_drought_satisficing"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

SSI_WINDOW_DEFAULT = 3
DATASETS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']
PANEL_LETTERS = list('abcdef')

# Show climate rows as change relative to baseline?
SHOW_CHANGE = True


def plot_satisficing_heatmaps(all_data, ssi_window, min_count=1,
                              show_change=SHOW_CHANGE):
    """Create the 3x2 publication figure with focal-region highlighting."""
    apply_publication_style()

    baseline_id = DATASETS[0]

    sev_edges, mag_edges, sev_centers, mag_centers = make_shared_edges_logmag(
        all_data, DATASETS, n_bins=GRID_N_BINS)

    # -- compute all grids per dataset --------------------------------------
    rate_grids, frac_grids, min_grids = {}, {}, {}
    for did in DATASETS:
        rg, _ = compute_exceedance_rate_grid(
            all_data[did], sev_edges, mag_edges, N_YEARS, min_count=min_count)
        rate_grids[did] = rg
        fg, _ = compute_emergency_grid(all_data[did], sev_edges, mag_edges, min_count=min_count)
        frac_grids[did] = fg
        mg, _ = compute_min_storage_grid(all_data[did], sev_edges, mag_edges, min_count=min_count)
        min_grids[did] = mg

    # -- identify focal region ----------------------------------------------
    focal_cells = identify_focal_region(rate_grids, frac_grids, min_grids, DATASETS)
    print(f"  Focal region: {len(focal_cells)} cells — {sorted(focal_cells)}")

    # -- colour maps & norms: absolute (row 0, always) ----------------------
    cmap_rate_abs = plt.cm.YlOrRd
    all_rates = np.concatenate([rg[~np.isnan(rg)] for rg in rate_grids.values()])
    if len(all_rates) > 0:
        rate_vmin = max(all_rates.min(), 1e-4)
        rate_vmax = all_rates.max()
    else:
        rate_vmin, rate_vmax = 1e-4, 1.0
    norm_rate_abs = mcolors.LogNorm(vmin=rate_vmin, vmax=rate_vmax)

    cmap_frac_abs = plt.cm.plasma_r
    norm_frac_abs = mcolors.Normalize(vmin=0.3, vmax=1.0)

    # -- colour maps & norms: divergent change (rows 1-2 when show_change) --
    if show_change:
        # Compute change grids
        rate_diff_grids, frac_diff_grids = {}, {}
        for did in DATASETS[1:]:
            rate_diff_grids[did] = rate_grids[did] - rate_grids[baseline_id]
            frac_diff_grids[did] = frac_grids[did] - frac_grids[baseline_id]

        # Symmetric limits for rate change
        all_rate_diffs = np.concatenate(
            [rd[~np.isnan(rd)] for rd in rate_diff_grids.values()])
        if len(all_rate_diffs) > 0:
            rate_diff_max = np.max(np.abs(all_rate_diffs))
        else:
            rate_diff_max = 0.01
        rate_diff_max = max(rate_diff_max, 1e-4)

        cmap_rate_div = plt.cm.RdBu_r
        norm_rate_div = mcolors.TwoSlopeNorm(vmin=-rate_diff_max, vcenter=0,
                                              vmax=rate_diff_max)

        # Symmetric limits for fraction change
        all_frac_diffs = np.concatenate(
            [fd[~np.isnan(fd)] for fd in frac_diff_grids.values()])
        if len(all_frac_diffs) > 0:
            frac_diff_max = np.max(np.abs(all_frac_diffs))
        else:
            frac_diff_max = 0.1
        frac_diff_max = max(frac_diff_max, 0.01)

        cmap_frac_div = plt.cm.RdBu
        norm_frac_div = mcolors.TwoSlopeNorm(vmin=-frac_diff_max, vcenter=0,
                                               vmax=frac_diff_max)

    # -- figure layout: 3 rows x 2 columns ---------------------------------
    fig = plt.figure(figsize=(10.0, 13.5))
    gs = gridspec.GridSpec(
        3, 2,
        width_ratios=[1, 1],
        hspace=0.12, wspace=0.12,
        left=0.10, right=0.95, bottom=0.06, top=0.90,
    )

    axes_rate = []
    axes_frac = []

    for row_idx, did in enumerate(DATASETS):
        label = DATASET_LABELS.get(did, did)
        is_change_row = show_change and row_idx > 0

        # ── Col 0: exceedance rate ───────────────────────────────────
        ax_rate = fig.add_subplot(gs[row_idx, 0])

        if is_change_row:
            plot_grid = rate_diff_grids[did]
            cmap_r, norm_r = cmap_rate_div, norm_rate_div
        else:
            plot_grid = rate_grids[did]
            cmap_r, norm_r = cmap_rate_abs, norm_rate_abs

        ax_rate.pcolormesh(
            sev_edges, mag_edges,
            np.ma.masked_invalid(plot_grid.T),
            cmap=cmap_r, norm=norm_r, rasterized=True,
        )
        ax_rate.set_facecolor('#f0f0f0')
        draw_focal_boundary(ax_rate, sev_edges, mag_edges, focal_cells)

        ax_rate.set_xlim(sev_edges[0], sev_edges[-1])
        ax_rate.set_yscale('log')
        ax_rate.set_ylim(mag_edges[0], mag_edges[-1])

        letter = PANEL_LETTERS[row_idx * 2]
        label_panel(ax_rate, letter, label=label, fontsize=FONTSIZE_LABEL)

        ax_rate.set_ylabel('Drought Magnitude\n(cumulative SSI deficit)',
                          fontsize=FONTSIZE_LABEL)
        if row_idx == 2:
            ax_rate.set_xlabel('Drought Severity\n(peak SSI deviation)',
                              fontsize=FONTSIZE_LABEL)
        else:
            ax_rate.set_xticklabels([])

        ax_rate.tick_params(labelsize=FONTSIZE_SMALL)
        axes_rate.append(ax_rate)

        # ── Col 1: fraction avoiding Drought Emergency ───────────────
        ax_frac = fig.add_subplot(gs[row_idx, 1])

        if is_change_row:
            plot_grid_f = frac_diff_grids[did]
            cmap_f, norm_f = cmap_frac_div, norm_frac_div
        else:
            plot_grid_f = frac_grids[did]
            cmap_f, norm_f = cmap_frac_abs, norm_frac_abs

        ax_frac.pcolormesh(
            sev_edges, mag_edges,
            np.ma.masked_invalid(plot_grid_f.T),
            cmap=cmap_f, norm=norm_f, rasterized=True,
        )
        ax_frac.set_facecolor('#f0f0f0')
        draw_focal_boundary(ax_frac, sev_edges, mag_edges, focal_cells)

        # "x" markers for worst-case storage
        min_grid = min_grids[did]
        for i, sc in enumerate(sev_centers):
            for j, mc in enumerate(mag_centers):
                if np.isnan(min_grid[i, j]):
                    continue
                if min_grid[i, j] < WORST_STORAGE_THRESH:
                    ax_frac.scatter(sc, mc, s=50, marker='x', color='black',
                                     linewidths=1.2, zorder=5)

        ax_frac.set_xlim(sev_edges[0], sev_edges[-1])
        ax_frac.set_yscale('log')
        ax_frac.set_ylim(mag_edges[0], mag_edges[-1])

        letter = PANEL_LETTERS[row_idx * 2 + 1]
        label_panel(ax_frac, letter, label=label, fontsize=FONTSIZE_LABEL)

        ax_frac.set_yticklabels([])
        if row_idx == 2:
            ax_frac.set_xlabel('Drought Severity\n(peak SSI deviation)',
                               fontsize=FONTSIZE_LABEL)
        else:
            ax_frac.set_xticklabels([])

        ax_frac.tick_params(labelsize=FONTSIZE_SMALL)
        axes_frac.append(ax_frac)

    # -- colorbars at top (for baseline / absolute row) ---------------------
    fig.canvas.draw()

    cbar_h = 0.012
    cbar_top = 0.92

    bb_left = axes_rate[0].get_position()
    cbar_ax_rate_abs = fig.add_axes([bb_left.x0, cbar_top, bb_left.width, cbar_h])
    cb_rate_abs = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_rate_abs, norm=norm_rate_abs),
        cax=cbar_ax_rate_abs, orientation='horizontal',
    )
    cbar_ax_rate_abs.xaxis.set_ticks_position('top')
    cbar_ax_rate_abs.xaxis.set_label_position('top')
    cb_rate_abs.set_label('Exceedance Rate (events yr$^{-1}$)',
                          fontsize=FONTSIZE_LABEL)
    cb_rate_abs.ax.tick_params(labelsize=FONTSIZE_SMALL)

    bb_right = axes_frac[0].get_position()
    cbar_ax_frac_abs = fig.add_axes([bb_right.x0, cbar_top, bb_right.width, cbar_h])
    cb_frac_abs = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_frac_abs, norm=norm_frac_abs),
        cax=cbar_ax_frac_abs, orientation='horizontal',
    )
    cbar_ax_frac_abs.xaxis.set_ticks_position('top')
    cbar_ax_frac_abs.xaxis.set_label_position('top')
    cb_frac_abs.set_label('Fraction Avoiding\nDrought Emergency',
                          fontsize=FONTSIZE_LABEL)
    cb_frac_abs.ax.tick_params(labelsize=FONTSIZE_SMALL)

    # -- divergent colorbars between rows 0 and 1 (when show_change) --------
    if show_change:
        # Position between baseline row bottom and first climate row top
        pos_baseline_rate = axes_rate[0].get_position()
        pos_climate_rate = axes_rate[1].get_position()
        cbar_div_y = pos_climate_rate.y0 + pos_climate_rate.height + \
            (pos_baseline_rate.y0 - pos_climate_rate.y0 - pos_climate_rate.height) * 0.35

        cbar_ax_rate_div = fig.add_axes([
            pos_climate_rate.x0, cbar_div_y,
            pos_climate_rate.width, cbar_h,
        ])
        cb_rate_div = fig.colorbar(
            plt.cm.ScalarMappable(cmap=cmap_rate_div, norm=norm_rate_div),
            cax=cbar_ax_rate_div, orientation='horizontal',
        )
        cb_rate_div.set_label('$\\Delta$ Rate (events yr$^{-1}$)',
                              fontsize=FONTSIZE_SMALL)
        cb_rate_div.ax.tick_params(labelsize=FONTSIZE_SMALL - 1)

        pos_climate_frac = axes_frac[1].get_position()
        cbar_ax_frac_div = fig.add_axes([
            pos_climate_frac.x0, cbar_div_y,
            pos_climate_frac.width, cbar_h,
        ])
        cb_frac_div = fig.colorbar(
            plt.cm.ScalarMappable(cmap=cmap_frac_div, norm=norm_frac_div),
            cax=cbar_ax_frac_div, orientation='horizontal',
        )
        cb_frac_div.set_label('$\\Delta$ Fraction Avoiding DE',
                              fontsize=FONTSIZE_SMALL)
        cb_frac_div.ax.tick_params(labelsize=FONTSIZE_SMALL - 1)

    # -- legend at bottom ---------------------------------------------------
    h_tri = Line2D([0], [0], marker='x', color='black', linestyle='none', markeredgewidth=1.2,
                   markersize=7,
                   label=f'Worst-case storage < {WORST_STORAGE_THRESH:.0f}%')
    h_nodata = Patch(facecolor='#f0f0f0', edgecolor='#cccccc', linewidth=0.8,
                     label='No drought events in this range')
    h_focal = Patch(facecolor='none', edgecolor='white', linewidth=2.0,
                    label=(f'Focal region (rate>{FOCAL_RATE_THRESH:.0e} all, '
                           f'frac<{FOCAL_FRAC_THRESH:.0%} all, '
                           f'min sto<{WORST_STORAGE_THRESH:.0f}% any)'))
    fig.legend(
        handles=[h_tri, h_nodata, h_focal], loc='lower center', ncol=3,
        fontsize=FONTSIZE_SMALL, frameon=True, framealpha=0.9,
        edgecolor='none', shadow=False,
        bbox_to_anchor=(0.52, -0.01),
    )

    # -- save ---------------------------------------------------------------
    change_tag = '_change' if show_change else ''
    fname = (f"{FIG_OUTPUT_DIR}/Fig9_satisficing_heatmap_ssi{ssi_window}"
             f"_rate{FOCAL_RATE_THRESH:.0e}_frac{FOCAL_FRAC_THRESH:.2f}"
             f"_sto{WORST_STORAGE_THRESH:.0f}{change_tag}.png")
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close(fig)


# -- main -------------------------------------------------------------------

def main():
    ssi_window = int(sys.argv[1]) if len(sys.argv) > 1 else SSI_WINDOW_DEFAULT
    print(f"Fig9: Drought Satisficing Heatmaps (SSI-{ssi_window},"
          f" show_change={SHOW_CHANGE})")

    all_data = {}
    for did in DATASETS:
        df = load_event_metrics(did, ssi_window)
        all_data[did] = df
        print(f"  {DATASET_LABELS.get(did, did)}: {len(df)} events")

    plot_satisficing_heatmaps(all_data, ssi_window, show_change=SHOW_CHANGE)
    print("Done.")


if __name__ == '__main__':
    main()
