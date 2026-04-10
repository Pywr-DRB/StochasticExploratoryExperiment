"""
Fig9alt: Drought Satisficing Heatmaps with Multi-Metric Focal Region

Six-panel figure (3 rows x 2 columns).
  Rows    = climate scenarios (Baseline, Mixed Future, Wet Future)
  Col 1   = Joint exceedance rate (events/year) per (severity, magnitude) bin
  Col 2   = Fraction of events avoiding Drought Emergency

A focal region is identified via multi-metric criteria applied across all
datasets and highlighted with white rectangles on both columns.

Usage:
    python Fig9alt_plot_drought_satisficing_heatmap_2col.py [ssi_window]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
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
    compute_exceedance_rate_grid, GRID_N_BINS,
    WORST_STORAGE_THRESH, SATISFICING_THRESHOLD,
)

# -- focal-region thresholds --------------------------------------------------
FOCAL_FRAC_THRESH = 0.95       # fraction avoiding emergency must be < this (ALL datasets)
FOCAL_RATE_THRESH = 10e-4      # exceedance rate must exceed this (ALL datasets)
# "highly consequential" = worst-case storage < WORST_STORAGE_THRESH in >= 1 dataset

# -- configuration -----------------------------------------------------------
FIG_OUTPUT_DIR = f"{FIG_DIR}/Fig9_drought_satisficing"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

SSI_WINDOW_DEFAULT = 3
DATASETS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']
PANEL_LETTERS = list('abcdef')


def _identify_focal_region(rate_grids, frac_grids, min_grids, datasets):
    """Identify grid cells meeting the multi-metric focal-region criteria.

    Criteria
    --------
    1. Fraction avoiding emergency < FOCAL_FRAC_THRESH in ALL datasets
    2. Exceedance rate > FOCAL_RATE_THRESH in ALL datasets
    3. Worst-case storage < WORST_STORAGE_THRESH in at least 1 dataset

    Returns
    -------
    focal_cells : set of (i, j) tuples
        Grid indices of qualifying cells.
    """
    ns, nm = rate_grids[datasets[0]].shape
    focal_cells = set()

    for i in range(ns):
        for j in range(nm):
            # Criterion 2: rate > threshold in ALL datasets
            if not all(
                not np.isnan(rate_grids[d][i, j]) and
                rate_grids[d][i, j] >= FOCAL_RATE_THRESH
                for d in datasets
            ):
                continue
            # Criterion 1: frac < threshold in ALL datasets
            if not all(
                not np.isnan(frac_grids[d][i, j]) and
                frac_grids[d][i, j] < FOCAL_FRAC_THRESH
                for d in datasets
            ):
                continue
            # Criterion 3: min storage < threshold in >= 1 dataset
            if not any(
                not np.isnan(min_grids[d][i, j]) and
                min_grids[d][i, j] < WORST_STORAGE_THRESH
                for d in datasets
            ):
                continue
            focal_cells.add((i, j))

    return focal_cells


def _add_focal_region(ax, sev_edges, mag_edges, focal_cells):
    """Draw white rectangles around all cells in the focal region."""
    for i, j in focal_cells:
        x = sev_edges[i]
        y = mag_edges[j]
        w = sev_edges[i + 1] - x
        h = mag_edges[j + 1] - y
        rect = Rectangle((x, y), w, h, linewidth=2.0,
                          edgecolor='white', facecolor='none', zorder=6)
        ax.add_patch(rect)


def plot_satisficing_heatmaps(all_data, ssi_window, min_count=1):
    """Create the 3x2 publication figure with focal-region highlighting."""
    apply_publication_style()

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
    focal_cells = _identify_focal_region(rate_grids, frac_grids, min_grids, DATASETS)
    print(f"  Focal region: {len(focal_cells)} cells — {sorted(focal_cells)}")

    # -- colour maps & norms ------------------------------------------------
    cmap_rate = plt.cm.YlOrRd
    all_rates = np.concatenate([rg[~np.isnan(rg)] for rg in rate_grids.values()])
    if len(all_rates) > 0:
        rate_vmin = max(all_rates.min(), 1e-4)
        rate_vmax = all_rates.max()
    else:
        rate_vmin, rate_vmax = 1e-4, 1.0
    norm_rate = mcolors.LogNorm(vmin=rate_vmin, vmax=rate_vmax)

    cmap_frac = plt.cm.plasma_r
    norm_frac = mcolors.Normalize(vmin=0.3, vmax=1.0)

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

        # ── Col 0: exceedance rate ───────────────────────────────────
        ax_rate = fig.add_subplot(gs[row_idx, 0])
        rate_grid = rate_grids[did]

        ax_rate.pcolormesh(
            sev_edges, mag_edges,
            np.ma.masked_invalid(rate_grid.T),
            cmap=cmap_rate, norm=norm_rate, rasterized=True,
        )
        ax_rate.set_facecolor('#f0f0f0')
        _add_focal_region(ax_rate, sev_edges, mag_edges, focal_cells)

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
        frac_grid = frac_grids[did]

        ax_frac.pcolormesh(
            sev_edges, mag_edges,
            np.ma.masked_invalid(frac_grid.T),
            cmap=cmap_frac, norm=norm_frac, rasterized=True,
        )
        ax_frac.set_facecolor('#f0f0f0')
        _add_focal_region(ax_frac, sev_edges, mag_edges, focal_cells)

        # Triangle markers for worst-case storage
        min_grid = min_grids[did]
        for i, sc in enumerate(sev_centers):
            for j, mc in enumerate(mag_centers):
                if np.isnan(min_grid[i, j]):
                    continue
                if min_grid[i, j] < WORST_STORAGE_THRESH:
                    ax_frac.scatter(sc, mc, s=40, marker='v', color='black',
                                     linewidths=0.6, zorder=5)

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

    # -- colorbars at top ---------------------------------------------------
    fig.canvas.draw()

    cbar_h = 0.012
    cbar_top = 0.92

    bb_left = axes_rate[0].get_position()
    cbar_ax1 = fig.add_axes([bb_left.x0, cbar_top, bb_left.width, cbar_h])
    cb1 = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_rate, norm=norm_rate),
        cax=cbar_ax1, orientation='horizontal',
    )
    cbar_ax1.xaxis.set_ticks_position('top')
    cbar_ax1.xaxis.set_label_position('top')
    cb1.set_label('Exceedance Rate (events yr$^{-1}$)',
                  fontsize=FONTSIZE_LABEL)
    cb1.ax.tick_params(labelsize=FONTSIZE_SMALL)

    bb_right = axes_frac[0].get_position()
    cbar_ax2 = fig.add_axes([bb_right.x0, cbar_top, bb_right.width, cbar_h])
    cb2 = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_frac, norm=norm_frac),
        cax=cbar_ax2, orientation='horizontal',
    )
    cbar_ax2.xaxis.set_ticks_position('top')
    cbar_ax2.xaxis.set_label_position('top')
    cb2.set_label('Fraction of Drought Events\nAvoiding FFMP Drought Emergency',
                  fontsize=FONTSIZE_LABEL)
    cb2.ax.tick_params(labelsize=FONTSIZE_SMALL)

    # -- legend at bottom ---------------------------------------------------
    h_tri = Line2D([0], [0], marker='v', color='black', linestyle='none',
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
    fname = (f"{FIG_OUTPUT_DIR}/Fig9_satisficing_heatmap_ssi{ssi_window}"
             f"_rate{FOCAL_RATE_THRESH:.0e}_frac{FOCAL_FRAC_THRESH:.2f}"
             f"_sto{WORST_STORAGE_THRESH:.0f}.png")
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close(fig)


# -- main -------------------------------------------------------------------

def main():
    ssi_window = int(sys.argv[1]) if len(sys.argv) > 1 else SSI_WINDOW_DEFAULT
    print(f"Fig9alt: Drought Satisficing Heatmaps (SSI-{ssi_window})")

    all_data = {}
    for did in DATASETS:
        df = load_event_metrics(did, ssi_window)
        all_data[did] = df
        print(f"  {DATASET_LABELS.get(did, did)}: {len(df)} events")

    plot_satisficing_heatmaps(all_data, ssi_window)
    print("Done.")


if __name__ == '__main__':
    main()
