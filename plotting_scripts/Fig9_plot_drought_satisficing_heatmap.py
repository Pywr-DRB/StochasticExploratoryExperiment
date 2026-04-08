"""
Usage:
    python Fig9_plot_drought_satisficing_heatmap.py [ssi_window]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import matplotlib.patheffects as pe
import warnings
warnings.filterwarnings("ignore")

from methods.config import ROOT_DIR, FIG_DIR
from methods.load import load_event_metrics
from methods.plotting.styles import (
    DATASET_LABELS, FONTSIZE_SMALL, FONTSIZE_MEDIUM,
    FONTSIZE_LABEL, FONTSIZE_TITLE, DPI_HIGH,
    apply_publication_style, label_panel,
)
from methods.plotting.heatmap import (
    MAG_MIN, make_shared_edges_logmag, compute_min_storage_grid,
    compute_emergency_grid, SATISFICING_THRESHOLD,
    GRID_N_BINS, GRID_TARGET_SEV_BIN, GRID_TARGET_MAG_BIN,
)

WORST_STORAGE_THRESH = 15.0  # local threshold for triangle markers

# -- configuration -----------------------------------------------------------
FIG_OUTPUT_DIR = f"{FIG_DIR}/Fig9_drought_satisficing"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

SSI_WINDOW_DEFAULT = 3
DATASETS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']
PANEL_LETTERS = list('abc')


def plot_combined_heatmap(all_data, ssi_window, n_bins=GRID_N_BINS, min_count=5):
    """Create the 3x1 combined heatmap figure.

    Parameters
    ----------
    all_data : dict
        ``{dataset_id: DataFrame}`` of event metrics.
    ssi_window : int
        SSI window used (for filename).
    n_bins : int
        Number of bins per axis.
    min_count : int
        Minimum number of events required to colour a bin (default 5).
        Bins below this threshold are shown as grey (NaN).
    """
    apply_publication_style()

    sev_edges, mag_edges, sev_centers, mag_centers = make_shared_edges_logmag(
        all_data, DATASETS, n_bins=n_bins)

    # -- colour map & norm (fraction avoiding emergency) --------------------
    cmap_frac = plt.cm.plasma_r
    norm_frac = mcolors.Normalize(vmin=0.3, vmax=1.0)

    # -- figure layout: 3 rows x 1 column ----------------------------------
    fig = plt.figure(figsize=(6.0, 13.5))
    gs = gridspec.GridSpec(
        3, 1,
        hspace=0.12,
        left=0.14, right=0.92, bottom=0.06, top=0.90,
    )

    axes = []

    for row_idx, did in enumerate(DATASETS):
        df = all_data[did]
        label = DATASET_LABELS.get(did, did)

        ax = fig.add_subplot(gs[row_idx, 0])

        # Compute both grids
        frac_grid, _ = compute_emergency_grid(df, sev_edges, mag_edges, min_count=min_count)
        min_grid, _ = compute_min_storage_grid(df, sev_edges, mag_edges, min_count=min_count)

        # Background: fraction avoiding emergency
        ax.pcolormesh(
            sev_edges, mag_edges,
            np.ma.masked_invalid(frac_grid.T),
            cmap=cmap_frac, norm=norm_frac, rasterized=True,
        )
        ax.set_facecolor('#f0f0f0')

        # Triangle markers where worst-case storage < threshold
        for i, sc in enumerate(sev_centers):
            for j, mc in enumerate(mag_centers):
                if np.isnan(min_grid[i, j]):
                    continue
                if min_grid[i, j] < WORST_STORAGE_THRESH:
                    ax.scatter(sc, mc, s=55, marker='v', color='black',
                               linewidths=0.8, zorder=5)

        # Highlight the focal grid cell (shared with Fig10)
        cell_x = sev_edges[GRID_TARGET_SEV_BIN]
        cell_y = mag_edges[GRID_TARGET_MAG_BIN]
        cell_w = sev_edges[GRID_TARGET_SEV_BIN + 1] - cell_x
        cell_h = mag_edges[GRID_TARGET_MAG_BIN + 1] - cell_y
        highlight = Rectangle(
            (cell_x, cell_y), cell_w, cell_h,
            linewidth=2.5, edgecolor='white', facecolor='none',
            zorder=6,
        )
        ax.add_patch(highlight)

        ax.set_xlim(sev_edges[0], sev_edges[-1])
        ax.set_ylim(mag_edges[0], mag_edges[-1])
        ax.set_yscale('log')

        # Panel label
        letter = PANEL_LETTERS[row_idx]
        label_panel(ax, letter, label=label, fontsize=FONTSIZE_LABEL)

        # Axis labels
        ax.set_ylabel('Drought Magnitude\n(cumulative SSI deficit)',
                       fontsize=FONTSIZE_LABEL)
        if row_idx == 2:
            ax.set_xlabel('Drought Severity\n(peak SSI deviation)',
                          fontsize=FONTSIZE_LABEL)
        else:
            ax.set_xticklabels([])

        ax.tick_params(labelsize=FONTSIZE_SMALL)
        axes.append(ax)

    # -- colorbar at top, spanning full subplot width -----------------------
    fig.canvas.draw()

    cbar_h = 0.012
    cbar_top = 0.92

    bb = axes[0].get_position()
    cbar_ax = fig.add_axes([bb.x0, cbar_top, bb.width, cbar_h])
    cb = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_frac, norm=norm_frac),
        cax=cbar_ax, orientation='horizontal',
    )
    cbar_ax.xaxis.set_ticks_position('top')
    cbar_ax.xaxis.set_label_position('top')
    cb.set_label('Fraction of Drought Events\nAvoiding FFMP Drought Emergency',
                 fontsize=FONTSIZE_LABEL)
    cb.ax.tick_params(labelsize=FONTSIZE_SMALL)

    # -- legend at bottom ---------------------------------------------------
    h_tri = Line2D([0], [0], marker='v', color='black', linestyle='none',
                   markersize=7,
                   label=f'Worst-case storage < {WORST_STORAGE_THRESH:.0f}%')
    h_nodata = Patch(facecolor='#f0f0f0', edgecolor='#cccccc', linewidth=0.8,
                     label='No drought events in this range')
    h_cell = Patch(facecolor='none', edgecolor='white', linewidth=2.5,
                   label='Focal cell (Fig. 10)')
    fig.legend(
        handles=[h_tri, h_nodata, h_cell], loc='lower center', ncol=3,
        fontsize=FONTSIZE_SMALL, frameon=True, framealpha=0.9,
        edgecolor='none', shadow=False,
        bbox_to_anchor=(0.53, -0.01),
    )

    # -- save ---------------------------------------------------------------
    fname = (f"{FIG_OUTPUT_DIR}/Fig9_satisficing_heatmap_ssi{ssi_window}"
             f"_focal_sev{GRID_TARGET_SEV_BIN}_mag{GRID_TARGET_MAG_BIN}.png")
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close(fig)


# -- main -------------------------------------------------------------------

def main():
    """Usage: python Fig9_plot_drought_satisficing_heatmap.py [ssi_window] [n_bins] [min_count]"""
    args = [a for a in sys.argv[1:] if not a.startswith('--')]

    ssi_window = int(args[0]) if len(args) > 0 else SSI_WINDOW_DEFAULT
    n_bins     = int(args[1]) if len(args) > 1 else GRID_N_BINS
    min_count  = int(args[2]) if len(args) > 2 else 1

    print(f"Fig9: Combined Drought Satisficing Heatmap (SSI-{ssi_window}, "
          f"n_bins={n_bins}, min_count={min_count})")

    all_data = {}
    for did in DATASETS:
        df = load_event_metrics(did, ssi_window)
        all_data[did] = df
        print(f"  {DATASET_LABELS.get(did, did)}: {len(df)} events")

    plot_combined_heatmap(all_data, ssi_window,
                          n_bins=n_bins,
                          min_count=min_count)
    print("Done.")


if __name__ == '__main__':
    main()
