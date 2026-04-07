"""
F5: Drought Satisficing Heatmaps

Six-panel figure (3 rows × 2 columns).
  Rows    = climate scenarios (Baseline, Mixed Future, Wet Future)
  Col 1   = Worst-case minimum NYC storage (%) during drought events
  Col 2   = Fraction of events avoiding Drought Emergency

Each cell in the severity × magnitude grid is coloured by its metric value.
Cells that breach a threshold are marked with an ×.

Usage:
    python F5_plot_drought_satisficing_heatmap.py [ssi_window]
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

from methods.config import ROOT_DIR, FIG_DIR
from methods.load import load_event_metrics
from methods.plotting.styles import (
    DATASET_LABELS, FONTSIZE_SMALL, FONTSIZE_MEDIUM,
    FONTSIZE_LABEL, FONTSIZE_TITLE, DPI_HIGH,
    apply_publication_style, label_panel,
)
from methods.plotting.heatmap import (
    make_shared_edges, compute_min_storage_grid, compute_emergency_grid,
    WORST_STORAGE_THRESH, SATISFICING_THRESHOLD,
)

# ── configuration ────────────────────────────────────────────────────
FIG_OUTPUT_DIR = f"{FIG_DIR}/Fig9_drought_satisficing"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

SSI_WINDOW_DEFAULT = 3
DATASETS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']
PANEL_LETTERS = list('abcdef')


def plot_satisficing_heatmaps(all_data, ssi_window):
    """Create the 3×2+colorbar publication figure."""
    apply_publication_style()

    sev_edges, mag_edges, sev_centers, mag_centers = make_shared_edges(
        all_data, DATASETS)

    # ── colour maps & norms ──────────────────────────────────────────
    cmap_sto = plt.cm.plasma_r
    norm_sto = mcolors.Normalize(vmin=0, vmax=60)

    cmap_frac = plt.cm.plasma_r
    norm_frac = mcolors.Normalize(vmin=0.3, vmax=1.0)

    # ── figure layout: 3 rows × 2 heatmaps, colorbars at top ──────────
    fig = plt.figure(figsize=(10.0, 13.5))
    gs = gridspec.GridSpec(
        3, 2,
        width_ratios=[1, 1],
        hspace=0.12, wspace=0.12,
        left=0.10, right=0.95, bottom=0.06, top=0.90,
    )

    axes_sto = []
    axes_frac = []

    for row_idx, did in enumerate(DATASETS):
        df = all_data[did]
        label = DATASET_LABELS.get(did, did)

        # ── Col 0: worst-case min storage ────────────────────────────
        ax_sto = fig.add_subplot(gs[row_idx, 0])
        min_grid, _ = compute_min_storage_grid(df, sev_edges, mag_edges)

        ax_sto.pcolormesh(
            sev_edges, mag_edges,
            np.ma.masked_invalid(min_grid.T),
            cmap=cmap_sto, norm=norm_sto, rasterized=True,
        )
        ax_sto.set_facecolor('#f0f0f0')

        # Mark bins where worst-case storage < threshold (▽)
        for i, sc in enumerate(sev_centers):
            for j, mc in enumerate(mag_centers):
                if np.isnan(min_grid[i, j]):
                    continue
                if min_grid[i, j] < WORST_STORAGE_THRESH:
                    ax_sto.scatter(sc, mc, s=55, marker='v', color='black',
                                   linewidths=0.8, zorder=5)

        ax_sto.set_xlim(sev_edges[0], sev_edges[-1])
        ax_sto.set_ylim(mag_edges[0], mag_edges[-1])

        # Panel label
        letter = PANEL_LETTERS[row_idx * 2]
        label_panel(ax_sto, letter, label=label, fontsize=FONTSIZE_LABEL)

        # Axis labels
        ax_sto.set_ylabel('Drought Magnitude\n(cumulative SSI deficit)',
                          fontsize=FONTSIZE_LABEL)
        if row_idx == 2:
            ax_sto.set_xlabel('Drought Severity\n(peak SSI deviation)',
                              fontsize=FONTSIZE_LABEL)
        else:
            ax_sto.set_xticklabels([])

        ax_sto.tick_params(labelsize=FONTSIZE_SMALL)
        axes_sto.append(ax_sto)

        # ── Col 1: fraction avoiding Drought Emergency ───────────────
        ax_frac = fig.add_subplot(gs[row_idx, 1])
        frac_grid, _ = compute_emergency_grid(df, sev_edges, mag_edges)

        ax_frac.pcolormesh(
            sev_edges, mag_edges,
            np.ma.masked_invalid(frac_grid.T),
            cmap=cmap_frac, norm=norm_frac, rasterized=True,
        )
        ax_frac.set_facecolor('#f0f0f0')

        # Mark bins below satisficing threshold
        for i, sc in enumerate(sev_centers):
            for j, mc in enumerate(mag_centers):
                if np.isnan(frac_grid[i, j]):
                    continue
                if frac_grid[i, j] < SATISFICING_THRESHOLD:
                    ax_frac.scatter(sc, mc, s=50, marker='x', color='black',
                                     linewidths=1.2, zorder=5)

        ax_frac.set_xlim(sev_edges[0], sev_edges[-1])
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

    # ── colorbars at top, spanning full subplot width ───────────────────
    fig.canvas.draw()

    cbar_h = 0.012
    cbar_top = 0.92          # close above the subplot grid (top=0.90)

    # Left column colorbar — full width of left subplot
    bb_left = axes_sto[0].get_position()
    cbar_ax1 = fig.add_axes([bb_left.x0, cbar_top,
                              bb_left.width, cbar_h])
    cb1 = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_sto, norm=norm_sto),
        cax=cbar_ax1, orientation='horizontal',
    )
    cbar_ax1.xaxis.set_ticks_position('top')
    cbar_ax1.xaxis.set_label_position('top')
    cb1.set_label('Worst-Case Minimum\nNYC Combined Reservoir Storage (%)',
                  fontsize=FONTSIZE_LABEL)
    cb1.ax.tick_params(labelsize=FONTSIZE_SMALL)

    # Right column colorbar — full width of right subplot
    bb_right = axes_frac[0].get_position()
    cbar_ax2 = fig.add_axes([bb_right.x0, cbar_top,
                              bb_right.width, cbar_h])
    cb2 = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_frac, norm=norm_frac),
        cax=cbar_ax2, orientation='horizontal',
    )
    cbar_ax2.xaxis.set_ticks_position('top')
    cbar_ax2.xaxis.set_label_position('top')
    cb2.set_label('Fraction of Drought Events\nAvoiding FFMP Drought Emergency',
                  fontsize=FONTSIZE_LABEL)
    cb2.ax.tick_params(labelsize=FONTSIZE_SMALL)

    # ── legend at bottom ──────────────────────────────────────────────
    h_sto = Line2D([0], [0], marker='v', color='black', linestyle='none',
                   markersize=7,
                   label=f'Worst-case < {WORST_STORAGE_THRESH:.0f}% storage')
    h_frac = Line2D([0], [0], marker='x', color='black', linestyle='none',
                    markeredgewidth=1.2, markersize=7,
                    label=f'< {SATISFICING_THRESHOLD:.0%} avoid Emergency')
    h_nodata = Patch(facecolor='#f0f0f0', edgecolor='#cccccc', linewidth=0.8,
                     label='No drought events in this range')
    fig.legend(
        handles=[h_sto, h_frac, h_nodata], loc='lower center', ncol=3,
        fontsize=FONTSIZE_SMALL, frameon=True, framealpha=0.9,
        edgecolor='none', shadow=False,
        bbox_to_anchor=(0.52, -0.01),
    )

    # ── save ─────────────────────────────────────────────────────────
    fname = f"{FIG_OUTPUT_DIR}/Fig9_satisficing_heatmap_ssi{ssi_window}.png"
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────

def main():
    ssi_window = int(sys.argv[1]) if len(sys.argv) > 1 else SSI_WINDOW_DEFAULT
    print(f"F5: Drought Satisficing Heatmaps (SSI-{ssi_window})")

    all_data = {}
    for did in DATASETS:
        df = load_event_metrics(did, ssi_window)
        all_data[did] = df
        print(f"  {DATASET_LABELS.get(did, did)}: {len(df)} events")

    plot_satisficing_heatmaps(all_data, ssi_window)
    print("Done.")


if __name__ == '__main__':
    main()
