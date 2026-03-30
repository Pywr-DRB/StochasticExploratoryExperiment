"""
F5alt: Combined Drought Satisficing Heatmap

Three-panel figure (3 rows x 1 column).
  Rows = climate scenarios (Baseline, Mixed Future, Wet Future)

Each cell background colour = fraction of events avoiding FFMP Drought Emergency.
Each cell text = worst-case minimum NYC combined reservoir storage (integer %).

Usage:
    python F5alt_plot_drought_satisficing_heatmap.py [ssi_window]
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
from collections import defaultdict
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
    SATISFICING_THRESHOLD,
)

WORST_STORAGE_THRESH = 15.0  # local threshold for triangle markers
BOUNDARY_THRESHOLD = 0.80   # fraction threshold for boundary line

# -- configuration -----------------------------------------------------------
FIG_OUTPUT_DIR = f"{FIG_DIR}/F5_drought_satisficing"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

SSI_WINDOW_DEFAULT = 3
DATASETS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']
PANEL_LETTERS = list('abc')


def _compute_boundary_segments(frac_grid, sev_edges, mag_edges,
                                threshold=BOUNDARY_THRESHOLD):
    """Grid-aligned boundary segments between cells above/below threshold.

    Returns a list of ((x1,y1),(x2,y2)) segments that lie on grid edges
    separating a cell >= threshold from a cell < threshold.
    """
    ns, nm = frac_grid.shape
    segments = []

    for i in range(ns):
        for j in range(nm):
            if np.isnan(frac_grid[i, j]):
                continue
            below = frac_grid[i, j] < threshold

            # Right neighbour
            if i + 1 < ns and not np.isnan(frac_grid[i + 1, j]):
                if (frac_grid[i + 1, j] < threshold) != below:
                    x = sev_edges[i + 1]
                    segments.append(((x, mag_edges[j]), (x, mag_edges[j + 1])))

            # Top neighbour
            if j + 1 < nm and not np.isnan(frac_grid[i, j + 1]):
                if (frac_grid[i, j + 1] < threshold) != below:
                    y = mag_edges[j + 1]
                    segments.append(((sev_edges[i], y), (sev_edges[i + 1], y)))

    return segments


def _order_segments(segments):
    """Connect boundary segments into continuous polyline paths."""
    if not segments:
        return []

    def rnd(pt):
        return (round(pt[0], 8), round(pt[1], 8))

    adj = defaultdict(list)
    for seg in segments:
        a, b = rnd(seg[0]), rnd(seg[1])
        adj[a].append(b)
        adj[b].append(a)

    visited = set()
    paths = []

    # Start from degree-1 nodes (open endpoints) first, then remaining
    starts = [p for p in adj if len(adj[p]) == 1]
    starts += [p for p in adj if len(adj[p]) != 1]

    for start in starts:
        if all((min(start, n), max(start, n)) in visited
               for n in adj[start]):
            continue

        path = [start]
        current = start
        while True:
            nxt = None
            for n in adj[current]:
                edge = (min(current, n), max(current, n))
                if edge not in visited:
                    nxt = n
                    break
            if nxt is None:
                break
            visited.add((min(current, nxt), max(current, nxt)))
            path.append(nxt)
            current = nxt

        if len(path) > 1:
            paths.append(path)

    return paths


def plot_combined_heatmap(all_data, ssi_window):
    """Create the 3x1 combined heatmap figure."""
    apply_publication_style()

    sev_edges, mag_edges, sev_centers, mag_centers = make_shared_edges(
        all_data, DATASETS, n_bins=10)

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
        frac_grid, _ = compute_emergency_grid(df, sev_edges, mag_edges)
        min_grid, _ = compute_min_storage_grid(df, sev_edges, mag_edges)

        # Background: fraction avoiding emergency
        ax.pcolormesh(
            sev_edges, mag_edges,
            np.ma.masked_invalid(frac_grid.T),
            cmap=cmap_frac, norm=norm_frac, rasterized=True,
        )
        ax.set_facecolor('#f0f0f0')

        # 80% avoidance boundary line
        boundary_segs = _compute_boundary_segments(
            frac_grid, sev_edges, mag_edges)
        boundary_effect = [pe.withStroke(linewidth=3.5, foreground='#333333')]
        for path in _order_segments(boundary_segs):
            xs, ys = zip(*path)
            ax.plot(xs, ys, color='white', linewidth=2,
                    solid_capstyle='round', solid_joinstyle='round',
                    zorder=4, path_effects=boundary_effect)

        # Triangle markers where worst-case storage < threshold
        for i, sc in enumerate(sev_centers):
            for j, mc in enumerate(mag_centers):
                if np.isnan(min_grid[i, j]):
                    continue
                if min_grid[i, j] < WORST_STORAGE_THRESH:
                    ax.scatter(sc, mc, s=55, marker='v', color='black',
                               linewidths=0.8, zorder=5)

        ax.set_xlim(sev_edges[0], sev_edges[-1])
        ax.set_ylim(mag_edges[0], mag_edges[-1])

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
    boundary_effect = [pe.withStroke(linewidth=3.5, foreground='#333333')]
    h_line = Line2D([0], [0], color='white', linewidth=2,
                    path_effects=boundary_effect,
                    label=f'{BOUNDARY_THRESHOLD:.0%} avoidance boundary')
    h_tri = Line2D([0], [0], marker='v', color='black', linestyle='none',
                   markersize=7,
                   label=f'Worst-case storage < {WORST_STORAGE_THRESH:.0f}%')
    h_nodata = Patch(facecolor='#f0f0f0', edgecolor='#cccccc', linewidth=0.8,
                     label='No drought events in this range')
    fig.legend(
        handles=[h_line, h_tri, h_nodata], loc='lower center', ncol=3,
        fontsize=FONTSIZE_SMALL, frameon=True, framealpha=0.9,
        edgecolor='none', shadow=False,
        bbox_to_anchor=(0.53, -0.01),
    )

    # -- save ---------------------------------------------------------------
    fname = f"{FIG_OUTPUT_DIR}/F5alt_satisficing_heatmap_ssi{ssi_window}.png"
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close(fig)


# -- main -------------------------------------------------------------------

def main():
    ssi_window = int(sys.argv[1]) if len(sys.argv) > 1 else SSI_WINDOW_DEFAULT
    print(f"F5alt: Combined Drought Satisficing Heatmap (SSI-{ssi_window})")

    all_data = {}
    for did in DATASETS:
        df = load_event_metrics(did, ssi_window)
        all_data[did] = df
        print(f"  {DATASET_LABELS.get(did, did)}: {len(df)} events")

    plot_combined_heatmap(all_data, ssi_window)
    print("Done.")


if __name__ == '__main__':
    main()
