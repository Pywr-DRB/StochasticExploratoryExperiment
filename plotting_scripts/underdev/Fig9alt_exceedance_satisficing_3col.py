"""
Fig9alt (3-column): Exceedance Rate + Satisficing Heatmap + Storage Slope Chart

9-panel figure (3 rows x 3 columns).
  Rows = climate scenarios (Baseline, Mixed Future, Wet Future)
  Col 1:  Joint exceedance rate (events/year) per (severity, magnitude) bin
  Col 2:  Fraction of events avoiding FFMP Drought Emergency
  Col 3:  Slope chart — median start storage → median min storage per bin

Usage:
    python Fig9alt_exceedance_satisficing_3col.py [ssi_window]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
    make_shared_edges_logmag, compute_exceedance_rate_grid,
    compute_emergency_grid, compute_min_storage_grid, draw_focal_boundary,
    GRID_N_BINS, GRID_TARGET_SEV_BIN, GRID_TARGET_MAG_BIN,
)

WORST_STORAGE_THRESH = 15.0

# -- focal-region thresholds --------------------------------------------------
# These define the multi-metric criteria for selecting "interesting" cells.
# On the HPC (many realizations) the defaults work well; on small local runs
# they are relaxed automatically — see _identify_focal_region().
FOCAL_FRAC_THRESH = 0.95       # fraction avoiding emergency must be < this (any dataset)
FOCAL_RATE_THRESH = 10e-4        # exceedance rate must exceed this in ALL datasets

# -- configuration -----------------------------------------------------------
FIG_OUTPUT_DIR = f"{FIG_DIR}/Fig9alt_exceedance_satisficing"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

SSI_WINDOW_DEFAULT = 3
DATASETS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']
PANEL_LETTERS = list('abcdefghi')


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



def _get_focal_region_events(df, sev_edges, mag_edges, focal_cells):
    """Return individual events that fall inside any focal-region cell.

    Returns
    -------
    start_storages : np.ndarray
    min_storages : np.ndarray
    """
    sev = df['severity'].values
    mag = df['magnitude'].values

    sev_idx = np.digitize(sev, sev_edges) - 1
    mag_idx = np.digitize(mag, mag_edges) - 1

    mask = np.zeros(len(df), dtype=bool)
    for i, j in focal_cells:
        mask |= (sev_idx == i) & (mag_idx == j)

    return (df.loc[mask, 'storage_at_start_pct'].values,
            df.loc[mask, 'event_min_storage_pct'].values)


def plot_combined_figure(all_data, ssi_window, n_bins=GRID_N_BINS, min_count=1):
    """Create the 3x3 combined figure."""
    apply_publication_style()

    sev_edges, mag_edges, sev_centers, mag_centers = make_shared_edges_logmag(
        all_data, DATASETS, n_bins=n_bins)

    # -- colour maps & norms ------------------------------------------------
    cmap_rate = plt.cm.YlOrRd
    cmap_frac = plt.cm.plasma_r
    norm_frac = mcolors.Normalize(vmin=0.3, vmax=1.0)

    # Compute all grids per dataset
    rate_grids, frac_grids, min_grids = {}, {}, {}
    for did in DATASETS:
        rg, _ = compute_exceedance_rate_grid(
            all_data[did], sev_edges, mag_edges, N_YEARS, min_count=min_count)
        rate_grids[did] = rg
        fg, _ = compute_emergency_grid(
            all_data[did], sev_edges, mag_edges, min_count=min_count)
        frac_grids[did] = fg
        mg, _ = compute_min_storage_grid(
            all_data[did], sev_edges, mag_edges, min_count=min_count)
        min_grids[did] = mg

    all_rates = np.concatenate([rg[~np.isnan(rg)] for rg in rate_grids.values()])
    if len(all_rates) > 0:
        rate_vmin = max(all_rates.min(), 1e-4)
        rate_vmax = all_rates.max()
    else:
        rate_vmin, rate_vmax = 1e-4, 1.0
    norm_rate = mcolors.LogNorm(vmin=rate_vmin, vmax=rate_vmax)

    # Identify focal region via multi-metric criteria
    focal_cells = _identify_focal_region(rate_grids, frac_grids, min_grids, DATASETS)
    print(f"  Focal region: {len(focal_cells)} cells — {sorted(focal_cells)}")

    # Shared drawdown colormap norm across all datasets / focal region
    cmap_dd = plt.cm.RdYlGn_r
    all_dd = []
    for did in DATASETS:
        s, m = _get_focal_region_events(all_data[did], sev_edges, mag_edges,
                                        focal_cells)
        if len(s) > 0:
            all_dd.extend(s - m)
    norm_dd = mcolors.Normalize(vmin=0, vmax=max(all_dd) if all_dd else 1.0)

    # -- figure layout: 3 rows x 3 columns ---------------------------------
    fig = plt.figure(figsize=(16.0, 13.5))
    gs = gridspec.GridSpec(
        3, 3,
        width_ratios=[1, 1, 1],
        hspace=0.15, wspace=0.30,
        left=0.07, right=0.95, bottom=0.08, top=0.88,
    )

    axes = np.empty((3, 3), dtype=object)
    for r in range(3):
        for c in range(3):
            axes[r, c] = fig.add_subplot(gs[r, c])

    # Make col 3 visually square via adjustable data limits
    for r in range(3):
        axes[r, 2].set_box_aspect(1)

    panel_idx = 0
    for row_idx, did in enumerate(DATASETS):
        df = all_data[did]
        ds_label = DATASET_LABELS.get(did, did)

        # ── Col 1: exceedance rate ──────────────────────────────────
        ax_rate = axes[row_idx, 0]
        rate_grid = rate_grids[did]

        ax_rate.pcolormesh(
            sev_edges, mag_edges,
            np.ma.masked_invalid(rate_grid.T),
            cmap=cmap_rate, norm=norm_rate, rasterized=True,
        )
        ax_rate.set_facecolor('#f0f0f0')
        draw_focal_boundary(ax_rate, sev_edges, mag_edges, focal_cells)

        label_panel(ax_rate, PANEL_LETTERS[panel_idx], label=ds_label,
                    fontsize=FONTSIZE_LABEL)
        panel_idx += 1

        ax_rate.set_yscale('log')
        ax_rate.set_xlim(sev_edges[0], sev_edges[-1])
        ax_rate.set_ylim(mag_edges[0], mag_edges[-1])
        ax_rate.set_ylabel('Drought Magnitude\n(cumulative SSI deficit)',
                           fontsize=FONTSIZE_LABEL)
        if row_idx == 2:
            ax_rate.set_xlabel('Drought Severity\n(peak SSI deviation)',
                               fontsize=FONTSIZE_LABEL)
        else:
            ax_rate.set_xticklabels([])
        ax_rate.tick_params(labelsize=FONTSIZE_SMALL)

        # ── Col 2: fraction avoiding emergency ─────────────────────
        ax_frac = axes[row_idx, 1]
        frac_grid = frac_grids[did]

        ax_frac.pcolormesh(
            sev_edges, mag_edges,
            np.ma.masked_invalid(frac_grid.T),
            cmap=cmap_frac, norm=norm_frac, rasterized=True,
        )
        ax_frac.set_facecolor('#f0f0f0')
        draw_focal_boundary(ax_frac, sev_edges, mag_edges, focal_cells)

        # Triangle markers for worst-case storage
        min_grid = min_grids[did]
        for i, sc in enumerate(sev_centers):
            for j, mc in enumerate(mag_centers):
                if np.isnan(min_grid[i, j]):
                    continue
                if min_grid[i, j] < WORST_STORAGE_THRESH:
                    ax_frac.scatter(sc, mc, s=40, marker='v', color='black',
                                    linewidths=0.6, zorder=5)

        label_panel(ax_frac, PANEL_LETTERS[panel_idx], label=ds_label,
                    fontsize=FONTSIZE_LABEL)
        panel_idx += 1

        ax_frac.set_yscale('log')
        ax_frac.set_xlim(sev_edges[0], sev_edges[-1])
        ax_frac.set_ylim(mag_edges[0], mag_edges[-1])
        if row_idx == 2:
            ax_frac.set_xlabel('Drought Severity\n(peak SSI deviation)',
                               fontsize=FONTSIZE_LABEL)
        else:
            ax_frac.set_xticklabels([])
        ax_frac.set_yticklabels([])
        ax_frac.tick_params(labelsize=FONTSIZE_SMALL)

        # ── Col 3: slope chart (individual focal-region events) ─────
        ax_slope = axes[row_idx, 2]
        start_sto, min_sto = _get_focal_region_events(
            df, sev_edges, mag_edges, focal_cells)

        # Color each line by its drawdown (shared norm across rows)
        if len(start_sto) > 0:
            drawdowns = start_sto - min_sto

            for s, m, dd in zip(start_sto, min_sto, drawdowns):
                color = cmap_dd(norm_dd(dd))
                ax_slope.plot([0, 1], [s, m],
                              color=color, alpha=0.6, linewidth=1.0, zorder=3)
                ax_slope.scatter([0], [s], color=color,
                                 s=18, alpha=0.7, zorder=4, edgecolors='none')
                ax_slope.scatter([1], [m], color=color,
                                 s=18, alpha=0.7, zorder=4, edgecolors='none')

        # Reference lines
        ax_slope.axhline(WORST_STORAGE_THRESH, color='#d32f2f', linewidth=1.0,
                         linestyle='--', alpha=0.7, zorder=2)

        ax_slope.set_xlim(-0.15, 1.15)
        ax_slope.set_ylim(0, 100)
        ax_slope.set_xticks([0, 1])
        ax_slope.set_xticklabels(['Start', 'Min'], fontsize=FONTSIZE_MEDIUM)
        if row_idx == 2:
            ax_slope.set_xlabel('Drought Phase', fontsize=FONTSIZE_LABEL)
        ax_slope.set_ylabel('NYC Storage (%)', fontsize=FONTSIZE_LABEL)
        ax_slope.tick_params(labelsize=FONTSIZE_SMALL)
        ax_slope.grid(axis='y', alpha=0.3, linewidth=0.5)

        label_panel(ax_slope, PANEL_LETTERS[panel_idx], label=ds_label,
                    fontsize=FONTSIZE_LABEL)
        panel_idx += 1

    # -- column titles ------------------------------------------------------
    axes[0, 0].set_title('Exceedance Rate\n(events yr$^{-1}$)',
                         fontsize=FONTSIZE_LABEL, pad=10)
    axes[0, 1].set_title('Fraction Avoiding\nDrought Emergency',
                         fontsize=FONTSIZE_LABEL, pad=10)
    axes[0, 2].set_title('Storage Drawdown\n(focal region events)',
                         fontsize=FONTSIZE_LABEL, pad=10)

    # -- colorbars at top ---------------------------------------------------
    fig.canvas.draw()

    cbar_h = 0.012
    cbar_top = 0.90

    # Left colorbar: exceedance rate
    bb_left = axes[0, 0].get_position()
    cbar_ax_rate = fig.add_axes([bb_left.x0, cbar_top, bb_left.width, cbar_h])
    cb_rate = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_rate, norm=norm_rate),
        cax=cbar_ax_rate, orientation='horizontal',
    )
    cbar_ax_rate.xaxis.set_ticks_position('top')
    cbar_ax_rate.xaxis.set_label_position('top')
    cb_rate.set_label('Exceedance Rate (events yr$^{-1}$)',
                      fontsize=FONTSIZE_SMALL)
    cb_rate.ax.tick_params(labelsize=FONTSIZE_SMALL - 1)

    # Middle colorbar: fraction avoiding emergency
    bb_mid = axes[0, 1].get_position()
    cbar_ax_frac = fig.add_axes([bb_mid.x0, cbar_top, bb_mid.width, cbar_h])
    cb_frac = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_frac, norm=norm_frac),
        cax=cbar_ax_frac, orientation='horizontal',
    )
    cbar_ax_frac.xaxis.set_ticks_position('top')
    cbar_ax_frac.xaxis.set_label_position('top')
    cb_frac.set_label('Fraction Avoiding Emergency',
                      fontsize=FONTSIZE_SMALL)
    cb_frac.ax.tick_params(labelsize=FONTSIZE_SMALL - 1)

    # Right colorbar: drawdown (shared norm computed earlier)
    bb_right = axes[0, 2].get_position()
    cbar_ax_dd = fig.add_axes([bb_right.x0, cbar_top, bb_right.width, cbar_h])
    cb_dd = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_dd, norm=norm_dd),
        cax=cbar_ax_dd, orientation='horizontal',
    )
    cbar_ax_dd.xaxis.set_ticks_position('top')
    cbar_ax_dd.xaxis.set_label_position('top')
    cb_dd.set_label('Drawdown (%-points)',
                    fontsize=FONTSIZE_SMALL)
    cb_dd.ax.tick_params(labelsize=FONTSIZE_SMALL - 1)

    # -- legend at bottom ---------------------------------------------------
    h_tri = Line2D([0], [0], marker='v', color='black', linestyle='none',
                   markersize=7,
                   label=f'Worst-case storage < {WORST_STORAGE_THRESH:.0f}%')
    h_nodata = Patch(facecolor='#f0f0f0', edgecolor='#cccccc', linewidth=0.8,
                     label='No drought events in this range')
    h_cell = Patch(facecolor='none', edgecolor='white', linewidth=2.0,
                   label=(f'Focal region (rate>{FOCAL_RATE_THRESH:.0e} all, '
                          f'frac<{FOCAL_FRAC_THRESH:.0%} all, '
                          f'min sto<{WORST_STORAGE_THRESH:.0f}% any)'))
    h_emerg = Line2D([0], [0], color='#d32f2f', linestyle='--', linewidth=1.0,
                     label=f'Emergency threshold ({WORST_STORAGE_THRESH:.0f}%)')
    fig.legend(
        handles=[h_tri, h_nodata, h_cell, h_emerg], loc='lower center', ncol=4,
        fontsize=FONTSIZE_SMALL, frameon=True, framealpha=0.9,
        edgecolor='none', shadow=False,
        bbox_to_anchor=(0.51, 0.01),
    )

    # -- save ---------------------------------------------------------------
    fname = (f"{FIG_OUTPUT_DIR}/Fig9alt_exceedance_satisficing_3col_ssi{ssi_window}"
             f"_rate{FOCAL_RATE_THRESH:.0e}_frac{FOCAL_FRAC_THRESH:.2f}"
             f"_sto{WORST_STORAGE_THRESH:.0f}.png")
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close(fig)


# -- main -------------------------------------------------------------------

def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    ssi_window = int(args[0]) if len(args) > 0 else SSI_WINDOW_DEFAULT
    n_bins = int(args[1]) if len(args) > 1 else GRID_N_BINS
    min_count = int(args[2]) if len(args) > 2 else 1

    print(f"Fig9alt (3-col): Exceedance + Satisficing + Slope (SSI-{ssi_window}, "
          f"n_bins={n_bins}, min_count={min_count})")

    all_data = {}
    for did in DATASETS:
        df = load_event_metrics(did, ssi_window)
        all_data[did] = df
        print(f"  {DATASET_LABELS.get(did, did)}: {len(df)} events")

    plot_combined_figure(all_data, ssi_window,
                         n_bins=n_bins, min_count=min_count)
    print("Done.")


if __name__ == '__main__':
    main()
