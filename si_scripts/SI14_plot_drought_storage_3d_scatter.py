"""
SI14: Drought Magnitude vs Minimum Storage

Three figure variants:

  (A) 2-D boxplots — events binned by drought magnitude only; one box per
      bin, all three climate scenarios side-by-side within each bin.

  (B) 3-D boxplots — events binned by (severity, magnitude); each bin is
      a 3-D box whose height = IQR of min storage, colored by median.

  (C) 2-D heatmap — severity × magnitude grid, cell color = median min
      storage, circle marker = absolute worst-case min storage per bin.

Usage:
    python SI14_plot_drought_storage_3d_scatter.py [ssi_window]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import warnings
warnings.filterwarnings("ignore")

from methods.config import FIG_DIR, EVENT_METRICS_DIR
from methods.plotting.styles import (
    DATASET_LABELS, DATASET_COLORS,
    FONTSIZE_SMALL, FONTSIZE_MEDIUM,
    DPI_HIGH, apply_publication_style,
)

FIG_OUTPUT_DIR = f"{FIG_DIR}/SI14_drought_storage_3d"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

SSI_WINDOW_DEFAULT = 3
MIN_DURATION = 30
N_BINS = 8          # bins per axis
MIN_COUNT = 5       # minimum events in a bin to draw a box

DATASETS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']


# ── data loading ─────────────────────────────────────────────────────

def load_events(dataset_id, ssi_window):
    df = pd.read_csv(
        f'{EVENT_METRICS_DIR}/'
        f'{dataset_id}_ssi{ssi_window}_event_metrics.csv'
    )
    df = df[df['duration_days'] >= MIN_DURATION].copy()
    df['severity'] = df['severity'].abs()
    df['magnitude'] = df['magnitude'].abs()
    return df


# ── bin statistics ───────────────────────────────────────────────────

def compute_bin_stats(df, sev_edges, mag_edges):
    """Compute storage quantiles per (severity, magnitude) bin.

    Returns list of dicts with keys:
        sev_center, mag_center, ds, dm,
        q25, q50, q75, lo, hi, count
    """
    sev = df['severity'].values
    mag = df['magnitude'].values
    sto = df['event_min_storage_pct'].values

    sev_idx = np.digitize(sev, sev_edges) - 1
    mag_idx = np.digitize(mag, mag_edges) - 1

    ns = len(sev_edges) - 1
    nm = len(mag_edges) - 1

    bins = []
    for i in range(ns):
        for j in range(nm):
            mask = (sev_idx == i) & (mag_idx == j)
            cnt = mask.sum()
            if cnt < MIN_COUNT:
                continue
            vals = sto[mask]
            bins.append({
                'sev_center': 0.5 * (sev_edges[i] + sev_edges[i + 1]),
                'mag_center': 0.5 * (mag_edges[j] + mag_edges[j + 1]),
                'ds': sev_edges[i + 1] - sev_edges[i],
                'dm': mag_edges[j + 1] - mag_edges[j],
                'q25': np.percentile(vals, 25),
                'q50': np.median(vals),
                'q75': np.percentile(vals, 75),
                'lo': vals.min(),
                'hi': vals.max(),
                'count': cnt,
            })
    return bins


# ── 2-D boxplots: magnitude bins only ────────────────────────────────

N_MAG_BINS = 6   # coarser for readability

def plot_magnitude_boxplots(all_data, ssi_window):
    """Side-by-side boxplots of min storage for each magnitude bin."""
    apply_publication_style()

    # Shared magnitude bin edges
    all_mag = np.concatenate([all_data[d]['magnitude'].values for d in DATASETS])
    mag_hi = np.percentile(all_mag, 99)
    mag_edges = np.linspace(all_mag.min(), mag_hi, N_MAG_BINS + 1)

    # Bin labels (range strings)
    bin_labels = [
        f'{mag_edges[j]:.0f}–{mag_edges[j+1]:.0f}'
        for j in range(N_MAG_BINS)
    ]

    # Collect per-bin storage arrays for each dataset
    box_data = {did: [] for did in DATASETS}
    bin_counts = {did: [] for did in DATASETS}
    for did in DATASETS:
        df = all_data[did]
        mag = df['magnitude'].values
        sto = df['event_min_storage_pct'].values
        mag_idx = np.digitize(mag, mag_edges) - 1
        for j in range(N_MAG_BINS):
            vals = sto[mag_idx == j]
            box_data[did].append(vals)
            bin_counts[did].append(len(vals))

    # ── figure ───────────────────────────────────────────────────────
    n_datasets = len(DATASETS)
    group_width = 0.75
    bw = group_width / n_datasets   # width of each box
    cap_hw = bw * 0.30              # half-width of min/max cap ticks

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for d_idx, did in enumerate(DATASETS):
        positions = np.arange(N_MAG_BINS) + (d_idx - (n_datasets - 1) / 2) * bw
        color = DATASET_COLORS.get(did, f'C{d_idx}')

        # Draw thin full-range bars (min→max) BEHIND the boxes
        for j, pos in enumerate(positions):
            vals = box_data[did][j]
            if len(vals) == 0:
                continue
            vmin, vmax = vals.min(), vals.max()
            # Vertical range line
            ax.plot([pos, pos], [vmin, vmax],
                    color=color, linewidth=1.0, alpha=0.35, zorder=1)
            # Small horizontal caps at min and max
            ax.plot([pos - cap_hw, pos + cap_hw], [vmin, vmin],
                    color=color, linewidth=1.0, alpha=0.50, zorder=1)
            ax.plot([pos - cap_hw, pos + cap_hw], [vmax, vmax],
                    color=color, linewidth=1.0, alpha=0.50, zorder=1)

        # IQR box with 5th–95th whiskers
        bp = ax.boxplot(
            box_data[did],
            positions=positions,
            widths=bw * 0.85,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
            medianprops=dict(color='black', linewidth=1.5),
            boxprops=dict(facecolor=color, alpha=0.7, edgecolor='black',
                          linewidth=0.6),
            whiskerprops=dict(color='black', linewidth=0.8),
            capprops=dict(color='black', linewidth=0.8),
            zorder=2,
        )

    # X-axis
    ax.set_xticks(np.arange(N_MAG_BINS))
    ax.set_xticklabels(bin_labels, fontsize=FONTSIZE_SMALL)
    ax.set_xlabel('Drought Magnitude Bin (cumulative SSI deficit)',
                  fontsize=FONTSIZE_MEDIUM)
    ax.set_ylabel('Min NYC Storage During Drought (%)',
                  fontsize=FONTSIZE_MEDIUM)

    # Count annotations
    for j in range(N_MAG_BINS):
        counts_str = '/'.join(str(bin_counts[d][j]) for d in DATASETS)
        ax.text(j, -3, f'n={counts_str}', ha='center',
                fontsize=FONTSIZE_SMALL - 2, fontstyle='italic', alpha=0.6)

    ax.set_ylim(-8, 105)
    ax.axhline(20, color='red', linewidth=1, linestyle='--', alpha=0.5,
               label='20% Storage Threshold')

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D as _L2D
    handles = [Patch(facecolor=DATASET_COLORS.get(d, f'C{i}'), alpha=0.7,
                     edgecolor='black', linewidth=0.6,
                     label=DATASET_LABELS.get(d, d))
               for i, d in enumerate(DATASETS)]
    handles += [
        _L2D([0], [0], color='black', linewidth=0.8,
             label='Whiskers: 5th–95th pctile'),
        _L2D([0], [0], color='grey', linewidth=1.0, alpha=0.45,
             label='Full range (min–max)'),
        _L2D([0], [0], color='red', linestyle='--',
             linewidth=1, alpha=0.5, label='20% Threshold'),
    ]
    ax.legend(handles=handles, fontsize=FONTSIZE_SMALL, loc='lower left',
              frameon=True, fancybox=True)

    ax.set_title(f'Min Storage by Drought Magnitude (SSI-{ssi_window})',
                 fontsize=FONTSIZE_MEDIUM, pad=8)

    fig.tight_layout()
    fname = (f"{FIG_OUTPUT_DIR}/"
             f"SI14_magnitude_storage_boxplot_ssi{ssi_window}.png")
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close(fig)


# ── 2-D heatmap with worst-case markers ──────────────────────────────

N_HEAT_BINS = 10

def _compute_grid(df, sev_edges, mag_edges):
    """Return 2-D arrays of median and min storage, plus event counts."""
    sev = df['severity'].values
    mag = df['magnitude'].values
    sto = df['event_min_storage_pct'].values

    sev_idx = np.digitize(sev, sev_edges) - 1
    mag_idx = np.digitize(mag, mag_edges) - 1

    ns = len(sev_edges) - 1
    nm = len(mag_edges) - 1

    median_grid = np.full((ns, nm), np.nan)
    min_grid = np.full((ns, nm), np.nan)
    count_grid = np.zeros((ns, nm), dtype=int)

    for i in range(ns):
        for j in range(nm):
            mask = (sev_idx == i) & (mag_idx == j)
            cnt = mask.sum()
            count_grid[i, j] = cnt
            if cnt < MIN_COUNT:
                continue
            vals = sto[mask]
            median_grid[i, j] = np.median(vals)
            min_grid[i, j] = vals.min()

    return median_grid, min_grid, count_grid


def plot_heatmap_with_min_marker(all_data, ssi_window):
    """Severity × magnitude heatmap: color = median storage, dot = worst case."""
    apply_publication_style()

    # Shared bin edges
    all_sev = np.concatenate([all_data[d]['severity'].values for d in DATASETS])
    all_mag = np.concatenate([all_data[d]['magnitude'].values for d in DATASETS])
    sev_hi = np.percentile(all_sev, 99)
    mag_hi = np.percentile(all_mag, 99)
    sev_edges = np.linspace(all_sev.min(), sev_hi, N_HEAT_BINS + 1)
    mag_edges = np.linspace(all_mag.min(), mag_hi, N_HEAT_BINS + 1)

    sev_centers = 0.5 * (sev_edges[:-1] + sev_edges[1:])
    mag_centers = 0.5 * (mag_edges[:-1] + mag_edges[1:])

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0, vmax=100)
    # Separate norm for the marker so color reflects worst-case value
    marker_norm = mcolors.Normalize(vmin=0, vmax=100)

    n = len(DATASETS)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 5.5), sharey=True)
    if n == 1:
        axes = [axes]

    panel_letters = 'abcdefghij'

    for idx, did in enumerate(DATASETS):
        ax = axes[idx]
        label = DATASET_LABELS.get(did, did)
        df = all_data[did]

        median_grid, min_grid, count_grid = _compute_grid(
            df, sev_edges, mag_edges)

        # Heatmap (pcolormesh) — axes are severity (x) × magnitude (y)
        im = ax.pcolormesh(
            sev_edges, mag_edges,
            np.ma.masked_invalid(median_grid.T),
            cmap=cmap, norm=norm, rasterized=True,
        )
        ax.set_facecolor('#f0f0f0')

        # Worst-case circle markers
        for i, sc in enumerate(sev_centers):
            for j, mc in enumerate(mag_centers):
                if np.isnan(min_grid[i, j]):
                    continue
                worst = min_grid[i, j]
                ax.scatter(
                    sc, mc, s=55,
                    c=[cmap(marker_norm(worst))],
                    edgecolors='black', linewidths=0.8,
                    marker='o', zorder=4,
                )

        ax.set_xlim(sev_edges[0], sev_edges[-1])
        ax.set_ylim(mag_edges[0], mag_edges[-1])
        ax.set_xlabel('Severity (max SSI deviation)', fontsize=FONTSIZE_MEDIUM)
        if idx == 0:
            ax.set_ylabel('Magnitude (cumulative deficit)',
                          fontsize=FONTSIZE_MEDIUM)
        ax.set_title(f'({panel_letters[idx]})  {label}',
                     fontsize=FONTSIZE_MEDIUM, pad=8)
        ax.tick_params(labelsize=FONTSIZE_SMALL)

    # ── colorbars ────────────────────────────────────────────────────
    # Main colorbar for heatmap fill
    cbar_ax = fig.add_axes([0.15, 0.03, 0.50, 0.025])
    cb = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm),
        cax=cbar_ax, orientation='horizontal',
    )
    cb.set_label('Median Min Storage (%)', fontsize=FONTSIZE_SMALL)
    cb.ax.tick_params(labelsize=FONTSIZE_SMALL - 1)

    # Marker legend (manual)
    from matplotlib.lines import Line2D as _L2D
    marker_handle = _L2D(
        [0], [0], marker='o', color='none', markerfacecolor='grey',
        markeredgecolor='black', markeredgewidth=0.8, markersize=7,
        label='Worst-case min storage\n(color = same scale)',
    )
    fig.legend(
        handles=[marker_handle], loc='lower right',
        fontsize=FONTSIZE_SMALL, frameon=True, framealpha=0.9,
        bbox_to_anchor=(0.97, 0.005),
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])

    fname = (f"{FIG_OUTPUT_DIR}/"
             f"SI14_heatmap_min_storage_ssi{ssi_window}.png")
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close(fig)


# ── 2-D heatmap: % avoiding Drought Emergency, with X for <90% ──────

SATISFICING_THRESHOLD = 0.90  # fraction that must stay above Emergency

def _compute_emergency_grid(df, sev_edges, mag_edges):
    """Return 2-D arrays: fraction avoiding Emergency, and event counts."""
    sev = df['severity'].values
    mag = df['magnitude'].values
    zone = df['ffmp_zone_at_min'].values

    sev_idx = np.digitize(sev, sev_edges) - 1
    mag_idx = np.digitize(mag, mag_edges) - 1

    ns = len(sev_edges) - 1
    nm = len(mag_edges) - 1

    frac_grid = np.full((ns, nm), np.nan)
    count_grid = np.zeros((ns, nm), dtype=int)

    for i in range(ns):
        for j in range(nm):
            mask = (sev_idx == i) & (mag_idx == j)
            cnt = mask.sum()
            count_grid[i, j] = cnt
            if cnt < MIN_COUNT:
                continue
            n_above = (zone[mask] != 'Emergency').sum()
            frac_grid[i, j] = n_above / cnt

    return frac_grid, count_grid


SEV_MAX = 4.0       # severity axis cap
MAG_MAX = 50.0      # magnitude axis cap
WORST_STORAGE_THRESH = 10.0  # % — mark X if worst-case below this

def _make_shared_edges(all_data):
    """Return (sev_edges, mag_edges, sev_centers, mag_centers)."""
    all_sev = np.concatenate([all_data[d]['severity'].values for d in DATASETS])
    all_mag = np.concatenate([all_data[d]['magnitude'].values for d in DATASETS])
    sev_edges = np.linspace(all_sev.min(), SEV_MAX, N_HEAT_BINS + 1)
    mag_edges = np.linspace(all_mag.min(), MAG_MAX, N_HEAT_BINS + 1)
    sev_centers = 0.5 * (sev_edges[:-1] + sev_edges[1:])
    mag_centers = 0.5 * (mag_edges[:-1] + mag_edges[1:])
    return sev_edges, mag_edges, sev_centers, mag_centers


def _compute_min_storage_grid(df, sev_edges, mag_edges):
    """Return 2-D array of worst-case (absolute min) storage per bin."""
    sev = df['severity'].values
    mag = df['magnitude'].values
    sto = df['event_min_storage_pct'].values

    sev_idx = np.digitize(sev, sev_edges) - 1
    mag_idx = np.digitize(mag, mag_edges) - 1

    ns = len(sev_edges) - 1
    nm = len(mag_edges) - 1

    min_grid = np.full((ns, nm), np.nan)
    count_grid = np.zeros((ns, nm), dtype=int)

    for i in range(ns):
        for j in range(nm):
            mask = (sev_idx == i) & (mag_idx == j)
            cnt = mask.sum()
            count_grid[i, j] = cnt
            if cnt < MIN_COUNT:
                continue
            min_grid[i, j] = sto[mask].min()

    return min_grid, count_grid


def plot_heatmap_emergency_satisficing(all_data, ssi_window):
    """Single-row heatmap: fraction avoiding Drought Emergency."""
    apply_publication_style()

    sev_edges, mag_edges, sev_centers, mag_centers = _make_shared_edges(all_data)

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0.15, vmax=1)

    n = len(DATASETS)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 5.5), sharey=True)
    if n == 1:
        axes = [axes]

    panel_letters = 'abcdefghij'

    for idx, did in enumerate(DATASETS):
        ax = axes[idx]
        label = DATASET_LABELS.get(did, did)
        df = all_data[did]

        frac_grid, _ = _compute_emergency_grid(df, sev_edges, mag_edges)

        im = ax.pcolormesh(
            sev_edges, mag_edges,
            np.ma.masked_invalid(frac_grid.T),
            cmap=cmap, norm=norm, rasterized=True,
        )
        ax.set_facecolor('#f0f0f0')

        for i, sc in enumerate(sev_centers):
            for j, mc in enumerate(mag_centers):
                if np.isnan(frac_grid[i, j]):
                    continue
                if frac_grid[i, j] < SATISFICING_THRESHOLD:
                    ax.scatter(sc, mc, s=60, marker='x', color='black',
                               linewidths=1.2, zorder=5)

        ax.set_xlim(sev_edges[0], sev_edges[-1])
        ax.set_ylim(mag_edges[0], mag_edges[-1])
        ax.set_xlabel('Severity (max SSI deviation)', fontsize=FONTSIZE_MEDIUM)
        if idx == 0:
            ax.set_ylabel('Magnitude (cumulative deficit)',
                          fontsize=FONTSIZE_MEDIUM)
        ax.set_title(f'({panel_letters[idx]})  {label}',
                     fontsize=FONTSIZE_MEDIUM, pad=8)
        ax.tick_params(labelsize=FONTSIZE_SMALL)

    cbar_ax = fig.add_axes([0.15, 0.03, 0.50, 0.025])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                      cax=cbar_ax, orientation='horizontal')
    cb.set_label('Fraction Avoiding Drought Emergency', fontsize=FONTSIZE_SMALL)
    cb.ax.tick_params(labelsize=FONTSIZE_SMALL - 1)

    from matplotlib.lines import Line2D as _L2D
    x_handle = _L2D([0], [0], marker='x', color='black', linestyle='none',
                     markeredgewidth=1.2, markersize=7,
                     label=f'< {SATISFICING_THRESHOLD:.0%} above Emergency')
    fig.legend(handles=[x_handle], loc='lower right',
               fontsize=FONTSIZE_SMALL, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.97, 0.005))

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fname = (f"{FIG_OUTPUT_DIR}/"
             f"SI14_heatmap_emergency_satisficing_ssi{ssi_window}.png")
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close(fig)


def plot_heatmap_combined(all_data, ssi_window):
    """Two-row figure: top = worst-case min storage, bottom = DE satisficing."""
    apply_publication_style()

    sev_edges, mag_edges, sev_centers, mag_centers = _make_shared_edges(all_data)

    n = len(DATASETS)
    fig, axes = plt.subplots(2, n, figsize=(6.0 * n, 10), sharey=True, sharex=True)

    panel_letters = 'abcdefghij'

    # ── Row 0: worst-case min storage ────────────────────────────────
    cmap_sto = plt.cm.RdYlGn
    norm_sto = mcolors.Normalize(vmin=0, vmax=60)

    for idx, did in enumerate(DATASETS):
        ax = axes[0, idx]
        label = DATASET_LABELS.get(did, did)
        df = all_data[did]

        min_grid, _ = _compute_min_storage_grid(df, sev_edges, mag_edges)

        im_sto = ax.pcolormesh(
            sev_edges, mag_edges,
            np.ma.masked_invalid(min_grid.T),
            cmap=cmap_sto, norm=norm_sto, rasterized=True,
        )
        ax.set_facecolor('#f0f0f0')

        # X where worst-case < 10 %
        for i, sc in enumerate(sev_centers):
            for j, mc in enumerate(mag_centers):
                if np.isnan(min_grid[i, j]):
                    continue
                if min_grid[i, j] < WORST_STORAGE_THRESH:
                    ax.scatter(sc, mc, s=60, marker='x', color='black',
                               linewidths=1.2, zorder=5)

        ax.set_xlim(sev_edges[0], sev_edges[-1])
        ax.set_ylim(mag_edges[0], mag_edges[-1])
        ax.set_title(f'({panel_letters[idx]})  {label}',
                     fontsize=FONTSIZE_MEDIUM, pad=8)
        ax.tick_params(labelsize=FONTSIZE_SMALL)
        if idx == 0:
            ax.set_ylabel('Magnitude (cumulative deficit)',
                          fontsize=FONTSIZE_MEDIUM)

    # ── Row 1: DE satisficing fraction ───────────────────────────────
    cmap_frac = plt.cm.RdYlGn
    norm_frac = mcolors.Normalize(vmin=0.15, vmax=1)

    for idx, did in enumerate(DATASETS):
        ax = axes[1, idx]
        df = all_data[did]

        frac_grid, _ = _compute_emergency_grid(df, sev_edges, mag_edges)

        im_frac = ax.pcolormesh(
            sev_edges, mag_edges,
            np.ma.masked_invalid(frac_grid.T),
            cmap=cmap_frac, norm=norm_frac, rasterized=True,
        )
        ax.set_facecolor('#f0f0f0')

        for i, sc in enumerate(sev_centers):
            for j, mc in enumerate(mag_centers):
                if np.isnan(frac_grid[i, j]):
                    continue
                if frac_grid[i, j] < SATISFICING_THRESHOLD:
                    ax.scatter(sc, mc, s=60, marker='x', color='black',
                               linewidths=1.2, zorder=5)

        ax.set_xlim(sev_edges[0], sev_edges[-1])
        ax.set_ylim(mag_edges[0], mag_edges[-1])
        ax.set_xlabel('Severity (max SSI deviation)', fontsize=FONTSIZE_MEDIUM)
        letter = panel_letters[n + idx]
        label = DATASET_LABELS.get(did, did)
        ax.set_title(f'({letter})  {label}',
                     fontsize=FONTSIZE_MEDIUM, pad=8)
        ax.tick_params(labelsize=FONTSIZE_SMALL)
        if idx == 0:
            ax.set_ylabel('Magnitude (cumulative deficit)',
                          fontsize=FONTSIZE_MEDIUM)

    # ── colorbars ────────────────────────────────────────────────────
    # Top row colorbar — worst-case storage
    cbar_ax1 = fig.add_axes([0.15, 0.52, 0.50, 0.012])
    cb1 = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_sto, norm=norm_sto),
                       cax=cbar_ax1, orientation='horizontal')
    cb1.set_label('Worst-Case Min Storage (%)', fontsize=FONTSIZE_SMALL)
    cb1.ax.tick_params(labelsize=FONTSIZE_SMALL - 1)

    # Bottom row colorbar — satisficing fraction
    cbar_ax2 = fig.add_axes([0.15, 0.03, 0.50, 0.012])
    cb2 = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_frac, norm=norm_frac),
                       cax=cbar_ax2, orientation='horizontal')
    cb2.set_label('Fraction Avoiding Drought Emergency',
                  fontsize=FONTSIZE_SMALL)
    cb2.ax.tick_params(labelsize=FONTSIZE_SMALL - 1)

    # ── legends ──────────────────────────────────────────────────────
    from matplotlib.lines import Line2D as _L2D
    x1 = _L2D([0], [0], marker='x', color='black', linestyle='none',
              markeredgewidth=1.2, markersize=7,
              label=f'Worst-case < {WORST_STORAGE_THRESH:.0f}% (top row)')
    x2 = _L2D([0], [0], marker='x', color='black', linestyle='none',
              markeredgewidth=1.2, markersize=7,
              label=f'< {SATISFICING_THRESHOLD:.0%} above Emergency (bottom row)')
    fig.legend(handles=[x1, x2], loc='lower right',
               fontsize=FONTSIZE_SMALL, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.97, 0.005))

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.subplots_adjust(hspace=0.35)

    fname = (f"{FIG_OUTPUT_DIR}/"
             f"SI14_heatmap_combined_ssi{ssi_window}.png")
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close(fig)


# ── draw a single 3-D box with whiskers ─────────────────────────────

def _draw_box(ax, x, y, dx, dy, z_lo, z_hi, facecolor, alpha=0.75):
    """Draw a rectangular prism from z_lo to z_hi centered at (x, y)."""
    x0 = x - dx / 2
    x1 = x + dx / 2
    y0 = y - dy / 2
    y1 = y + dy / 2

    # six faces: bottom, top, front, back, left, right
    verts = [
        [(x0, y0, z_lo), (x1, y0, z_lo), (x1, y1, z_lo), (x0, y1, z_lo)],  # bottom
        [(x0, y0, z_hi), (x1, y0, z_hi), (x1, y1, z_hi), (x0, y1, z_hi)],  # top
        [(x0, y0, z_lo), (x1, y0, z_lo), (x1, y0, z_hi), (x0, y0, z_hi)],  # front
        [(x0, y1, z_lo), (x1, y1, z_lo), (x1, y1, z_hi), (x0, y1, z_hi)],  # back
        [(x0, y0, z_lo), (x0, y1, z_lo), (x0, y1, z_hi), (x0, y0, z_hi)],  # left
        [(x1, y0, z_lo), (x1, y1, z_lo), (x1, y1, z_hi), (x1, y0, z_hi)],  # right
    ]
    poly = Poly3DCollection(verts, alpha=alpha, facecolor=facecolor,
                            edgecolor='black', linewidths=0.3)
    ax.add_collection3d(poly)


def _draw_whisker(ax, x, y, z_start, z_end, color='black', lw=0.8):
    """Draw a vertical whisker line at (x, y)."""
    ax.plot([x, x], [y, y], [z_start, z_end],
            color=color, linewidth=lw, alpha=0.6)


# ── 3-D box-plot figure ─────────────────────────────────────────────

def plot_3d_boxplot(all_data, ssi_window):
    apply_publication_style()

    n = len(DATASETS)
    fig = plt.figure(figsize=(6.5 * n, 6.5))

    # Shared bin edges across datasets
    all_sev = np.concatenate([all_data[d]['severity'].values for d in DATASETS])
    all_mag = np.concatenate([all_data[d]['magnitude'].values for d in DATASETS])

    sev_hi = np.percentile(all_sev, 99)
    mag_hi = np.percentile(all_mag, 99)
    sev_edges = np.linspace(all_sev.min(), sev_hi, N_BINS + 1)
    mag_edges = np.linspace(all_mag.min(), mag_hi, N_BINS + 1)

    # Colormap: median storage 0-100 %
    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0, vmax=100)

    panel_letters = 'abcdefghij'

    for i, did in enumerate(DATASETS):
        ax = fig.add_subplot(1, n, i + 1, projection='3d')
        df = all_data[did]
        label = DATASET_LABELS.get(did, did)

        bins = compute_bin_stats(df, sev_edges, mag_edges)

        for b in bins:
            fc = cmap(norm(b['q50']))

            # IQR box
            _draw_box(ax, b['sev_center'], b['mag_center'],
                      b['ds'] * 0.85, b['dm'] * 0.85,
                      b['q25'], b['q75'], facecolor=fc)

            # Whiskers: min → Q25, Q75 → max
            _draw_whisker(ax, b['sev_center'], b['mag_center'],
                          b['lo'], b['q25'])
            _draw_whisker(ax, b['sev_center'], b['mag_center'],
                          b['q75'], b['hi'])

        ax.set_xlim(sev_edges[0], sev_edges[-1])
        ax.set_ylim(mag_edges[0], mag_edges[-1])
        ax.set_zlim(0, 100)

        ax.set_xlabel('Severity\n(max SSI deviation)',
                      fontsize=FONTSIZE_SMALL, labelpad=8)
        ax.set_ylabel('Magnitude\n(cumulative deficit)',
                      fontsize=FONTSIZE_SMALL, labelpad=8)
        ax.set_zlabel('Min Storage (%)',
                      fontsize=FONTSIZE_SMALL, labelpad=8)

        ax.set_title(f'({panel_letters[i]})  {label}',
                     fontsize=FONTSIZE_MEDIUM, pad=12)

        ax.tick_params(labelsize=FONTSIZE_SMALL - 1)
        ax.view_init(elev=25, azim=-50)

    # ── colorbar ─────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.15, 0.02, 0.70, 0.02])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cb.set_label('Median Min Storage (%)', fontsize=FONTSIZE_SMALL)
    cb.ax.tick_params(labelsize=FONTSIZE_SMALL - 1)

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    fname = f"{FIG_OUTPUT_DIR}/SI14_drought_storage_3d_boxplot_ssi{ssi_window}.png"
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────

def main():
    ssi_window = int(sys.argv[1]) if len(sys.argv) > 1 else SSI_WINDOW_DEFAULT
    print(f"SI14: 3-D Drought Storage Box Plot (SSI-{ssi_window})")

    all_data = {}
    for did in DATASETS:
        df = load_events(did, ssi_window)
        all_data[did] = df
        print(f"  {DATASET_LABELS.get(did, did)}: {len(df)} events")

    plot_magnitude_boxplots(all_data, ssi_window)
    plot_heatmap_with_min_marker(all_data, ssi_window)
    plot_heatmap_emergency_satisficing(all_data, ssi_window)
    plot_heatmap_combined(all_data, ssi_window)
    plot_3d_boxplot(all_data, ssi_window)
    print("Done.")


if __name__ == '__main__':
    main()
