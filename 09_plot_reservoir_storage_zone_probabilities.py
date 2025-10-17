"""
Plot reservoir storage zone probabilities from pre-calculated CSVs.

Uses cached zone probability calculations from 09a_calculate_storage_zone_probabilities.py
to generate heatmaps and comparison plots.

Usage:
  python 09b_plot_storage_zone_probabilities.py [dataset_id]
  python 09b_plot_storage_zone_probabilities.py --all
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LogNorm
from matplotlib import colors
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from config import *


# Input/output directories
ZONE_PROB_DIR = f"{ROOT_DIR}/pywrdrb/zone_probabilities"
FIG_OUTPUT_DIR = f"{FIG_DIR}/storage_zones"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)


def load_zone_probabilities(dataset_id, period='weekly'):
    """Load zone probabilities from CSV."""
    csv_file = f"{ZONE_PROB_DIR}/{dataset_id}_zone_probs_{period}.csv"
    if not os.path.exists(csv_file):
        print(f"ERROR: Zone probabilities not found: {csv_file}")
        print("Run 09a_calculate_storage_zone_probabilities.py first!")
        return None
    
    df = pd.read_csv(csv_file, index_col='period')
    return df


def load_ffmp_boundaries():
    """Load FFMP level boundaries once (cached)."""
    if not hasattr(load_ffmp_boundaries, '_cache'):
        ffmp_data = pywrdrb.Data(results_sets=["ffmp_level_boundaries"])
        ffmp_data.load_output(output_filenames=[RECONSTRUCTION_OUTPUT_FNAME])
        boundaries = ffmp_data.ffmp_level_boundaries['reconstruction'][0] * 100
        load_ffmp_boundaries._cache = boundaries
    return load_ffmp_boundaries._cache


def get_ordered_threshold_columns(ffmp_boundaries):
    """Get threshold columns ordered by median value."""
    num_cols = ffmp_boundaries.select_dtypes(include=[np.number]).columns.tolist()
    med = ffmp_boundaries[num_cols].median(axis=0).sort_values()
    return list(med.index)


def _period_index(dts: pd.DatetimeIndex, period: str = 'daily', origin: str = 'june1') -> np.ndarray:
    """Map dates to generic-year period index."""
    dts = pd.DatetimeIndex(dts)
    june1_this = pd.to_datetime(dts.year.astype(str) + '-06-01')
    is_after = dts >= june1_this
    june1_prev = pd.to_datetime((dts.year - 1).astype(str) + '-06-01')
    
    doy_wy = np.where(is_after,
                      (dts - june1_this).days + 1,
                      (dts - june1_prev).days + 1)
    
    if period == 'daily':
        return doy_wy
    elif period == 'weekly':
        return ((doy_wy - 1) // 7) + 1
    else:  # monthly
        wy_month = ((dts.month - 6) % 12) + 1
        return wy_month


def build_y_edges_grid(ffmp_boundaries, period='weekly', pct_extents=(0.0, 100.0)):
    """
    Build Y-axis edges for pcolormesh from FFMP boundaries.
    
    Returns
    -------
    y_edges_grid : np.ndarray
        Shape (Z+1, P+1) for pcolormesh
    periods_sorted : np.ndarray
        Unique period values
    """
    thr_cols = get_ordered_threshold_columns(ffmp_boundaries)
    
    # Get period indices
    p_idx = _period_index(ffmp_boundaries.index, period=period, origin='june1')
    
    # Group by period and get median thresholds
    df_b = ffmp_boundaries.copy()
    df_b['__p__'] = p_idx
    grouped = df_b.groupby('__p__')[thr_cols].median()
    periods_sorted = grouped.index.to_numpy()
    
    # Build edges: [0, t1, t2, ..., tK, 100] for each period
    lo, hi = pct_extents
    edges_mat = np.column_stack([
        np.full((grouped.shape[0], 1), lo),
        grouped.to_numpy(copy=False),
        np.full((grouped.shape[0], 1), np.nextafter(hi, np.inf)),
    ])  # shape: (P, Z+1)
    
    # Transpose to (Z+1, P) and append last column for pcolormesh
    y_edges = edges_mat.T
    y_edges_grid = np.concatenate([y_edges, y_edges[:, [-1]]], axis=1)  # (Z+1, P+1)
    
    return y_edges_grid, periods_sorted


def build_x_edges(periods_sorted):
    """Build X-axis edges from period centers."""
    x_centers = periods_sorted.astype(float)
    if x_centers.size == 1:
        x_edges = np.array([x_centers[0]-0.5, x_centers[0]+0.5])
    else:
        mid = 0.5 * (x_centers[:-1] + x_centers[1:])
        x_edges = np.empty(x_centers.size + 1)
        x_edges[1:-1] = mid
        x_edges[0] = x_centers[0] - (mid[0] - x_centers[0])
        x_edges[-1] = x_centers[-1] + (x_centers[-1] - mid[-1])
    return x_edges


def plot_storage_zone_probabilities(prob_df, 
                                     ffmp_boundaries,
                                     period='weekly',
                                     figsize=(14, 6),
                                     cmap='magma_r',
                                     vmin=0.01,
                                     vmax=100,
                                     title=None,
                                     fname=None):
    """
    Plot storage zone probability heatmap.
    
    Parameters
    ----------
    prob_df : pd.DataFrame
        Zone probabilities with index=period, columns=zone_0, zone_1, ...
    ffmp_boundaries : pd.DataFrame
        FFMP level boundaries for computing Y edges
    period : str
        Time aggregation period
    """
    # Build grids
    y_edges_grid, periods_sorted = build_y_edges_grid(ffmp_boundaries, period)
    x_edges = build_x_edges(periods_sorted)
    
    # Align probability data to periods_sorted
    M = prob_df.loc[periods_sorted].to_numpy().T  # (Z, P)
    
    # Handle zeros for LogNorm
    M = np.where(M > 0, M, vmin)
    
    # Build meshgrid
    X = np.tile(x_edges, (y_edges_grid.shape[0], 1))  # (Z+1, P+1)
    Y = y_edges_grid  # (Z+1, P+1)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color='#f0f0f0')
    
    norm = LogNorm(vmin=vmin, vmax=vmax)
    quad = ax.pcolormesh(X, Y, M, cmap=cmap_obj, norm=norm, shading='flat')
    
    # Zone boundary lines
    for j in range(Y.shape[1]):
        ax.plot(X[:, j], Y[:, j], color='white', linewidth=0.6, alpha=0.7)
    
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(0, 100)
    ax.set_xlabel('Period of year')
    ax.set_ylabel('Total NYC storage (% of capacity)')
    if title:
        ax.set_title(title)
    
    cbar = plt.colorbar(quad, ax=ax, pad=0.02)
    cbar.set_label('Probability (%)')
    
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
    return fig, ax


def plot_storage_zone_comparison(prob_df_ref,
                                  prob_df_comp,
                                  ffmp_boundaries,
                                  period='weekly',
                                  figsize=(14, 6),
                                  title=None,
                                  fname=None):
    """
    Plot percentage difference between two zone probability datasets.
    
    Parameters
    ----------
    prob_df_ref : pd.DataFrame
        Reference zone probabilities (e.g., stationary)
    prob_df_comp : pd.DataFrame
        Comparison zone probabilities
    ffmp_boundaries : pd.DataFrame
        FFMP level boundaries for computing Y edges
    """
    # Build grids
    y_edges_grid, periods_sorted = build_y_edges_grid(ffmp_boundaries, period)
    x_edges = build_x_edges(periods_sorted)
    
    # Align data
    M_ref = prob_df_ref.loc[periods_sorted].to_numpy().T  # (Z, P)
    M_comp = prob_df_comp.loc[periods_sorted].to_numpy().T  # (Z, P)
    
    # Calculate percentage difference
    eps = 1e-8
    prob_diff = 100.0 * (M_comp - M_ref) / np.maximum(M_ref, eps)
    prob_ratio = np.log10(np.maximum(M_comp, eps) / np.maximum(M_ref, eps))
    
    # Build meshgrid
    X = np.tile(x_edges, (y_edges_grid.shape[0], 1))
    Y = y_edges_grid
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # set vmin and vmax as 90% percentiles
    vmin = -100
    vmax = 100

    if vmin < 0 and vmax > 0:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

    pcm = ax.pcolormesh(X, Y, prob_diff, cmap='BrBG_r', norm=norm, shading='flat')
    
    # Zone boundary lines
    for j in range(Y.shape[1]):
        ax.plot(X[:, j], Y[:, j], color='white', linewidth=0.6, alpha=0.7)
    
    ax.set_xlabel('Period of year')
    ax.set_ylabel('Total NYC storage (% of capacity)')
    if title:
        ax.set_title(title)
    
    cbar = plt.colorbar(pcm, ax=ax, pad=0.02, extend='both')
    cbar.set_label('Δ Probability (%)')
    
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
    return fig, ax


def plot_dataset(dataset_id, period='weekly', figsize=(14, 6)):
    """Plot zone probabilities for a single dataset."""
    print(f"Plotting {dataset_id}...")
    
    # Load zone probabilities
    prob_df = load_zone_probabilities(dataset_id, period)
    if prob_df is None:
        return False
    
    # Load FFMP boundaries
    ffmp_boundaries = load_ffmp_boundaries()
    
    # Generate plot
    fname = f"{FIG_OUTPUT_DIR}/{dataset_id}_storage_zone_probabilities_{period}.png"
    plot_storage_zone_probabilities(
        prob_df,
        ffmp_boundaries,
        period=period,
        figsize=figsize,
        title=f"{dataset_id} - NYC Storage Zone Probabilities",
        fname=fname
    )
    
    print(f"  Saved: {fname}")
    return True


def plot_comparison(dataset_id_ref, dataset_id_comp, period='weekly',
                    figsize=(14, 6)):
    """Plot comparison between two datasets."""
    print(f"Plotting {dataset_id_comp} vs {dataset_id_ref}...")
    
    # Load zone probabilities
    prob_df_ref = load_zone_probabilities(dataset_id_ref, period)
    prob_df_comp = load_zone_probabilities(dataset_id_comp, period)
    
    if prob_df_ref is None or prob_df_comp is None:
        return False
    
    # Load FFMP boundaries
    ffmp_boundaries = load_ffmp_boundaries()
    
    # Generate comparison plot
    fname = f"{FIG_OUTPUT_DIR}/{dataset_id_comp}_vs_{dataset_id_ref}_diff_{period}.png"
    plot_storage_zone_comparison(
        prob_df_ref,
        prob_df_comp,
        ffmp_boundaries,
        period=period,
        figsize=figsize,
        title=f"Storage Zone Probability Difference: {dataset_id_comp} - {dataset_id_ref}",
        fname=fname
    )
    
    print(f"  Saved: {fname}")
    return True


def plot_all_datasets(period='weekly', figsize=(14, 6)):
    """Plot all datasets and comparisons."""
    print("=" * 60)
    print("PLOTTING ZONE PROBABILITIES")
    print("=" * 60)
    
    
    
    # Plot individual datasets
    print("\nPlotting individual datasets...")
    for dataset_id in DATASET_CONFIGS.keys():
        plot_dataset(dataset_id, period, figsize=figsize)
    
    # Plot comparisons (all climate-adjusted vs stationary)
    if 'stationary_ensemble' in DATASET_CONFIGS:
        print("\nPlotting comparisons vs stationary...")
        
        for dataset_id in DATASET_CONFIGS.keys():
            if dataset_id == 'stationary_ensemble':
                continue

            plot_comparison('stationary_ensemble', dataset_id, period, figsize=figsize)

    print("\n" + "=" * 60)
    print(f"All plots saved to {FIG_OUTPUT_DIR}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        print(f"\nAvailable datasets: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)
    
    arg = sys.argv[1]
    period = 'weekly'
    
    figsize = (10, 8)
    if arg == '--all':
        plot_all_datasets(period, figsize=figsize)
    else:
        dataset_id = arg
        verify_dataset_id(dataset_id)
        
        print("=" * 60)
        print(f"PLOTTING ZONE PROBABILITIES: {dataset_id}")
        print("=" * 60)
        
        # Plot this dataset
        success = plot_dataset(dataset_id, period, figsize=figsize)
        
        # If not stationary, also plot comparison
        if success and dataset_id != 'stationary_ensemble':
            print()
            plot_comparison('stationary_ensemble', dataset_id, period, figsize=figsize)
        
        print("=" * 60)
        print("Done!")


if __name__ == "__main__":
    main()