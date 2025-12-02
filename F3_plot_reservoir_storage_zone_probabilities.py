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
from matplotlib.colors import TwoSlopeNorm, LogNorm, BoundaryNorm, ListedColormap
from matplotlib import colors
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from methods.config import *

# Default probability bins (percent) - log-scale discrete bins
# Designed to emphasize low probability values (<5%) while covering full range
DEFAULT_PROB_BINS = [0.01, 0.1,  1, 5, 10, 25, 50, 75, 99]


def create_discrete_colormap(bin_edges, base_cmap='magma_r'):
    """
    Create a discrete colormap with log-scale spacing for better low-value visibility.

    The colormap samples colors at log-scaled positions, giving more visual
    distinction to low probability values (e.g., <5%) while still representing
    the full range up to 99%.

    Parameters
    ----------
    bin_edges : list or array
        Bin edges (e.g., [0.1, 1, 5, 10, 25, 50, 75, 99])
    base_cmap : str or colormap
        Base colormap to sample from

    Returns
    -------
    cmap : ListedColormap
        Discrete colormap with colors sampled at log-scaled positions
    norm : BoundaryNorm
        Normalization for the discrete bins
    """
    bin_edges = np.array(bin_edges)
    n_bins = len(bin_edges) - 1

    # Use log10 transform for color sampling positions
    # This gives more visual separation to low values
    log_edges = np.log10(np.maximum(bin_edges, 0.01))

    # Normalize log-transformed edges to [0, 1] for colormap sampling
    log_min = log_edges[0]
    log_max = log_edges[-1]
    normalized_positions = (log_edges - log_min) / (log_max - log_min)

    # Sample colors at the midpoint of each bin (in log space)
    # This gives each bin a representative color
    bin_midpoints = 0.5 * (normalized_positions[:-1] + normalized_positions[1:])

    # Sample colors from base colormap
    base_cmap_obj = plt.get_cmap(base_cmap)
    colors_list = [base_cmap_obj(pos) for pos in bin_midpoints]

    # Create discrete colormap
    cmap = ListedColormap(colors_list)
    cmap.set_bad(color='#f0f0f0')

    # Create boundary normalization
    norm = BoundaryNorm(boundaries=bin_edges, ncolors=n_bins, extend='neither')

    return cmap, norm


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


def _period_index(dts: pd.DatetimeIndex, period: str = 'daily', origin: str = 'jan1') -> np.ndarray:
    """Map dates to generic-year period index."""
    dts = pd.DatetimeIndex(dts)

    if origin == 'june1':
        # Water year starting June 1
        june1_this = pd.to_datetime(dts.year.astype(str) + '-06-01')
        is_after = dts >= june1_this
        june1_prev = pd.to_datetime((dts.year - 1).astype(str) + '-06-01')

        doy = np.where(is_after,
                       (dts - june1_this).days + 1,
                       (dts - june1_prev).days + 1)

        if period == 'monthly':
            return ((dts.month - 6) % 12) + 1
    else:  # origin == 'jan1'
        # Calendar year starting January 1
        jan1_this = pd.to_datetime(dts.year.astype(str) + '-01-01')
        doy = (dts - jan1_this).days + 1

        if period == 'monthly':
            return dts.month

    if period == 'daily':
        return doy
    elif period == 'weekly':
        return ((doy - 1) // 7) + 1


def build_y_edges_grid(ffmp_boundaries, period='weekly', origin='jan1', pct_extents=(0.0, 100.0)):
    """
    Build Y-axis edges for pcolormesh from FFMP boundaries.

    Parameters
    ----------
    ffmp_boundaries : pd.DataFrame
        FFMP level boundaries
    period : str
        Time aggregation period
    origin : str
        Period origin: 'jan1' or 'june1'
    pct_extents : tuple
        Min/max storage percentage bounds

    Returns
    -------
    y_edges_grid : np.ndarray
        Shape (Z+1, P+1) for pcolormesh
    periods_sorted : np.ndarray
        Unique period values
    """
    thr_cols = get_ordered_threshold_columns(ffmp_boundaries)

    # Get period indices
    p_idx = _period_index(ffmp_boundaries.index, period=period, origin=origin)
    
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


def _plot_single_storage_panel(ax,
                                M,
                                X,
                                Y,
                                cmap,
                                norm,
                                origin='jan1',
                                show_zone_lines=True,
                                xlabel='',
                                ylabel='Total NYC storage (% of capacity)'):
    """
    Plot a single storage zone probability panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    M : np.ndarray
        Probability matrix (Z, P)
    X, Y : np.ndarray
        Meshgrid coordinates (Z+1, P+1)
    cmap : str or matplotlib colormap
        Colormap to use (can be string or colormap object)
    norm : matplotlib normalization
        Color normalization
    origin : str
        Period origin: 'jan1' or 'june1'
    show_zone_lines : bool
        Whether to show white zone boundary lines
    xlabel, ylabel : str
        Axis labels

    Returns
    -------
    pcolormesh
        The pcolormesh object for creating colorbars
    """
    # Handle both string and colormap object inputs
    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color='#f0f0f0')
    else:
        cmap_obj = cmap  # Already a colormap object with set_bad applied

    pcm = ax.pcolormesh(X, Y, M, cmap=cmap_obj, norm=norm, shading='flat')

    # Zone boundary lines
    if show_zone_lines:
        for j in range(Y.shape[1]):
            ax.plot(X[:, j], Y[:, j], color='white', linewidth=0.6, alpha=0.7)

    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(0, 100)

    # Set up monthly ticks based on origin
    # Approximate week at start of each month (4.33 weeks per month)
    month_week_starts = [1, 5, 9, 14, 18, 23, 27, 32, 36, 40, 45, 49]

    if origin == 'jan1':
        # Calendar year: Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    else:  # origin == 'june1'
        # Water year: Jun, Jul, Aug, Sep, Oct, Nov, Dec, Jan, Feb, Mar, Apr, May
        month_labels = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']

    ax.set_xticks(month_week_starts)
    ax.set_xticklabels(month_labels)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(labelsize=10)

    return pcm


def plot_storage_zone_probabilities(prob_df,
                                     ffmp_boundaries,
                                     period='weekly',
                                     origin=None,
                                     figsize=(14, 6),
                                     cmap='magma_r',
                                     prob_bins=None,
                                     title=None,
                                     fname=None):
    """
    Plot storage zone probability heatmap with discrete colormap.

    Parameters
    ----------
    prob_df : pd.DataFrame
        Zone probabilities with index=period, columns=zone_0, zone_1, ...
    ffmp_boundaries : pd.DataFrame
        FFMP level boundaries for computing Y edges
    period : str
        Time aggregation period
    origin : str, optional
        Period origin: 'jan1' or 'june1'. If None, uses PERIOD_ORIGIN from config
    prob_bins : list or array, optional
        Probability bin edges (percent). If None, uses DEFAULT_PROB_BINS
    """
    # Use configured origin if not specified
    if origin is None:
        origin = PERIOD_ORIGIN

    # Use default bins if not specified
    if prob_bins is None:
        prob_bins = DEFAULT_PROB_BINS

    # Build grids
    y_edges_grid, periods_sorted = build_y_edges_grid(ffmp_boundaries, period, origin)
    x_edges = build_x_edges(periods_sorted)

    # Align probability data to periods_sorted
    M = prob_df.loc[periods_sorted].to_numpy().T  # (Z, P)

    # Build meshgrid
    X = np.tile(x_edges, (y_edges_grid.shape[0], 1))  # (Z+1, P+1)
    Y = y_edges_grid  # (Z+1, P+1)

    # Create discrete colormap
    cmap_discrete, norm_discrete = create_discrete_colormap(prob_bins, base_cmap=cmap)

    # Plot using helper
    fig, ax = plt.subplots(figsize=figsize)

    quad = _plot_single_storage_panel(
        ax=ax,
        M=M,
        X=X,
        Y=Y,
        cmap=cmap_discrete,
        norm=norm_discrete,
        origin=origin,
        show_zone_lines=True,
        ylabel='Total NYC storage (% of capacity)'
    )

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
                                  origin=None,
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
    period : str
        Time aggregation period
    origin : str, optional
        Period origin: 'jan1' or 'june1'. If None, uses PERIOD_ORIGIN from config
    """
    # Use configured origin if not specified
    if origin is None:
        origin = PERIOD_ORIGIN

    # Build grids
    y_edges_grid, periods_sorted = build_y_edges_grid(ffmp_boundaries, period, origin)
    x_edges = build_x_edges(periods_sorted)

    # Align data
    M_ref = prob_df_ref.loc[periods_sorted].to_numpy().T  # (Z, P)
    M_comp = prob_df_comp.loc[periods_sorted].to_numpy().T  # (Z, P)

    # Calculate percentage difference
    eps = 1e-8
    prob_diff = 100.0 * (M_comp - M_ref) / np.maximum(M_ref, eps)

    # Build meshgrid
    X = np.tile(x_edges, (y_edges_grid.shape[0], 1))
    Y = y_edges_grid

    # Plot using helper
    fig, ax = plt.subplots(figsize=figsize)

    vmin = -100
    vmax = 100
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    pcm = _plot_single_storage_panel(
        ax=ax,
        M=prob_diff,
        X=X,
        Y=Y,
        cmap='BrBG_r',
        norm=norm,
        origin=origin,
        show_zone_lines=True,
        ylabel='Total NYC storage (% of capacity)'
    )

    if title:
        ax.set_title(title)

    cbar = plt.colorbar(pcm, ax=ax, pad=0.02, extend='both')
    cbar.set_label('Δ Probability (%)')

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
    return fig, ax


def plot_4panel_storage_comparison(period='weekly',
                                    origin=None,
                                    figsize=(14, 8),
                                    prob_bins=None,
                                    vmin_diff=-100,
                                    vmax_diff=100,
                                    fname=None):
    """
    Create a 3-panel comparison figure showing storage zone probabilities for selected scenarios.

    Layout:
    - Left panel: Stationary ensemble (absolute probability)
    - Right panels (stacked): Low, High climate scenarios (% difference from stationary)

    Parameters
    ----------
    period : str
        Time aggregation period ('weekly', 'monthly', 'daily')
    origin : str, optional
        Period origin: 'jan1' or 'june1'. If None, uses PERIOD_ORIGIN from config
    figsize : tuple
        Figure size in inches
    prob_bins : list or array, optional
        Probability bin edges (percent) for discrete colormap. If None, uses DEFAULT_PROB_BINS
    vmin_diff, vmax_diff : float
        Color scale limits for percentage difference (right panels)
    fname : str
        Output filename (if None, will auto-generate)
    """
    # Use configured origin if not specified
    if origin is None:
        origin = PERIOD_ORIGIN

    # Use default bins if not specified
    if prob_bins is None:
        prob_bins = DEFAULT_PROB_BINS

    print(f"\n{'='*60}")
    print("Creating 3-Panel Storage Zone Comparison Figure")
    print(f"{'='*60}")

    # Define datasets to plot (excluding climate_adjusted_medium)
    datasets = {
        'stationary_ensemble': 'Stationary',
        'climate_adjusted_low': 'Low',
        'climate_adjusted_high': 'High'
    }

    # Load zone probabilities for all datasets
    all_prob_dfs = {}
    for dataset_id, label in datasets.items():
        print(f"\nLoading {dataset_id} ({label})...")
        prob_df = load_zone_probabilities(dataset_id, period)
        if prob_df is None:
            print(f"ERROR: Could not load {dataset_id}")
            return None
        all_prob_dfs[dataset_id] = prob_df

    # Load FFMP boundaries
    ffmp_boundaries = load_ffmp_boundaries()

    # Build grids (same for all panels)
    y_edges_grid, periods_sorted = build_y_edges_grid(ffmp_boundaries, period, origin)
    x_edges = build_x_edges(periods_sorted)
    X = np.tile(x_edges, (y_edges_grid.shape[0], 1))  # (Z+1, P+1)
    Y = y_edges_grid  # (Z+1, P+1)

    # Calculate percentage differences for climate scenarios
    print(f"\n{'='*60}")
    print("Calculating percentage differences from stationary...")
    print(f"{'='*60}")

    M_ref = all_prob_dfs['stationary_ensemble'].loc[periods_sorted].to_numpy().T  # (Z, P)
    eps = 1e-8

    diff_matrices = {}
    for dataset_id in ['climate_adjusted_low', 'climate_adjusted_high']:
        M_comp = all_prob_dfs[dataset_id].loc[periods_sorted].to_numpy().T  # (Z, P)
        prob_diff = 100.0 * (M_comp - M_ref) / np.maximum(M_ref, eps)
        diff_matrices[dataset_id] = prob_diff

    print(f"\n{'='*60}")
    print("Creating multi-panel figure...")
    print(f"{'='*60}")

    # Set up figure with GridSpec
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
                          hspace=0.15, wspace=0.25,
                          left=0.08, right=0.95, top=0.95, bottom=0.12)

    # Create axes
    ax_stat = fig.add_subplot(gs[:, 0])  # Left panel spans both rows
    ax_low = fig.add_subplot(gs[0, 1])   # Top right
    ax_high = fig.add_subplot(gs[1, 1])  # Bottom right

    axes = [ax_stat, ax_low, ax_high]
    dataset_list = list(datasets.keys())

    # Set up colormaps and norms
    # Left panel: absolute probability (discrete colormap)
    cmap_abs, norm_abs = create_discrete_colormap(prob_bins, base_cmap='magma_r')

    # Right panels: percentage difference (diverging colormap - continuous)
    cmap_diff = 'BrBG_r'
    norm_diff = TwoSlopeNorm(vmin=vmin_diff, vcenter=0, vmax=vmax_diff)

    # Storage for pcolormeshes
    pm_abs = None
    pm_diff = None

    # Plot each panel
    for idx, (ax, dataset_id) in enumerate(zip(axes, dataset_list)):

        if idx == 0:  # Stationary panel (absolute values)
            M = all_prob_dfs[dataset_id].loc[periods_sorted].to_numpy().T  # (Z, P)

            pm_abs = _plot_single_storage_panel(
                ax=ax,
                M=M,
                X=X,
                Y=Y,
                cmap=cmap_abs,
                norm=norm_abs,
                show_zone_lines=True,
                ylabel='Total NYC storage (% of capacity)'
            )

        else:  # Climate scenario panels (percentage difference)
            M_diff = diff_matrices[dataset_id]

            pm_diff = _plot_single_storage_panel(
                ax=ax,
                M=M_diff,
                X=X,
                Y=Y,
                cmap=cmap_diff,
                norm=norm_diff,
                show_zone_lines=True,
                ylabel=''  # No ylabel for right panels
            )

    # Add two colorbars at bottom
    # Left colorbar for absolute probability
    cbar_abs_ax = fig.add_axes([0.08, 0.04, 0.35, 0.02])
    cbar_abs = fig.colorbar(pm_abs, cax=cbar_abs_ax, orientation='horizontal', extend='max')
    cbar_abs.set_label('Probability (%)', fontsize=11, fontweight='bold')
    cbar_abs.ax.tick_params(labelsize=9)

    # Right colorbar for percentage difference
    cbar_diff_ax = fig.add_axes([0.56, 0.04, 0.35, 0.02])
    cbar_diff = fig.colorbar(pm_diff, cax=cbar_diff_ax, orientation='horizontal', extend='both')
    cbar_diff.set_label('Δ Probability (%)', fontsize=11, fontweight='bold')
    cbar_diff.ax.tick_params(labelsize=9)

    # Save figure
    if fname is None:
        fname = f"{FIG_OUTPUT_DIR}/comparison_3panel_storage_zone_probabilities_{period}.png"

    plt.savefig(fname, dpi=400, bbox_inches='tight')
    # Also save vector version
    base = fname.rsplit('.', 1)[0]
    plt.savefig(f"{base}.svg", bbox_inches='tight')

    print(f"\nSaved: {fname}")
    print(f"Saved: {base}.svg")

    return fig, axes


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
        print("Special option: 'comparison' - generates 3-panel comparison figure")
        sys.exit(1)

    arg = sys.argv[1]
    period = 'weekly'

    figsize = (10, 8)

    # Handle special 'comparison' option for 3-panel figure
    if arg.lower() == 'comparison':
        print("=" * 60)
        print("PLOTTING 3-PANEL STORAGE ZONE COMPARISON")
        print("=" * 60)
        plot_4panel_storage_comparison(period=period)
        print("=" * 60)
        print("3-panel comparison figure completed successfully!")
        return

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