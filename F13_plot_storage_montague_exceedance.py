"""
Storage and Montague Release Exceedance Figure.

2-panel figure showing:
  - Top: Storage exceedance curve (CDF) for annual minimum storage
  - Bottom: NYC releases to Montague summed across the year

Each CDF reflects a given percentile of the ensemble (e.g., median, 10th percentile).
The x-axis ordering is determined by one metric (default: minimum storage).

Usage:
  python F_plot_storage_montague_exceedance.py [ensemble_percentile] [--sort-by montague]

Examples:
  python F_plot_storage_montague_exceedance.py 0.5  # Plot median across realizations
  python F_plot_storage_montague_exceedance.py 0.1  # Plot 10th percentile (worst)
  python F_plot_storage_montague_exceedance.py 0.5 --sort-by montague  # Sort by Montague instead of storage
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from methods.config import (
    FIG_DIR, N_YEARS, NYC_RESERVOIRS, NYC_TOTAL_CAPACITY, ROOT_DIR,
    RECONSTRUCTION_OUTPUT_FNAME
)
from methods.plotting.styles import (
    DATASET_COLORS, DATASET_LINESTYLES, DATASET_LABELS,
    HISTORIC_COLOR, HISTORIC_LABEL,
    LINEWIDTH_MEDIUM, DPI_HIGH,
    FONTSIZE_SMALL, FONTSIZE_MEDIUM, FONTSIZE_LARGE,
)

# Output directory
FIG_OUTPUT_DIR = f"{FIG_DIR}/F_storage_montague_exceedance"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

# All datasets to compare
ALL_DATASETS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']


def load_dataset(dataset_id):
    """
    Load postprocessed data for a dataset.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier

    Returns
    -------
    data : pywrdrb.Data
        Data object with res_storage and contribution loaded
    """
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'

    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Postprocessed data not found: {fname}\n"
            f"Run 04_postprocess_data.py first!"
        )

    print(f"  Loading {dataset_id} from {fname}")
    data = pywrdrb.Data()
    data.load_from_export(
        fname,
        results_sets=['res_storage', 'contribution']
    )

    return data


def load_historic_storage():
    """
    Load historic/reconstruction storage data from pub_nhmv10_BC_withObsScaled simulation.

    Returns
    -------
    annual_min_storage : np.ndarray
        Array of annual minimum storage percentages for historic period
    annual_montague_releases : np.ndarray
        Array of annual total Montague releases for historic period
    """
    if not os.path.exists(RECONSTRUCTION_OUTPUT_FNAME):
        print(f"  Warning: Historic data not found at {RECONSTRUCTION_OUTPUT_FNAME}")
        return None, None

    print(f"  Loading historic data from {RECONSTRUCTION_OUTPUT_FNAME}")
    historic_data = pywrdrb.Data(results_sets=['res_storage', 'nyc_release_components'])
    historic_data.load_output(
        output_filenames=[RECONSTRUCTION_OUTPUT_FNAME],
        results_sets=['res_storage', 'nyc_release_components']
    )

    # Get reconstruction data (should be keyed by filename stem)
    file_stem = os.path.splitext(os.path.basename(RECONSTRUCTION_OUTPUT_FNAME))[0]
    dataset_key = file_stem

    # Should have realization 0
    realization_key = 0

    # Calculate NYC storage percentage
    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']
    nyc_storage = historic_data.res_storage[dataset_key][realization_key][nyc_reservoirs].sum(axis=1)
    nyc_storage_pct = 100.0 * nyc_storage / NYC_TOTAL_CAPACITY

    # Calculate NYC→Montague contribution from release components
    # (same logic as in postprocess.py _compute_contribution)
    release_components = historic_data.nyc_release_components[dataset_key][realization_key]
    contribution_columns = [f'mrf_montagueTrenton_{res}' for res in nyc_reservoirs]
    montague_releases = release_components.loc[:, contribution_columns].sum(axis=1)

    # Calculate annual metrics
    annual_min_storage = nyc_storage_pct.resample('YS').min().values
    annual_montague_total = montague_releases.resample('YS').sum().values

    print(f"    Loaded {len(annual_min_storage)} years of historic data")

    return annual_min_storage, annual_montague_total


def compute_annual_metrics(data, dataset_id):
    """
    Compute annual minimum storage and annual total Montague releases for all realizations.

    Parameters
    ----------
    data : pywrdrb.Data
        Data object with res_storage and contribution
    dataset_id : str
        Dataset identifier

    Returns
    -------
    annual_metrics : dict
        Dictionary mapping realization_id to DataFrame with columns:
        - year: int
        - min_storage_pct: float (minimum NYC storage percentage for that year)
        - montague_release_total: float (total NYC→Montague releases for that year, MG)
    """
    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']
    realizations = sorted(data.res_storage[dataset_id].keys())

    annual_metrics = {}

    for r in realizations:
        # Calculate NYC storage percentage
        nyc_storage = data.res_storage[dataset_id][r][nyc_reservoirs].sum(axis=1)
        nyc_storage_pct = 100.0 * nyc_storage / NYC_TOTAL_CAPACITY

        # Get NYC→Montague releases
        montague_releases = data.contribution[dataset_id][r]['mrf_montagueTrenton_nyc']

        # Calculate annual metrics
        annual_min_storage = nyc_storage_pct.resample('YS').min()
        annual_montague_total = montague_releases.resample('YS').sum()

        # Combine into single DataFrame
        metrics_df = annual_min_storage.to_frame(name='min_storage_pct')
        metrics_df['montague_release_total'] = annual_montague_total
        metrics_df['year'] = metrics_df.index.year
        metrics_df = metrics_df.reset_index(drop=True)

        annual_metrics[r] = metrics_df

    return annual_metrics


def extract_ensemble_percentile_series(annual_metrics, metric, ensemble_percentile):
    """
    Extract a specific percentile across realizations for each year.

    For each year, computes the specified percentile across all realizations.

    Parameters
    ----------
    annual_metrics : dict
        Dictionary mapping realization_id to annual metrics DataFrame
    metric : str
        Metric name ('min_storage_pct' or 'montague_release_total')
    ensemble_percentile : float
        Percentile to extract (0.0 to 1.0, e.g., 0.5 for median)

    Returns
    -------
    percentile_series : np.ndarray
        Array of metric values, one per year, representing the specified percentile
    """
    # Get number of years from first realization
    n_years = len(annual_metrics[list(annual_metrics.keys())[0]])
    n_realizations = len(annual_metrics)

    # Create matrix: rows = realizations, columns = years
    metric_matrix = np.zeros((n_realizations, n_years))

    for i, (r, df) in enumerate(annual_metrics.items()):
        metric_matrix[i, :] = df[metric].values

    # Compute percentile across realizations (axis=0) for each year
    percentile_series = np.percentile(metric_matrix, ensemble_percentile * 100, axis=0)

    return percentile_series


def plot_storage_montague_exceedance(
    ensemble_percentile=0.5,
    sort_by='storage',
    datasets=None,
    figsize=None,
    fname=None,
):
    """
    Create 2-panel exceedance figure for storage and Montague releases.

    Parameters
    ----------
    ensemble_percentile : float
        Percentile of the ensemble to plot (0.0 to 1.0)
        - 0.5 = median across realizations
        - 0.1 = 10th percentile (worst performing)
        - 0.9 = 90th percentile (best performing)
    sort_by : str
        Metric to use for x-axis ordering ('storage' or 'montague')
        Default: 'storage' (sort by minimum storage)
    datasets : list of str, optional
        List of dataset IDs to plot. Default: ALL_DATASETS
    figsize : tuple, optional
        Figure size. Default: (8, 10)
    fname : str, optional
        Output filename. Auto-generated if None.

    Returns
    -------
    fig, axes
    """
    if datasets is None:
        datasets = ALL_DATASETS

    if figsize is None:
        figsize = (8, 10)

    # Validate parameters
    if not 0.0 <= ensemble_percentile <= 1.0:
        raise ValueError(f"ensemble_percentile must be between 0 and 1, got {ensemble_percentile}")

    if sort_by not in ['storage', 'montague']:
        raise ValueError(f"sort_by must be 'storage' or 'montague', got {sort_by}")

    print(f"\nGenerating storage-Montague exceedance figure...")
    print(f"  Ensemble percentile: {ensemble_percentile*100:.0f}%")
    print(f"  Sort by: {sort_by}")
    print(f"  Datasets: {datasets}")

    # ------------------------------------------------------------------
    # Load data for all datasets
    # ------------------------------------------------------------------
    dataset_annual_metrics = {}

    for dataset_id in datasets:
        print(f"\n  Loading {dataset_id}...")
        data = load_dataset(dataset_id)
        annual_metrics = compute_annual_metrics(data, dataset_id)
        dataset_annual_metrics[dataset_id] = annual_metrics

    # Load historic data
    print(f"\n  Loading historic data...")
    historic_storage, historic_montague = load_historic_storage()

    # ------------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    ax_storage, ax_montague = axes

    # ------------------------------------------------------------------
    # Plot each dataset
    # ------------------------------------------------------------------
    for dataset_id in datasets:
        print(f"\n  Processing {dataset_id}...")
        annual_metrics = dataset_annual_metrics[dataset_id]

        # Extract percentile series for both metrics
        storage_series = extract_ensemble_percentile_series(
            annual_metrics, 'min_storage_pct', ensemble_percentile
        )
        montague_series = extract_ensemble_percentile_series(
            annual_metrics, 'montague_release_total', ensemble_percentile
        )

        # Sort by selected metric
        if sort_by == 'storage':
            sort_idx = np.argsort(storage_series)  # Ascending order
        else:  # sort_by == 'montague'
            sort_idx = np.argsort(montague_series)  # Ascending order

        storage_sorted = storage_series[sort_idx]
        montague_sorted = montague_series[sort_idx]

        # Create exceedance probabilities (fraction of years)
        n_years = len(storage_sorted)
        exceedance_prob = np.arange(1, n_years + 1) / n_years

        # Get plotting style
        color = DATASET_COLORS.get(dataset_id, '#808080')
        linestyle = DATASET_LINESTYLES.get(dataset_id, '-')
        label = DATASET_LABELS.get(dataset_id, dataset_id)

        # Plot storage CDF (swap axes: exceedance on x, values on y)
        ax_storage.plot(
            exceedance_prob, storage_sorted,
            color=color, linestyle=linestyle,
            linewidth=LINEWIDTH_MEDIUM,
            label=label,
            zorder=5
        )

        # Plot Montague CDF (swap axes: exceedance on x, values on y)
        ax_montague.plot(
            exceedance_prob, montague_sorted,
            color=color, linestyle=linestyle,
            linewidth=LINEWIDTH_MEDIUM,
            label=label,
            zorder=5
        )

    # ------------------------------------------------------------------
    # Plot historic data as reference (horizontal line at specified percentile)
    # ------------------------------------------------------------------
    if historic_storage is not None and historic_montague is not None:
        # Calculate the specified percentile from historic annual values
        historic_storage_percentile = np.percentile(historic_storage, ensemble_percentile * 100)
        historic_montague_percentile = np.percentile(historic_montague, ensemble_percentile * 100)

        # Plot historic storage as horizontal line
        ax_storage.axhline(
            historic_storage_percentile,
            color=HISTORIC_COLOR, linestyle='-',
            linewidth=LINEWIDTH_MEDIUM, alpha=0.8,
            label=HISTORIC_LABEL, zorder=10
        )

        # Plot historic Montague as horizontal line
        ax_montague.axhline(
            historic_montague_percentile,
            color=HISTORIC_COLOR, linestyle='-',
            linewidth=LINEWIDTH_MEDIUM, alpha=0.8,
            label=HISTORIC_LABEL, zorder=10
        )

    # ------------------------------------------------------------------
    # Format storage panel (top)
    # ------------------------------------------------------------------
    ax_storage.set_xlabel('Exceedance Probability', fontsize=FONTSIZE_MEDIUM)
    ax_storage.set_ylabel('Annual Minimum Storage (%)', fontsize=FONTSIZE_MEDIUM)
    ax_storage.set_title(
        f'Annual Minimum Storage\n(Ensemble Percentile: {ensemble_percentile*100:.0f}%)',
        fontsize=FONTSIZE_MEDIUM, pad=10
    )

    # Add reference lines for critical storage levels
    for level, color_line, label in [(30, 'orange', '30%'), (20, 'red', '20%')]:
        ax_storage.axhline(level, color=color_line, linestyle='--',
                          linewidth=1, alpha=0.5, zorder=1)
        ax_storage.text(0.98, level, label,
                       transform=ax_storage.get_yaxis_transform(),
                       fontsize=FONTSIZE_SMALL-1, color=color_line,
                       ha='right', va='bottom')

    ax_storage.set_xlim(0, 1)
    ax_storage.set_ylim(bottom=0, top=100)
    ax_storage.grid(True, color='gray', alpha=0.2, linewidth=0.5, linestyle='--')
    ax_storage.set_axisbelow(True)
    ax_storage.tick_params(labelsize=FONTSIZE_SMALL)
    ax_storage.spines['top'].set_visible(False)
    ax_storage.spines['right'].set_visible(False)

    # Panel label
    ax_storage.text(
        0.02, 0.98, '(a)',
        transform=ax_storage.transAxes,
        fontsize=FONTSIZE_MEDIUM,
        va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none')
    )

    # ------------------------------------------------------------------
    # Format Montague panel (bottom)
    # ------------------------------------------------------------------
    ax_montague.set_xlabel('Exceedance Probability', fontsize=FONTSIZE_MEDIUM)
    ax_montague.set_ylabel('Annual NYC Releases to Montague (MG)', fontsize=FONTSIZE_MEDIUM)
    ax_montague.set_title(
        f'Annual NYC Releases to Montague\n(Ensemble Percentile: {ensemble_percentile*100:.0f}%)',
        fontsize=FONTSIZE_MEDIUM, pad=10
    )

    ax_montague.set_xlim(0, 1)
    ax_montague.set_ylim(bottom=0)
    ax_montague.grid(True, color='gray', alpha=0.2, linewidth=0.5, linestyle='--')
    ax_montague.set_axisbelow(True)
    ax_montague.tick_params(labelsize=FONTSIZE_SMALL)
    ax_montague.spines['top'].set_visible(False)
    ax_montague.spines['right'].set_visible(False)

    # Panel label
    ax_montague.text(
        0.02, 0.98, '(b)',
        transform=ax_montague.transAxes,
        fontsize=FONTSIZE_MEDIUM,
        va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none')
    )

    # ------------------------------------------------------------------
    # Add legend
    # ------------------------------------------------------------------
    ax_storage.legend(
        loc='lower left',
        frameon=True,
        fontsize=FONTSIZE_SMALL,
        framealpha=0.9
    )

    # ------------------------------------------------------------------
    # Add sorting info text
    # ------------------------------------------------------------------
    sort_text = f"Years sorted by: {'Minimum Storage' if sort_by == 'storage' else 'Montague Releases'}"
    fig.text(
        0.5, 0.01, sort_text,
        ha='center', fontsize=FONTSIZE_SMALL,
        style='italic', color='gray'
    )

    # ------------------------------------------------------------------
    # Layout and save
    # ------------------------------------------------------------------
    plt.tight_layout(rect=[0, 0.02, 1, 1])

    if fname is None:
        percentile_str = f"{int(ensemble_percentile*100):02d}"
        fname = f"{FIG_OUTPUT_DIR}/storage_montague_exceedance_p{percentile_str}_sortby_{sort_by}.png"

    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"\nSaved: {fname}")

    return fig, axes


def main():
    """Generate storage-Montague exceedance figure."""
    # Parse command line arguments
    ensemble_percentile = 0.5  # Default: median
    sort_by = 'storage'  # Default: sort by minimum storage

    if len(sys.argv) > 1:
        try:
            ensemble_percentile = float(sys.argv[1])
            if not 0.0 <= ensemble_percentile <= 1.0:
                raise ValueError
        except ValueError:
            print(f"ERROR: ensemble_percentile must be between 0.0 and 1.0, got {sys.argv[1]}")
            sys.exit(1)

    if len(sys.argv) > 2:
        if sys.argv[2] == '--sort-by' and len(sys.argv) > 3:
            sort_by = sys.argv[3]
            if sort_by not in ['storage', 'montague']:
                print(f"ERROR: sort_by must be 'storage' or 'montague', got {sort_by}")
                sys.exit(1)

    print(f"Storage-Montague Exceedance Figure")
    print(f"  Ensemble Percentile: {ensemble_percentile*100:.0f}%")
    print(f"  Sort By: {sort_by}")

    plot_storage_montague_exceedance(
        ensemble_percentile=ensemble_percentile,
        sort_by=sort_by
    )

    plt.close('all')


if __name__ == "__main__":
    main()
