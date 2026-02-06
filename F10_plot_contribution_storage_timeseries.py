"""
F10: Contribution-Storage Timeseries Figure.

Shows the temporal evolution of NYC contribution/inflow ratio with color
encoding the aggregate NYC storage level.

Layout:
  - X-axis: Day of water year (Jun 1 = day 1, Jun-May water year)
  - Y-axis: Contribution/inflow ratio (rolling n-day sum, %)
  - Color: NYC storage level (% of total capacity)

Each water year is plotted as its own line, with color varying along the line
based on storage level at each time point.

Usage:
    python F10_plot_contribution_storage_timeseries.py [dataset_id]
    python F10_plot_contribution_storage_timeseries.py --multipanel
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from methods.config import (
    FIG_DIR, DATASET_CONFIGS, NYC_RESERVOIRS, NYC_TOTAL_CAPACITY,
    verify_dataset_id,
)
from methods.plotting.styles import DPI_HIGH, DATASET_LABELS

# Output directory
FIG_OUTPUT_DIR = f"{FIG_DIR}/F10_contribution_storage_timeseries"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Rolling window for contribution/inflow ratio (days)
ROLLING_WINDOW_DAYS = 7

# Minimum inflow threshold to avoid division issues (MG)
MIN_INFLOW_THRESHOLD = 100

# Filter: only plot years where minimum storage falls at or below this threshold (%)
# Set to None to plot all years, or a value like 20 to focus on drought years
MIN_STORAGE_THRESHOLD = 20

# Sampling: set to None to plot all years, or an integer to randomly sample
# With filtering by min storage, this may not be needed
MAX_YEARS_TO_PLOT = None

# Color map for storage
STORAGE_CMAP = 'RdYlBu'  # Red (low) -> Yellow (mid) -> Blue (high)
STORAGE_VMIN = 0
STORAGE_VMAX = 100

# Line styling for individual year traces
LINE_WIDTH = 0.5
LINE_ALPHA = 0.5


# ============================================================================
# DATA PROCESSING
# ============================================================================

def get_water_year_day(dates):
    """Convert dates to water year day (Jun 1 = day 1)."""
    # Water year starts Jun 1
    month = dates.month
    day = dates.day

    # Days since Jun 1
    # Jun=1-30 (days 1-30), Jul=31-61, Aug=62-92, Sep=93-122, Oct=123-153, etc.
    days_in_month = [30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 31]  # Jun-May
    month_to_wy_month = {6: 0, 7: 1, 8: 2, 9: 3, 10: 4, 11: 5,
                         12: 6, 1: 7, 2: 8, 3: 9, 4: 10, 5: 11}

    cumulative_days = [0]
    for d in days_in_month[:-1]:
        cumulative_days.append(cumulative_days[-1] + d)

    wy_day = np.zeros(len(dates), dtype=int)
    for i, (m, d) in enumerate(zip(month, day)):
        wy_month = month_to_wy_month[m]
        wy_day[i] = cumulative_days[wy_month] + d

    return wy_day


def get_water_year(date):
    """Get water year for a date (Jun-May)."""
    if date.month >= 6:
        return date.year + 1
    return date.year


def calculate_yearly_traces(data, dataset_id, rolling_days=7, min_storage_threshold=None):
    """
    Calculate rolling contribution/inflow ratio for all realizations,
    organized by individual water years.

    Parameters
    ----------
    data : pywrdrb.Data
        Loaded data object.
    dataset_id : str
        Dataset identifier.
    rolling_days : int
        Rolling window for ratio calculation.
    min_storage_threshold : float or None
        If provided, only include years where minimum storage falls at or below
        this percentage. E.g., 20 means only years with min storage <= 20%.

    Returns
    -------
    list of dicts, each containing:
        - realization_id
        - water_year
        - wy_days: array of water year days (1-365, Jun 1 = day 1)
        - ratios: array of contribution/inflow ratios
        - storages: array of storage percentages
        - min_storage: minimum storage during that year
    """
    yearly_traces = []

    realization_ids = list(data.contribution[dataset_id].keys())

    for real_id in realization_ids:
        # Get contribution data
        contribution_df = data.contribution[dataset_id][real_id]
        nyc_contribution = contribution_df['mrf_montagueTrenton_nyc']

        # Get inflow data
        inflow_df = data.inflow[dataset_id][real_id]
        nyc_inflow = inflow_df[NYC_RESERVOIRS].sum(axis=1)

        # Get storage data
        storage_df = data.res_storage[dataset_id][real_id]
        nyc_storage = storage_df[NYC_RESERVOIRS].sum(axis=1)
        storage_pct = (nyc_storage / NYC_TOTAL_CAPACITY) * 100

        # Align indices
        common_idx = nyc_contribution.index.intersection(nyc_inflow.index).intersection(storage_pct.index)
        nyc_contribution = nyc_contribution.loc[common_idx]
        nyc_inflow = nyc_inflow.loc[common_idx]
        storage_pct = storage_pct.loc[common_idx]

        # Rolling sums
        contribution_rolling = nyc_contribution.rolling(rolling_days, min_periods=1).sum()
        inflow_rolling = nyc_inflow.rolling(rolling_days, min_periods=1).sum()

        # Calculate ratio (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(
                inflow_rolling > MIN_INFLOW_THRESHOLD,
                (contribution_rolling / inflow_rolling) * 100,
                np.nan
            )

        # Get water year day and water year
        wy_day = get_water_year_day(common_idx)
        water_years = np.array([get_water_year(d) for d in common_idx])

        # Group by water year
        unique_wys = np.unique(water_years)
        for wy in unique_wys:
            mask = water_years == wy
            wy_days = wy_day[mask]
            wy_ratios = ratio[mask]
            wy_storages = storage_pct.values[mask]

            # Filter out NaN ratios
            valid_mask = ~np.isnan(wy_ratios)
            if valid_mask.sum() < 30:  # Skip years with too few valid days
                continue

            # Calculate minimum storage for this year
            min_storage = np.nanmin(wy_storages[valid_mask])

            # Apply minimum storage filter if specified
            if min_storage_threshold is not None and min_storage > min_storage_threshold:
                continue

            # Sort by water year day
            sort_idx = np.argsort(wy_days[valid_mask])

            yearly_traces.append({
                'realization_id': real_id,
                'water_year': wy,
                'wy_days': wy_days[valid_mask][sort_idx],
                'ratios': wy_ratios[valid_mask][sort_idx],
                'storages': wy_storages[valid_mask][sort_idx],
                'min_storage': min_storage,
            })

    return yearly_traces


# ============================================================================
# PLOTTING
# ============================================================================

def create_colored_line(ax, x, y, colors, cmap, vmin, vmax, linewidth=2, alpha=1.0):
    """
    Create a line where color varies along the line based on `colors` array.

    Uses LineCollection for efficient rendering.
    """
    # Create line segments
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Normalize colors
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Create LineCollection
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    lc.set_array(colors[:-1])  # Color for each segment

    ax.add_collection(lc)
    return lc


def plot_contribution_storage_timeseries(
    dataset_id,
    rolling_days=ROLLING_WINDOW_DAYS,
    figsize=(12, 6),
    max_years=MAX_YEARS_TO_PLOT,
    min_storage_threshold=MIN_STORAGE_THRESHOLD,
):
    """
    Create the contribution-storage timeseries figure.

    Each water year is plotted as its own line, with color varying
    based on storage at each time point.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier.
    rolling_days : int
        Rolling window for contribution/inflow ratio.
    figsize : tuple
        Figure size.
    max_years : int or None
        Maximum number of water years to plot. If None, plot all.
    min_storage_threshold : float or None
        Only plot years where min storage <= this value (%).
    """
    verify_dataset_id(dataset_id)

    # Load data
    print(f"Loading data for {dataset_id}...")
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Data not found: {fname}")

    data = pywrdrb.Data()
    data.load_from_export(fname, results_sets=['contribution', 'inflow', 'res_storage'])

    # Calculate yearly traces
    print("Calculating yearly traces...")
    yearly_traces = calculate_yearly_traces(data, dataset_id, rolling_days,
                                            min_storage_threshold=min_storage_threshold)
    filter_msg = f" (min storage <= {min_storage_threshold}%)" if min_storage_threshold else ""
    print(f"  {len(yearly_traces)} water years{filter_msg} from {len(set(t['realization_id'] for t in yearly_traces))} realizations")

    # Sample if needed
    if max_years is not None and len(yearly_traces) > max_years:
        print(f"  Sampling {max_years} years for plotting...")
        np.random.seed(42)
        indices = np.random.choice(len(yearly_traces), max_years, replace=False)
        yearly_traces = [yearly_traces[i] for i in indices]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each year as a colored line
    for trace in yearly_traces:
        x = trace['wy_days']
        y = trace['ratios']
        colors = trace['storages']

        if len(x) > 1:
            create_colored_line(
                ax, x, y, colors,
                cmap=STORAGE_CMAP, vmin=STORAGE_VMIN, vmax=STORAGE_VMAX,
                linewidth=LINE_WIDTH, alpha=LINE_ALPHA
            )

    # Set axis limits
    all_ratios = np.concatenate([t['ratios'] for t in yearly_traces])
    ax.set_xlim(1, 365)
    ax.set_ylim(0, np.nanpercentile(all_ratios, 99.5))

    # X-axis: water year months (Jun-May)
    month_starts = [1, 31, 62, 93, 123, 154, 184, 215, 246, 274, 305, 335]
    month_labels = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',
                    'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels, fontsize=10)
    ax.set_xlabel('Month (Jun-May Water Year)', fontsize=12)

    # Y-axis
    ax.set_ylabel(f'NYC Contribution / Inflow\n({rolling_days}-day rolling, %)', fontsize=12)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=STORAGE_CMAP,
                                norm=mcolors.Normalize(vmin=STORAGE_VMIN, vmax=STORAGE_VMAX))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('NYC Storage (%)', fontsize=11)

    # Title annotation
    dataset_label = DATASET_LABELS.get(dataset_id, dataset_id)
    if min_storage_threshold is not None:
        annotation = f'{dataset_label}\n{len(yearly_traces)} years (min storage <= {min_storage_threshold}%)'
    else:
        annotation = f'{dataset_label}\n({len(yearly_traces)} water years)'
    ax.text(0.02, 0.98, annotation,
            transform=ax.transAxes, fontsize=11,
            va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save
    suffix = f"_minStorage{min_storage_threshold}" if min_storage_threshold else ""
    fname_out = f"{FIG_OUTPUT_DIR}/F10_{dataset_id}_contribution_storage_ts{suffix}.png"
    plt.savefig(fname_out, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname_out}")

    plt.close()
    return fig, ax


def plot_multipanel_comparison(figsize=(14, 12), max_years=MAX_YEARS_TO_PLOT,
                               min_storage_threshold=MIN_STORAGE_THRESHOLD):
    """
    Create a 3-panel comparison showing all datasets.

    Each panel shows individual water year traces colored by storage.
    """
    datasets = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Store data for plotting
    all_traces = {}

    for dataset_id in datasets:
        verify_dataset_id(dataset_id)
        fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
        if not os.path.exists(fname):
            print(f"Skipping {dataset_id}: data not found")
            continue

        print(f"Processing {dataset_id}...")
        data = pywrdrb.Data()
        data.load_from_export(fname, results_sets=['contribution', 'inflow', 'res_storage'])

        traces = calculate_yearly_traces(data, dataset_id, ROLLING_WINDOW_DAYS,
                                         min_storage_threshold=min_storage_threshold)

        # Sample if needed
        if max_years is not None and len(traces) > max_years:
            np.random.seed(42)
            indices = np.random.choice(len(traces), max_years, replace=False)
            traces = [traces[i] for i in indices]

        all_traces[dataset_id] = traces
        filter_msg = f" (min storage <= {min_storage_threshold}%)" if min_storage_threshold else ""
        print(f"  {len(traces)} water years{filter_msg}")

    # Find global y-max for consistent scaling
    y_max = 0
    for traces in all_traces.values():
        for t in traces:
            y_max = max(y_max, np.nanpercentile(t['ratios'], 99.5))

    # Plot each dataset
    for i, (dataset_id, ax) in enumerate(zip(datasets, axes)):
        if dataset_id not in all_traces:
            ax.text(0.5, 0.5, f'{dataset_id}: No data', transform=ax.transAxes,
                   ha='center', va='center')
            continue

        traces = all_traces[dataset_id]

        # Plot each year as a colored line
        for trace in traces:
            x = trace['wy_days']
            y = trace['ratios']
            colors = trace['storages']

            if len(x) > 1:
                create_colored_line(
                    ax, x, y, colors,
                    cmap=STORAGE_CMAP, vmin=STORAGE_VMIN, vmax=STORAGE_VMAX,
                    linewidth=LINE_WIDTH, alpha=LINE_ALPHA
                )

        ax.set_xlim(1, 365)
        ax.set_ylim(0, y_max)

        # Label
        dataset_label = DATASET_LABELS.get(dataset_id, dataset_id)
        ax.text(0.02, 0.95, f'{dataset_label}\n({len(traces)} years)',
                transform=ax.transAxes, fontsize=11,
                va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_ylabel('Contribution/Inflow (%)', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

    # X-axis labels on bottom panel only (Jun-May)
    month_starts = [1, 31, 62, 93, 123, 154, 184, 215, 246, 274, 305, 335]
    month_labels = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',
                    'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']
    axes[-1].set_xticks(month_starts)
    axes[-1].set_xticklabels(month_labels, fontsize=10)
    axes[-1].set_xlabel('Month (Jun-May Water Year)', fontsize=12)

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=STORAGE_CMAP,
                                norm=mcolors.Normalize(vmin=STORAGE_VMIN, vmax=STORAGE_VMAX))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label('NYC Storage (%)', fontsize=11)

    # Add filter info as figure annotation
    if min_storage_threshold is not None:
        fig.text(0.5, 0.01, f'Years with minimum storage <= {min_storage_threshold}%',
                 ha='center', fontsize=10, style='italic')

    plt.tight_layout()

    suffix = f"_minStorage{min_storage_threshold}" if min_storage_threshold else ""
    fname_out = f"{FIG_OUTPUT_DIR}/F10_multipanel_contribution_storage_ts{suffix}.png"
    plt.savefig(fname_out, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname_out}")

    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        if sys.argv[1] == '--multipanel':
            plot_multipanel_comparison()
        else:
            dataset_id = sys.argv[1]
            verify_dataset_id(dataset_id)
            plot_contribution_storage_timeseries(dataset_id)
    else:
        # Default: generate multipanel comparison
        plot_multipanel_comparison()


if __name__ == "__main__":
    main()
