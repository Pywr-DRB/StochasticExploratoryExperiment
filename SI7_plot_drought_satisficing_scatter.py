"""
SI7: Drought Event Satisficing Scatter Plot

Creates scatter plots showing satisficing outcomes for each SSI-defined drought event.
- X-axis: Drought severity (minimum SSI)
- Y-axis: Drought magnitude (cumulative deficit)
- Marker color: Satisficing outcome category
- Marker size: Drought duration

Satisficing criteria:
  - Storage: NYC storage stays ≥ 20% throughout drought
  - Montague: Montague flow violations ≤ 3 consecutive days during drought

Pre-calculated satisficing data is loaded from pywrdrb/satisficing_analysis/.

Usage:
    python SI7_plot_drought_satisficing_scatter.py [ssi_window]
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

from methods.config import FIG_DIR, SSI_WINDOWS
from methods.plotting.styles import (
    DPI_HIGH, DATASET_COLORS, DATASET_LABELS,
    FONTSIZE_SMALL, FONTSIZE_MEDIUM, FONTSIZE_LABEL,
)

# Output directory
FIG_OUTPUT_DIR = f"{FIG_DIR}/SI7_drought_satisficing"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

# Pre-calculated satisficing data directory
SATISFICING_DATA_DIR = "./pywrdrb/satisficing_analysis"

# ============================================================================
# CONFIGURATION
# ============================================================================

# Satisficing thresholds (used for categorization)
STORAGE_THRESHOLD = 20.0       # Minimum storage % during drought
VIOLATION_DAYS_THRESHOLD = 3   # Max consecutive Montague violation days

# Datasets to analyze
DATASETS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']

# Scatter plot settings
MARKER_SIZE_SCALE = 3.0  # Scale factor for marker size based on duration
MIN_MARKER_SIZE = 20
MAX_MARKER_SIZE = 200

# Colors for satisficing categories
SATISFICING_COLORS = {
    'all_pass': '#2ca02c',        # Green - all criteria met
    'storage_fail': '#ff7f0e',     # Orange - storage failed only
    'montague_fail': '#d62728',    # Red - Montague failed only
    'multiple_fail': '#1f77b4',    # Blue - multiple failures
}

SATISFICING_LABELS = {
    'all_pass': 'All Criteria Met',
    'storage_fail': 'Storage Failure',
    'montague_fail': 'Montague Failure',
    'multiple_fail': 'Multiple Failures',
}


# ============================================================================
# DATA LOADING
# ============================================================================

# Drought metrics directory
DROUGHT_METRICS_DIR = "./pywrdrb/drought_metrics"


def _add_satisficing_category(df):
    """Add satisficing category based on storage and montague thresholds."""
    df['storage_pass'] = df['min_storage_pct'] >= STORAGE_THRESHOLD
    df['montague_pass'] = df['max_violation_days'] <= VIOLATION_DAYS_THRESHOLD

    def get_category(row):
        failures = []
        if not row['storage_pass']:
            failures.append('storage')
        if not row['montague_pass']:
            failures.append('montague')

        if len(failures) == 0:
            return 'all_pass'
        elif len(failures) > 1:
            return 'multiple_fail'
        elif failures[0] == 'storage':
            return 'storage_fail'
        else:
            return 'montague_fail'

    df['satisficing_category'] = df.apply(get_category, axis=1)
    return df


def load_satisficing_data(dataset_id, ssi_window):
    """
    Load pre-calculated satisficing data for drought events.

    First tries to load the detailed _during_droughts.csv file.
    If not available, loads drought events and merges with annual satisficing.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    ssi_window : int
        SSI window (3, 6, or 12)

    Returns
    -------
    pd.DataFrame
        Drought events with satisficing columns
    """
    # Try detailed drought-level satisficing first
    fname_during = f"{SATISFICING_DATA_DIR}/{dataset_id}_ssi{ssi_window}_during_droughts.csv"

    if os.path.exists(fname_during):
        print(f"  Loading: {fname_during}")
        df = pd.read_csv(fname_during)
        df = _add_satisficing_category(df)
        print(f"    Loaded {len(df)} drought events")
        return df

    # Fall back to merging drought events with annual satisficing
    fname_events = f"{DROUGHT_METRICS_DIR}/{dataset_id}_ssi{ssi_window}_drought_events.csv"
    fname_annual = f"{SATISFICING_DATA_DIR}/{dataset_id}_ssi{ssi_window}_years_with_droughts.csv"

    if not os.path.exists(fname_events):
        print(f"  Drought events not found: {fname_events}")
        return None

    if not os.path.exists(fname_annual):
        print(f"  Annual satisficing not found: {fname_annual}")
        return None

    print(f"  Loading drought events: {fname_events}")
    events_df = pd.read_csv(fname_events)

    # Extract start year from drought events
    events_df['start'] = pd.to_datetime(events_df['start'])
    events_df['year'] = events_df['start'].dt.year

    print(f"  Loading annual satisficing: {fname_annual}")
    annual_df = pd.read_csv(fname_annual)

    # Rename realization column for merge
    if 'realization' in annual_df.columns:
        annual_df = annual_df.rename(columns={'realization': 'realization_id'})

    # Merge on year and realization
    merged = events_df.merge(
        annual_df[['year', 'realization_id', 'min_storage_pct', 'max_violation_days', 'satisficing']],
        on=['year', 'realization_id'],
        how='left'
    )

    # Fill missing values (droughts in years not in satisficing data)
    merged = merged.dropna(subset=['min_storage_pct'])

    merged = _add_satisficing_category(merged)
    print(f"    Merged {len(merged)} drought events with annual satisficing")

    return merged


# ============================================================================
# PLOTTING
# ============================================================================

def plot_single_dataset_scatter(df, dataset_id, ssi_window, ax=None, show_legend=True):
    """
    Create scatter plot for a single dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Drought events with satisficing columns
    dataset_id : str
        Dataset identifier
    ssi_window : int
        SSI window
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    show_legend : bool
        Whether to show legend

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate marker sizes based on duration
    durations = df['duration'].values
    sizes = np.clip(durations * MARKER_SIZE_SCALE, MIN_MARKER_SIZE, MAX_MARKER_SIZE)

    # Plot each category
    for category, color in SATISFICING_COLORS.items():
        mask = df['satisficing_category'] == category
        if mask.sum() > 0:
            ax.scatter(
                df.loc[mask, 'severity'].abs(),  # Use absolute value for plotting
                df.loc[mask, 'magnitude'].abs(),
                s=sizes[mask],
                c=color,
                alpha=0.6,
                edgecolors='black',
                linewidths=0.5,
                label=f"{SATISFICING_LABELS[category]} ({mask.sum()})",
            )

    # Formatting
    ax.set_xlabel('Drought Severity (|min SSI|)', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Drought Magnitude (|cumulative deficit|)', fontsize=FONTSIZE_LABEL)

    dataset_label = DATASET_LABELS.get(dataset_id, dataset_id)
    ax.set_title(f'{dataset_label}\nSSI-{ssi_window} Drought Events (n={len(df)})',
                 fontsize=FONTSIZE_MEDIUM)

    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    if show_legend:
        ax.legend(loc='upper left', fontsize=FONTSIZE_SMALL, frameon=True, fancybox=True)

    return ax


def plot_multipanel_scatter(ssi_window=12, figsize=(16, 5)):
    """
    Create 3-panel figure comparing satisficing across datasets.

    Parameters
    ----------
    ssi_window : int
        SSI window
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

    all_results = {}

    for idx, dataset_id in enumerate(DATASETS):
        print(f"\nProcessing {dataset_id}...")
        df = load_satisficing_data(dataset_id, ssi_window)

        if df is not None:
            all_results[dataset_id] = df
            plot_single_dataset_scatter(
                df, dataset_id, ssi_window,
                ax=axes[idx],
                show_legend=(idx == 0)
            )
        else:
            axes[idx].text(0.5, 0.5, 'No Data',
                          transform=axes[idx].transAxes,
                          ha='center', va='center', fontsize=14)
            axes[idx].set_title(DATASET_LABELS.get(dataset_id, dataset_id))

    # Shared legend at bottom
    handles = [
        mpatches.Patch(color=color, alpha=0.6, label=SATISFICING_LABELS[cat])
        for cat, color in SATISFICING_COLORS.items()
    ]
    # Add size legend
    size_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=np.sqrt(s/np.pi), label=f'{d} months')
        for d, s in [(3, MIN_MARKER_SIZE), (12, 100), (24, MAX_MARKER_SIZE)]
    ]

    fig.legend(handles + size_legend,
               [h.get_label() for h in handles] + [l.get_label() for l in size_legend],
               loc='lower center', ncol=4, fontsize=FONTSIZE_SMALL,
               bbox_to_anchor=(0.5, -0.02), frameon=True)

    # Only show y-label on first subplot
    axes[0].set_ylabel('Drought Magnitude (|cumulative deficit|)', fontsize=FONTSIZE_LABEL)
    for ax in axes[1:]:
        ax.set_ylabel('')

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    # Save
    fname = f"{FIG_OUTPUT_DIR}/SI7_drought_satisficing_scatter_ssi{ssi_window}.png"
    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"\nSaved: {fname}")

    plt.close()

    return all_results


def plot_combined_hexbin_scatter(ssi_window=12, figsize=(14, 10)):
    """
    Create combined figure with hexbin background and satisficing overlay.

    Parameters
    ----------
    ssi_window : int
        SSI window
    figsize : tuple
        Figure size
    """
    print(f"\nGenerating combined hexbin + scatter figure for SSI-{ssi_window}...")

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Collect all data first
    all_data = []
    dataset_dfs = {}

    for dataset_id in DATASETS:
        df = load_satisficing_data(dataset_id, ssi_window)
        if df is not None:
            df['dataset_id'] = dataset_id
            all_data.append(df)
            dataset_dfs[dataset_id] = df

    if not all_data:
        print("No data available!")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # Top-left: Hexbin of all droughts (colored by count)
    ax = axes[0, 0]
    hb = ax.hexbin(
        combined_df['severity'].abs(),
        combined_df['magnitude'].abs(),
        gridsize=25,
        cmap='YlOrRd',
        mincnt=1,
        yscale='log',
    )
    cb = fig.colorbar(hb, ax=ax, shrink=0.8)
    cb.set_label('Count', fontsize=FONTSIZE_SMALL)
    ax.set_xlabel('Severity (|min SSI|)', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Magnitude (|cum. deficit|)', fontsize=FONTSIZE_LABEL)
    ax.set_title('(a) All Drought Events', fontsize=FONTSIZE_MEDIUM)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Top-right: Scatter colored by satisficing category (all datasets)
    ax = axes[0, 1]
    for category, color in SATISFICING_COLORS.items():
        mask = combined_df['satisficing_category'] == category
        if mask.sum() > 0:
            ax.scatter(
                combined_df.loc[mask, 'severity'].abs(),
                combined_df.loc[mask, 'magnitude'].abs(),
                s=30,
                c=color,
                alpha=0.5,
                label=f"{SATISFICING_LABELS[category]}",
            )
    ax.set_xlabel('Severity (|min SSI|)', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Magnitude (|cum. deficit|)', fontsize=FONTSIZE_LABEL)
    ax.set_title('(b) Satisficing Outcomes', fontsize=FONTSIZE_MEDIUM)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=FONTSIZE_SMALL - 1, frameon=True)

    # Bottom panels: Compare stationary vs climate scenarios
    # Bottom-left: Stationary only
    ax = axes[1, 0]
    if 'stationary_ensemble' in dataset_dfs:
        df = dataset_dfs['stationary_ensemble']
        for category, color in SATISFICING_COLORS.items():
            mask = df['satisficing_category'] == category
            if mask.sum() > 0:
                ax.scatter(
                    df.loc[mask, 'severity'].abs(),
                    df.loc[mask, 'magnitude'].abs(),
                    s=40,
                    c=color,
                    alpha=0.6,
                    edgecolors='black',
                    linewidths=0.3,
                )
    ax.set_xlabel('Severity (|min SSI|)', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Magnitude (|cum. deficit|)', fontsize=FONTSIZE_LABEL)
    ax.set_title(f'(c) {DATASET_LABELS.get("stationary_ensemble", "Stationary")}',
                 fontsize=FONTSIZE_MEDIUM)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Bottom-right: Climate scenarios combined
    ax = axes[1, 1]
    climate_dfs = [dataset_dfs.get(d) for d in ['climate_adjusted_low', 'climate_adjusted_high']
                   if d in dataset_dfs]
    if climate_dfs:
        climate_df = pd.concat(climate_dfs, ignore_index=True)
        for category, color in SATISFICING_COLORS.items():
            mask = climate_df['satisficing_category'] == category
            if mask.sum() > 0:
                ax.scatter(
                    climate_df.loc[mask, 'severity'].abs(),
                    climate_df.loc[mask, 'magnitude'].abs(),
                    s=40,
                    c=color,
                    alpha=0.6,
                    edgecolors='black',
                    linewidths=0.3,
                )
    ax.set_xlabel('Severity (|min SSI|)', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Magnitude (|cum. deficit|)', fontsize=FONTSIZE_LABEL)
    ax.set_title('(d) Climate-Adjusted Scenarios', fontsize=FONTSIZE_MEDIUM)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Print summary statistics
    print("\nSatisficing Summary:")
    print("-" * 60)
    for cat in SATISFICING_COLORS.keys():
        n = (combined_df['satisficing_category'] == cat).sum()
        pct = 100 * n / len(combined_df)
        print(f"  {SATISFICING_LABELS[cat]:25s}: {n:5d} ({pct:5.1f}%)")

    plt.tight_layout()

    # Save
    fname = f"{FIG_OUTPUT_DIR}/SI7_drought_satisficing_combined_ssi{ssi_window}.png"
    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"\nSaved: {fname}")

    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    ssi_window = int(sys.argv[1]) if len(sys.argv) > 1 else 12

    if ssi_window not in SSI_WINDOWS:
        print(f"ERROR: Invalid SSI window. Must be one of {SSI_WINDOWS}")
        sys.exit(1)

    print("=" * 70)
    print(f"SI7: Drought Satisficing Scatter Plot (SSI-{ssi_window})")
    print("=" * 70)

    # Generate multi-panel comparison
    plot_multipanel_scatter(ssi_window)

    # Generate combined hexbin + scatter
    plot_combined_hexbin_scatter(ssi_window)

    print("\nDone!")


if __name__ == "__main__":
    main()
